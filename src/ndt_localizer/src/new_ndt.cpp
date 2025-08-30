#include "ndt.h"

NdtLocalizer::NdtLocalizer(const rclcpp::NodeOptions &options) : rclcpp::Node("ndt_localizer", options),
last_init_time_(this->now() - rclcpp::Duration(100, 0))
{
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
    tf2_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    key_value_stdmap_["state"] = "Initializing";
    init_params();

    // Publishers
    sensor_aligned_pose_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("points_aligned", 10);
    ndt_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("ndt_pose", 10);
    ndt_pose_cov_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("ndt_pose_with_cov", 10);
    exe_time_pub_ = this->create_publisher<std_msgs::msg::Float32>("exe_time_ms", 10);
    transform_probability_pub_ = this->create_publisher<std_msgs::msg::Float32>("transform_probability", 10);
    iteration_num_pub_ = this->create_publisher<std_msgs::msg::Float32>("iteration_num", 10);
    diagnostics_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("diagnostics", 10);
    tf_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(33),  // ~30 Hz
        [this]() { this->timer_tf_callback(); }
      );
      
    create_initialpose_subscription();

    // Subscribers
    // initial_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        // "initialpose", 100, std::bind(&NdtLocalizer::callback_init_pose, this, std::placeholders::_1));
    map_points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "points_map", 1, std::bind(&NdtLocalizer::callback_pointsmap, this, std::placeholders::_1));
    sensor_points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "filtered_points", rclcpp::SensorDataQoS(), std::bind(&NdtLocalizer::callback_pointcloud, this, std::placeholders::_1));

    diagnostic_thread_ = std::thread(&NdtLocalizer::timer_diagnostic, this);
    diagnostic_thread_.detach();
}

void NdtLocalizer::create_initialpose_subscription() {
    std::lock_guard<std::mutex> lock(sub_mutex_);
    if (!subscription_active_) {
        initial_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "initialpose", 100, std::bind(&NdtLocalizer::callback_init_pose, this, std::placeholders::_1));
        subscription_active_ = true;
        RCLCPP_INFO(this->get_logger(), "Initial pose subscription created");
    }
}

void NdtLocalizer::destroy_initialpose_subscription() {
    std::lock_guard<std::mutex> lock(sub_mutex_);
    if (subscription_active_) {
        initial_pose_sub_.reset();
        subscription_active_ = false;
        RCLCPP_INFO(this->get_logger(), "Initial pose subscription destroyed");
    }
}

void NdtLocalizer::timer_diagnostic()
{
    rclcpp::Rate rate(100);
    while (rclcpp::ok()) {
        diagnostic_msgs::msg::DiagnosticStatus diag_status_msg;
        diag_status_msg.name = "ndt_scan_matcher";
        diag_status_msg.hardware_id = "";

        for (const auto &key_value : key_value_stdmap_) {
            diagnostic_msgs::msg::KeyValue key_value_msg;
            key_value_msg.key = key_value.first;
            key_value_msg.value = key_value.second;
            diag_status_msg.values.push_back(key_value_msg);
        }

        diag_status_msg.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        diag_status_msg.message = "";
        if (key_value_stdmap_.count("state") && key_value_stdmap_["state"] == "Initializing") {
            diag_status_msg.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            diag_status_msg.message += "Initializing State. ";
        }
        if (key_value_stdmap_.count("skipping_publish_num") &&
            std::stoi(key_value_stdmap_["skipping_publish_num"]) > 1) {
            diag_status_msg.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            diag_status_msg.message += "skipping_publish_num > 1. ";
        }
        if (key_value_stdmap_.count("skipping_publish_num") &&
            std::stoi(key_value_stdmap_["skipping_publish_num"]) >= 5) {
            diag_status_msg.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
            diag_status_msg.message += "skipping_publish_num exceed limit. ";
        }

        diagnostic_msgs::msg::DiagnosticArray diag_msg;
        diag_msg.header.stamp = this->now();
        diag_msg.status.push_back(diag_status_msg);

        diagnostics_pub_->publish(diag_msg);

        rate.sleep();
    }
}

void NdtLocalizer::callback_init_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr initial_pose_msg_ptr)
{
    // Process the message
    if (initial_pose_msg_ptr->header.frame_id == map_frame_) {
        initial_pose_cov_msg_ = *initial_pose_msg_ptr;
    } else {
        geometry_msgs::msg::TransformStamped TF_pose_to_map;
        get_transform(map_frame_, initial_pose_msg_ptr->header.frame_id, TF_pose_to_map);
        geometry_msgs::msg::PoseWithCovarianceStamped mapTF_initial_pose_msg;
        tf2::doTransform(*initial_pose_msg_ptr, mapTF_initial_pose_msg, TF_pose_to_map);
        initial_pose_cov_msg_ = mapTF_initial_pose_msg;
    }

    // Destroy subscription after processing
    destroy_initialpose_subscription();
    last_init_time_ = this->now();
    RCLCPP_INFO(this->get_logger(), "Processed initial pose, subscription destroyed");
}

void NdtLocalizer::callback_pointsmap(const sensor_msgs::msg::PointCloud2::SharedPtr map_points_msg_ptr)
{
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_new;

    ndt_new.setTransformationEpsilon(trans_epsilon_);
    ndt_new.setStepSize(step_size_);
    ndt_new.setResolution(resolution_);
    ndt_new.setMaximumIterations(max_iterations_);

    pcl::PointCloud<pcl::PointXYZ>::Ptr map_points_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*map_points_msg_ptr, *map_points_ptr);
    ndt_new.setInputTarget(map_points_ptr);

    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    ndt_new.align(*output_cloud, Eigen::Matrix4f::Identity());

    ndt_map_mtx_.lock();
    ndt_ = ndt_new;
    ndt_map_mtx_.unlock();
}

void NdtLocalizer::timer_tf_callback() {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    if (latest_pose_.header.stamp.sec == 0) return;  // Skip if no valid pose
    publish_tf(map_frame_, base_frame_, latest_pose_);
  }

void NdtLocalizer::callback_pointcloud(
    const sensor_msgs::msg::PointCloud2::SharedPtr sensor_points_sensorTF_msg_ptr)
{
    const auto exe_start_time = std::chrono::system_clock::now();

    // mutex lock for Map
    std::lock_guard<std::mutex> lock(ndt_map_mtx_);

    const std::string sensor_frame = sensor_points_sensorTF_msg_ptr->header.frame_id;
    const auto sensor_ros_time = sensor_points_sensorTF_msg_ptr->header.stamp;

    // Convert PointCloud2 message to pcl PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr sensor_points_sensorTF_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*sensor_points_sensorTF_msg_ptr, *sensor_points_sensorTF_ptr);

    // Get TF base to sensor
    geometry_msgs::msg::TransformStamped TF_base_to_sensor;
    if (!get_transform(base_frame_, sensor_frame, TF_base_to_sensor)) {
        RCLCPP_WARN(this->get_logger(), "Failed to get transform from %s to %s", base_frame_.c_str(), sensor_frame.c_str());
        return;
    }

    // Transform sensor points to base_link frame
    const Eigen::Affine3d base_to_sensor_affine = tf2::transformToEigen(TF_base_to_sensor);
    const Eigen::Matrix4f base_to_sensor_matrix = base_to_sensor_affine.matrix().cast<float>();
    pcl::PointCloud<pcl::PointXYZ>::Ptr sensor_points_baselinkTF_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*sensor_points_sensorTF_ptr, *sensor_points_baselinkTF_ptr, base_to_sensor_matrix);

    // Set input point cloud to NDT
    ndt_.setInputSource(sensor_points_baselinkTF_ptr);

    if (ndt_.getInputTarget() == nullptr) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "No MAP!");
        return;
    }

    // Align the point cloud
    Eigen::Matrix4f initial_pose_matrix;
    if (!init_pose_) {
        Eigen::Affine3d initial_pose_affine;
        tf2::fromMsg(initial_pose_cov_msg_.pose.pose, initial_pose_affine);
        initial_pose_matrix = initial_pose_affine.matrix().cast<float>();

        // Set the initial transformation matrix
        pre_trans_ = initial_pose_matrix;
        init_pose_ = true;
    } else {
        // Use predicted pose as initial guess
        initial_pose_matrix = pre_trans_ * delta_trans_;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    const auto align_start_time = std::chrono::system_clock::now();
    key_value_stdmap_["state"] = "Aligning";

    // Perform alignment using NDT
    ndt_.align(*output_cloud, initial_pose_matrix);

    key_value_stdmap_["state"] = "Sleeping";
    const auto align_end_time = std::chrono::system_clock::now();
    const double align_time = std::chrono::duration_cast<std::chrono::microseconds>(align_end_time - align_start_time).count() / 1000.0;

    // Retrieve the result of the alignment
    const Eigen::Matrix4f result_pose_matrix = ndt_.getFinalTransformation();
    Eigen::Affine3d result_pose_affine;
    result_pose_affine.matrix() = result_pose_matrix.cast<double>();
    const geometry_msgs::msg::Pose result_pose_msg = tf2::toMsg(result_pose_affine);

    const auto exe_end_time = std::chrono::system_clock::now();
    const double exe_time = std::chrono::duration_cast<std::chrono::microseconds>(exe_end_time - exe_start_time).count() / 1000.0;

    double fitness_score = ndt_.getFitnessScore(); 
    const float transform_probability = ndt_.getTransformationProbability();
    const int iteration_num = ndt_.getFinalNumIteration();

    bool is_converged = true;
    static size_t skipping_publish_num = 0;

    if (iteration_num >= ndt_.getMaximumIterations() + 2 || transform_probability < converged_param_transform_probability_) {
        is_converged = false;
        ++skipping_publish_num;
        RCLCPP_WARN(this->get_logger(), "NDT did not converge!");

        if (transform_probability < converged_param_transform_probability_) {
            const auto time_since_last_init = (this->now() - last_init_time_).seconds();
            if (time_since_last_init >= reinit_delay_duration_) {
                create_initialpose_subscription();
                RCLCPP_INFO(this->get_logger(), "Recreated initial pose subscription for relocalization");
            }
        }
    
        
    } else {
        skipping_publish_num = 0;
    }

    // Calculate the delta transformation from the previous transformation to the current transformation
    delta_trans_ = pre_trans_.inverse() * result_pose_matrix;
    Eigen::Vector3f delta_trans_lation = delta_trans_.block<3, 1>(0, 3);
    RCLCPP_INFO(this->get_logger(), "Delta x: %f, y: %f, z: %f", delta_trans_lation(0), delta_trans_lation(1), delta_trans_lation(2));

    Eigen::Matrix3f delta_rotation_matrix = delta_trans_.block<3, 3>(0, 0);
    Eigen::Vector3f delta_euler = delta_rotation_matrix.eulerAngles(2, 1, 0);
    RCLCPP_INFO(this->get_logger(), "Delta yaw: %f, pitch: %f, roll: %f", delta_euler(0), delta_euler(1), delta_euler(2));

    // After getting result_pose_matrix (3D):
    Eigen::Affine3d result_pose_affine_3d;
    result_pose_affine_3d.matrix() = result_pose_matrix.cast<double>();

    // Update the previous transformation
    pre_trans_ = result_pose_matrix;

    Eigen::Matrix3d R = result_pose_affine_3d.rotation().matrix();
    double yaw = std::atan2(R(1, 0), R(0, 0)); // Corrected line
    Eigen::Affine3d result_pose_affine_2d = Eigen::Affine3d::Identity();
    result_pose_affine_2d.translation().x() = result_pose_affine_3d.translation().x();
    result_pose_affine_2d.translation().y() = result_pose_affine_3d.translation().y();
    result_pose_affine_2d.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    

    // Publish NDT pose if converged
    geometry_msgs::msg::PoseStamped result_pose_stamped_msg;
    result_pose_stamped_msg.header.stamp = sensor_ros_time;
    result_pose_stamped_msg.header.frame_id = map_frame_;
    result_pose_stamped_msg.pose = tf2::toMsg(result_pose_affine_2d);

    // // Compute residuals for covariance estimation
    std::vector<Eigen::Vector3f> residuals;
    for (const auto& point : output_cloud->points) {
        Eigen::Vector3f transformed_pt(point.x, point.y, point.z);
        residuals.push_back(transformed_pt);
    }
    Eigen::MatrixXf J(residuals.size() * 3, 6);
    for (size_t i = 0; i < residuals.size(); ++i) {
        float x = residuals[i](0);
        float y = residuals[i](1);
        float z = residuals[i](2);
        J.block<3, 1>(i * 3, 0) << 1, 0, 0;
        J.block<3, 1>(i * 3, 1) << 0, 1, 0;
        J.block<3, 1>(i * 3, 2) << 0, 0, 1;
        J.block<3, 1>(i * 3, 3) << 0, z, -y;
        J.block<3, 1>(i * 3, 4) << -z, 0, x;
        J.block<3, 1>(i * 3, 5) << y, -x, 0;
    }
    Eigen::Matrix<float, 6, 6> JTJ = J.transpose() * J;
    Eigen::Matrix<float, 6, 6> covariance_matrix = JTJ.inverse();

    covariance_matrix(2, 2) = 1e6;
    covariance_matrix(3, 3) = 1e6;
    covariance_matrix(4, 4) = 1e6;
 
    // Convert to ROS PoseWithCovarianceStamped message
    geometry_msgs::msg::PoseWithCovarianceStamped result_pose_cov_msg;
    result_pose_cov_msg.header.stamp = sensor_ros_time;
    result_pose_cov_msg.header.frame_id = map_frame_;
    result_pose_cov_msg.pose.pose = tf2::toMsg(result_pose_affine_2d);

    
    // Fill covariance matrix
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            result_pose_cov_msg.pose.covariance[i * 6 + j] = covariance_matrix(i, j);
        }
    }


    if (is_converged) {
        ndt_pose_pub_->publish(result_pose_stamped_msg);
        ndt_pose_cov_pub_->publish(result_pose_cov_msg);
    }

    

    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        latest_pose_ = result_pose_stamped_msg;
    }
    // Publish transform (base frame to map frame)
    // publish_tf(map_frame_, odom_frame_, result_pose_stamped_msg);

   

    // Publish aligned point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr sensor_points_mapTF_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*sensor_points_baselinkTF_ptr, *sensor_points_mapTF_ptr, result_pose_matrix);
    sensor_msgs::msg::PointCloud2 sensor_points_mapTF_msg;
    pcl::toROSMsg(*sensor_points_mapTF_ptr, sensor_points_mapTF_msg);
    sensor_points_mapTF_msg.header.stamp = sensor_ros_time;
    sensor_points_mapTF_msg.header.frame_id = map_frame_;
    sensor_aligned_pose_pub_->publish(sensor_points_mapTF_msg);

    // Publish execution time
    std_msgs::msg::Float32 exe_time_msg;
    exe_time_msg.data = exe_time;
    exe_time_pub_->publish(exe_time_msg);

    // Publish transform probability
    std_msgs::msg::Float32 transform_probability_msg;
    transform_probability_msg.data = transform_probability;
    transform_probability_pub_->publish(transform_probability_msg);

    // Publish iteration number
    std_msgs::msg::Float32 iteration_num_msg;
    iteration_num_msg.data = iteration_num;
    iteration_num_pub_->publish(iteration_num_msg);

    // Update diagnostics data
    key_value_stdmap_["seq"] = std::to_string(sensor_points_sensorTF_msg_ptr->header.stamp.nanosec);  // seq field is no longer available in ROS 2
    key_value_stdmap_["transform_probability"] = std::to_string(transform_probability);
    key_value_stdmap_["iteration_num"] = std::to_string(iteration_num);
    key_value_stdmap_["skipping_publish_num"] = std::to_string(skipping_publish_num);

    RCLCPP_INFO(this->get_logger(), "------------------------------------------------");
    RCLCPP_INFO(this->get_logger(), "align_time: %f ms", align_time);
    RCLCPP_INFO(this->get_logger(), "exe_time: %f ms", exe_time);
    RCLCPP_INFO(this->get_logger(), "transform_probability: %f", transform_probability);
    RCLCPP_INFO(this->get_logger(), "iteration_num: %d", iteration_num);
    RCLCPP_INFO(this->get_logger(), "skipping_publish_num: %lu", skipping_publish_num);
}

void NdtLocalizer::init_params()
{
    this->declare_parameter<std::string>("base_frame", "base_link");
    base_frame_ = this->get_parameter("base_frame").as_string();
    RCLCPP_INFO(this->get_logger(), "base_frame: %s", base_frame_.c_str());

    this->declare_parameter<double>("trans_epsilon", ndt_.getTransformationEpsilon());
    this->declare_parameter<double>("step_size", ndt_.getStepSize());
    this->declare_parameter<double>("resolution", ndt_.getResolution());
    this->declare_parameter<int>("max_iterations", ndt_.getMaximumIterations());

    trans_epsilon_ = this->get_parameter("trans_epsilon").as_double();
    step_size_ = this->get_parameter("step_size").as_double();
    resolution_ = this->get_parameter("resolution").as_double();
    max_iterations_ = this->get_parameter("max_iterations").as_int();
    
    this->declare_parameter<double>("reinit_delay_duration", 5.0); // Default 5 seconds
    reinit_delay_duration_ = this->get_parameter("reinit_delay_duration").as_double();
    RCLCPP_INFO(this->get_logger(), "reinit_delay_duration: %f", reinit_delay_duration_);

    // Print out the parameters
    RCLCPP_INFO(this->get_logger(), "trans_epsilon: %f", trans_epsilon_);
    RCLCPP_INFO(this->get_logger(), "step_size: %f", step_size_);
    RCLCPP_INFO(this->get_logger(), "resolution: %f", resolution_);
    RCLCPP_INFO(this->get_logger(), "max_iterations: %d", max_iterations_);

    map_frame_ = "map";
    odom_frame_ = "odom";

    ndt_.setTransformationEpsilon(trans_epsilon_);
    ndt_.setStepSize(step_size_);
    ndt_.setResolution(resolution_);
    ndt_.setMaximumIterations(max_iterations_);

    this->declare_parameter<double>("converged_param_transform_probability", 3.0);
    converged_param_transform_probability_ = this->get_parameter("converged_param_transform_probability").as_double();
}


bool NdtLocalizer::get_transform(const std::string &target_frame, const std::string &source_frame,
                    geometry_msgs::msg::TransformStamped &transform_stamped)
{
    if (target_frame == source_frame) {
        transform_stamped.header.stamp = this->now();
        transform_stamped.header.frame_id = target_frame;
        transform_stamped.child_frame_id = source_frame;
        transform_stamped.transform.translation.x = 0.0;
        transform_stamped.transform.translation.y = 0.0;
        transform_stamped.transform.translation.z = 0.0;
        transform_stamped.transform.rotation.x = 0.0;
        transform_stamped.transform.rotation.y = 0.0;
        transform_stamped.transform.rotation.z = 0.0;
        transform_stamped.transform.rotation.w = 1.0;
        return true;
    }

    try {
        transform_stamped = tf2_buffer_->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "%s", ex.what());
        return false;
    }
    return true;
}

// void NdtLocalizer::publish_tf(const std::string &frame_id, const std::string &child_frame_id,
//                 const geometry_msgs::msg::PoseStamped &pose_msg)
// {
    
    
    
//     geometry_msgs::msg::TransformStamped transform_stamped;
//     transform_stamped.header.frame_id = frame_id;
//     transform_stamped.child_frame_id = child_frame_id;
//     transform_stamped.header.stamp = pose_msg.header.stamp;
//     // transform_stamped.transform.translation.x = pose_msg.pose.position.x;
//     // transform_stamped.transform.translation.y = pose_msg.pose.position.y;
//     // transform_stamped.transform.translation.z = pose_msg.pose.position.z;
//     // transform_stamped.transform.rotation = pose_msg.pose.orientation;
//     // Convert pose msg to transform msg
//     geometry_msgs::msg::Transform transform;
//     transform.translation.x = pose_msg.pose.position.x;
//     transform.translation.y = pose_msg.pose.position.y;
//     transform.translation.z = pose_msg.pose.position.z;
//     transform.rotation = pose_msg.pose.orientation;

//     geometry_msgs::msg::TransformStamped odom_to_base_tf;
//     try {
//         odom_to_base_tf = tf2_buffer_->lookupTransform(
//             odom_frame_, base_frame_, tf2::TimePointZero);
//     } catch (tf2::TransformException &ex) {
//         RCLCPP_WARN(this->get_logger(), "Could not lookup odom to base transform: %s", ex.what());
//         return;
//     }

//     // Convert pose to transform

//     Eigen::Isometry3d transform_eigen = tf2::transformToEigen(transform);

//     Eigen::Isometry3d odom_to_base = tf2::transformToEigen(odom_to_base_tf.transform);

//     // Invert the transform to go from base_link to map
//     Eigen::Isometry3d inverted_transform = transform_eigen * odom_to_base.inverse();

//     // Set the inverted transform's translation
//     transform_stamped.transform.translation.x = inverted_transform.translation().x();
//     transform_stamped.transform.translation.y = inverted_transform.translation().y();
//     transform_stamped.transform.translation.z = inverted_transform.translation().z();

//     // Set the inverted transform's rotation
//     Eigen::Quaterniond quat(inverted_transform.rotation());
//     transform_stamped.transform.rotation.x = quat.x();
//     transform_stamped.transform.rotation.y = quat.y();
//     transform_stamped.transform.rotation.z = quat.z();
//     transform_stamped.transform.rotation.w = quat.w();

    

//     tf2_broadcaster_->sendTransform(transform_stamped);

//     // geometry_msgs::msg::TransformStamped map_to_odom;
//     // map_to_odom.header.stamp = pose_msg.header.stamp;
//     // map_to_odom.header.frame_id = map_frame_;
//     // map_to_odom.child_frame_id = odom_frame_;
//     // map_to_odom.transform.translation.x = 0.0;
//     // map_to_odom.transform.translation.y = 0.0;
//     // map_to_odom.transform.translation.z = 0.0;
//     // map_to_odom.transform.rotation.x = 0.0;
//     // map_to_odom.transform.rotation.y = 0.0;
//     // map_to_odom.transform.rotation.z = 0.0;
//     // map_to_odom.transform.rotation.w = 1.0;  // Identity rotation

//     // tf2_broadcaster_->sendTransform(map_to_odom);
// }
void NdtLocalizer::publish_tf(const std::string &frame_id, const std::string &child_frame_id,
    const geometry_msgs::msg::PoseStamped &pose_msg)
{
geometry_msgs::msg::TransformStamped transform_stamped;
transform_stamped.header.frame_id = frame_id;
transform_stamped.child_frame_id = child_frame_id;
transform_stamped.header.stamp = pose_msg.header.stamp;
// transform_stamped.transform.translation.x = pose_msg.pose.position.x;
// transform_stamped.transform.translation.y = pose_msg.pose.position.y;
// transform_stamped.transform.translation.z = pose_msg.pose.position.z;
// transform_stamped.transform.rotation = pose_msg.pose.orientation;
// Convert pose msg to transform msg
geometry_msgs::msg::Transform transform;
transform.translation.x = pose_msg.pose.position.x;
transform.translation.y = pose_msg.pose.position.y;
transform.translation.z = pose_msg.pose.position.z;
transform.rotation = pose_msg.pose.orientation;
// Convert pose to transform
Eigen::Isometry3d transform_eigen = tf2::transformToEigen(transform);

// Invert the transform to go from base_link to map
Eigen::Isometry3d inverted_transform = transform_eigen;//.inverse();

// Set the inverted transform's translation
transform_stamped.transform.translation.x = inverted_transform.translation().x();
transform_stamped.transform.translation.y = inverted_transform.translation().y();
transform_stamped.transform.translation.z = inverted_transform.translation().z();

// Set the inverted transform's rotation
Eigen::Quaterniond quat(inverted_transform.rotation());
transform_stamped.transform.rotation.x = quat.x();
transform_stamped.transform.rotation.y = quat.y();
transform_stamped.transform.rotation.z = quat.z();
transform_stamped.transform.rotation.w = quat.w();

tf2_broadcaster_->sendTransform(transform_stamped);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    auto ndt_localizer_node = std::make_shared<NdtLocalizer>(node_options);
    rclcpp::spin(ndt_localizer_node);
    rclcpp::shutdown();
    return 0;
}