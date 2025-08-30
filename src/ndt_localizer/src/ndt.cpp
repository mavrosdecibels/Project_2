#include "ndt.h"

NdtLocalizer::NdtLocalizer(const rclcpp::NodeOptions &options) : rclcpp::Node("ndt_localizer", options)
{
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
    tf2_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    key_value_stdmap_["state"] = "Initializing";
    init_params();

    // Publishers
    sensor_aligned_pose_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("points_aligned", 10);
    ndt_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("ndt_pose", 10);
    exe_time_pub_ = this->create_publisher<std_msgs::msg::Float32>("exe_time_ms", 10);
    transform_probability_pub_ = this->create_publisher<std_msgs::msg::Float32>("transform_probability", 10);
    iteration_num_pub_ = this->create_publisher<std_msgs::msg::Float32>("iteration_num", 10);
    diagnostics_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("diagnostics", 10);
    
    // Subscribers
    initial_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "initialpose", 100, std::bind(&NdtLocalizer::callback_init_pose, this, std::placeholders::_1));
    map_points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "points_map", 1, std::bind(&NdtLocalizer::callback_pointsmap, this, std::placeholders::_1));
    sensor_points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "filtered_points", rclcpp::SensorDataQoS(), std::bind(&NdtLocalizer::callback_pointcloud, this, std::placeholders::_1));

    diagnostic_thread_ = std::thread(&NdtLocalizer::timer_diagnostic, this);
    diagnostic_thread_.detach();
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
    if (initial_pose_msg_ptr->header.frame_id == map_frame_) {
        initial_pose_cov_msg_ = *initial_pose_msg_ptr;
    } else {
        geometry_msgs::msg::TransformStamped TF_pose_to_map;
        get_transform(map_frame_, initial_pose_msg_ptr->header.frame_id, TF_pose_to_map);

        geometry_msgs::msg::PoseWithCovarianceStamped mapTF_initial_pose_msg;
        tf2::doTransform(*initial_pose_msg_ptr, mapTF_initial_pose_msg, TF_pose_to_map);
        initial_pose_cov_msg_ = mapTF_initial_pose_msg;
    }
    init_pose_ = false;
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

    const float transform_probability = ndt_.getTransformationProbability();
    const int iteration_num = ndt_.getFinalNumIteration();

    bool is_converged = true;
    static size_t skipping_publish_num = 0;

    if (iteration_num >= ndt_.getMaximumIterations() + 2 || transform_probability < converged_param_transform_probability_) {
        is_converged = false;
        ++skipping_publish_num;
        RCLCPP_WARN(this->get_logger(), "NDT did not converge!");
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

    // Update the previous transformation
    pre_trans_ = result_pose_matrix;

    // Publish NDT pose if converged
    geometry_msgs::msg::PoseStamped result_pose_stamped_msg;
    result_pose_stamped_msg.header.stamp = sensor_ros_time;
    result_pose_stamped_msg.header.frame_id = map_frame_;
    result_pose_stamped_msg.pose = result_pose_msg;

    if (is_converged) {
        ndt_pose_pub_->publish(result_pose_stamped_msg);
    }

    // Publish transform (base frame to map frame)
    publish_tf(map_frame_, base_frame_, result_pose_stamped_msg);

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

    // Print out the parameters
    RCLCPP_INFO(this->get_logger(), "trans_epsilon: %f", trans_epsilon_);
    RCLCPP_INFO(this->get_logger(), "step_size: %f", step_size_);
    RCLCPP_INFO(this->get_logger(), "resolution: %f", resolution_);
    RCLCPP_INFO(this->get_logger(), "max_iterations: %d", max_iterations_);

    map_frame_ = "map";

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