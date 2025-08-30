#include "map_loader.h"

MapLoader::MapLoader(const rclcpp::NodeOptions &options) : rclcpp::Node("map_loader", options) {
    // Declare and get parameters
    std::string pcd_file_path = this->get_parameter("pcd_path").as_string();
    std::string map_topic = this->get_parameter("map_topic").as_string();
    // Dump the parameters
    RCLCPP_INFO(this->get_logger(), "PCD file path: %s", pcd_file_path.c_str());
    RCLCPP_INFO(this->get_logger(), "Map topic: %s", map_topic.c_str());

    RCLCPP_INFO(this->get_logger(), "Map loader node started");

    init_tf_params();

    // Create publisher
    pc_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(map_topic, rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());

    // Store PCD file path in the file list
    file_list_.push_back(pcd_file_path);

    // Create and publish transformed map
    auto pc_msg = CreatePcd();
    auto out_msg = TransformMap(pc_msg);

    if (out_msg.width != 0) {
        out_msg.header.frame_id = "map";
        // Publish the transformed map
        RCLCPP_INFO(this->get_logger(), "Publishing transformed map with width %d", out_msg.width);
        pc_map_pub_->publish(out_msg);
    }
}

void MapLoader::init_tf_params() {
    // Get parameters
    double  tf_x_ = this->get_parameter("x").as_double();
    double  tf_y_ = this->get_parameter("y").as_double();
    double  tf_z_ = this->get_parameter("z").as_double();
    double  tf_roll_ = this->get_parameter("roll").as_double();
    double  tf_pitch_ = this->get_parameter("pitch").as_double();
    double  tf_yaw_ = this->get_parameter("yaw").as_double();

    RCLCPP_INFO(this->get_logger(), "x: %f, y: %f, z: %f, roll: %f, pitch: %f, yaw: %f",
                tf_x_, tf_y_, tf_z_, tf_roll_, tf_pitch_, tf_yaw_);
}

sensor_msgs::msg::PointCloud2 MapLoader::TransformMap(sensor_msgs::msg::PointCloud2 &in) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr in_pc(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(in, *in_pc);

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_pc_ptr(new pcl::PointCloud<pcl::PointXYZ>);

    Eigen::Translation3f tl_m2w(tf_x_, tf_y_, tf_z_);
    Eigen::AngleAxisf rot_x_m2w(tf_roll_, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf rot_y_m2w(tf_pitch_, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rot_z_m2w(tf_yaw_, Eigen::Vector3f::UnitZ());
    Eigen::Matrix4f tf_m2w = (tl_m2w * rot_z_m2w * rot_y_m2w * rot_x_m2w).matrix();

    pcl::transformPointCloud(*in_pc, *transformed_pc_ptr, tf_m2w);

    SaveMap(transformed_pc_ptr);

    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*transformed_pc_ptr, output_msg);
    return output_msg;
}

void MapLoader::SaveMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr map_pc_ptr) {
    pcl::io::savePCDFile("/tmp/transformed_map.pcd", *map_pc_ptr);
}

sensor_msgs::msg::PointCloud2 MapLoader::CreatePcd() {
    sensor_msgs::msg::PointCloud2 pcd, part;
    for (const std::string& path : file_list_) {
        if (pcd.width == 0) {
            if (pcl::io::loadPCDFile(path.c_str(), pcd) == -1) {
                std::cerr << "load failed " << path << std::endl;
            }
        } else {
            if (pcl::io::loadPCDFile(path.c_str(), part) == -1) {
                std::cerr << "load failed " << path << std::endl;
            }
            pcd.width += part.width;
            pcd.row_step += part.row_step;
            pcd.data.insert(pcd.data.end(), part.data.begin(), part.data.end());
        }
        std::cout << "load " << path << "successfully" << std::endl;
    }

    return pcd;
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.allow_undeclared_parameters(true);
    node_options.automatically_declare_parameters_from_overrides(true);
    rclcpp::Node::SharedPtr map_loader = std::make_shared<MapLoader>(node_options);
    rclcpp::spin(map_loader);
    rclcpp::shutdown();
    return 0;
}
