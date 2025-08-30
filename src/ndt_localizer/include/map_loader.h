#ifndef _MAP_LOADER_H_
#define _MAP_LOADER_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vector>
#include <pcl_ros/transforms.hpp>

class MapLoader : public rclcpp::Node {
public:
    MapLoader(const rclcpp::NodeOptions &options);
    ~MapLoader() {}
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_map_pub_;
    std::vector<std::string> file_list_;

private:

    float tf_x_, tf_y_, tf_z_, tf_roll_, tf_pitch_, tf_yaw_; 

    void init_tf_params();
    sensor_msgs::msg::PointCloud2 CreatePcd();
    sensor_msgs::msg::PointCloud2 TransformMap(sensor_msgs::msg::PointCloud2 &in);
    void SaveMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr map_pc_ptr);
}; // MapLoader

#endif
