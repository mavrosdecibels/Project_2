#pragma once

#include <chrono>
#include <mutex>
#include <thread>
#include <sstream>
#include <string>
#include <rclcpp/rclcpp.hpp>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float32.hpp>

#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/transform_listener.h>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/ndt.h>
#include <pcl_ros/transforms.hpp>

class NdtLocalizer : public rclcpp::Node {
public:
    NdtLocalizer(const rclcpp::NodeOptions &options);
    ~NdtLocalizer() {};

private:
    // Subscriptions
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_points_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sensor_points_sub_;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr sensor_aligned_pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ndt_pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr ndt_pose_cov_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr exe_time_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr transform_probability_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr iteration_num_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
    geometry_msgs::msg::PoseStamped latest_pose_;
    std::mutex pose_mutex_;
    rclcpp::TimerBase::SharedPtr tf_timer_;


    // NDT algorithm
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_;

    // TF2 components
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf2_broadcaster_;

    // Transform matrices
    Eigen::Matrix4f base_to_sensor_matrix_;
    Eigen::Matrix4f pre_trans_, delta_trans_;
    bool init_pose_ = false;       // Flag for the first initial pose
    bool allow_reinitialization_ = false; // Flag to allow reinitialization
    rclcpp::Time last_init_time_;        // Timestamp of last initial pose
    double reinit_delay_duration_;       // Delay (seconds) before allowing reinit

    // Frame names
    std::string base_frame_;
    std::string map_frame_;
    std::string odom_frame_;

    // NDT Parameters
    double trans_epsilon_;
    double step_size_;
    double resolution_;
    int max_iterations_;

    // Initial guess for NDT
    geometry_msgs::msg::PoseWithCovarianceStamped initial_pose_cov_msg_;

    std::mutex ndt_map_mtx_;

    // Convergence parameters
    double converged_param_transform_probability_;
    std::thread diagnostic_thread_;
    std::map<std::string, std::string> key_value_stdmap_;

    std::mutex sub_mutex_; // Mutex for subscription control
    bool subscription_active_ = false;
    void create_initialpose_subscription();
    void destroy_initialpose_subscription();    

    // Function prototypes
    void init_params();
    void timer_diagnostic();

    bool get_transform(const std::string &target_frame, const std::string &source_frame,
                       geometry_msgs::msg::TransformStamped &transform_stamped,
                       const rclcpp::Time &time_stamp);
    bool get_transform(const std::string &target_frame, const std::string &source_frame,
                       geometry_msgs::msg::TransformStamped &transform_stamped);
    void publish_tf(const std::string &frame_id, const std::string &child_frame_id,
                    const geometry_msgs::msg::PoseStamped &pose_msg);

    void callback_pointsmap(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud2_msg_ptr);
    void callback_init_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose_conv_msg_ptr);
    void callback_pointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud2_msg_ptr);
    void timer_tf_callback();
};
