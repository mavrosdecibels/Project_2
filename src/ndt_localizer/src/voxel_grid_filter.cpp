#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

#include <fstream>
#include <ctime>

#define MAX_MEASUREMENT_RANGE 150.0

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_points_pub;

// Leaf size of VoxelGrid filter.
static double voxel_leaf_size = 2.0;

static std::ofstream ofs;
static std::string filename;

static std::string points_topic;

static pcl::PointCloud<pcl::PointXYZ> removePointsByRange(pcl::PointCloud<pcl::PointXYZ> scan, double min_range, double max_range)
{
    pcl::PointCloud<pcl::PointXYZ> narrowed_scan;
    narrowed_scan.header = scan.header;

    if (min_range >= max_range) {
        RCLCPP_ERROR_ONCE(rclcpp::get_logger("voxel_grid_filter"), "min_range >= max_range @(%lf, %lf)", min_range, max_range);
        return scan;
    }

    double square_min_range = min_range * min_range;
    double square_max_range = max_range * max_range;

    for (const auto &p : scan.points)
    {
        double square_distance = p.x * p.x + p.y * p.y;

        if (square_min_range <= square_distance && square_distance <= square_max_range)
        {
            narrowed_scan.points.push_back(p);
        }
    }

    return narrowed_scan;
}

static void scan_callback(const sensor_msgs::msg::PointCloud2::SharedPtr input)
{
    pcl::PointCloud<pcl::PointXYZ> scan;
    pcl::fromROSMsg(*input, scan);
    scan = removePointsByRange(scan, 0, MAX_MEASUREMENT_RANGE);

    pcl::PointCloud<pcl::PointXYZ>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZ>(scan));
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZ>());

    sensor_msgs::msg::PointCloud2 filtered_msg;

    // If voxel_leaf_size < 0.1 voxel_grid_filter cannot down sample (PCL specification)
    if (voxel_leaf_size >= 0.1)
    {
        // Downsampling the point cloud scan using VoxelGrid filter
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_filter;
        voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
        voxel_grid_filter.setInputCloud(scan_ptr);
        voxel_grid_filter.filter(*filtered_scan_ptr);
        pcl::toROSMsg(*filtered_scan_ptr, filtered_msg);
    }
    else
    {
        pcl::toROSMsg(*scan_ptr, filtered_msg);
    }
    filtered_msg.header = input->header;
    filtered_points_pub->publish(filtered_msg);
}

int main(int argc, char** argv)
{
    // Initialize ROS 2
    rclcpp::init(argc, argv);
    std::cout << "Voxel Grid Filter Node Started" << std::endl;

    // Create the Node
    auto node = rclcpp::Node::make_shared("voxel_grid_filter");

    // Declare parameters
    node->declare_parameter<std::string>("points_topic", "/points_raw");
    node->declare_parameter<bool>("output_log", false);
    node->declare_parameter<double>("leaf_size", 0.2);

    // Get parameters
    points_topic = node->get_parameter("points_topic").as_string();
    voxel_leaf_size = node->get_parameter("leaf_size").as_double();
    bool output_log = node->get_parameter("output_log").as_bool();

    RCLCPP_INFO(node->get_logger(), "Voxel leaf size is: %f", voxel_leaf_size);

    if (output_log)
    {
        char buffer[80];
        std::time_t now = std::time(NULL);
        std::tm *pnow = std::localtime(&now);
        std::strftime(buffer, 80, "%Y%m%d_%H%M%S", pnow);
        filename = "voxel_grid_filter_" + std::string(buffer) + ".csv";
        ofs.open(filename.c_str(), std::ios::app);
    }

    // Publishers
    filtered_points_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_points", rclcpp::SensorDataQoS());

    // Subscribers
    auto scan_sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
        points_topic, rclcpp::SensorDataQoS(), scan_callback);

    // Spin the node
    rclcpp::spin(node);

    // Shutdown ROS 2
    rclcpp::shutdown();
    return 0;
}
