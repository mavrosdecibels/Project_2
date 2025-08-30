cmake_minimum_required(VERSION 3.5)
project(ndt_localizer)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# Include directories
include_directories(include ${PCL_INCLUDE_DIRS})

# Build flags
set(CMAKE_CXX_FLAGS "-O2 -g -Wall ${CMAKE_CXX_FLAGS}")

#############
## Executables ##
#############

# voxel_grid_filter node
add_executable(voxel_grid_filter src/voxel_grid_filter.cpp)
target_link_libraries(voxel_grid_filter ${PCL_LIBRARIES})
ament_target_dependencies(voxel_grid_filter
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  diagnostic_msgs
  pcl_conversions
  pcl_ros
)

# map_loader node
add_executable(map_loader src/map_loader.cpp)
target_link_libraries(map_loader ${PCL_LIBRARIES})
ament_target_dependencies(map_loader
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  diagnostic_msgs
  pcl_conversions
  pcl_ros
)

# ndt_localizer_node
add_executable(ndt_localizer_node src/ndt.cpp)
target_link_libraries(ndt_localizer_node ${PCL_LIBRARIES})
ament_target_dependencies(ndt_localizer_node
  rclcpp
  tf2
  tf2_eigen
  tf2_ros
  tf2_geometry_msgs
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  diagnostic_msgs
  pcl_conversions
  pcl_ros
)

#############
## Install ##
#############

# Install targets
install(TARGETS voxel_grid_filter map_loader ndt_localizer_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install include files
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.h"
)

# Install other directories
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
# Finalize package
ament_package()
