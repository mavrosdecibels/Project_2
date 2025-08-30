from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Define paths for your package and files
    # ndt_localizer_path = os.path.join(
    #     get_package_share_directory('ndt_localizer'), 'map', 'pointcloud_map.pcd')
    ndt_localizer_path = "/home/bunty/wipod_ws/pcd_maps/map/pointcloud_map.pcd"
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument('roll', default_value='0.0'),
        DeclareLaunchArgument('pitch', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.0'),
        DeclareLaunchArgument('pcd_path', default_value=ndt_localizer_path),
        DeclareLaunchArgument('map_topic', default_value='/initial_map'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        # Node for Map Loader
        Node(
            package='ndt_localizer',
            executable='map_loader',
            name='map_loader',
            output='both',
            parameters=[
                {'pcd_path': LaunchConfiguration('pcd_path')},
                {'map_topic': LaunchConfiguration('map_topic')},
                {'roll': LaunchConfiguration('roll')},
                {'pitch': LaunchConfiguration('pitch')},
                {'yaw': LaunchConfiguration('yaw')},
                {'x': LaunchConfiguration('x')},
                {'y': LaunchConfiguration('y')},
                {'z': LaunchConfiguration('z')},
            ]
        ),
    ])
