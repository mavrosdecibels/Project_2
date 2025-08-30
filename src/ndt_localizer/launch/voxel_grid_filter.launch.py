from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'sync', default_value='false', description='Synchronize drivers'
        ),
        DeclareLaunchArgument(
            'points_topic', default_value='/velodyne_points', description='Topic to subscribe to for points'
        ),
        DeclareLaunchArgument(
            'output_log', default_value='false', description='Enable output log'
        ),
        DeclareLaunchArgument(
            'leaf_size', default_value='0.2', description='Leaf size for the voxel grid filter'
        ),
        DeclareLaunchArgument(
            'use_sim_time', default_value='true', description='Whether to use sime time'
        ),

        # Node declaration
        Node(
            package='ndt_localizer',
            executable="voxel_grid_filter",
            name="voxel_grid_filter",
            output='screen',
            parameters=[
                {'points_topic': LaunchConfiguration('points_topic')},
                {'output_log': LaunchConfiguration('output_log')},
                {'leaf_size': LaunchConfiguration('leaf_size')}
            ],
            remappings=[
                ('/points_raw', '/sync_drivers/points_raw')
            ],
        )
    ])
