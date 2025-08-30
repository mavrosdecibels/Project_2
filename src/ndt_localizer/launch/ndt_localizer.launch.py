from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Define the paths for included launch files
    ndt_localizer_launch_dir = get_package_share_directory('ndt_localizer')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument('input_sensor_points_topic', default_value='/filtered_points', description='Sensor points topic'),
        DeclareLaunchArgument('input_initial_pose_topic', default_value='/initialpose', description='Initial position topic to align'),
        DeclareLaunchArgument('input_map_points_topic', default_value='/initial_map', description='Map points topic'),
        DeclareLaunchArgument('output_pose_topic', default_value='ndt_pose', description='Estimated self position'),
        DeclareLaunchArgument('output_pose_with_covariance_topic', default_value='ndt_pose_with_covariance', description='Estimated self position with covariance'),
        DeclareLaunchArgument('output_diagnostics_topic', default_value='diagnostics', description='Diagnostic topic'),
        DeclareLaunchArgument('base_frame', default_value='base_link', description='Vehicle reference frame'),
        DeclareLaunchArgument('trans_epsilon', default_value='0.01', description='The maximum difference between two consecutive transformations for convergence'),
        DeclareLaunchArgument('step_size', default_value='0.1', description='The newton line search maximum step length'),
        DeclareLaunchArgument('resolution', default_value='3.0', description='The ND voxel grid resolution'),
        DeclareLaunchArgument('max_iterations', default_value='30', description='The number of iterations required to calculate alignment'),
        DeclareLaunchArgument('converged_param_transform_probability', default_value='3.0', description='Convergence probability parameter'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        # Include map_loader launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(ndt_localizer_launch_dir, 'launch', 'map_loader.launch.py')
            )
        ),
        
        # Include voxel_grid_filter launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(ndt_localizer_launch_dir, 'launch', 'voxel_grid_filter.launch.py')
            )
        ),

        # NDT Localizer node
        Node(
            package='ndt_localizer',
            executable='ndt_localizer_node',
            name='ndt_localizer_node',
            output='screen',
            parameters=[
                {'base_frame': LaunchConfiguration('base_frame')},
                {'trans_epsilon': LaunchConfiguration('trans_epsilon')},
                {'step_size': LaunchConfiguration('step_size')},
                {'resolution': LaunchConfiguration('resolution')},
                {'max_iterations': LaunchConfiguration('max_iterations')},
                {'converged_param_transform_probability': LaunchConfiguration('converged_param_transform_probability')},
            ],
            remappings=[
                ('filtered_points', LaunchConfiguration('input_sensor_points_topic')),
                ('initialpose', LaunchConfiguration('input_initial_pose_topic')),
                ('points_map', LaunchConfiguration('input_map_points_topic')),
                ('ndt_pose', LaunchConfiguration('output_pose_topic')),
                ('ndt_pose_with_covariance', LaunchConfiguration('output_pose_with_covariance_topic')),
                ('diagnostics', LaunchConfiguration('output_diagnostics_topic'))
            ]
        ),
    ])
