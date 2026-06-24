"""
Launch only the Odin1 hardware driver nodes (no RViz, no gaussian_lic).

Used by the web monitor to start the sensor stack automatically when
Start Recording is pressed and the driver is not yet running.

Can also be launched standalone:
    ros2 launch gaussian_lic odin1_drivers.launch.py
"""

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('odin_ros_driver')
    config_path = os.path.join(pkg, 'config', 'control_command.yaml')
    calib_path  = os.path.join(pkg, 'config', 'calib.yaml')

    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    params['calib_file_path'] = calib_path

    return LaunchDescription([
        Node(
            package='odin_ros_driver',
            executable='host_sdk_sample',
            name='host_sdk_sample',
            output='screen',
            parameters=[{'config_file': config_path}],
        ),
        Node(
            package='odin_ros_driver',
            executable='pcd2depth_ros2_node',
            name='pcd2depth_ros2_node',
            output='screen',
            parameters=[params],
        ),
        Node(
            package='odin_ros_driver',
            executable='cloud_reprojection_ros2_node',
            name='cloud_reprojection_ros2_node',
            output='screen',
            parameters=[params],
        ),
    ])
