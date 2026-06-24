"""
ROS2 launch file for Gaussian-LIC with the Odin1 sensor.

Bag replay example:
    ros2 bag play /data/Downtown1 --clock -r 1.0

Live sensor example:
    ros2 launch odin_ros_driver odin1.launch.py   (start driver first)
    ros2 launch gaussian_lic odin1.launch.py

Optional launch arguments:
    config       – path to a YAML config (default: <pkg>/config/odin1.yaml)
    result       – directory to write the Gaussian map (default: /mnt/Volume/Ubuntu/3d_vision/results/<timestamp>)
    web_monitor  – start live web dashboard on port 8765 (default: true)

Web dashboard (when running in Docker with -p 8765:8765):
    Open http://localhost:8765 in any browser on the host.
"""

import os
from datetime import datetime
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("gaussian_lic")

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=os.path.join(pkg_share, "config", "odin1.yaml"),
        description="Path to the Gaussian-LIC YAML config file",
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_arg = DeclareLaunchArgument(
        "result",
        default_value=os.path.join("/home/jetson/results", timestamp),
        description="Directory where the output Gaussian map will be written. "
                    "Pass result:=/path/to/result/<bag_name> to keep runs separate.",
    )
    web_monitor_arg = DeclareLaunchArgument(
        "web_monitor",
        default_value="true",
        description="Start the live web dashboard on port 8765",
    )
    wait_for_start_arg = DeclareLaunchArgument(
        "wait_for_start",
        default_value="false",
        description="Hold off processing until the Start Recording button is pressed in the web dashboard",
    )

    gs_node = Node(
        package="gaussian_lic",
        executable="gs_mapping",
        name="gs_mapping",
        output="screen",
        parameters=[
            {
                "config_path":     LaunchConfiguration("config"),
                "result_path":     LaunchConfiguration("result"),
                "lpips_path":      os.path.join(pkg_share, "src", "lpips"),
                "wait_for_start":  LaunchConfiguration("wait_for_start"),
            }
        ],
    )

    web_node = Node(
        package="gaussian_lic",
        executable="web_monitor",
        name="web_monitor",
        output="screen",
        condition=IfCondition(LaunchConfiguration("web_monitor")),
        parameters=[{
            "config_path": LaunchConfiguration("config"),
            "lpips_path":  os.path.join(pkg_share, "src", "lpips"),
        }],
    )

    return LaunchDescription([config_arg, result_arg, web_monitor_arg, wait_for_start_arg, gs_node, web_node])
