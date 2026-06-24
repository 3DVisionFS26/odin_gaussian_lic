#!/bin/bash
source /opt/ros/humble/setup.bash
source /home/jetson/ws/install/setup.bash
export LD_LIBRARY_PATH=/home/jetson/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}
exec ros2 launch gaussian_lic odin1.launch.py wait_for_start:=true
