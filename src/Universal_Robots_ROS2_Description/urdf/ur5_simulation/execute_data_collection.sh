#!/bin/bash

# Run Isaac sim in vertual environment
cd ..
source isaac_env/bin/activate
cd ur5_simulation/src/data_collection/scripts/ || exit
python launch_isaac.py &

isaac_pid=$!
echo "Isaac Sim started (PID: $isaac_pid)"

# Run IK
gnome-terminal -- bash -c "\
source /opt/ros/humble/setup.bash; \
source ~/ur5_simulation/install/setup.bash; \
ros2 launch data_collection pickik_arm_control.launch.py; \
exec bash" &

# Run Data Collection (blocking, determines when we finish)
source /opt/ros/humble/setup.bash
source ~/ur5_simulation/install/setup.bash
ros2 run data_collection automatic_data_collection.py

# Once ROS data collection ends, kill Isaac Sim and IK
pkill -f "launch_isaac.py"
pkill -f "ros2 launch data_collection pickik_arm_control.launch.py"