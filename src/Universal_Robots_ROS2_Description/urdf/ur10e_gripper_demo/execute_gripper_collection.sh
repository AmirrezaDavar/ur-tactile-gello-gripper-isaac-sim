#!/bin/bash
# execute_gripper_collection.sh
#
# Runs one complete data-collection episode:
#   1. Launches Isaac Sim (background) — loads scene, controls gripper, publishes ROS2 topics.
#   2. Runs the data collector (blocking) — records camera + joint data, writes parquet + videos.
#   3. Kills Isaac Sim when the data collector exits.
#
# Called repeatedly by run_gripper_collection.py (50 times total).
#
# ADJUST THESE PATHS to match your environment:
#   ISAAC_VENV  — path to the Isaac Sim Python virtual environment
#   ROS2_SETUP  — ROS2 setup script (Humble default)
#   WS_SETUP    — your ROS2 workspace install/setup.bash (if needed)

set -e

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Path configuration ────────────────────────────────────────────────────────
# Isaac Sim virtual environment (adjust to your installation).
# Common locations:
#   ~/.local/share/ov/pkg/isaac-sim-*/isaac_env   (Linux package install)
#   ~/isaac_env                                    (manual venv)
ISAAC_ROOT="${HOME}/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64"
ISAAC_PYTHON="${ISAAC_ROOT}/python.sh"

ROS2_SETUP="/opt/ros/humble/setup.bash"

# Your ROS2 workspace (leave empty if not needed).
WS_SETUP="${HOME}/ur_ws/install/setup.bash"

# ── Isaac Sim ROS2 environment ────────────────────────────────────────────────
# Isaac Sim uses Python 3.11. The system ROS2 rclpy is compiled for Python 3.10
# and will crash under Python 3.11. We must NOT source /opt/ros/humble/setup.bash
# for the Isaac Sim process.
#
# Instead we:
#   1. Source Isaac Sim's setup_ros_env.sh  → sets LD_LIBRARY_PATH + RMW
#   2. Prepend Isaac Sim's bundled Python 3.11 rclpy to PYTHONPATH
#      so `import rclpy` finds the correct version.
source "${ISAAC_ROOT}/setup_ros_env.sh"

BRIDGE_RCLPY="${ISAAC_ROOT}/exts/isaacsim.ros2.bridge/humble/rclpy"
export PYTHONPATH="${BRIDGE_RCLPY}:${PYTHONPATH}"

echo "[execute] Starting Isaac Sim..."
"${ISAAC_PYTHON}" "${DEMO_DIR}/launch_isaac_ur10e.py" &
ISAAC_PID=$!
echo "[execute] Isaac Sim PID: ${ISAAC_PID}"

# Give Isaac Sim time to start up before launching the ROS2 recorder.
sleep 20

# ── Source ROS2 and launch data collector (blocking) ──────────────────────────
echo "[execute] Starting data collector..."
source "${ROS2_SETUP}"
if [ -f "${WS_SETUP}" ]; then
    source "${WS_SETUP}"
fi

python "${DEMO_DIR}/gripper_data_collection.py"

echo "[execute] Data collection finished."

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo "[execute] Killing Isaac Sim (PID: ${ISAAC_PID})..."
kill "${ISAAC_PID}" 2>/dev/null || true
pkill -f "launch_isaac_ur10e.py" 2>/dev/null || true

echo "[execute] Done."
