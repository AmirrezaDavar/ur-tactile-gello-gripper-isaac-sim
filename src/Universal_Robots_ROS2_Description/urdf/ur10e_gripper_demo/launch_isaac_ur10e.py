#!/usr/bin/python3
"""
Isaac Sim launcher for UR10e + custom gripper demo.

Responsibilities:
  1. Load ur10e_gripper_scene.usda (UR10e + custom gripper + two cameras).
  2. Set up ROS2 camera publishers (top and jaw) via OmniGraph.
  3. Teleport the arm to the target pose at physics start using the Articulation API
     so the robot is ALREADY at the correct configuration from frame 1.
  4. Run a per-episode state machine that opens/closes the gripper.
  5. Publish gripper joint positions, commanded targets, and episode state
     over ROS2 so gripper_data_collection.py can record them.
  6. Exit after one complete episode (close → hold → open → done).
     The outer run_gripper_collection.py loop re-launches this 50 times.

Episode state machine (published on /episode_state as Int32):
  0 = SETTLING   - robot held at initial pose, no recording
  1 = CLOSING    - gripper closing, record frames
  2 = HOLDING    - gripper held closed, record frames
  3 = OPENING    - gripper opening, record frames
  4 = DONE       - episode finished, write data and exit

Randomization per episode:
  - close_target_per_joint : slightly different per joint (-0.007 to -0.0093 m)
  - close_step_size        : closing speed (0.0002 – 0.0005 m / physics step)
  - open_step_size         : opening speed (0.0002 – 0.0005 m / physics step)
"""

import os
import random
from math import radians
import numpy as np

import isaacsim
from isaacsim import SimulationApp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless_mode", type=str, default=None,
                    help="To run headless, use one of [native, websocket].")
args = parser.parse_args()

simulation_app = SimulationApp({
    "headless": args.headless_mode is not None,
    "width": "1280",
    "height": "720",
})

import omni
import omni.graph.core as og
from isaacsim.core.api import SimulationContext
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation          # moved in Isaac Sim 5.x
from isaacsim.core.utils import extensions

# ── ROS2 bridge ────────────────────────────────────────────────────────────────
extensions.enable_extension("isaacsim.ros2.bridge")

# ── Fix sys.path: force Isaac Sim's Python 3.11 rclpy over system Python 3.10 ─
# .bashrc sources /opt/ros/humble/setup.bash which puts Python 3.10 rclpy into
# sys.path. Isaac Sim runs Python 3.11, so the system rclpy will fail to load
# _rclpy_pybind11. We strip /opt/ros paths and insert the bundled 3.11 rclpy.
import sys
_ISAAC_ROOT = os.path.expanduser("~/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64")
_ISAAC_RCLPY = os.path.join(_ISAAC_ROOT, "exts/isaacsim.ros2.bridge/humble/rclpy")
sys.path = [p for p in sys.path if "/opt/ros" not in p]
if _ISAAC_RCLPY not in sys.path:
    sys.path.insert(0, _ISAAC_RCLPY)

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32

# ── Physics setup ──────────────────────────────────────────────────────────────
simulation_context = SimulationContext(stage_units_in_meters=1.0)
physics_context = simulation_context.get_physics_context()
physics_context.enable_ccd(True)
physics_context.enable_gpu_dynamics(True)
physics_context.set_broadphase_type("gpu")
physics_context.enable_stablization(False)

# ── Load scene ─────────────────────────────────────────────────────────────────
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
usd_path = os.path.join(DEMO_DIR, "ur10e_gripper_scene.usda")

usd_context = omni.usd.get_context()
usd_context.open_stage(usd_path)

simulation_app.update()
while isaacsim.core.utils.stage.is_stage_loading():
    simulation_app.update()

stage = usd_context.get_stage()
my_world = World(stage_units_in_meters=1.0, physics_dt=1/100, rendering_dt=1/50)

# ── Register UR10e articulation ────────────────────────────────────────────────
# In Isaac Sim 5.x, Articulation uses prim_paths_expr (not prim_path).
# The gripper's ArticulationRootAPI is deleted in 1_fixed.usda so the arm
# and gripper form one combined articulation rooted at /World/ur10e.
ur10e_art = Articulation(prim_paths_expr="/World/ur10e", name="ur10e_arm")

# ── Camera ROS2 publishers (OmniGraph) ─────────────────────────────────────────
def setup_camera_publisher(camera_prim_path: str, topic_name: str, graph_path: str):
    """Create an OmniGraph that publishes a camera prim as an RGB ROS2 Image."""
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick",      "omni.graph.action.OnPlaybackTick"),
                ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("ROS2CameraHelper",    "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                ("CreateRenderProduct.inputs:cameraPrimPath", camera_prim_path),
                ("CreateRenderProduct.inputs:resolution",     [640, 480]),
                ("ROS2CameraHelper.inputs:topicName",         topic_name),
                ("ROS2CameraHelper.inputs:type",              "rgb"),
                ("ROS2CameraHelper.inputs:frameId",           topic_name),
                ("ROS2CameraHelper.inputs:nodeNamespace",     ""),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick",
                 "CreateRenderProduct.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut",
                 "ROS2CameraHelper.inputs:execIn"),
                ("CreateRenderProduct.outputs:renderProductPath",
                 "ROS2CameraHelper.inputs:renderProductPath"),
            ],
        },
    )

setup_camera_publisher("/World/CameraTop",                    "rgb_top", "/ActionGraph_CameraTop")
setup_camera_publisher("/World/ur10e/wrist_3_link/CameraJaw", "rgb_jaw", "/ActionGraph_CameraJaw")

# ── Target arm configuration (degrees → radians for Articulation API) ─────────
# Matches Physics Inspector values: shoulder_pan=0, shoulder_lift=-83.3,
# elbow=103.3, wrist_1=-110.4, wrist_2=-88.1, wrist_3=-172.5
#
# Joint order in the combined UR10e+gripper articulation (depth-first):
#   indices 0-5  : arm joints (pan, lift, elbow, w1, w2, w3)
#   indices 6-9  : gripper prismatic joints (PrismaticJoint1-4)
ARM_ANGLES_DEG = [0.0, -83.3, 103.3, -110.4, -88.1, -172.5]
ARM_ANGLES_RAD = np.array([radians(d) for d in ARM_ANGLES_DEG], dtype=float)
ARM_INDICES    = np.arange(6)

GRIPPER_OPEN_M   = np.zeros(4, dtype=float)   # 0 m = fully open
GRIPPER_INDICES  = np.arange(6, 10)

def teleport_arm_to_initial():
    """
    Use the Articulation API to instantly set arm joint positions (radians)
    and zero velocities. This is the only reliable way to place the robot
    at the target pose from frame 1, bypassing USD visual defaults.

    Isaac Sim 5.x Articulation.set_joint_positions expects shape (1, N):
    the outer dimension is the batch/robot count (always 1 here).
    """
    try:
        ur10e_art.set_joint_positions(
            ARM_ANGLES_RAD.reshape(1, -1), joint_indices=ARM_INDICES)
        ur10e_art.set_joint_velocities(
            np.zeros((1, 6)),              joint_indices=ARM_INDICES)
        ur10e_art.set_joint_positions(
            GRIPPER_OPEN_M.reshape(1, -1), joint_indices=GRIPPER_INDICES)
        ur10e_art.set_joint_velocities(
            np.zeros((1, 4)),              joint_indices=GRIPPER_INDICES)
    except Exception as e:
        print(f"[WARN] teleport_arm_to_initial: {e}")

# ── Arm drive-target locking (USD attribute level, in degrees) ─────────────────
ARM_DRIVE_TARGETS = {
    "/World/ur10e/joints/shoulder_pan_joint":  0.0,
    "/World/ur10e/joints/shoulder_lift_joint": -83.3,
    "/World/ur10e/joints/elbow_joint":          103.3,
    "/World/ur10e/joints/wrist_1_joint":       -110.4,
    "/World/ur10e/joints/wrist_2_joint":        -88.1,
    "/World/ur10e/joints/wrist_3_joint":       -172.5,
}

def lock_arm_drives():
    """Keep arm drive targets pinned to the initial configuration."""
    for path, deg in ARM_DRIVE_TARGETS.items():
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.GetAttribute("drive:angular:physics:targetPosition").Set(float(deg))

lock_arm_drives()

# ── Gripper joint prim paths (USD attribute level) ─────────────────────────────
JOINT_PATHS = [
    "/World/custom_gripper/custom_gripper/left_finger_1/PrismaticJoint1",
    "/World/custom_gripper/custom_gripper/left_finger_2/PrismaticJoint2",
    "/World/custom_gripper/custom_gripper/right_finger_1/PrismaticJoint3",
    "/World/custom_gripper/custom_gripper/right_finger_2/PrismaticJoint4",
]

OPEN_POS  =  0.0
MAX_CLOSE = -0.0093

def set_gripper_targets(targets: list):
    """Write drive:linear:physics:targetPosition for each gripper joint."""
    for path, target in zip(JOINT_PATHS, targets):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.GetAttribute("drive:linear:physics:targetPosition").Set(float(target))

def get_gripper_positions() -> list:
    """Read state:linear:physics:position for each gripper joint."""
    positions = []
    for path in JOINT_PATHS:
        prim = stage.GetPrimAtPath(path)
        val = 0.0
        if prim.IsValid():
            v = prim.GetAttribute("state:linear:physics:position").Get()
            if v is not None:
                val = float(v)
        positions.append(val)
    return positions

# ── ROS2 publishers ────────────────────────────────────────────────────────────
if not rclpy.ok():
    rclpy.init()

class GripperPublisher(Node):
    def __init__(self):
        super().__init__("isaac_gripper_publisher")
        self.joint_state_pub   = self.create_publisher(Float32MultiArray, "/gripper_joint_states", 10)
        self.action_pub        = self.create_publisher(Float32MultiArray, "/gripper_action",        10)
        self.episode_state_pub = self.create_publisher(Int32,             "/episode_state",          10)

    def publish_all(self, positions: list, targets: list, episode_state: int):
        js_msg = Float32MultiArray(data=[float(p) for p in positions])
        self.joint_state_pub.publish(js_msg)
        act_msg = Float32MultiArray(data=[float(t) for t in targets])
        self.action_pub.publish(act_msg)
        self.episode_state_pub.publish(Int32(data=episode_state))

pub_node = GripperPublisher()

# ── Per-episode randomization ──────────────────────────────────────────────────
base_close   = -(0.007 + random.random() * 0.0023)
close_targets = [max(MAX_CLOSE, base_close + random.uniform(-0.0005, 0.0005)) for _ in range(4)]
close_step    = 0.0002 + random.random() * 0.0003
open_step     = 0.0002 + random.random() * 0.0003

print(f"[Episode] base_close={base_close:.4f} m  close_step={close_step:.5f}  open_step={open_step:.5f}")

# ── Episode state machine ──────────────────────────────────────────────────────
SETTLING = 0
CLOSING  = 1
HOLDING  = 2
OPENING  = 3
DONE     = 4

SETTLE_STEPS = 150
HOLD_STEPS   = 80

episode_state   = SETTLING
current_targets = [OPEN_POS] * 4
settle_count    = 0
hold_count      = 0
done_count      = 0

# ── Initialize physics and TELEPORT arm to target pose ────────────────────────
simulation_context.initialize_physics()
my_world.reset()          # required so Articulation objects are initialized
teleport_arm_to_initial() # place arm at correct pose from frame 1
set_gripper_targets([OPEN_POS] * 4)  # ensure gripper starts open

simulation_context.play()

# ── Main sim loop ──────────────────────────────────────────────────────────────
while simulation_app.is_running():
    simulation_context.step(render=True)

    gripper_pos = get_gripper_positions()

    if episode_state == SETTLING:
        # During settling: hold arm in place and keep gripper open.
        # Teleport for the first 10 steps in case reset() wasn't enough.
        if settle_count < 10:
            teleport_arm_to_initial()
        elif settle_count % 20 == 0:
            lock_arm_drives()
        set_gripper_targets([OPEN_POS] * 4)
        settle_count += 1
        if settle_count >= SETTLE_STEPS:
            episode_state = CLOSING
            print("[State] → CLOSING")

    elif episode_state == CLOSING:
        all_reached = True
        for i in range(4):
            if current_targets[i] > close_targets[i]:
                current_targets[i] = max(close_targets[i], current_targets[i] - close_step)
                all_reached = False
        set_gripper_targets(current_targets)
        if all_reached:
            episode_state = HOLDING
            print("[State] → HOLDING")

    elif episode_state == HOLDING:
        hold_count += 1
        if hold_count >= HOLD_STEPS:
            episode_state = OPENING
            print("[State] → OPENING")

    elif episode_state == OPENING:
        all_reached = True
        for i in range(4):
            if current_targets[i] < OPEN_POS:
                current_targets[i] = min(OPEN_POS, current_targets[i] + open_step)
                all_reached = False
        set_gripper_targets(current_targets)
        if all_reached:
            episode_state = DONE
            print("[State] → DONE")

    elif episode_state == DONE:
        done_count += 1
        if done_count >= 30:
            break

    pub_node.publish_all(gripper_pos, current_targets, episode_state)
    rclpy.spin_once(pub_node, timeout_sec=0)

# ── Cleanup ────────────────────────────────────────────────────────────────────
simulation_context.stop()
pub_node.destroy_node()
rclpy.shutdown()
simulation_app.close()
