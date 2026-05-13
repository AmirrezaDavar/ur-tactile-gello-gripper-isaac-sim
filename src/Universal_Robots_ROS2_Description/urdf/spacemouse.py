import omni
import carb
import numpy as np
from pxr import UsdPhysics, UsdGeom
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
from omni.isaac.core.robots import Robot
from scipy.spatial.transform import Rotation as R
import threading
import hid

# ─────────────────────────────────────────────
# PATHS — match your USDA exactly
# ─────────────────────────────────────────────
ROBOT_PRIM_PATH   = "/World/ur10e"
GRIPPER_BASE_PATH = "/World/custom_gripper/custom_gripper"

LEFT_JOINT_1  = f"{GRIPPER_BASE_PATH}/left_finger_1/PrismaticJoint1"
LEFT_JOINT_2  = f"{GRIPPER_BASE_PATH}/left_finger_2/PrismaticJoint2"
RIGHT_JOINT_1 = f"{GRIPPER_BASE_PATH}/right_finger_1/PrismaticJoint3"
RIGHT_JOINT_2 = f"{GRIPPER_BASE_PATH}/right_finger_2/PrismaticJoint4"

ISAAC_SIM_ROOT     = "/home/wanglab22/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64"
RMPFLOW_CONFIG_DIR = f"{ISAAC_SIM_ROOT}/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur10e"

STAGE = omni.usd.get_context().get_stage()

# ─────────────────────────────────────────────
# GRIPPER SETTINGS
# ─────────────────────────────────────────────
OPEN_POS  =  0.0
CLOSE_POS = -0.0093

gripper_target = OPEN_POS

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def set_joint_target(joint_path, value):
    prim = STAGE.GetPrimAtPath(joint_path)
    if not prim.IsValid():
        carb.log_warn(f"Joint not found: {joint_path}"); return
    drive = UsdPhysics.DriveAPI.Get(prim, "linear")
    if not drive:
        carb.log_warn(f"No linear drive on: {joint_path}"); return
    drive.GetTargetPositionAttr().Set(float(value))

def apply_gripper(target):
    for path in [LEFT_JOINT_1, LEFT_JOINT_2, RIGHT_JOINT_1, RIGHT_JOINT_2]:
        set_joint_target(path, target)

# ─────────────────────────────────────────────
# SPACEMOUSE READER
# ─────────────────────────────────────────────
SPACEMOUSE_VENDOR_ID   = 0x256F
SPACEMOUSE_PRODUCT_IDS = [
    0xC62E,
    0xC631,
    0xC633,
    0xC635,  # SpaceMouse Compact — your device
    0xC652,
]

TRANSLATE_SCALE = 0.00005
ROTATE_SCALE    = 0.00008
DEADZONE        = 50

_sm_axes    = [0.0] * 6
_sm_lock    = threading.Lock()
_sm_running = True

def _signed16(lo, hi):
    v = lo | (hi << 8)
    return v - 65536 if v > 32767 else v

def _deadzone(v):
    return v if abs(v) > DEADZONE else 0

def spacemouse_reader():
    global _sm_running
    device = None
    for pid in SPACEMOUSE_PRODUCT_IDS:
        try:
            d = hid.device()
            d.open(SPACEMOUSE_VENDOR_ID, pid)
            d.set_nonblocking(True)
            device = d
            carb.log_info(f"SpaceMouse connected (PID=0x{pid:04X})")
            break
        except Exception:
            continue

    if device is None:
        carb.log_warn("SpaceMouse not found.")
        return

    while _sm_running:
        try:
            data = device.read(13)
            if data and len(data) >= 7:
                rid = data[0]
                x = _deadzone(_signed16(data[1], data[2]))
                y = _deadzone(_signed16(data[3], data[4]))
                z = _deadzone(_signed16(data[5], data[6]))
                with _sm_lock:
                    if rid == 1:
                        _sm_axes[0] = x
                        _sm_axes[1] = y
                        _sm_axes[2] = z
                    elif rid == 2:
                        _sm_axes[3] = x
                        _sm_axes[4] = y
                        _sm_axes[5] = z
        except Exception as e:
            carb.log_warn(f"SpaceMouse read error: {e}")
            break

    try:
        device.close()
    except Exception:
        pass

# ─────────────────────────────────────────────
# ROBOT CONTROLLER
# ─────────────────────────────────────────────
_articulation = None
_rmpflow      = None
_policy       = None

def setup_robot_controller():
    global _articulation, _rmpflow, _policy

    _articulation = Articulation(prim_path=ROBOT_PRIM_PATH)
    _articulation.initialize()

    _rmpflow = RmpFlow(
        robot_description_path  = f"{RMPFLOW_CONFIG_DIR}/rmpflow/ur10e_robot_description.yaml",
        rmpflow_config_path     = f"{RMPFLOW_CONFIG_DIR}/rmpflow/ur10e_rmpflow_config.yaml",
        urdf_path               = f"{RMPFLOW_CONFIG_DIR}/ur10e.urdf",
        end_effector_frame_name = "tool0",
        maximum_substep_size    = 0.00334,
    )

    _policy = ArticulationMotionPolicy(_articulation, _rmpflow, 1.0/60.0)
    carb.log_info("UR10e controller ready.")

# ─────────────────────────────────────────────
# EE POSE HELPER
# ─────────────────────────────────────────────
def _get_current_ee_pose():
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/ur10e/tool0")
    if not prim.IsValid():
        prim = stage.GetPrimAtPath("/World/ur10e/wrist_3_link")
    if not prim.IsValid():
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    xform     = UsdGeom.Xformable(prim)
    transform = xform.ComputeLocalToWorldTransform(0)
    pos       = transform.ExtractTranslation()
    rot       = transform.ExtractRotationQuat()
    img       = rot.GetImaginary()
    return (
        np.array([pos[0], pos[1], pos[2]]),
        np.array([rot.GetReal(), img[0], img[1], img[2]])
    )

# ─────────────────────────────────────────────
# PER-FRAME UPDATE
# ─────────────────────────────────────────────
_ee_pos = None
_ee_rot = None

def on_physics_step(step_size):
    global _ee_pos, _ee_rot

    if _articulation is None or _rmpflow is None:
        return

    # Guard: skip if physics not ready
    try:
        joint_pos = _articulation.get_joint_positions()
        if joint_pos is None:
            return
    except Exception:
        return

    # Lazy-init EE target to current pose
    if _ee_pos is None:
        _ee_pos, q = _get_current_ee_pose()
        # q is [w,x,y,z], scipy wants [x,y,z,w]
        _ee_rot = R.from_quat([q[1], q[2], q[3], q[0]])

    # Read SpaceMouse deltas
    with _sm_lock:
        tx, ty, tz, rx, ry, rz = list(_sm_axes)

    # Apply translation delta
    _ee_pos[0] += ty * TRANSLATE_SCALE
    _ee_pos[1] -= tx * TRANSLATE_SCALE
    _ee_pos[2] += tz * TRANSLATE_SCALE

    # Apply rotation delta
    delta_rot = R.from_euler('xyz', [
        rx * ROTATE_SCALE,
        ry * ROTATE_SCALE,
        rz * ROTATE_SCALE,
    ])
    _ee_rot = delta_rot * _ee_rot

    # Convert to Isaac Sim [w,x,y,z]
    q_xyzw      = _ee_rot.as_quat()
    target_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    _rmpflow.set_end_effector_target(_ee_pos, target_quat)
    action = _policy.get_next_articulation_action()
    if action is not None:
        _articulation.apply_action(action)

# ─────────────────────────────────────────────
# KEYBOARD (gripper only)
# ─────────────────────────────────────────────
_keyboard     = None
_keyboard_sub = None

def on_keyboard_event(event, *args, **kwargs):
    global gripper_target
    if event.type != carb.input.KeyboardEventType.KEY_PRESS:
        return True
    key = event.input
    if key == carb.input.KeyboardInput.O:
        gripper_target = OPEN_POS
        apply_gripper(OPEN_POS)
        carb.log_info("Gripper: OPEN")
    elif key == carb.input.KeyboardInput.C:
        gripper_target = CLOSE_POS
        apply_gripper(CLOSE_POS)
        carb.log_info("Gripper: CLOSE")
    elif key == carb.input.KeyboardInput.H:
        print_help()
    return True

def print_help():
    print("\n=== Control mapping ===")
    print("  SpaceMouse push/pull  → robot XYZ translation")
    print("  SpaceMouse twist      → robot Roll/Pitch/Yaw")
    print("  O key                 → open gripper fully")
    print("  C key                 → close gripper fully")
    print("  H key                 → this help\n")

def setup_keyboard():
    global _keyboard, _keyboard_sub
    win       = omni.appwindow.get_default_app_window()
    _keyboard = win.get_keyboard()
    iface     = carb.input.acquire_input_interface()
    _keyboard_sub = iface.subscribe_to_keyboard_events(_keyboard, on_keyboard_event)

# ─────────────────────────────────────────────
# STARTUP — press Play in Isaac Sim BEFORE running this script
# ─────────────────────────────────────────────
# 1. Start SpaceMouse reader thread
_sm_thread = threading.Thread(target=spacemouse_reader, daemon=True)
_sm_thread.start()

# 2. Set up robot IK
setup_robot_controller()

# 3. Register per-frame physics callback
_physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(on_physics_step)

# 4. Set up keyboard for gripper
setup_keyboard()

print_help()
carb.log_info("SpaceMouse + keyboard gripper control ACTIVE.")