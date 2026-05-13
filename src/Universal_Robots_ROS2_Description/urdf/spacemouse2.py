import omni
import carb
import numpy as np
from pxr import UsdPhysics, UsdGeom
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
from scipy.spatial.transform import Rotation as R
import threading
import hid
import h5py
import time
import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROBOT_PRIM_PATH   = "/World/ur10e"
GRIPPER_BASE_PATH = "/World/custom_gripper/custom_gripper"

LEFT_JOINT_1  = f"{GRIPPER_BASE_PATH}/left_finger_1/PrismaticJoint1"
LEFT_JOINT_2  = f"{GRIPPER_BASE_PATH}/left_finger_2/PrismaticJoint2"
RIGHT_JOINT_1 = f"{GRIPPER_BASE_PATH}/right_finger_1/PrismaticJoint3"
RIGHT_JOINT_2 = f"{GRIPPER_BASE_PATH}/right_finger_2/PrismaticJoint4"

ISAAC_SIM_ROOT     = "/home/wanglab22/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64"
RMPFLOW_CONFIG_DIR = f"{ISAAC_SIM_ROOT}/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur10e"

# ─────────────────────────────────────────────
# DATA RECORDING SETTINGS
# ─────────────────────────────────────────────
RECORDING_DIR = "/home/wanglab22/robot_demos"
os.makedirs(RECORDING_DIR, exist_ok=True)

_recording     = False
_record_buffer = []
_demo_count    = 0

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
SPACEMOUSE_PRODUCT_IDS = [0xC62E, 0xC631, 0xC633, 0xC635, 0xC652]

TRANSLATE_SCALE = 0.00005
ROTATE_SCALE    = 0.00008
DEADZONE        = 50

_sm_axes      = [0.0] * 6
_sm_btn_left  = False
_sm_btn_right = False
_sm_lock      = threading.Lock()
_sm_running   = True

def _signed16(lo, hi):
    v = lo | (hi << 8)
    return v - 65536 if v > 32767 else v

def _deadzone(v):
    return v if abs(v) > DEADZONE else 0

def spacemouse_reader():
    global _sm_running, _sm_btn_left, _sm_btn_right

    try:
        d = hid.device()
        d.open_path(b'/dev/hidraw1')
        d.set_nonblocking(False)   # BLOCKING — never misses a report
        carb.log_info("SpaceMouse opened on /dev/hidraw1 (blocking mode)")
    except Exception as e:
        carb.log_warn(f"SpaceMouse failed to open: {e}")
        return

    while _sm_running:
        try:
            data = d.read(13, timeout_ms=100)  # 100ms timeout so we can check _sm_running
            if not data or len(data) < 2:
                continue

            rid = data[0]
            with _sm_lock:
                if rid == 1 and len(data) >= 7:
                    _sm_axes[0] = _deadzone(_signed16(data[1], data[2]))
                    _sm_axes[1] = _deadzone(_signed16(data[3], data[4]))
                    _sm_axes[2] = _deadzone(_signed16(data[5], data[6]))
                elif rid == 2 and len(data) >= 7:
                    _sm_axes[3] = _deadzone(_signed16(data[1], data[2]))
                    _sm_axes[4] = _deadzone(_signed16(data[3], data[4]))
                    _sm_axes[5] = _deadzone(_signed16(data[5], data[6]))
                elif rid == 3:
                    btn           = data[1]
                    _sm_btn_left  = bool(btn & 0x01)
                    _sm_btn_right = bool(btn & 0x02)

        except Exception as e:
            carb.log_warn(f"SpaceMouse read error: {e}")
            break

    try:
        d.close()
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
    prim  = stage.GetPrimAtPath("/World/ur10e/tool0")
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
# DATA RECORDING HELPERS
# ─────────────────────────────────────────────
def start_recording():
    global _recording, _record_buffer
    if _recording:
        print("Already recording.")
        return
    _recording     = True
    _record_buffer = []
    print(f"\n[REC] Recording started. Press S to stop and save.\n")

def stop_and_save():
    global _recording, _record_buffer, _demo_count
    if not _recording:
        print("Not currently recording.")
        return
    _recording = False
    if len(_record_buffer) == 0:
        print("[REC] No data recorded.")
        return

    demo_path = os.path.join(RECORDING_DIR, f"demo_{_demo_count:04d}.hdf5")
    with h5py.File(demo_path, "w") as f:
        ee_pos      = np.array([s["ee_pos"]    for s in _record_buffer])
        ee_quat     = np.array([s["ee_quat"]   for s in _record_buffer])
        joint_pos   = np.array([s["joint_pos"] for s in _record_buffer])
        gripper_arr = np.array([s["gripper"]   for s in _record_buffer])
        timestamps  = np.array([s["timestamp"] for s in _record_buffer])
        sm_axes_arr = np.array([s["sm_axes"]   for s in _record_buffer])

        f.create_dataset("ee_pos",     data=ee_pos)
        f.create_dataset("ee_quat",    data=ee_quat)
        f.create_dataset("joint_pos",  data=joint_pos)
        f.create_dataset("gripper",    data=gripper_arr)
        f.create_dataset("timestamps", data=timestamps)
        f.create_dataset("sm_axes",    data=sm_axes_arr)

        f.attrs["n_steps"]    = len(_record_buffer)
        f.attrs["demo_index"] = _demo_count

    print(f"[REC] Saved {len(_record_buffer)} steps → {demo_path}")
    _demo_count   += 1
    _record_buffer = []

def discard_recording():
    global _recording, _record_buffer
    _recording     = False
    _record_buffer = []
    print("[REC] Recording discarded.")

# ─────────────────────────────────────────────
# PER-FRAME UPDATE
# ─────────────────────────────────────────────
_ee_pos       = None
_ee_rot       = None
_debug_btn    = True   # set to False once buttons confirmed working

def on_physics_step(step_size):
    global _ee_pos, _ee_rot, _debug_btn

    if _articulation is None or _rmpflow is None:
        setup_robot_controller()
        return

    try:
        joint_pos = _articulation.get_joint_positions()
        if joint_pos is None:
            return
    except Exception:
        return

    # Lazy-init EE target
    if _ee_pos is None:
        _ee_pos, q = _get_current_ee_pose()
        _ee_rot = R.from_quat([q[1], q[2], q[3], q[0]])

    # Read SpaceMouse state
    with _sm_lock:
        tx, ty, tz, rx, ry, rz = list(_sm_axes)
        btn_left  = _sm_btn_left
        btn_right = _sm_btn_right

    # DEBUG — prints when buttons pressed; remove once confirmed working
    if _debug_btn and (btn_left or btn_right):
        print(f"[BTN] left={btn_left}  right={btn_right}")

    # ── Translation ──────────────────────────
    # XY always active
    _ee_pos[0] -= ty * TRANSLATE_SCALE
    _ee_pos[1] -= tx * TRANSLATE_SCALE
    # Z only when LEFT button held
    if btn_left:
        _ee_pos[2] -= tz * TRANSLATE_SCALE

    # ── Rotation ─────────────────────────────
    # Only when RIGHT button held
    if btn_right:
        delta_rot = R.from_euler('xyz', [
            rx * ROTATE_SCALE,
            ry * ROTATE_SCALE,
            rz * ROTATE_SCALE,
        ])
        _ee_rot = delta_rot * _ee_rot

    # Send to RMPflow
    q_xyzw      = _ee_rot.as_quat()
    target_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    _rmpflow.set_end_effector_target(_ee_pos, target_quat)
    action = _policy.get_next_articulation_action()
    if action is not None:
        _articulation.apply_action(action)

    # ── Record frame if active ────────────────
    if _recording:
        ee_pos_now, ee_quat_now = _get_current_ee_pose()
        _record_buffer.append({
            "timestamp": time.time(),
            "ee_pos":    ee_pos_now.copy(),
            "ee_quat":   ee_quat_now.copy(),
            "joint_pos": joint_pos.copy(),
            "gripper":   float(gripper_target),
            "sm_axes":   np.array([tx, ty, tz, rx, ry, rz]),
        })

# ─────────────────────────────────────────────
# KEYBOARD
# ─────────────────────────────────────────────
_keyboard     = None
_keyboard_sub = None

def on_keyboard_event(event, *args, **kwargs):
    global gripper_target, _debug_btn
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

    elif key == carb.input.KeyboardInput.R:
        start_recording()

    elif key == carb.input.KeyboardInput.S:
        stop_and_save()

    elif key == carb.input.KeyboardInput.X:
        discard_recording()

    elif key == carb.input.KeyboardInput.D:
        _debug_btn = not _debug_btn
        print(f"[DEBUG] Button debug printing: {'ON' if _debug_btn else 'OFF'}")

    elif key == carb.input.KeyboardInput.H:
        print_help()

    return True

def print_help():
    print("\n=== Control mapping ===")
    print("  SpaceMouse XY              → robot XY translation (always)")
    print("  LEFT button held + puck Z  → robot Z translation")
    print("  RIGHT button held + twist  → robot Roll/Pitch/Yaw")
    print("  O key                      → open gripper")
    print("  C key                      → close gripper")
    print("  ── Recording ──────────────")
    print("  R key                      → start recording demo")
    print("  S key                      → stop and save demo")
    print("  X key                      → discard recording")
    print("  ── Debug ───────────────────")
    print("  D key                      → toggle button debug printing")
    print(f"  Demos saved to: {RECORDING_DIR}")
    print("  H key                      → this help\n")

def setup_keyboard():
    global _keyboard, _keyboard_sub
    win           = omni.appwindow.get_default_app_window()
    _keyboard     = win.get_keyboard()
    iface         = carb.input.acquire_input_interface()
    _keyboard_sub = iface.subscribe_to_keyboard_events(_keyboard, on_keyboard_event)

# ─────────────────────────────────────────────
# STARTUP — press Play in Isaac Sim BEFORE running
# ─────────────────────────────────────────────
_sm_thread = threading.Thread(target=spacemouse_reader, daemon=True)
_sm_thread.start()

_physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(on_physics_step)

setup_keyboard()

print_help()
carb.log_info("Ready. Press Play first, then use SpaceMouse + keyboard.")