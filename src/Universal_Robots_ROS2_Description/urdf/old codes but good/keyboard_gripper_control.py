import omni
import carb
from pxr import UsdPhysics

# -----------------------------
# JOINT PATHS FROM YOUR USDA
# -----------------------------
LEFT_JOINT_1 = "/World/custom_gripper/left_finger_1/PrismaticJoint1"
LEFT_JOINT_2 = "/World/custom_gripper/left_finger_2/PrismaticJoint2"
RIGHT_JOINT_1 = "/World/custom_gripper/right_finger_1/PrismaticJoint3"
RIGHT_JOINT_2 = "/World/custom_gripper/right_finger_2/PrismaticJoint4"

STAGE = omni.usd.get_context().get_stage()

# -----------------------------
# MOTION SETTINGS
# Only targetPosition is changed.
# These should match your joint limits.
# -----------------------------
OPEN_POS = 0.0
CLOSE_POS = -0.0093
STEP = 0.001   # 1 mm per key press

keyboard = None
keyboard_sub = None

left_target = OPEN_POS
right_target = OPEN_POS


def clamp(val, low, high):
    return max(low, min(high, val))


def set_joint_target(joint_path, target_value):
    prim = STAGE.GetPrimAtPath(joint_path)
    if not prim.IsValid():
        carb.log_warn(f"Joint not found: {joint_path}")
        return

    drive_api = UsdPhysics.DriveAPI.Get(prim, "linear")
    if not drive_api:
        carb.log_warn(f"Linear drive API not found on: {joint_path}")
        return

    target_attr = drive_api.GetTargetPositionAttr()
    if not target_attr:
        carb.log_warn(f"targetPosition attribute not found on: {joint_path}")
        return

    target_attr.Set(float(target_value))


def apply_left():
    set_joint_target(LEFT_JOINT_1, left_target)
    set_joint_target(LEFT_JOINT_2, left_target)
    carb.log_info(f"Left fingers target = {left_target:.4f} m")


def apply_right():
    set_joint_target(RIGHT_JOINT_1, right_target)
    set_joint_target(RIGHT_JOINT_2, right_target)
    carb.log_info(f"Right fingers target = {right_target:.4f} m")


def print_help():
    print("\nKeyboard control:")
    print("  A  -> open left pair")
    print("  Z  -> close left pair")
    print("  K  -> open right pair")
    print("  M  -> close right pair")
    print("  O  -> open both pairs")
    print("  C  -> close both pairs")
    print("  H  -> help\n")


def on_keyboard_event(event, *args, **kwargs):
    global left_target, right_target

    if event.type != carb.input.KeyboardEventType.KEY_PRESS:
        return True

    key = event.input

    # LEFT PAIR
    if key == carb.input.KeyboardInput.A:
        left_target = clamp(left_target + STEP, CLOSE_POS, OPEN_POS)
        apply_left()

    elif key == carb.input.KeyboardInput.Z:
        left_target = clamp(left_target - STEP, CLOSE_POS, OPEN_POS)
        apply_left()

    # RIGHT PAIR
    elif key == carb.input.KeyboardInput.K:
        right_target = clamp(right_target + STEP, CLOSE_POS, OPEN_POS)
        apply_right()

    elif key == carb.input.KeyboardInput.M:
        right_target = clamp(right_target - STEP, CLOSE_POS, OPEN_POS)
        apply_right()

    # BOTH PAIRS
    elif key == carb.input.KeyboardInput.O:
        left_target = OPEN_POS
        right_target = OPEN_POS
        apply_left()
        apply_right()

    elif key == carb.input.KeyboardInput.C:
        left_target = CLOSE_POS
        right_target = CLOSE_POS
        apply_left()
        apply_right()

    elif key == carb.input.KeyboardInput.H:
        print_help()

    return True


def setup_keyboard():
    global keyboard, keyboard_sub

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_iface = carb.input.acquire_input_interface()

    keyboard_sub = input_iface.subscribe_to_keyboard_events(
        keyboard, on_keyboard_event
    )

    print_help()
    carb.log_info("Keyboard gripper control is active.")


setup_keyboard()