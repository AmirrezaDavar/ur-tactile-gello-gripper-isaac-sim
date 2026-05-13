#!/usr/bin/python3

import os
import isaacsim
from isaacsim import SimulationApp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
args = parser.parse_args()
simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1280",
        "height": "720",
    }
)

import omni
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.world.world import World
import omni.physx as physx
from pxr import UsdGeom, Gf
from isaacsim.core.utils import extensions

import random

# enable ROS2 bridge extension
extensions.enable_extension("isaacsim.ros2.bridge")

simulation_context = SimulationContext(stage_units_in_meters = 1.0)
physics_context = simulation_context.get_physics_context()
physics_context.enable_ccd(True)
physics_context.enable_gpu_dynamics(True)
physics_context.set_broadphase_type("gpu")
physics_context.enable_stablization(False)

usd_path = os.environ["HOME"] + "/ur5_simulation/pushT.usd"
usd_context = omni.usd.get_context()
usd_context.open_stage(usd_path)

# wait for things to load
simulation_app.update()
while isaacsim.core.utils.stage.is_stage_loading():
    simulation_app.update()

stage = usd_context.get_stage()

my_world = World(stage_units_in_meters = 1.0, physics_dt = 1/100, rendering_dt = 1/50)

Tbar_path = "/World/Tbar"
prim = stage.GetPrimAtPath(Tbar_path )
# Update the translate op to move the object
xform = UsdGeom.Xformable(prim)
xform_ops = xform.GetOrderedXformOps()

init_x = 0.59 - random.random()/10
init_y = random.random()/10 - 0.07

for op in xform_ops:
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        op.Set((init_x, init_y, 0.76))
        print(f"Updated position of {Tbar_path} to ({init_x}, {init_y}, 0.76)")
        break
else:
    raise RuntimeError("Translate op not found on prim!")

theta = 0
while True:
    theta = random.random()*360 - 180
    # to avoid Tbar collision with tool at the begining of the simulation
    if init_y < 0.02 and init_x > 0.57:
        break
    else:
        if theta < -150 or theta > -105:
            break

for op in xform_ops:
    print(op.GetOpType())
    if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), theta)
        quatd = rot.GetQuat()
        quatf = Gf.Quatf(quatd)
        op.Set(quatf)
        print(f"inittial angle is {theta}")
        break
else:
    raise RuntimeError("Translate op not found on prim!")

simulation_context.initialize_physics()
simulation_context.play()

while simulation_app.is_running():
    simulation_context.step(render=True)

