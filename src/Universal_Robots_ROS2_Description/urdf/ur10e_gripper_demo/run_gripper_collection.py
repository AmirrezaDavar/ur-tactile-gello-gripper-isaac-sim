#!/usr/bin/python3
"""
Master loop for UR10e gripper demo data collection.

Calls execute_gripper_collection.sh 50 times sequentially.
Each iteration = one episode (close + hold + open).

Usage:
    python run_gripper_collection.py

Output (written by gripper_data_collection.py each run):
    ur10e_gripper_demo/training_data/gripper_demo/
        data/chunk-000/episode_000000.parquet
        data/chunk-000/episode_000001.parquet
        ...
        videos/chunk-000/observation.images.top/episode_000000.mp4
        videos/chunk-000/observation.images.jaw/episode_000000.mp4
        ...
"""

import os
import subprocess
import time

NUM_EPISODES = 50
SLEEP_BETWEEN = 5   # seconds between episodes (let ROS2/ports settle)

SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "execute_gripper_collection.sh")

if not os.path.isfile(SCRIPT_PATH):
    raise FileNotFoundError(f"Script not found: {SCRIPT_PATH}")

failed = []

for i in range(1, NUM_EPISODES + 1):
    print(f"\n{'='*60}")
    print(f"  Episode {i} / {NUM_EPISODES}")
    print(f"{'='*60}")

    try:
        subprocess.run(["bash", SCRIPT_PATH], check=True)
        print(f"  ✓ Episode {i} complete.")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Episode {i} failed (exit code {e.returncode}). Continuing...")
        failed.append(i)

    if i < NUM_EPISODES:
        time.sleep(SLEEP_BETWEEN)

print(f"\n{'='*60}")
print(f"Collection finished.  {NUM_EPISODES - len(failed)}/{NUM_EPISODES} episodes succeeded.")
if failed:
    print(f"Failed episode numbers: {failed}")
print(f"{'='*60}")
