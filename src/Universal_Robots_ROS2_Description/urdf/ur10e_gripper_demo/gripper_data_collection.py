#!/usr/bin/python3
"""
ROS2 data collection node for the UR10e gripper open/close demo.

Subscribes to:
  /rgb_top              (sensor_msgs/Image)  — top-down camera
  /rgb_jaw              (sensor_msgs/Image)  — jaw-view camera
  /gripper_joint_states (std_msgs/Float32MultiArray) — actual joint positions [j1..j4] (m)
  /gripper_action       (std_msgs/Float32MultiArray) — commanded targets     [j1..j4] (m)
  /episode_state        (std_msgs/Int32)              — 0=settling 1=closing 2=holding
                                                        3=opening  4=done

Records frames during states 1, 2, 3 (CLOSING, HOLDING, OPENING).
Writes one episode per run:
  data/chunk-000/episode_XXXXX.parquet
  videos/chunk-000/observation.images.top/episode_XXXXX.mp4
  videos/chunk-000/observation.images.jaw/episode_XXXXX.mp4

DataFrame columns (LeRobot-compatible):
  observation.state  — list[4 floats] actual joint positions (m)
  action             — list[4 floats] commanded targets (m)
  episode_index      — int
  frame_index        — int (within episode)
  timestamp          — float (s, 0-based per episode)
  next.reward        — float [0–1], mean closure fraction across joints
  next.done          — bool
  next.success       — bool (reward >= 0.90)
  index              — int (global, continues across episodes)
  task_index         — int (always 0)
"""

import os
import copy
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32

import cv2
import numpy as np
from cv_bridge import CvBridge
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Constants ──────────────────────────────────────────────────────────────────
EPISODE_STATES = {0: "SETTLING", 1: "CLOSING", 2: "HOLDING", 3: "OPENING", 4: "DONE"}
MAX_CLOSE = 0.0093          # |fully closed| in metres  (used for reward normalisation)
SUCCESS_THRESHOLD = 0.90    # mean closure fraction for success
RECORD_STATES = {1, 2, 3}   # CLOSING, HOLDING, OPENING

VID_H, VID_W = 240, 240
HZ = 10                     # recording frequency (timer callback Hz)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "training_data", "gripper_demo")

bridge = CvBridge()

# ── Shared globals (written by subscribers, read by Data_Recorder timer) ───────
top_camera_image   = np.zeros((VID_H, VID_W, 3), np.uint8)
jaw_camera_image   = np.zeros((VID_H, VID_W, 3), np.uint8)
gripper_positions  = [0.0, 0.0, 0.0, 0.0]   # actual joint positions
gripper_action     = [0.0, 0.0, 0.0, 0.0]   # commanded targets
current_ep_state   = 0                        # latest episode_state from Isaac

# ── Subscriber nodes ───────────────────────────────────────────────────────────

class TopCameraSubscriber(Node):
    def __init__(self):
        super().__init__("top_camera_subscriber")
        self.sub = self.create_subscription(Image, "/rgb_top", self._cb, 10)

    def _cb(self, msg):
        global top_camera_image
        img = bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = img.shape[:2]
        # Centre-crop to square, then resize to VID_HxVID_W
        side = min(h, w)
        top  = (h - side) // 2
        left = (w - side) // 2
        cropped = img[top:top+side, left:left+side]
        top_camera_image = cv2.resize(cropped, (VID_W, VID_H), cv2.INTER_LINEAR)


class JawCameraSubscriber(Node):
    def __init__(self):
        super().__init__("jaw_camera_subscriber")
        self.sub = self.create_subscription(Image, "/rgb_jaw", self._cb, 10)

    def _cb(self, msg):
        global jaw_camera_image
        img = bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = img.shape[:2]
        side  = min(h, w)
        top   = (h - side) // 2
        left  = (w - side) // 2
        cropped = img[top:top+side, left:left+side]
        jaw_camera_image = cv2.resize(cropped, (VID_W, VID_H), cv2.INTER_LINEAR)


class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__("joint_state_subscriber")
        self.sub_pos = self.create_subscription(
            Float32MultiArray, "/gripper_joint_states", self._pos_cb, 10)
        self.sub_act = self.create_subscription(
            Float32MultiArray, "/gripper_action", self._act_cb, 10)
        self.sub_ep  = self.create_subscription(
            Int32, "/episode_state", self._ep_cb, 10)

    def _pos_cb(self, msg):
        global gripper_positions
        gripper_positions = list(msg.data)[:4]

    def _act_cb(self, msg):
        global gripper_action
        gripper_action = list(msg.data)[:4]

    def _ep_cb(self, msg):
        global current_ep_state
        current_ep_state = msg.data


# ── Main recorder node ─────────────────────────────────────────────────────────

class DataRecorder(Node):

    def __init__(self):
        super().__init__("data_recorder")
        self.timer = self.create_timer(1.0 / HZ, self._timer_cb)

        # ── Output directories ───────────────────────────────────────────────
        self.log_dir = os.path.join(BASE_DIR, "data", "chunk-000")
        os.makedirs(self.log_dir, exist_ok=True)

        vid_base = os.path.join(BASE_DIR, "videos", "chunk-000", "observation.images.")
        self.top_vid_dir = vid_base + "top"
        self.jaw_vid_dir = vid_base + "jaw"
        os.makedirs(self.top_vid_dir, exist_ok=True)
        os.makedirs(self.jaw_vid_dir, exist_ok=True)

        # ── Episode / frame counters (persist across runs via parquet files) ──
        existing = sorted(
            f for f in os.listdir(self.log_dir)
            if f.endswith(".parquet")
        )
        if existing:
            last_df = pd.read_parquet(os.path.join(self.log_dir, existing[-1]))
            self.global_index   = int(last_df["index"].iloc[-1]) + 1
            self.episode_index  = int(last_df["episode_index"].iloc[-1]) + 1
        else:
            self.global_index  = 0
            self.episode_index = 0

        self.get_logger().info(
            f"Starting from episode {self.episode_index}, global index {self.global_index}")

        self.frame_index = 0
        self.timestamp   = 0.0
        self.column_idx  = 0
        self.done        = False
        self.success     = False
        self.data_written = False

        # ── In-memory buffers ────────────────────────────────────────────────
        self.df = pd.DataFrame(columns=[
            "observation.state", "action",
            "episode_index", "frame_index", "timestamp",
            "next.reward", "next.done", "next.success",
            "index", "task_index",
        ])
        self.top_frames = []
        self.jaw_frames = []

        # ── State tracking ───────────────────────────────────────────────────
        self._prev_ep_state = -1
        self._recording     = False

    # ── reward: mean closure fraction across 4 joints ─────────────────────────
    @staticmethod
    def _compute_reward(positions: list) -> float:
        if not positions:
            return 0.0
        fractions = [min(1.0, abs(p) / MAX_CLOSE) for p in positions]
        return float(np.mean(fractions))

    # ── episode filename helper ────────────────────────────────────────────────
    def _episode_filename(self) -> str:
        return f"episode_{self.episode_index:06d}"

    # ── main timer callback (10 Hz) ────────────────────────────────────────────
    def _timer_cb(self):
        global current_ep_state, gripper_positions, gripper_action

        ep_state = current_ep_state

        if ep_state != self._prev_ep_state:
            self.get_logger().info(
                f"Episode state → {EPISODE_STATES.get(ep_state, ep_state)}")
            self._prev_ep_state = ep_state

        # ── Recording active? ────────────────────────────────────────────────
        if ep_state in RECORD_STATES:
            self._recording = True

        # ── Record frame ─────────────────────────────────────────────────────
        if self._recording and not self.data_written:
            pos  = copy.copy(gripper_positions)
            act  = copy.copy(gripper_action)
            rew  = self._compute_reward(pos)

            is_done    = (ep_state == DONE_STATE := 4)
            is_success = rew >= SUCCESS_THRESHOLD and is_done

            self.df.loc[self.column_idx] = [
                pos, act,
                self.episode_index, self.frame_index, self.timestamp,
                rew, is_done, is_success,
                self.global_index, 0,
            ]
            self.column_idx  += 1
            self.frame_index += 1
            self.timestamp   += 1.0 / HZ
            self.global_index += 1

            self.top_frames.append(copy.copy(top_camera_image))
            self.jaw_frames.append(copy.copy(jaw_camera_image))

        # ── Done: write files ────────────────────────────────────────────────
        if ep_state == 4 and self._recording and not self.data_written:
            self._write_episode()

    def _write_episode(self):
        name = self._episode_filename()
        self.get_logger().info(f"Writing episode: {name}")

        # ── Parquet ──────────────────────────────────────────────────────────
        parquet_path = os.path.join(self.log_dir, name + ".parquet")
        table = pa.Table.from_pandas(self.df)
        pq.write_table(table, parquet_path)
        self.get_logger().info(f"  Parquet written → {parquet_path}")

        # ── Videos ──────────────────────────────────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        top_path = os.path.join(self.top_vid_dir, name + ".mp4")
        out_top  = cv2.VideoWriter(top_path, fourcc, HZ, (VID_W, VID_H))
        for frame in self.top_frames:
            out_top.write(frame)
        out_top.release()
        self.get_logger().info(f"  Top video written → {top_path}")

        jaw_path = os.path.join(self.jaw_vid_dir, name + ".mp4")
        out_jaw  = cv2.VideoWriter(jaw_path, fourcc, HZ, (VID_W, VID_H))
        for frame in self.jaw_frames:
            out_jaw.write(frame)
        out_jaw.release()
        self.get_logger().info(f"  Jaw video written → {jaw_path}")

        self.data_written = True
        self.get_logger().info("Episode saved. Shutting down data recorder.")
        rclpy.shutdown()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    rclpy.init(args=None)

    top_sub   = TopCameraSubscriber()
    jaw_sub   = JawCameraSubscriber()
    joint_sub = JointStateSubscriber()
    recorder  = DataRecorder()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(top_sub)
    executor.add_node(jaw_sub)
    executor.add_node(joint_sub)
    executor.add_node(recorder)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        rate = recorder.create_rate(2)
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        print("Ctrl+C pressed.")
    except rclpy.exceptions.ROSInterruptException:
        print("ROS shutdown triggered.")
    finally:
        executor.shutdown()
        rclpy.shutdown()
        executor_thread.join()
        print("Data collector stopped.")


if __name__ == "__main__":
    main()
