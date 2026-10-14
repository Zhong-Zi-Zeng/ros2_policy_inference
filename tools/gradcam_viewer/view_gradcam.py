#!/usr/bin/env python3
"""
Live view of the policy's Grad-CAM overlays (policy_inference_node with gradcam_enable:=true).

Each camera the policy uses is shown with its map on top: blue = no influence on the predicted
actions, red = most. The percentage is that camera's share of the total; the bottom line is the
explained target and the motion the policy predicts over those steps. Details: ros2_policy_inference/gradcam.py.

Keys: s = save the current image as PNG, r = start / stop recording an MP4 (at the rate overlays
arrive), q / Esc = quit. Files go to --save-dir as gradcam_<time>.png / .mp4.

Read-only: it never publishes, so it can run alongside the policy and the sim.

Usage (inside a ROS 2 Jazzy environment, e.g. conda env lerobot312-ros):
    python3 tools/gradcam_viewer/view_gradcam.py [--topic /policy/gradcam/image] [--save-dir recordings]
"""
import argparse
import os
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default="/policy/gradcam/image")
    ap.add_argument("--save-dir", default=str(REPO / "recordings"))
    args = ap.parse_args()
    out_dir = Path(args.save_dir)

    rclpy.init()
    node = rclpy.create_node("gradcam_viewer")
    state = {"msg": None, "new": False, "img": None, "writer": None, "path": None, "times": []}

    def on_image(msg):
        state["msg"], state["new"] = msg, True

    node.create_subscription(Image, args.topic, on_image, 1)
    print(f"[gradcam_viewer] waiting for {args.topic} (ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}); "
          f"start the policy with gradcam_enable:=true (./run_docker.sh --policy-name ... --gradcam)")

    # Tk rather than cv2.imshow: the pip opencv in the ROS conda env is headless.
    root = tk.Tk()
    root.title("Grad-CAM   (s = save, r = record, q = quit)")
    label = tk.Label(root, text=f"waiting for {args.topic} ...", width=60, height=10)
    label.pack()

    def save(_e=None):
        if state["img"] is None:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"gradcam_{time.strftime('%Y%m%d-%H%M%S')}.png"
        cv2.imwrite(str(path), cv2.cvtColor(state["img"], cv2.COLOR_RGB2BGR))
        print(f"[gradcam_viewer] saved {path}")

    def toggle_record(_e=None):
        if state["writer"] is not None:
            state["writer"].release()
            print(f"[gradcam_viewer] recording stopped: {state['path']}")
            state["writer"] = None
            root.title("Grad-CAM   (s = save, r = record, q = quit)")
            return
        if state["img"] is None:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = state["times"]
        fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 5.0
        h, w = state["img"].shape[:2]
        state["path"] = out_dir / f"gradcam_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        state["writer"] = cv2.VideoWriter(str(state["path"]), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"[gradcam_viewer] recording at {fps:.1f} fps to {state['path']} (r to stop)")
        root.title("Grad-CAM   ● RECORDING (r = stop)")

    def quit_(_e=None):
        if state["writer"] is not None:
            toggle_record()
        root.destroy()

    root.bind("s", save)
    root.bind("r", toggle_record)
    root.bind("q", quit_)
    root.bind("<Escape>", quit_)
    root.protocol("WM_DELETE_WINDOW", quit_)

    def tick():
        try:
            rclpy.spin_once(node, timeout_sec=0.0)
        except ExternalShutdownException:
            quit_()
            return
        msg = state["msg"]
        if state["new"] and msg is not None:
            state["new"] = False
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            if msg.encoding == "bgr8":
                img = img[..., ::-1]
            state["img"] = np.ascontiguousarray(img)
            state["times"] = (state["times"] + [time.monotonic()])[-20:]
            if state["writer"] is not None:
                state["writer"].write(cv2.cvtColor(state["img"], cv2.COLOR_RGB2BGR))
            ppm = f"P6 {img.shape[1]} {img.shape[0]} 255 ".encode() + state["img"].tobytes()
            photo = tk.PhotoImage(data=ppm, format="PPM")
            label.configure(image=photo, width=img.shape[1], height=img.shape[0])
            label.image = photo        # keep a reference or Tk drops the image
        root.after(20, tick)

    tick()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        quit_()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
