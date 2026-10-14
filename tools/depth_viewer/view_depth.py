#!/usr/bin/env python3
"""
Live view of the wrist depth image: what the camera (or Isaac Sim) publishes, and what the depth
policies (depth_act, depth_inference*) actually feed the model.

Window, side by side:
  left   raw depth, colour-mapped from 0 m (blue) to --view-max-m (dark red); black = no reading
  right  the policy input: resized to image_size with nearest-neighbour and encoded by the policy's
         own encode_depth() (white 255 = depth_min_m ... 1 = depth_max_m, 0 = no reading), so
         anything beyond depth_max_m (1 m) is near-black, almost the same as no reading. Shown at 2x.
Once a second it prints the encoding, size and depth statistics. Keys: s = save both images as
PNG (and the raw depth as .npy) into --save-dir, q / Esc = quit. --no-window only prints.

Read-only: it never publishes, so it can run alongside the policy and the sim.

Usage (inside a ROS 2 Jazzy environment, e.g. conda env lerobot312-ros):
    python3 tools/depth_viewer/view_depth.py [--topic /camera_L/depth/aligned_depth_to_color]
        [--view-max-m 2.0] [--save-dir recordings] [--no-window]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.executors import ExternalShutdownException
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ros2_policy_inference.policies.depth_inference.depth_policy import encode_depth  # noqa: E402
from ros2_policy_inference.preprocessing import resize_image  # noqa: E402
from ros2_policy_inference.ros_bridge import RosBridge  # noqa: E402

PARAMS = REPO / "config" / "params.yaml"
DEPTH_CFG = REPO / "ros2_policy_inference" / "policies" / "act_depth" / "config.yaml"


def to_metres(depth: np.ndarray) -> np.ndarray:
    """uint16 millimetres (real camera) or float metres (sim) -> float metres, NaN = no reading."""
    if np.issubdtype(depth.dtype, np.floating):
        d = depth.astype(np.float32)
        d[~np.isfinite(d) | (d <= 0)] = np.nan
    else:
        d = depth.astype(np.float32) / 1000.0
        d[depth == 0] = np.nan
    return d


def colour_map(metres: np.ndarray, view_max: float) -> np.ndarray:
    valid = np.isfinite(metres)
    x = np.clip(np.nan_to_num(metres, nan=0.0) / view_max, 0.0, 1.0)
    img = cv2.applyColorMap(np.rint(x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def main():
    params = yaml.safe_load(PARAMS.read_text())["policy_inference_node"]["ros__parameters"]
    depth_cfg = yaml.safe_load(DEPTH_CFG.read_text())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default=params.get("depth_topic", "/camera_L/depth/aligned_depth_to_color"))
    ap.add_argument("--image-size", type=int, default=int(params.get("image_size", 256)))
    ap.add_argument("--view-max-m", type=float, default=2.0, help="far end of the raw colour map")
    ap.add_argument("--save-dir", default=str(REPO / "recordings"))
    ap.add_argument("--no-window", action="store_true", help="print stats only (s/q keys unavailable)")
    args = ap.parse_args()
    d_min, d_max = float(depth_cfg["depth_min_m"]), float(depth_cfg["depth_max_m"])

    rclpy.init()
    node = rclpy.create_node("depth_viewer")
    latest = {}
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    node.create_subscription(Image, args.topic, lambda m: latest.update(msg=m, t=time.monotonic()), qos)
    print(f"[depth_viewer] waiting for {args.topic} (ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}); "
          f"policy encoding {d_min:g}-{d_max:g} m -> 255-1, image_size {args.image_size}")

    state = {"last_print": 0.0, "frame": None}

    def process():
        """Spin once; returns (raw, left_bgr, policy_in) for the newest message, or None."""
        rclpy.spin_once(node, timeout_sec=0.0 if not args.no_window else 0.03)
        msg = latest.get("msg")
        if msg is None:
            return None
        raw = RosBridge._image_msg_to_array(msg)
        metres = to_metres(raw)
        policy_in = encode_depth(resize_image(raw, args.image_size, is_depth=True), d_min, d_max)[..., 0]
        now = time.monotonic()
        if now - state["last_print"] >= 1.0:
            state["last_print"] = now
            v = metres[np.isfinite(metres)]
            stats = (f"min {v.min():.3f}  median {np.median(v):.3f}  max {v.max():.3f} m, "
                     f"beyond {d_max:g} m {np.mean(v > d_max) * 100:.1f}%") if v.size else "no valid pixels"
            print(f"[depth_viewer] {msg.encoding} {msg.width}x{msg.height}  {stats}  "
                  f"no reading {np.mean(~np.isfinite(metres)) * 100:.1f}%  "
                  f"(last msg {now - latest['t']:.2f} s ago)")
        return raw, colour_map(metres, args.view_max_m), policy_in

    def save(_event=None):
        if state["frame"] is None:
            return
        raw, left, policy_in = state["frame"]
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(str(out / f"depth_raw_{stamp}.png"), left)
        cv2.imwrite(str(out / f"depth_policy_input_{stamp}.png"), policy_in)
        np.save(out / f"depth_raw_{stamp}.npy", raw)
        print(f"[depth_viewer] saved depth_raw_{stamp}.png/.npy and depth_policy_input_{stamp}.png to {out}")

    try:
        if args.no_window:
            while rclpy.ok():
                process()
            return

        # Tk rather than cv2.imshow: the pip opencv in the ROS conda env is headless.
        import tkinter as tk
        root = tk.Tk()
        root.title("depth_viewer   left: raw depth   right: policy input   (s = save, q = quit)")
        label = tk.Label(root, text=f"waiting for {args.topic} ...", width=60, height=10)
        label.pack()
        root.bind("s", save)
        root.bind("q", lambda _e: root.destroy())
        root.bind("<Escape>", lambda _e: root.destroy())

        def tick():
            if not rclpy.ok():
                root.destroy()
                return
            frame = process()
            if frame is not None:
                state["frame"] = frame
                _, left, policy_in = frame
                h = 2 * args.image_size
                left = cv2.resize(left, (round(left.shape[1] * h / left.shape[0]), h),
                                  interpolation=cv2.INTER_NEAREST)
                right = cv2.resize(cv2.cvtColor(policy_in, cv2.COLOR_GRAY2BGR), (h, h),
                                   interpolation=cv2.INTER_NEAREST)
                cv2.putText(left, f"raw 0-{args.view_max_m:g} m", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                cv2.putText(right, "policy input", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                rgb = cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_BGR2RGB)
                ppm = f"P6 {rgb.shape[1]} {rgb.shape[0]} 255 ".encode() + rgb.tobytes()
                photo = tk.PhotoImage(data=ppm, format="PPM")
                label.configure(image=photo, width=rgb.shape[1], height=rgb.shape[0])
                label.image = photo        # keep a reference or Tk drops the image
            root.after(30, tick)

        tick()
        root.mainloop()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
