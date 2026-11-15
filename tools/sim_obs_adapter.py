#!/usr/bin/env python3
"""Feed the inference node from Isaac Sim (delta_extension scene), for sim-only tests.

The scene's Camera_L graph publishes /camera_L/color/image_raw (raw, no compressed) and
/camera_L/depth/aligned_depth_to_color, and there is no F/T sensor. The node waits for all four of its input topics
(config/params.yaml), so this node fills in what is missing:
  - JPEG-compresses /camera_L/color/image_raw        -> /camera_L/color/image_raw/compressed
  - publishes a zero /left_ft/wrench_compensated at 30 Hz (the omnireset policy does not use F/T)
  - republishes   /sim/left_arm_joint_states         -> /left_arm/joint_states (the extension's virtual arm)
An output topic that something already publishes is left alone (its publishers are logged), so nothing is doubled.
Depth goes to the node as Isaac publishes it.

Run it on the host next to Isaac Sim, with Isaac Sim's bundled ROS 2 (conda env of Isaac Sim activated):
  bash tools/run_sim_obs_adapter_host.sh
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.executors import ExternalShutdownException
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState

SENSOR_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)


def image_to_bgr(msg):
    """sensor_msgs/Image (rgb8 / bgr8 / rgba8 / bgra8) -> BGR uint8 for cv2.imencode."""
    h, w, enc = msg.height, msg.width, msg.encoding
    c = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(enc)
    if c is None:
        raise ValueError(f"unsupported colour encoding {enc!r}")
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, msg.step)[:, : w * c].reshape(h, w, c)
    if enc.startswith("rgb"):
        return np.ascontiguousarray(a[..., 2::-1])
    return np.ascontiguousarray(a[..., :3])


class SimObsAdapter:
    def __init__(self, node, args, outputs):
        self._node = node
        self._args = args
        self._log = node.get_logger()
        self._counts = {"rgb": 0, "joint": 0, "ft": 0}
        self._seen = set()

        self._outputs = outputs
        if "rgb" in outputs:
            self._rgb_pub = node.create_publisher(CompressedImage, args.rgb_out, SENSOR_QOS)
            node.create_subscription(Image, args.rgb_in, self._on_rgb, SENSOR_QOS)
        if "joint" in outputs:
            self._joint_pub = node.create_publisher(JointState, args.joint_out, 10)
            node.create_subscription(JointState, args.joint_in, self._on_joint, 10)
        if "ft" in outputs:
            self._ft_pub = node.create_publisher(WrenchStamped, args.ft_out, 10)
            node.create_timer(1.0 / args.ft_rate, self._on_ft_timer)
        # only to log the depth format once: a 1280x720 float image at 30 Hz is not free to receive
        self._depth_sub = node.create_subscription(Image, args.depth_in, self._on_depth, SENSOR_QOS)
        node.create_timer(5.0, self._report)

    def _first(self, key, text):
        if key not in self._seen:
            self._seen.add(key)
            self._log.info(text)

    def _on_rgb(self, msg):
        try:
            bgr = image_to_bgr(msg)
        except ValueError as e:
            self._log.error(f"{self._args.rgb_in}: {e}", throttle_duration_sec=5.0)
            return
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._args.jpeg_quality])
        if not ok:
            self._log.error("JPEG encoding failed", throttle_duration_sec=5.0)
            return
        out = CompressedImage()
        out.header = msg.header
        out.format = f"{msg.encoding}; jpeg compressed bgr8"
        out.data = buf.tobytes()
        self._rgb_pub.publish(out)
        self._counts["rgb"] += 1
        self._first("rgb", f"colour {msg.width}x{msg.height} {msg.encoding} -> {self._args.rgb_out} (JPEG)")

    def _on_joint(self, msg):
        out = JointState()
        out.header = msg.header
        out.name, out.position, out.velocity, out.effort = msg.name, msg.position, msg.velocity, msg.effort
        self._joint_pub.publish(out)
        self._counts["joint"] += 1
        self._first("joint", f"joints {list(msg.name)} -> {self._args.joint_out}")

    def _on_depth(self, msg):
        self._first("depth", f"depth {msg.width}x{msg.height} {msg.encoding} on {self._args.depth_in} (used by the node as is)")
        if self._depth_sub is not None:
            self._node.destroy_subscription(self._depth_sub)
            self._depth_sub = None

    def _on_ft_timer(self):
        out = WrenchStamped()
        out.header.stamp = self._node.get_clock().now().to_msg()
        out.header.frame_id = "left_ft"
        self._ft_pub.publish(out)
        self._counts["ft"] += 1

    def _report(self):
        rates = {k: v / 5.0 for k, v in self._counts.items() if k in self._outputs}
        self._counts = dict.fromkeys(self._counts, 0)
        self._log.info("publishing Hz: " + ", ".join(f"{k} {v:.1f}" for k, v in rates.items()))
        if rates.get("rgb") == 0:
            self._log.warn(f"nothing on {self._args.rgb_in}: is Isaac Sim playing?")
        if rates.get("joint") == 0:
            self._log.warn(f"nothing on {self._args.joint_in}: is Isaac Sim playing (the extension publishes it after Play)?")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgb-in", default="/camera_L/color/image_raw")
    ap.add_argument("--rgb-out", default="/camera_L/color/image_raw/compressed")
    ap.add_argument("--depth-in", default="/camera_L/depth/aligned_depth_to_color")
    ap.add_argument("--joint-in", default="/sim/left_arm_joint_states")
    ap.add_argument("--joint-out", default="/left_arm/joint_states")
    ap.add_argument("--ft-out", default="/left_ft/wrench_compensated")
    ap.add_argument("--ft-rate", type=float, default=30.0)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--discovery-sec", type=float, default=3.0, help="wait this long for the ROS graph before the checks")
    args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = rclpy.create_node("sim_obs_adapter")
    log = node.get_logger()

    # only fill in topics nobody publishes yet, so nothing reaches the node twice
    end = time.monotonic() + args.discovery_sec
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    outputs = []
    for key, topic in (("rgb", args.rgb_out), ("joint", args.joint_out), ("ft", args.ft_out)):
        publishers = sorted({f"{i.node_namespace.rstrip('/')}/{i.node_name}" for i in node.get_publishers_info_by_topic(topic)})
        if publishers:
            log.info(f"{topic}: already published by {', '.join(publishers)}, left alone")
        else:
            outputs.append(key)
    if not outputs:
        log.info("all of the node's inputs are already published, nothing to do")
        node.destroy_node()
        rclpy.shutdown()
        return 0

    SimObsAdapter(node, args, outputs)
    log.info(f"publishing {', '.join(outputs)} (ROS_DOMAIN_ID {os.environ.get('ROS_DOMAIN_ID', '0')})")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):   # Ctrl+C / docker stop (SIGTERM)
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
