"""
Subscribes to the real robot's camera + proprioception topics used as
policy observations.

Each topic is subscribed independently; there is NO time alignment.
get_latest() simply returns the most recent message seen on every topic —
the same approach ros2_data_collector's RosBridge uses, and for the same
reason: robust to topics running at different rates or with
un-synchronised header.stamps.

QoS — camera topics are typically published BEST_EFFORT (sensor-data QoS);
a RELIABLE subscriber silently receives nothing from them, so cameras use
a BEST_EFFORT profile while joint_state/F-T stay RELIABLE.

Gripper state is intentionally NOT subscribed here — see
PolicyInferenceNode, which feeds the policy's own previous gripper output
back in as the next step's gripper observation instead.
"""
import time

import numpy as np
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState
from geometry_msgs.msg import WrenchStamped

# image source -> (config topic attribute, message type)
_IMAGE_STREAMS = {
    "rgb": ("rgb_topic", CompressedImage),
    "depth": ("depth_topic", Image),
    "head_rgb": ("head_rgb_topic", CompressedImage),
    "head_depth": ("head_depth_topic", Image),
}


class RosBridge:
    def __init__(self, node, cfg, image_sources=("rgb", "depth")):
        self._node = node
        self._image_sources = tuple(image_sources)

        self._joint = None
        self._ft = None
        self._images = {}

        # topic key -> topic name, for check_health(); last_seen tracked
        # separately (monotonic seconds) so health checks don't depend on
        # any particular clock source.
        self._topics = {"joint": cfg.joint_state_topic}
        self._last_seen = {}

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        log = node.get_logger()
        log.info("RosBridge topics:")
        node.create_subscription(JointState, cfg.joint_state_topic, self._on_joint, 10)
        log.info(f"  joint       = {cfg.joint_state_topic}")
        for source in self._image_sources:
            attr, msg_type = _IMAGE_STREAMS[source]
            topic = getattr(cfg, attr)
            self._topics[source] = topic
            node.create_subscription(msg_type, topic, self._image_callback(source, msg_type), sensor_qos)
            log.info(f"  {source:<11} = {topic}  ({msg_type.__name__}, BEST_EFFORT)")
        node.create_subscription(WrenchStamped, cfg.ft_sensor_topic, self._on_ft, 10)
        self._topics["ft"] = cfg.ft_sensor_topic
        log.info(f"  ft          = {cfg.ft_sensor_topic}")

    # ---------- per-topic callbacks ----------

    def _on_joint(self, msg):
        self._joint = {"joint_pos": np.asarray(msg.position, dtype=np.float32)}
        self._last_seen["joint"] = time.monotonic()

    def _image_callback(self, source, msg_type):
        decode = self._compressed_msg_to_array if msg_type is CompressedImage else self._image_msg_to_array

        def on_image(msg):
            self._images[source] = decode(msg)
            self._last_seen[source] = time.monotonic()
        return on_image

    def _on_ft(self, msg):
        f = msg.wrench.force
        t = msg.wrench.torque
        self._ft = {
            "ft_wrench": np.array([f.x, f.y, f.z, t.x, t.y, t.z], dtype=np.float32),
        }
        self._last_seen["ft"] = time.monotonic()

    @staticmethod
    def _image_msg_to_array(msg):
        h, w, enc = msg.height, msg.width, msg.encoding

        if enc in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            if enc == "bgr8":
                arr = arr[..., ::-1]
            return np.ascontiguousarray(arr)

        if enc in ("rgba8", "bgra8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)[..., :3]
            if enc == "bgra8":
                arr = arr[..., ::-1]
            return np.ascontiguousarray(arr)

        if enc in ("16UC1", "mono16"):
            return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).copy()

        if enc == "32FC1":
            return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()

        if enc == "mono8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w).copy()

        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1).copy()

    @staticmethod
    def _compressed_msg_to_array(msg):
        import cv2
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(
                f"cv2.imdecode failed (format='{msg.format}', {len(msg.data)} bytes)"
            )
        return np.ascontiguousarray(bgr[..., ::-1])

    def get_latest(self):
        """Most recent message from every topic, merged into one dict.
        Returns None until every topic has been received at least once."""
        if (self._joint is None or self._ft is None
                or any(s not in self._images for s in self._image_sources)):
            return None
        out = {}
        out.update(self._joint)
        out.update(self._ft)
        out.update({s: self._images[s] for s in self._image_sources})
        return out

    def received_since(self, t):
        """True once joint + every camera topic has delivered a message after monotonic time t."""
        return all(self._last_seen.get(k, -1.0) > t for k in ("joint", *self._image_sources))

    def check_health(self, timeout_sec):
        """{topic_name: status} for every subscribed topic, status one of:
          - "ok"           received a message within the last timeout_sec
          - "no_publisher" nothing on the ROS graph is publishing this topic
          - "no_data"      a publisher exists, but no message has arrived yet
                           (e.g. a QoS mismatch — publisher BEST_EFFORT vs a
                           RELIABLE subscription silently drops everything)
          - "stale"        was receiving messages, but not within timeout_sec
                           (e.g. the camera/driver died mid-session)
        """
        now = time.monotonic()
        result = {}
        for key, topic in self._topics.items():
            last = self._last_seen.get(key)
            if last is None:
                publishers = self._node.count_publishers(topic)
                status = "no_data" if publishers > 0 else "no_publisher"
            elif now - last > timeout_sec:
                status = "stale"
            else:
                status = "ok"
            result[topic] = status
        return result
