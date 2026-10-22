"""
Standalone ROS2 policy-inference node: subscribes to the real robot's
camera + proprioception topics, preprocesses them into the observation
format an IL/VLA policy was trained on, runs one prediction per timer
tick, and publishes the resulting EE-offset action.

Gripper state is not subscribed from the robot: the arm is always open
before an episode starts, and afterwards the policy's own previous
gripper output is fed back in as the next step's gripper observation
(see _gripper_state / _build_obs).

To deploy a trained model, add a PolicyModel subclass under policies/ (see
policies/base_policy.py for the interface), register it in
policies/__init__.py's create_policy(), and select it via the policy_name
parameter (config/params.yaml or `-p policy_name:=...`).
"""
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node

from .policies import create_policy
from .policies.base_policy import ACTION_DIM
from .preprocessing import resize_image
from .ros_bridge import RosBridge

_DEFAULT_PARAMS = dict(
    joint_state_topic="/left_arm/joint_states",
    rgb_topic="/camera_L/color/image_raw/compressed",
    depth_topic="/camera_L/depth/aligned_depth_to_color",
    ft_sensor_topic="/left_ft/wrench_compensated",
    action_topic="/il/grasp/action",
    # Which policies/*.py implementation to run — see policies/__init__.py.
    policy_name="dummy",
    inference_rate_hz=10.0,
    topic_timeout_sec=2.0,
    image_size=256,
    initial_gripper_open=True,
    gripper_open_threshold=0.5,
)


@dataclass
class InferenceConfig:
    joint_state_topic: str
    rgb_topic: str
    depth_topic: str
    ft_sensor_topic: str
    action_topic: str
    policy_name: str
    inference_rate_hz: float
    topic_timeout_sec: float
    image_size: int
    initial_gripper_open: bool
    gripper_open_threshold: float


class PolicyInferenceNode(Node):
    def __init__(self, policy=None):
        super().__init__("policy_inference_node")
        cfg = self._load_config()
        self.cfg = cfg

        self.bridge = RosBridge(self, cfg)
        self.policy = policy if policy is not None else create_policy(cfg.policy_name)
        self.policy.reset()

        self._gripper_state = 1.0 if cfg.initial_gripper_open else 0.0

        self.create_timer(1.0 / cfg.inference_rate_hz, self._on_tick)
        self.create_timer(1.0, self._on_health_check)

    def _load_config(self):
        for name, default in _DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)
        values = {name: self.get_parameter(name).value for name in _DEFAULT_PARAMS}
        return InferenceConfig(**values)

    def _on_tick(self):
        ros_data = self.bridge.get_latest()
        if ros_data is None:
            self.get_logger().warn(
                "waiting for all observation topics before first inference...",
                throttle_duration_sec=5.0,
            )
            return

        health = self.bridge.check_health(self.cfg.topic_timeout_sec)
        if any(status != "ok" for status in health.values()):
            self.get_logger().warn(
                "observation data is stale, skipping inference tick",
                throttle_duration_sec=5.0,
            )
            return

        obs = self._build_obs(ros_data)
        action = np.asarray(self.policy.predict(obs), dtype=np.float64)
        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f"policy returned action of shape {action.shape}, "
                f"expected ({ACTION_DIM},)"
            )

        self.bridge.publish_action(action)
        self.get_logger().info(f"action: {np.array2string(action, precision=4)}")
        # close ends this task; next task starts right after, so feed back "open" either way
        self._gripper_state = 1.0

    def _build_obs(self, ros_data):
        return {
            "rgb": resize_image(ros_data["rgb"], self.cfg.image_size, is_depth=False),
            "depth": resize_image(ros_data["depth"], self.cfg.image_size, is_depth=True),
            "joint_pos": ros_data["joint_pos"],
            "gripper_state": np.float32(self._gripper_state),
            "ft_wrench": ros_data["ft_wrench"],
        }

    def _on_health_check(self):
        """Warn (throttled) about any subscribed topic that has no
        publisher, has never delivered a message, or has gone quiet."""
        messages = {
            "no_publisher": "no publisher found for {topic} — check the topic name / that its node is running",
            "no_data": "publisher exists for {topic} but no message received yet — check QoS (reliability) compatibility",
            "stale": "no message on {topic} in over {timeout:.1f}s — it may have died",
        }
        for topic, status in self.bridge.check_health(self.cfg.topic_timeout_sec).items():
            if status == "ok":
                continue
            self.get_logger().warn(
                messages[status].format(topic=topic, timeout=self.cfg.topic_timeout_sec),
                throttle_duration_sec=5.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
