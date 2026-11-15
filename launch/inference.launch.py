import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Node settings a policy's config.yaml may set in a `node:` section. For each one the value comes from, in order:
# the key:=value launch argument, the policy's `node:` section, config/params.yaml.
_POLICY_NODE_KEYS = {"action_topic": str, "inference_rate_hz": float, "use_sim_time": bool}


def _as(kind, value):
    if kind is bool and isinstance(value, str):
        if value.strip().lower() not in ("true", "false"):
            raise ValueError(f"expected true / false, got {value!r}")
        return value.strip().lower() == "true"
    return kind(value)


def _policy_node_settings(policy_name):
    """The `node:` section of policies/<policy_name>/config.yaml ({} without one) and that file's path."""
    import ros2_policy_inference

    candidates = (
        os.path.join(os.path.dirname(ros2_policy_inference.__file__), "policies", policy_name, "config.yaml"),
        # colcon only installs .py files: the mounted source tree in the container
        f"/workspace/src/ros2_policy_inference/ros2_policy_inference/policies/{policy_name}/config.yaml",
    )
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        return {}, None
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get("node") or {}, path


def _node(context):
    policy_name = LaunchConfiguration("policy_name").perform(context)
    policy_node, path = _policy_node_settings(policy_name)
    params = {"policy_name": policy_name, "ckpt_path": LaunchConfiguration("ckpt_path").perform(context)}
    origin = {}
    for key, kind in _POLICY_NODE_KEYS.items():
        given = LaunchConfiguration(key).perform(context)
        if given != "":
            params[key], origin[key] = _as(kind, given), "launch argument"
        elif key in policy_node:
            params[key], origin[key] = _as(kind, policy_node[key]), f"policies/{policy_name}/config.yaml"
    unknown = sorted(set(policy_node) - set(_POLICY_NODE_KEYS))
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {unknown} in `node:` (allowed: {sorted(_POLICY_NODE_KEYS)})")

    actions = [LogInfo(msg=f"[inference.launch] policy {policy_name}: "
                           + (", ".join(f"{k}={params[k]} ({origin[k]})" for k in origin) or "settings from config/params.yaml"))]
    if params.get("use_sim_time"):
        actions.append(LogInfo(msg="[inference.launch] use_sim_time: the node only runs while /clock advances "
                                   "(delta_extension Test Place publishes it while inserting)"))
    params_file = os.path.join(get_package_share_directory("ros2_policy_inference"), "config", "params.yaml")
    actions.append(Node(
        package="ros2_policy_inference",
        executable="inference_node",
        name="policy_inference_node",
        output="screen",
        parameters=[params_file, params],
    ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("policy_name", default_value="dummy"),
        DeclareLaunchArgument("ckpt_path", default_value=""),
        # "" = not given: see _POLICY_NODE_KEYS
        DeclareLaunchArgument("inference_rate_hz", default_value=""),
        DeclareLaunchArgument("action_topic", default_value=""),
        DeclareLaunchArgument("use_sim_time", default_value=""),
        OpaqueFunction(function=_node),
    ])
