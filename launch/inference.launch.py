import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory("ros2_policy_inference")
    params_file = os.path.join(pkg_share, "config", "params.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("policy_name", default_value="dummy"),
        DeclareLaunchArgument("inference_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("action_topic", default_value="/il/grasp/action"),
        Node(
            package="ros2_policy_inference",
            executable="inference_node",
            name="policy_inference_node",
            output="screen",
            parameters=[
                params_file,
                {
                    "policy_name": LaunchConfiguration("policy_name"),
                    "inference_rate_hz": ParameterValue(LaunchConfiguration("inference_rate_hz"), value_type=float),
                    "action_topic": ParameterValue(LaunchConfiguration("action_topic"), value_type=str),
                },
            ],
        ),
    ])
