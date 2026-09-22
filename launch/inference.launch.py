import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("ros2_policy_inference")
    params_file = os.path.join(pkg_share, "config", "params.yaml")

    return LaunchDescription([
        Node(
            package="ros2_policy_inference",
            executable="inference_node",
            name="policy_inference_node",
            output="screen",
            parameters=[params_file],
        ),
    ])
