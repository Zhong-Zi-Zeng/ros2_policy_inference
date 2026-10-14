#!/usr/bin/env bash
# Runs view_depth.py in the lerobot312-ros conda env (ROS 2 Jazzy + rclpy), on the same
# ROS_DOMAIN_ID as the sim/policy (default 0). Any view_depth.py args pass through, e.g.
#   ./run_depth_viewer.sh --view-max-m 1.5
#   ./run_depth_viewer.sh --topic /camera_R/depth/aligned_depth_to_color
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${DEPTH_VIEWER_PYTHON:-$HOME/miniconda3/envs/lerobot312-ros/bin/python}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
exec "$PY" "$SCRIPT_DIR/view_depth.py" "$@"
