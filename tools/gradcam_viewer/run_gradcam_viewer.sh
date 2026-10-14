#!/usr/bin/env bash
# Runs view_gradcam.py in the lerobot312-ros conda env (ROS 2 Jazzy + rclpy), on the same
# ROS_DOMAIN_ID as the policy (default 0). Any view_gradcam.py args pass through, e.g.
#   ./run_gradcam_viewer.sh --save-dir /tmp/cams
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${GRADCAM_VIEWER_PYTHON:-$HOME/miniconda3/envs/lerobot312-ros/bin/python}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
exec "$PY" "$SCRIPT_DIR/view_gradcam.py" "$@"
