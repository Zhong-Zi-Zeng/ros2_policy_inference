#!/usr/bin/env bash
# Runs action_monitor.py in the lerobot_ros2 image (no ROS needed on the host),
# then open http://localhost:8765 in a browser. Read-only: it never publishes.
#
# Usage:
#   ./run_action_monitor.sh                         # watch /il/grasp/action on port 8765
#   ./run_action_monitor.sh --topic /x --port 9000  # any action_monitor.py args
#
# Save PNG / Save CSV write <policy>_01, <policy>_02, ... into RECORDINGS_DIR
# (default: <repo>/recordings), owned by you rather than root.
#
# ROS_DOMAIN_ID must match the policy and the sim (default 0). Runs on Fast DDS,
# the same middleware as Isaac Sim (Isaac crashed in libfastrtps once while a
# CycloneDDS node was connected); override with ACTION_MONITOR_RMW=rmw_cyclonedds_cpp.
# Fast DDS is forced to UDP: with shared memory a container discovers host
# publishers but never receives their messages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${ACTION_MONITOR_IMAGE:-lerobot_ros2:latest}"
RECORDINGS_DIR="$(realpath -m -- "${RECORDINGS_DIR:-$SCRIPT_DIR/../../recordings}")"
mkdir -p "$RECORDINGS_DIR"

if ! docker image inspect "$IMAGE" &> /dev/null; then
    echo "[ERROR] $IMAGE not found. Run ../../run_docker.sh once (it pulls or tells you how to build it)."
    exit 1
fi

docker run --rm -it \
    --name action_monitor \
    --network host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e ROS_LOG_DIR=/tmp/ros_log \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    -e RMW_IMPLEMENTATION="${ACTION_MONITOR_RMW:-rmw_fastrtps_cpp}" \
    -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    -v "$SCRIPT_DIR":/action_monitor:ro \
    -v "$RECORDINGS_DIR":/recordings \
    --entrypoint bash \
    "$IMAGE" -c 'source /opt/ros/jazzy/setup.bash && exec python3 /action_monitor/action_monitor.py \
        --save-dir /recordings --save-dir-label "$0" "$@"' "$RECORDINGS_DIR" "$@"
