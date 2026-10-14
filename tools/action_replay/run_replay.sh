#!/usr/bin/env bash
# Runs replay_actions.py in the lerobot_ros2 image (no ROS needed on the host).
# Results (.csv + .html report) go to tools/action_replay/results/, owned by you.
#
# Usage:
#   ./run_replay.sh                        # wait for the sim's IL hand-off, replay demo 12
#   ./run_replay.sh --max-actions 300      # any replay_actions.py option (see --help)
#
# Uses Fast DDS, the same middleware as Isaac Sim, instead of the image's default
# CycloneDDS: Isaac crashed in libfastrtps at the IL hand-off while a CycloneDDS
# node was subscribed to /il/grasp/handoff (2026-10-09 15:11). UDPv4 because Fast
# DDS's shared-memory transport doesn't work between the host and a container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/results"

docker run --rm -it \
    --name action_replay \
    --network host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e ROS_LOG_DIR=/tmp/ros_log \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    -v "$SCRIPT_DIR":/action_replay \
    --entrypoint bash \
    lerobot_ros2:latest -c 'source /opt/ros/jazzy/setup.bash && exec python3 /action_replay/replay_actions.py "$@"' _ "$@"
