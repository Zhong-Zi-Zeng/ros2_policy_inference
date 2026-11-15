#!/usr/bin/env bash
# Runs tools/sim_obs_adapter.py on the host with Isaac Sim's bundled ROS 2 (rclpy + Fast DDS), i.e. on the same
# DDS stack as Isaac Sim, whose topics the inference node's container already receives. No system ROS needed.
#
#   conda activate <the env Isaac Sim is pip-installed in>
#   bash tools/run_sim_obs_adapter_host.sh [sim_obs_adapter.py options]
# PYTHON=<python of that env> works too, without activating it.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
ROOT="$("$PYTHON" -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("isaacsim").origin))')" \
    || { echo "[sim_obs_adapter] $PYTHON has no isaacsim package: activate the Isaac Sim env (or set PYTHON=...)" >&2; exit 1; }

export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"   # same default as delta_extension's launch_isaacsim.sh
BRIDGE="$ROOT/exts/isaacsim.ros2.bridge/$ROS_DISTRO"
[ -d "$BRIDGE/rclpy" ] || { echo "[sim_obs_adapter] no bundled rclpy at $BRIDGE/rclpy" >&2; exit 1; }
export PYTHONPATH="$BRIDGE/rclpy${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$BRIDGE/lib"

echo "[sim_obs_adapter] $PYTHON, ROS 2 $ROS_DISTRO from $BRIDGE, $RMW_IMPLEMENTATION, ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
exec "$PYTHON" "$DIR/sim_obs_adapter.py" "$@"
