#!/usr/bin/env bash
# Auto-pulls the environment image from the GitLab Package Registry if it
# isn't already present locally (and, for liteflow, fetches its checkpoint
# if missing too — pi's checkpoint comes from Hugging Face instead, see README),
# then runs docker compose.
#
# --policy-name picks which policy to launch and, from that, which image:
# liteflow runs in its own image (ROS2 + CUDA + InternVL3/FLOWER — see
# Dockerfile.liteflow), everything else (act/pi/dummy) in the shared
# lerobot one. Either way, code is mounted in and colcon-built fresh by
# entrypoint.sh on every container start, so pulling an older image is
# fine as long as its Dockerfile hasn't changed. Rebuild + push_docker.sh
# only when it has.
#
# Usage:
#   ./run_docker.sh --policy-name act        # docker compose run, act policy
#   ./run_docker.sh --policy-name liteflow   # docker compose run, liteflow policy
#   ./run_docker.sh --policy-name liteflow bash   # interactive shell instead
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "${1:-}" != "--policy-name" ] || [ -z "${2:-}" ]; then
    echo "Usage: $0 --policy-name <dummy|act|pi|liteflow> [command...]" >&2
    exit 1
fi
POLICY_NAME="$2"
shift 2

if [ "$POLICY_NAME" = "liteflow" ]; then
    IMAGE_NAME="liteflow_ros2:latest"
    PACKAGE_NAME="liteflow-ros2-docker"
    SERVICE="policy-inference-liteflow"
else
    IMAGE_NAME="lerobot_ros2:latest"
    PACKAGE_NAME="lerobot-ros2-docker"
    SERVICE="policy-inference"
fi
PACKAGE_URL="http://miislab-gitlab.synology.me:32769/api/v4/projects/mis-miis-robotics%2Fdelta%2Fros2_policy_inference/packages/generic/$PACKAGE_NAME/1.0.0/${IMAGE_NAME%%:*}.tar.gz"

# Falls back to a token saved on disk so you don't have to export it every
# time. Create it once with: printf '%s' 'glpat-...' > ~/.gitlab_token && chmod 600 ~/.gitlab_token
: "${GITLAB_TOKEN:=$(cat "$HOME/.gitlab_token" 2>/dev/null || true)}"

if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "[run_docker] $IMAGE_NAME not found locally, pulling from GitLab Package Registry..."
    if [ -z "${GITLAB_TOKEN:-}" ]; then
        echo "[ERROR] No GitLab token found. Either export \$GITLAB_TOKEN, save one to"
        echo "        ~/.gitlab_token (scope: read_api), or build locally instead:"
        echo "        docker compose -f docker/docker-compose.yml build $SERVICE"
        exit 1
    fi

    TMP_TAR="$(mktemp --suffix=.tar.gz)"
    trap 'rm -f "$TMP_TAR"' EXIT
    curl --fail --location --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
        -o "$TMP_TAR" "$PACKAGE_URL"
    gunzip -c "$TMP_TAR" | docker load
fi

if [ "$POLICY_NAME" = "liteflow" ] && [ ! -s "checkpoints/liteflow/step_20000_ema/config.json" ]; then
    echo "[run_docker] liteflow checkpoint not found locally, fetching..."
    ./ros2_policy_inference/policies/liteflow/fetch_weights.sh
fi

if [ $# -eq 0 ]; then
    docker compose -f docker/docker-compose.yml run --rm "$SERVICE" \
        ros2 launch ros2_policy_inference inference.launch.py "policy_name:=$POLICY_NAME"
else
    docker compose -f docker/docker-compose.yml run --rm "$SERVICE" "$@"
fi
