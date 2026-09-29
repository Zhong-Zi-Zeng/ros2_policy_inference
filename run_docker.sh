#!/usr/bin/env bash
# Auto-pulls the environment image from the GitLab Package Registry if it
# isn't already present locally, then runs docker compose.
#
# The image (ROS2 + CUDA + torch + lerobot) is environment-only — this
# repo's code is mounted in and colcon-built fresh by entrypoint.sh on every
# container start, so pulling an older image is fine as long as the
# Dockerfile itself hasn't changed. Rebuild + push_docker.sh only when it has.
#
# Usage:
#   ./run_docker.sh          # docker compose up (launches inference_node)
#   ./run_docker.sh bash     # interactive shell instead
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="lerobot_ros2:latest"
PACKAGE_URL="http://miislab-gitlab.synology.me:32769/api/v4/projects/mis-miis-robotics%2Fdelta%2Fros2_policy_inference/packages/generic/lerobot-ros2-docker/1.0.0/lerobot_ros2.tar.gz"

# Falls back to a token saved on disk so you don't have to export it every
# time. Create it once with: printf '%s' 'glpat-...' > ~/.gitlab_token && chmod 600 ~/.gitlab_token
: "${GITLAB_TOKEN:=$(cat "$HOME/.gitlab_token" 2>/dev/null || true)}"

if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "[run_docker] $IMAGE_NAME not found locally, pulling from GitLab Package Registry..."
    if [ -z "${GITLAB_TOKEN:-}" ]; then
        echo "[ERROR] No GitLab token found. Either export \$GITLAB_TOKEN, save one to"
        echo "        ~/.gitlab_token (scope: read_api), or build locally instead:"
        echo "        docker compose build"
        exit 1
    fi

    TMP_TAR="$(mktemp --suffix=.tar.gz)"
    trap 'rm -f "$TMP_TAR"' EXIT
    curl --fail --location --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
        -o "$TMP_TAR" "$PACKAGE_URL"
    gunzip -c "$TMP_TAR" | docker load
fi

if [ $# -eq 0 ]; then
    docker compose up
else
    docker compose run --rm policy-inference "$@"
fi
