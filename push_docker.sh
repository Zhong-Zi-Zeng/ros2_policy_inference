#!/usr/bin/env bash
# Builds the environment image and uploads it to the GitLab Package
# Registry so other machines can `./run_docker.sh` without rebuilding.
#
# Only needed when a Dockerfile changes (new deps, CUDA/torch version bump,
# etc) — code changes don't need this, they're picked up by entrypoint.sh's
# colcon build on every container start.
#
# Usage:
#   ./push_docker.sh [version]             # default image, default version 1.0.0
#   ./push_docker.sh --liteflow [version]  # liteflow image
#   (reads GITLAB_TOKEN from the env, or falls back to ~/.gitlab_token; needs 'api' scope)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "${1:-}" = "--liteflow" ]; then
    shift
    IMAGE_NAME="liteflow_ros2:latest"
    PACKAGE_NAME="liteflow-ros2-docker"
    SERVICE="policy-inference-liteflow"
else
    IMAGE_NAME="lerobot_ros2:latest"
    PACKAGE_NAME="lerobot-ros2-docker"
    SERVICE="policy-inference"
fi
VERSION="${1:-1.0.0}"
PACKAGE_URL="http://miislab-gitlab.synology.me:32769/api/v4/projects/mis-miis-robotics%2Fdelta%2Fros2_policy_inference/packages/generic/$PACKAGE_NAME/${VERSION}/${IMAGE_NAME%%:*}.tar.gz"

: "${GITLAB_TOKEN:=$(cat "$HOME/.gitlab_token" 2>/dev/null || true)}"

if [ -z "${GITLAB_TOKEN:-}" ]; then
    echo "[ERROR] No GitLab token found (needs 'api' scope to upload). Either export"
    echo "        \$GITLAB_TOKEN or save one to ~/.gitlab_token. Get one at:"
    echo "        http://miislab-gitlab.synology.me:32769/-/user_settings/personal_access_tokens"
    exit 1
fi

echo "[push_docker] Building $IMAGE_NAME..."
docker compose build "$SERVICE"

TMP_TAR="$(mktemp --suffix=.tar.gz)"
trap 'rm -f "$TMP_TAR"' EXIT
echo "[push_docker] Saving image..."
docker save "$IMAGE_NAME" | gzip > "$TMP_TAR"

echo "[push_docker] Uploading to $PACKAGE_URL..."
curl --fail --location --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
    --upload-file "$TMP_TAR" "$PACKAGE_URL"
echo "[push_docker] Done."
