#!/usr/bin/env bash
# Downloads the pi05 (π₀.₅) grasp_fan checkpoint from the GitLab Package
# Registry into ./checkpoints/pi05_grasp_fan/ (repo root), where docker-compose.yml's checkpoints
# volume mounts them into the container at /checkpoints/pi05_grasp_fan/ (see config.yaml).
#
# Usage:
#   ./fetch_weights.sh [checkpoint_version]   # default: grasp_fan
#   The checkpoint (~8.8G) is over GitLab's per-file limit, so it is stored as an uncompressed
#   tar split into 4G parts, plus parts.txt (part names) and sha256sums.txt. To upload a new one:
#     tar -cf - -C checkpoints/pi05_grasp_fan pretrained_model depth_info.json \
#       | split -b 4G -d -a 2 - checkpoint.tar.part
#     ls checkpoint.tar.part* > parts.txt && sha256sum checkpoint.tar.part* > sha256sums.txt
#   then upload every file to generic package pi05-weights/<version>/.
#   (reads GITLAB_TOKEN from the env, or falls back to ~/.gitlab_token; scope: read_api)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT_DIR="$REPO_ROOT/checkpoints/pi05_grasp_fan"

CKPT_VERSION="${1:-grasp_fan}"
PROJECT="mis-miis-robotics%2Fdelta%2Fros2_policy_inference"
API="http://miislab-gitlab.synology.me:32769/api/v4/projects/$PROJECT/packages/generic"

: "${GITLAB_TOKEN:=$(cat "$HOME/.gitlab_token" 2>/dev/null || true)}"
if [ -z "${GITLAB_TOKEN:-}" ]; then
    echo "[ERROR] No GitLab token found. Either export \$GITLAB_TOKEN or save one to"
    echo "        ~/.gitlab_token (scope: read_api)."
    exit 1
fi

if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
    echo "[fetch_weights] $OUT_DIR already populated, skipping"
    exit 0
fi

URL="$API/pi05-weights/$CKPT_VERSION"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

get() {
    curl --fail --location --header "PRIVATE-TOKEN: $GITLAB_TOKEN" -o "$TMP/$1" "$URL/$1"
}

echo "[fetch_weights] downloading pi05-weights/$CKPT_VERSION..."
get parts.txt
get sha256sums.txt
while read -r part; do
    echo "[fetch_weights]   $part"
    get "$part"
done < "$TMP/parts.txt"
(cd "$TMP" && sha256sum --quiet -c sha256sums.txt)

mkdir -p "$OUT_DIR"
cat $(sed "s|^|$TMP/|" "$TMP/parts.txt") | tar -xf - -C "$OUT_DIR"
echo "[fetch_weights] extracted into $OUT_DIR"

echo "[fetch_weights] done."
