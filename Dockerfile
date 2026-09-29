# syntax=docker/dockerfile:1
#
# Environment-only image — ROS2 Jazzy + CUDA-enabled torch + lerobot. No
# application code is baked in: code lives in this repo (GitLab-managed) and
# is mounted into the container at runtime. entrypoint.sh colcon-builds
# whatever's currently mounted every time the container starts, so a
# `git pull` on the host + container restart always runs the latest code.
#
#   docker build -t lerobot_ros2:latest .
#   docker run -it --network host --gpus all \
#     -e ROS_DOMAIN_ID=0 \
#     -v $(pwd):/workspace/src/ros2_policy_inference \
#     -v /path/to/checkpoints:/checkpoints:ro \
#     lerobot_ros2:latest
#
# or: ./run_docker.sh (auto-pulls a prebuilt image from the GitLab Package
# Registry if one isn't present locally) / docker compose up

FROM ros:jazzy-ros-base
SHELL ["/bin/bash", "-c"]

# Match TORCH_INDEX_URL to whatever CUDA build your host's NVIDIA driver
# actually supports, and pin LEROBOT_VERSION to whatever you trained /
# received the checkpoint with — e.g.:
#   docker build --build-arg LEROBOT_VERSION=0.1.0 .
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG LEROBOT_VERSION=

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-venv \
        python3-colcon-common-extensions \
        ffmpeg \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# A plain `pip install` here hits two walls: PEP 668's "externally managed"
# guard (the same one that blocks a bare `pip install` on the host — see
# README Troubleshooting), and — worse — lerobot's transitive deps want to
# upgrade packages apt already installed (e.g. `packaging`), which pip
# refuses since it has no RECORD file for anything dpkg installed. A venv
# sidesteps both: pip-installed packages land in its own isolated
# site-packages instead of fighting the system ones. --system-site-packages
# keeps rclpy/ament_index_python (apt-installed) visible from inside it.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# torch first, with the CUDA-enabled wheel, so `pip install lerobot`
# afterwards sees it already satisfied instead of silently pulling in a
# CPU-only build as a transitive dependency.
RUN pip install --no-cache-dir \
        --extra-index-url ${TORCH_INDEX_URL} torch torchvision \
    && pip install --no-cache-dir \
        lerobot${LEROBOT_VERSION:+==${LEROBOT_VERSION}} \
        scipy

WORKDIR /workspace
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "ros2_policy_inference", "inference.launch.py"]
