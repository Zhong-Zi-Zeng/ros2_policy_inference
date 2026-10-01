# syntax=docker/dockerfile:1
#
# Environment-only image — ROS2 Jazzy + CUDA-enabled torch + lerobot. No
# application code is baked in: code lives in this repo (GitLab-managed) and
# is mounted into the container at runtime. entrypoint.sh colcon-builds
# whatever's currently mounted every time the container starts, so a
# `git pull` on the host + container restart always runs the latest code.
#
#   docker build -f docker/Dockerfile -t lerobot_ros2:latest .   # run from repo root
#
# or: ./run_docker.sh (auto-pulls a prebuilt image from the GitLab Package
# Registry if one isn't present locally) / docker compose -f docker/docker-compose.yml up

FROM ros:jazzy-ros-base
SHELL ["/bin/bash", "-c"]

# Match TORCH_INDEX_URL to whatever CUDA build your host's NVIDIA driver
# actually supports, and pin LEROBOT_VERSION to whatever you trained /
# received the checkpoint with — e.g.:
#   docker build --build-arg LEROBOT_VERSION=0.1.0 -f docker/Dockerfile .
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
        ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Avoids Fast-DDS's same-host shared-memory transport, which silently drops
# everything since the container's /dev/shm isn't the host's.
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir \
        --extra-index-url ${TORCH_INDEX_URL} torch torchvision \
    && pip install --no-cache-dir \
        lerobot${LEROBOT_VERSION:+==${LEROBOT_VERSION}} \
        scipy

# colcon itself must resolve to the venv's python, or built scripts can't see pip-installed deps.
RUN pip install --no-cache-dir --force-reinstall --no-deps colcon-core

WORKDIR /workspace
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "ros2_policy_inference", "inference.launch.py"]
