# 🤖 ROS 2 Policy Inference

Run IL / VLA policies with camera and robot observations.
Publish predicted end-effector actions to `/il/grasp/action`.

## 🚀 Run

Set `policy_name` in `config/params.yaml`: `dummy`, `act`, or `pi`.

> 💡 Start with `dummy` to check inputs and preprocessing without a checkpoint.

### 🐳 Docker

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
./run_docker.sh        # Start inference
./run_docker.sh bash   # Open a shell
```

The script pulls `lerobot_ros2` from the GitLab Package Registry if unavailable locally.
The image contains dependencies only; repository code is mounted and rebuilt at startup.
After `git pull`, restart the container to apply code changes.

### 💻 Local

Requires ROS 2 Jazzy and the policy's dependencies.

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ros2_policy_inference
source install/setup.bash
ros2 launch ros2_policy_inference inference.launch.py
```

### 📡 Check output

```bash
ros2 topic echo /il/grasp/action
```

## 🔌 ROS Interface

### Inputs

| Topic | Message type | Preprocessed observation |
|:--|:--|:--|
| `/camera_L/color/image_raw/compressed` | `CompressedImage` | RGB: `256 × 256 × 3` |
| `/camera_L/depth/aligned_depth_to_color` | `Image` | Depth: `256 × 256`, nearest-neighbor resize |
| `/left_arm/joint_states` | `JointState` | Position: `(6,)`; velocity: `(6,)` |
| `/left_arm/tcp_pose_world` | `PoseStamped` | EE-offset position: `(3,)`; quaternion: `(4,)` |
| `/left_ft/wrench_compensated` | `WrenchStamped` | `[fx, fy, fz, tx, ty, tz]` |

### Output

| Topic | Message type | Format |
|:--|:--|:--|
| `/il/grasp/action` | `Float64MultiArray` | `[dx, dy, dz, drx, dry, drz, gripper]` |

## 🛠️ Development

### Add a policy

1. Subclass `PolicyModel` in `policies/your_policy.py`; see `policies/dummy_policy.py`.
2. Define observation and action specs in `policies/base_policy.py`.
3. Register the policy in `policies/__init__.py` through `create_policy()`.

### Publish a Docker image

Rebuild and publish only when the Dockerfile or image dependencies change:

```bash
GITLAB_TOKEN=<your-token> ./push_docker.sh
```