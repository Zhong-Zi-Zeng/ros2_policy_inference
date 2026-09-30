# 🤖 ROS 2 Policy Inference

Run IL / VLA policies with camera and robot observations.
Publish predicted end-effector actions to `/il/grasp/action`.

## 🔑 Setup

Get a GitLab token (scope: `read_api`) from your GitLab profile → Access Tokens, then save it once — scripts pick it up automatically:

```bash
printf '%s' 'glpat-...' > ~/.gitlab_token && chmod 600 ~/.gitlab_token
```

## 🚀 Run

> 💡 Start with `dummy` to check inputs and preprocessing without a checkpoint.

### 🐳 Docker

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
./run_docker.sh --policy-name act
./run_docker.sh --policy-name act bash   # shell instead of launching
```

| Policy | Status |
|:--|:--|
| `dummy` | no-op placeholder |
| `act` | not implemented |
| `pi` | not implemented |
| `liteflow` | InternVL3-1B + FLOWER, working |

### 💻 Local

Requires ROS 2 Jazzy and the policy's dependencies.

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ros2_policy_inference
source install/setup.bash
ros2 launch ros2_policy_inference inference.launch.py policy_name:=act
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

Rebuild and publish only when a Dockerfile or its dependencies change. Needs
a token with `api` scope (higher than the `read_api` one from Setup) in
`~/.gitlab_token` or `$GITLAB_TOKEN`:

```bash
./push_docker.sh             # act/pi/dummy image
./push_docker.sh --liteflow   # liteflow image
```