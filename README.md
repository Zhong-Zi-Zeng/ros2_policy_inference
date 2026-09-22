# ROS 2 Policy Inference

Run a trained **IL / VLA policy** on a real robot arm.

**Input:** camera + robot proprioception  
**Output:** predicted end-effector action

---

## Quick Start

### Build

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ros2_policy_inference
source install/setup.bash
```

### Launch

```bash
ros2 launch ros2_policy_inference inference.launch.py
```

### Inspect Actions

```bash
ros2 topic echo /il/grasp/action
```

---

## Data Flow

```text
Camera / Robot Sensors
          │
          ▼
    Preprocessing
          │
          ▼
    Policy Inference
          │
          ▼
   /il/grasp/action
```

---

## ROS Topics

### Inputs

| Source | Topic | Type | Processed as |
|:--|:--|:--|:--|
| RGB camera | `/camera_L/color/image_raw/compressed` | `CompressedImage` | RGB image, `256 × 256 × 3` |
| Depth camera | `/camera_L/depth/aligned_depth_to_color` | `Image` | Depth image, `256 × 256` *(nearest resize)* |
| Joint state | `/left_arm/joint_states` | `JointState` | Position `(6,)`, velocity `(6,)` |
| TCP pose | `/left_arm/tcp_pose_world` | `PoseStamped` | EE-offset position `(3,)`, quaternion `(4,)` |
| Force / torque | `/left_ft/wrench_compensated` | `WrenchStamped` | `[fx, fy, fz, tx, ty, tz]` `(6,)` |

### Output

| Topic | Type | Action format |
|:--|:--|:--|
| `/il/grasp/action` | `Float64MultiArray` | `[dx, dy, dz, drx, dry, drz, gripper]` |

---

## Select a Policy

Set `policy_name` to any registered policy, such as `dummy`, `act`, or `pi`.

```bash
ros2 launch ros2_policy_inference inference.launch.py \
  --ros-args -p policy_name:=act
```

Or set the default in:

```text
config/params.yaml
```

---

## Add a New Policy

1. Create a `PolicyModel` subclass under `policies/`.
2. Define its observation and action specs in `policies/base_policy.py`.
3. Use `policies/dummy_policy.py` as a minimal reference.
4. Register the policy in `policies/__init__.py` via `create_policy()`.

```text
policies/
├── base_policy.py      # Policy interface and specs
├── dummy_policy.py     # Minimal example
└── your_policy.py      # Your implementation
```

---

> **Tip:** Start with `policy_name:=dummy` to verify ROS topics and preprocessing before loading a real checkpoint.