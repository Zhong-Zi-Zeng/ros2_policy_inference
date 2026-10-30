"""OmniReset student (rl_games image_vec actor): camera + proprioception insertion policy from rl-template's
OmniResetEnv, either DAgger-distilled (distill_student.py) or RL fine-tuned (rl_games), configured by config.yaml."""
import os
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import yaml
from rclpy.logging import get_logger
from scipy.spatial.transform import Rotation

from ..base_policy import ACTION_DIM, PolicyModel

# colcon only installs .py files, so fall back to the mounted source tree in the container
_CONFIG_PATH = next(
    (p for p in (
        os.path.join(os.path.dirname(__file__), "config.yaml"),
        "/workspace/src/ros2_policy_inference/ros2_policy_inference/policies/omnireset/config.yaml",
    ) if os.path.isfile(p)),
    None,
)

IMG_SIZE = 256
VEC_DIM = 6 + 1 + ACTION_DIM   # arm joints, gripper state, previous action (OmniResetEnv._proprio)
CLIP_OBS = 5.0                 # rl_games clip_observations
# depth clipping / normalization of OmniResetEnvCfg (same as PickAndPlaceEnv, metres)
DEPTH_NEAR, DEPTH_FAR = 0.02, 0.60
DEPTH_MEAN, DEPTH_SCALE = 0.4463, 0.9062
CAM_KEYS = ("cam_channels", "cam_compact", "real_cam_model", "obs_history")


class _ResBlock(nn.Module):
    """agents/image_vec_network.py _ResBlock."""

    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.GroupNorm(8, c), nn.ReLU(), nn.Conv2d(c, c, 3, 1, 1),
                                  nn.GroupNorm(8, c), nn.ReLU(), nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x):
        return x + self.body(x)


def _resnet(in_c, channels):
    """agents/image_vec_network.py ResNetEncoder (cnn type resnet)."""
    layers = []
    for i, c in enumerate(channels):
        layers.append(_ResBlock(c))
        if i + 1 < len(channels):
            layers.append(nn.Conv2d(c, channels[i + 1], 3, 2, 1))
    layers += [nn.GroupNorm(8, channels[-1]), nn.ReLU()]
    return nn.Sequential(OrderedDict(stem=nn.Sequential(nn.Conv2d(in_c, channels[0], 8, 4, 2)), body=nn.Sequential(*layers)))


def _conv2d(in_c):
    """cnn type conv2d of agents/rl_games_omnireset.yaml."""
    return nn.Sequential(
        nn.Conv2d(in_c, 32, 8, 4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
    )


class _Actor(nn.Module):
    """Actor half of rl_games' image_vec network, built from the shapes in the state dict (cnn conv2d or resnet,
    any number of image channels). Attribute names match the checkpoint's a2c_network.* keys."""

    def __init__(self, sd):
        super().__init__()
        if "actor_cnn.stem.0.weight" in sd:
            in_c = sd["actor_cnn.stem.0.weight"].shape[1]
            # stage channels: the stem, then each stride-2 conv between stages
            down = sorted((int(m.group(1)), v.shape[0]) for k, v in sd.items()
                          if (m := re.fullmatch(r"actor_cnn\.body\.(\d+)\.weight", k)) and v.dim() == 4)
            self.cnn_type = "resnet"
            self.actor_cnn = _resnet(in_c, (sd["actor_cnn.stem.0.weight"].shape[0], *(c for _, c in down)))
        else:
            in_c = sd["actor_cnn.0.weight"].shape[1]
            self.cnn_type = "conv2d"
            self.actor_cnn = _conv2d(in_c)
        self.in_c = in_c
        mlp = sorted((int(m.group(1)), v.shape) for k, v in sd.items() if (m := re.fullmatch(r"actor_mlp\.(\d+)\.weight", k)))
        self.actor_mlp = nn.Sequential(*[m for _, (o, i) in mlp for m in (nn.Linear(i, o), nn.ELU())])
        self.mu = nn.Linear(*sd["mu.weight"].shape[::-1])
        with torch.no_grad():
            feat = self.actor_cnn(torch.zeros(1, in_c, IMG_SIZE, IMG_SIZE)).flatten(1).shape[1]
        if mlp[0][1][1] != feat + VEC_DIM:
            raise ValueError(f"actor_mlp input {mlp[0][1][1]} != cnn features {feat} + proprio {VEC_DIM}")

    def forward(self, img, vec):
        feat = self.actor_cnn(img).flatten(1)
        return self.mu(self.actor_mlp(torch.cat([feat, vec], dim=1)))


class _EnvYamlLoader(yaml.SafeLoader):
    """Isaac Lab's params/env.yaml has python tags (tuples, class names); only plain scalars are read here."""


_EnvYamlLoader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def _camera_setup(ckpt, ckpt_path):
    """How the actor's image was made (OmniResetEnvCfg cam_channels / cam_compact / real_cam_model / obs_history).
    distill_student.py stores it in the checkpoint; for rl_games (RL fine-tune) checkpoints it comes from the run's
    params/env.yaml, copied next to the checkpoint as <name>.env.yaml."""
    if "student" in ckpt:
        src, meta = ckpt_path, ckpt
    else:
        src = os.path.splitext(ckpt_path)[0] + ".env.yaml"
        if not os.path.isfile(src):
            raise FileNotFoundError(f"{ckpt_path} is an rl_games checkpoint: copy its run's params/env.yaml to {src}")
        with open(src) as f:
            meta = yaml.load(f, Loader=_EnvYamlLoader)
        if meta.get("ee_target_servo"):
            raise ValueError(f"{src}: ee_target_servo=True runs use a different action meaning, not supported")
    # keys missing in older checkpoints have the OmniResetEnvCfg defaults
    setup = {k: meta.get(k, d) for k, d in zip(CAM_KEYS, ("rgbd", False, False, 1))}
    return setup, src


class OmniResetPolicyModel(PolicyModel):
    def __init__(self):
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        self.sync_chunk_size = cfg.get("sync_chunk_size", 1)
        self.request_topic = cfg.get("request_topic")
        self.chunk_topic = cfg.get("chunk_topic")
        self._log = get_logger("omnireset_policy")
        self._cfg = cfg
        self._device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._rng = np.random.default_rng()

        ckpt = torch.load(cfg["ckpt_path"], map_location="cpu", weights_only=False)
        # distill_student.py checkpoints keep the network under "student", rl_games ones under "model"
        state = ckpt["student"] if "student" in ckpt else ckpt["model"]
        setup, src = _camera_setup(ckpt, cfg["ckpt_path"])
        if setup["obs_history"] != 1:
            raise ValueError(f"{src}: obs_history={setup['obs_history']}, only single-frame students are supported")
        self._cam_channels, self._real_cam = setup["cam_channels"], bool(setup["real_cam_model"])
        # cam_compact: only the used channels (depth 1, rgb 3); otherwise RGB-D with the unused ones zeroed
        self._compact = bool(setup["cam_compact"]) and self._cam_channels != "rgbd"
        keep = ("a2c_network.actor_cnn.", "a2c_network.actor_mlp.", "a2c_network.mu.")
        sd = {k[len("a2c_network."):]: v for k, v in state.items() if k.startswith(keep)}
        self._net = _Actor(sd)
        self._net.load_state_dict(sd)
        self._net.to(self._device).eval()
        in_c = {"rgb": 3, "depth": 1}[self._cam_channels] if self._compact else 4
        if self._net.in_c != in_c:
            raise ValueError(f"{cfg['ckpt_path']}: network takes {self._net.in_c} image channels, but {src} says "
                             f"cam_channels={self._cam_channels} cam_compact={setup['cam_compact']} ({in_c} channels)")

        self._scale = np.array([cfg["pos_scale"]] * 3 + [cfg["rot_scale"]] * 3) * cfg.get("action_scale_mult", 1.0)
        # env frame -> frame of /il/grasp/action
        w, x, y, z = cfg["out_frame_quat_wxyz"]
        self._R = Rotation.from_quat([x, y, z, w]).inv().as_matrix()

        self.reset()
        self._log.info(
            f"omnireset actor loaded from {cfg['ckpt_path']} on {self._device}: cnn={self._net.cnn_type}, "
            f"cam_channels={self._cam_channels}, cam_compact={self._compact}, real_cam_model={self._real_cam}, "
            f"action_scale_mult={cfg.get('action_scale_mult', 1.0)}, step_shift_px={cfg.get('step_shift_px', 0)}, "
            f"allow_gripper_open={cfg.get('allow_gripper_open', False)}"
        )
        if not self._real_cam:
            self._log.warning("this student was trained with the old square sim camera, not the real wrist camera model")
        if cfg.get("warmup", True):
            with torch.inference_mode():
                self._net(torch.zeros(1, in_c, IMG_SIZE, IMG_SIZE, device=self._device),
                          torch.zeros(1, VEC_DIM, device=self._device))

    def reset(self):
        self._step = 0
        # OmniResetEnv._reset_idx: every episode starts holding the fan, "gripper closed" (-1) in the action history
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._prev_action[6] = -1.0
        self._gripper_state = 0.0

    def predict(self, obs: dict) -> np.ndarray:
        self._step += 1
        if self._step > self._cfg["max_steps"]:
            if self._step == self._cfg["max_steps"] + 1:
                self._log.info(f"done (timeout) at step {self._step}, holding still")
            return np.zeros(ACTION_DIM, dtype=np.float32)

        img = self._preprocess(obs["rgb"], obs["depth"])
        vec = self._proprio(obs["joint_pos"])
        with torch.inference_mode():
            a = torch.clamp(self._net(img, vec), -1.0, 1.0)[0].cpu().numpy()
        self._prev_action = a.astype(np.float32)

        # gripper: sim publishes (a + 1) / 2 and opens at >= 0.5 (a >= 0). Opening drops the fan, so it is
        # held closed unless allow_gripper_open; the fed-back state is what was actually sent.
        grip = float(a[6] >= 0.0) if self._cfg.get("allow_gripper_open", False) else 0.0
        self._gripper_state = grip
        return self._to_output(a[:6] * self._scale, grip)

    def _proprio(self, joint_pos):
        """OmniResetEnv._proprio without the training noise -> (1, 14) on the policy device."""
        q = np.asarray(joint_pos, dtype=np.float32)[:6]   # Joint_1..6 in rad
        vec = np.concatenate([q, [self._gripper_state], self._prev_action]).astype(np.float32)
        vec = np.clip(vec, -CLIP_OBS, CLIP_OBS)
        return torch.from_numpy(vec)[None].to(self._device)

    def _preprocess(self, rgb, depth):
        """OmniResetEnv._get_observations without the training augmentation -> (1, C, 256, 256)."""
        if rgb.shape[:2] != (IMG_SIZE, IMG_SIZE):
            raise ValueError(f"omnireset expects {IMG_SIZE}x{IMG_SIZE} images (node image_size), got {rgb.shape}")
        d = np.asarray(depth, dtype=np.float32)
        if np.issubdtype(np.asarray(depth).dtype, np.integer):
            d = d / 1000.0   # 16UC1 / mono16 is mm
        if self._real_cam:
            # real camera model: no return reads 0 m, which the clamp turns into depth_near, as on the robot
            d = np.where(np.isfinite(d), d, 0.0)
        else:
            # old sim camera: no hit = inf -> depth_far
            d = np.where(np.isfinite(d) & (d > 0), d, DEPTH_FAR)
        rgb = rgb.astype(np.float32) / 255.0
        s = self._cfg.get("step_shift_px", 0)
        if s > 0:
            # the env's per-step random image shift (aug_step_shift_px, border padding); off = the steady real image
            dy, dx = self._rng.integers(-s, s + 1, 2)
            ys = np.clip(np.arange(IMG_SIZE) + dy, 0, IMG_SIZE - 1)
            xs = np.clip(np.arange(IMG_SIZE) + dx, 0, IMG_SIZE - 1)
            rgb, d = rgb[ys][:, xs], d[ys][:, xs]
        d = (np.clip(d, DEPTH_NEAR, DEPTH_FAR) - DEPTH_MEAN) / DEPTH_SCALE
        rgb -= rgb.mean(axis=(0, 1), keepdims=True)
        if self._compact:
            img = d[..., None] if self._cam_channels == "depth" else rgb
        else:
            # the env zeroes the unused channels after normalization, so the image layout stays RGB-D
            if self._cam_channels == "rgb":
                d = np.zeros_like(d)
            elif self._cam_channels == "depth":
                rgb = np.zeros_like(rgb)
            img = np.concatenate([rgb, d[..., None]], axis=-1)
        img = np.clip(img, -CLIP_OBS, CLIP_OBS)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].to(self._device)

    def _to_output(self, d_env, grip):
        dpos = self._R @ d_env[:3]
        drot = Rotation.from_rotvec(self._R @ d_env[3:]).as_euler(self._cfg.get("euler_order", "xyz"))
        return np.array([*dpos, *drot, grip], dtype=np.float32)
