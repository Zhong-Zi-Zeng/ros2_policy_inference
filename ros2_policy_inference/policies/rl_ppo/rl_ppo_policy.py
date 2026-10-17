"""Place (rl_games): RL alignment policy from rl-template's PickAndPlaceEnv, followed by the
scripted push used in training (env.scripted_push=True), configured by config.yaml."""
import math
import os

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
        "/workspace/src/ros2_policy_inference/ros2_policy_inference/policies/rl_ppo/config.yaml",
    ) if os.path.isfile(p)),
    None,
)

IMG_SIZE = 256
# depth normalization of PickAndPlaceEnv._get_observations (metres)
DEPTH_NEAR, DEPTH_FAR = 0.02, 0.60
DEPTH_MEAN, DEPTH_SCALE = 0.4463, 0.9062


class _Actor(nn.Module):
    """Actor half of rl_games' image_vec network (agents/rl_games_place.yaml). Attribute names
    match the checkpoint's a2c_network.* keys, so its state dict loads as is."""

    def __init__(self):
        super().__init__()
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        self.actor_mlp = nn.Sequential(nn.Linear(64 * 13 * 13 + ACTION_DIM, 512), nn.ELU())
        self.mu = nn.Linear(512, ACTION_DIM)

    def forward(self, img, prev_action):
        feat = self.actor_cnn(img).flatten(1)
        return self.mu(self.actor_mlp(torch.cat([feat, prev_action], dim=1)))


class RLPPOPolicyModel(PolicyModel):
    def __init__(self):
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        self._log = get_logger("rl_ppo_policy")
        self._cfg = cfg
        self._device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

        state = torch.load(cfg["ckpt_path"], map_location="cpu", weights_only=False)["model"]
        keep = ("a2c_network.actor_cnn.", "a2c_network.actor_mlp.", "a2c_network.mu.")
        self._net = _Actor()
        self._net.load_state_dict({k[len("a2c_network."):]: v for k, v in state.items() if k.startswith(keep)})
        self._net.to(self._device).eval()

        self._scale = np.array([cfg["pos_scale"]] * 3 + [cfg["rot_scale"]] * 3) * cfg.get("action_scale_mult", 1.0)
        # env frame -> frame of /il/grasp/action
        w, x, y, z = cfg["out_frame_quat_wxyz"]
        self._R = Rotation.from_quat([x, y, z, w]).inv().as_matrix()

        self.reset()
        self._log.info(
            f"rl_ppo place model loaded from {cfg['ckpt_path']} on {self._device}: "
            f"action_scale_mult={cfg.get('action_scale_mult', 1.0)}, push_speed={cfg['push_speed']} m/step"
        )
        if cfg.get("warmup", True):
            with torch.inference_mode():
                self._net(torch.zeros(1, 4, IMG_SIZE, IMG_SIZE, device=self._device),
                          torch.zeros(1, ACTION_DIM, device=self._device))

    def reset(self):
        self._phase = "align"           # align (RL) -> push (scripted) -> done (hold still)
        self._step = 0
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._dpos_hist = []            # commanded EE deltas (env frame), for stall detection
        self._drot_hist = []
        self._pushed = 0.0
        self._f0 = np.zeros(3)

    def predict(self, obs: dict) -> np.ndarray:
        self._step += 1
        if self._phase != "done" and self._step > self._cfg["max_steps"]:
            self._finish("timeout")
        if self._phase == "align" and self._stalled():
            self._phase = "push"
            self._f0 = np.asarray(obs["ft_wrench"][:3], dtype=np.float64)
            self._log.info(f"EE stalled at step {self._step}, handing over to the scripted push")

        if self._phase == "align":
            d_env = self._align(obs)
        elif self._phase == "push":
            d_env = self._push(obs)
        else:
            d_env = np.zeros(6)
        return self._to_output(d_env)

    def _align(self, obs):
        img = self._preprocess(obs["rgb"], obs["depth"])
        prev = torch.from_numpy(self._prev_action)[None].to(self._device)
        with torch.inference_mode():
            a = torch.clamp(self._net(img, prev), -1.0, 1.0)[0].cpu().numpy()
        self._prev_action = a.astype(np.float32)   # raw 7-D output (gripper dim included) is fed back
        d_env = a[:6] * self._scale
        self._dpos_hist = (self._dpos_hist + [d_env[:3]])[-self._cfg["push_stall_window"]:]
        self._drot_hist = (self._drot_hist + [d_env[3:]])[-self._cfg["push_stall_window"]:]
        return d_env

    def _stalled(self):
        """Same trigger as PickAndPlaceEnv._scripted_push, with the EE motion over the window taken
        from the commanded deltas (the real arm is position-controlled and follows them)."""
        cfg = self._cfg
        if self._step <= cfg["push_min_steps"] or len(self._dpos_hist) < cfg["push_stall_window"]:
            return False
        moved = np.linalg.norm(np.sum(self._dpos_hist, axis=0))
        turned = np.linalg.norm(np.sum(self._drot_hist, axis=0))
        return moved < cfg["push_stall_dist"] and turned < math.radians(cfg["push_stall_ang_deg"])

    def _push(self, obs):
        # sim also servoes x / z / orientation back to the pose locked at hand-over; the real arm
        # holds them by itself, so only the step along env +y (the insertion direction) is sent
        cfg = self._cfg
        force = float(np.linalg.norm(np.asarray(obs["ft_wrench"][:3]) - self._f0))
        if force > cfg["push_force_stop"]:
            self._finish(f"contact force {force:.1f} N")
            return np.zeros(6)
        if self._pushed >= cfg["push_max_travel"]:
            self._finish(f"pushed {self._pushed * 1000:.0f} mm")
            return np.zeros(6)
        self._pushed += cfg["push_speed"]
        return np.array([0.0, cfg["push_speed"], 0.0, 0.0, 0.0, 0.0])

    def _finish(self, reason):
        self._phase = "done"
        self._log.info(f"done ({reason}) at step {self._step}, holding still")

    def _preprocess(self, rgb, depth):
        """Same as PickAndPlaceEnv._get_observations -> (1, 4, 256, 256) on the policy device."""
        if rgb.shape[:2] != (IMG_SIZE, IMG_SIZE):
            raise ValueError(f"rl_ppo place model expects {IMG_SIZE}x{IMG_SIZE} images (node image_size), got {rgb.shape}")
        rgb = rgb.astype(np.float32) / 255.0
        rgb -= rgb.mean(axis=(0, 1), keepdims=True)
        d = np.asarray(depth, dtype=np.float32)
        if np.issubdtype(np.asarray(depth).dtype, np.integer):
            d = d / 1000.0   # 16UC1 is mm
        d = np.where(np.isfinite(d) & (d > 0), d, DEPTH_FAR)   # invalid (0) = no return, like sim's inf
        d = (np.clip(d, DEPTH_NEAR, DEPTH_FAR) - DEPTH_MEAN) / DEPTH_SCALE
        img = np.clip(np.concatenate([rgb, d[..., None]], axis=-1), -5.0, 5.0)   # rl_games clip_observations
        return torch.from_numpy(img).permute(2, 0, 1)[None].to(self._device)

    def _to_output(self, d_env):
        dpos = self._R @ d_env[:3]
        drot = Rotation.from_rotvec(self._R @ d_env[3:]).as_euler(self._cfg.get("euler_order", "xyz"))
        # gripper always closed: sim overwrites the policy's gripper output with "close"
        return np.array([*dpos, *drot, 0.0], dtype=np.float32)