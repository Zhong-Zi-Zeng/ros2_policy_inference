"""Diffusion Policy (lerobot): runs a LeRobot diffusion checkpoint, configured by config.yaml."""
import os
import time

import numpy as np
import yaml
from rclpy.logging import get_logger

from ..base_policy import ACTION_DIM, PolicyModel
from ..cameras import build_cameras

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors, prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_STATE

# colcon only installs .py files, so fall back to the mounted source tree in the container
_CONFIG_PATH = next(
    (p for p in (
        os.path.join(os.path.dirname(__file__), "config.yaml"),
        "/workspace/src/ros2_policy_inference/ros2_policy_inference/policies/diffusion/config.yaml",
    ) if os.path.isfile(p)),
    None,
)

_STATE_SEGMENTS = {"joint_pos": 6, "gripper_state": 1, "ft_wrench": 6}


class DiffusionPolicyModel(PolicyModel):
    def __init__(self):
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        self._log = get_logger("diffusion_policy")

        ckpt_path = cfg["ckpt_path"]
        self._task = cfg.get("task", "")
        self._robot_type = cfg.get("robot_type", "")

        policy_cfg = PreTrainedConfig.from_pretrained(ckpt_path)
        policy_cfg.pretrained_path = ckpt_path
        if cfg.get("device"):
            policy_cfg.device = cfg["device"]
        if cfg.get("n_action_steps"):
            policy_cfg.n_action_steps = cfg["n_action_steps"]
        if cfg.get("num_inference_steps"):
            policy_cfg.num_inference_steps = cfg["num_inference_steps"]
        assert policy_cfg.output_features[ACTION].shape[0] == ACTION_DIM, "checkpoint action dim mismatch"
        self._device = torch.device(policy_cfg.device)

        self._cameras = build_cameras(cfg["cameras"], policy_cfg.image_features, ckpt_path)

        state_dim = policy_cfg.input_features[OBS_STATE].shape[0]
        self._state_keys = cfg["state_keys"]
        assert sum(_STATE_SEGMENTS[k] for k in self._state_keys) == state_dim, "state_keys don't match checkpoint"

        self._policy = get_policy_class(policy_cfg.type).from_pretrained(ckpt_path, config=policy_cfg)
        self._policy.to(self._device).eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg, pretrained_path=ckpt_path,
            preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
        )

        self._log.info(
            f"diffusion loaded from {ckpt_path}: task='{self._task}', state={list(self._state_keys)}, "
            f"cameras={list(self._cameras)}, n_action_steps={policy_cfg.n_action_steps}"
        )
        if cfg.get("warmup", True):
            self._warmup({k: policy_cfg.input_features[k].shape[1:] for k in self._cameras})

    def reset(self):
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def predict(self, obs: dict) -> np.ndarray:
        # DiffusionPolicy.select_action() caches its own obs-history and action-chunk queues
        # internally, so unlike pi/groot this needs no deque of our own.
        frame = {key: encode(obs[source]) for key, (source, encode) in self._cameras.items()}
        frame[OBS_STATE] = self._build_state(obs)
        batch = prepare_observation_for_inference(frame, self._device, self._task, self._robot_type)
        with torch.inference_mode():
            action = self._policy.select_action(self._preprocessor(batch))
        action = self._postprocessor(action)
        return action.squeeze(0).float().cpu().numpy()

    def _build_state(self, obs):
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in self._state_keys])

    def _warmup(self, shapes):
        """shapes: image feature -> (H, W). Both encodings produce (H, W, 3) uint8."""
        frame = {key: np.zeros((*hw, 3), np.uint8) for key, hw in shapes.items()}
        frame[OBS_STATE] = np.zeros(sum(_STATE_SEGMENTS[k] for k in self._state_keys), np.float32)
        batch = prepare_observation_for_inference(frame, self._device, self._task, self._robot_type)
        started = time.perf_counter()
        with torch.inference_mode():
            self._policy.select_action(self._preprocessor(batch))
        self._log.info(f"warmup inference in {(time.perf_counter() - started) * 1e3:.0f} ms")
        self.reset()
