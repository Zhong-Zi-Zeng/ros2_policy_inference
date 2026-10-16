"""GR00T N1.7 policy: runs a LeRobot-trained GR00T checkpoint, configured by config.yaml."""
import os
import time
from collections import deque

import numpy as np
import yaml
from rclpy.logging import get_logger

from ..base_policy import ACTION_DIM, PolicyModel
from ..cameras import build_cameras

try:
    import transformers  # noqa: F401  needed by lerobot's groot, only installed with lerobot[groot]
except ImportError as exc:
    raise ImportError("groot policy needs transformers — rebuild the image with lerobot[groot]") from exc

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors, prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_STATE

# colcon only installs .py files, so fall back to the mounted source tree in the container
_CONFIG_PATH = next(
    (p for p in (
        os.path.join(os.path.dirname(__file__), "config.yaml"),
        "/workspace/src/ros2_policy_inference/ros2_policy_inference/policies/groot/config.yaml",
    ) if os.path.isfile(p)),
    None,
)

_STATE_SEGMENTS = {"joint_pos": 6, "gripper_state": 1, "ft_wrench": 6}


class GrootPolicyModel(PolicyModel):
    def __init__(self):
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        self._log = get_logger("groot_policy")

        ckpt_path = cfg["ckpt_path"]
        self._task = cfg["task"]
        self._robot_type = cfg.get("robot_type", "")

        policy_cfg = PreTrainedConfig.from_pretrained(ckpt_path)
        policy_cfg.pretrained_path = ckpt_path
        if cfg.get("device"):
            policy_cfg.device = cfg["device"]
        if cfg.get("n_action_steps"):
            policy_cfg.n_action_steps = cfg["n_action_steps"]
        assert policy_cfg.output_features[ACTION].shape[0] == ACTION_DIM, "checkpoint action dim mismatch"
        self._device = torch.device(policy_cfg.device)
        self._n_action_steps = policy_cfg.n_action_steps
        self._chunk = deque()

        self._cameras = build_cameras(cfg["cameras"], policy_cfg.image_features, ckpt_path)

        state_dim = policy_cfg.input_features[OBS_STATE].shape[0]
        self._state_keys = cfg["state_keys"]
        assert sum(_STATE_SEGMENTS[k] for k in self._state_keys) == state_dim, "state_keys don't match checkpoint"

        self._policy = get_policy_class(policy_cfg.type).from_pretrained(ckpt_path, config=policy_cfg)
        self._policy.to(self._device).eval()
        self._drop_lm_head()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg, pretrained_path=ckpt_path,
            preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
        )

        self._log.info(
            f"groot loaded from {ckpt_path}: task='{self._task}', state={list(self._state_keys)}, "
            f"cameras={list(self._cameras)}, n_action_steps={self._n_action_steps}"
        )
        if cfg.get("warmup", True):
            self._warmup({k: policy_cfg.input_features[k].shape[1:] for k in self._cameras})

    def reset(self):
        self._chunk.clear()
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def predict(self, obs: dict) -> np.ndarray:
        # Only touch the GPU when the chunk runs out; queued actions are popped straight from CPU.
        if not self._chunk:
            frame = {key: encode(obs[source]) for key, (source, encode) in self._cameras.items()}
            frame[OBS_STATE] = self._build_state(obs)
            self._chunk.extend(self._infer_chunk(frame))
        return self._chunk.popleft()

    def _drop_lm_head(self):
        """GR00T reads the backbone's last decoder layer through a hook and never uses the VLM's
        text logits, yet Qwen3-VL still runs its 151k-vocab lm_head (plus a bf16 weight copy
        under autocast) on every call. Identity makes that a free view of the hidden states."""
        vlm = self._policy._groot_model.backbone.model
        head = vlm.lm_head
        tied = head.weight.data_ptr() == vlm.get_input_embeddings().weight.data_ptr()
        freed = 0 if tied else head.weight.numel() * head.weight.element_size()
        vlm.lm_head = torch.nn.Identity()
        del head  # drop the last reference before empty_cache, or the weights stay allocated
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        self._log.info(f"replaced unused backbone lm_head (tied={tied}, weights freed {freed / 2**20:.0f} MiB)")

    def _build_state(self, obs):
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in self._state_keys])

    def _warmup(self, shapes):
        """shapes: image feature -> (H, W). Both encodings produce (H, W, 3) uint8."""
        frame = {key: np.zeros((*hw, 3), np.uint8) for key, hw in shapes.items()}
        frame[OBS_STATE] = np.zeros(sum(_STATE_SEGMENTS[k] for k in self._state_keys), np.float32)
        self._infer_chunk(frame)
        self.reset()

    def _infer_chunk(self, frame):
        """frame -> (n_action_steps, ACTION_DIM) unnormalized actions on CPU.
        GR00T runs its own bf16 autocast inside predict_action_chunk."""
        started = time.perf_counter()
        with torch.inference_mode():
            batch = prepare_observation_for_inference(frame, self._device, self._task, self._robot_type)
            chunk = self._policy.predict_action_chunk(self._preprocessor(batch))[:, : self._n_action_steps]
            chunk = self._postprocessor(chunk)
        self._log.info(f"denoised new chunk in {(time.perf_counter() - started) * 1e3:.0f} ms")
        return chunk.squeeze(0).float().cpu().numpy()
