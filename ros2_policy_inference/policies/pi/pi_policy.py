"""π₀.₅ (pi05) policy: runs a LeRobot pi05 checkpoint, configured by config.yaml."""
import dataclasses
import json
import os
import tempfile
import time
from collections import deque
from contextlib import nullcontext

import numpy as np
import yaml
from rclpy.logging import get_logger

from ..base_policy import ACTION_DIM, PolicyModel
from ..cameras import build_cameras

try:
    import transformers  # noqa: F401  needed by lerobot's pi05, only installed with lerobot[pi]
except ImportError as exc:
    raise ImportError("pi policy needs transformers — rebuild the image (docker/Dockerfile installs lerobot[pi])") from exc

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors, prepare_observation_for_inference
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import ACTION, OBS_STATE

# colcon only installs .py files, so fall back to the mounted source tree in the container
_CONFIG_PATH = next(
    (p for p in (
        os.path.join(os.path.dirname(__file__), "config.yaml"),
        "/workspace/src/ros2_policy_inference/ros2_policy_inference/policies/pi/config.yaml",
    ) if os.path.isfile(p)),
    None,
)

_STATE_SEGMENTS = {"joint_pos": 6, "gripper_state": 1, "ft_wrench": 6}

# Fields the forked lerobot writes into config.json; stock PI05Config rejects them.
# Safe to drop only while these features are off.
_FORK_FEATURE_FLAGS = ("use_visual_memory", "use_proprioceptive_memory")


class PIPolicyModel(PolicyModel):
    def __init__(self):
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        self._log = get_logger("pi_policy")
        self.request_topic = cfg.get("request_topic")
        self.chunk_topic = cfg.get("chunk_topic")

        ckpt_path = cfg["ckpt_path"]
        self._task = cfg["task"]
        self._robot_type = cfg.get("robot_type", "")

        policy_cfg = self._load_config(ckpt_path, cfg)
        self._device = torch.device(policy_cfg.device)
        self._n_action_steps = policy_cfg.n_action_steps
        self.sync_chunk_size = policy_cfg.n_action_steps
        self._chunk = deque()
        self._use_amp = policy_cfg.use_amp and self._device.type == "cuda"

        # empty_camera_* are placeholders pi05 fills with blank frames itself, so they are not mapped
        declared = [k for k in policy_cfg.image_features if "empty_camera" not in k]
        self._cameras = build_cameras(cfg["cameras"], declared, ckpt_path)

        state_dim = policy_cfg.input_features[OBS_STATE].shape[0]
        self._state_keys = cfg["state_keys"]
        assert sum(_STATE_SEGMENTS[k] for k in self._state_keys) == state_dim, "state_keys don't match checkpoint"

        self._policy = self._load_policy(ckpt_path, policy_cfg)
        self._policy.to(self._device).eval()
        self._drop_lm_heads()
        overrides = {"device_processor": {"device": policy_cfg.device}}
        # the fork saves its tokenizer next to the checkpoint
        if os.path.isdir(os.path.join(ckpt_path, "tokenizer")):
            overrides["tokenizer_processor"] = {"tokenizer_name": os.path.join(ckpt_path, "tokenizer")}
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg, pretrained_path=ckpt_path, preprocessor_overrides=overrides
        )

        self._log.info(
            f"pi05 loaded from {ckpt_path}: task='{self._task}', state={list(self._state_keys)}, "
            f"cameras={list(self._cameras)}, n_action_steps={policy_cfg.n_action_steps}"
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

    def _load_config(self, ckpt_path, cfg):
        with open(os.path.join(ckpt_path, "config.json")) as f:
            raw = json.load(f)
        known = {f.name for f in dataclasses.fields(PI05Config)} | {"type"}
        unknown = set(raw) - known
        if any(raw.get(k) for k in _FORK_FEATURE_FLAGS):
            raise ValueError(f"{ckpt_path} uses fork-only memory features stock lerobot lacks")
        for k in unknown:
            del raw[k]
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(raw, f)
            policy_cfg = PreTrainedConfig.from_pretrained(tmp)

        policy_cfg.pretrained_path = ckpt_path
        policy_cfg.rtc_config = None  # select_action() refuses RTC
        if cfg.get("device"):
            policy_cfg.device = cfg["device"]
        if cfg.get("n_action_steps"):
            policy_cfg.n_action_steps = cfg["n_action_steps"]
        if cfg.get("num_inference_steps"):
            policy_cfg.num_inference_steps = cfg["num_inference_steps"]
        assert policy_cfg.output_features[ACTION].shape[0] == ACTION_DIM, "checkpoint action dim mismatch"
        return policy_cfg

    def _load_policy(self, ckpt_path, policy_cfg):        
        adapter_config_path = os.path.join(ckpt_path, "adapter_config.json")
        if not os.path.isfile(adapter_config_path):
            return get_policy_class(policy_cfg.type).from_pretrained(ckpt_path, config=policy_cfg)

        with open(adapter_config_path) as f:
            base_model_id = json.load(f)["base_model_name_or_path"]
        base = get_policy_class(policy_cfg.type).from_pretrained(base_model_id, config=policy_cfg, strict=False)
        from peft import PeftModel
        return PeftModel.from_pretrained(base, ckpt_path)

    def _unwrap_policy(self):
        """self._policy may be a peft.PeftModel wrapping the real PI05Policy (LoRA checkpoints)."""
        from peft import PeftModel
        return self._policy.get_base_model() if isinstance(self._policy, PeftModel) else self._policy

    def _drop_lm_heads(self):
        """pi05 never computes text logits, so the 257k-vocab lm_heads (~1.5 GiB bf16) are dead
        weight on the GPU. Keep one only if it shares storage with embed_tokens (tied)."""
        pwe = self._unwrap_policy().model.paligemma_with_expert
        embed = pwe.paligemma.model.language_model.embed_tokens.weight
        freed = 0
        for owner in (pwe.paligemma, pwe.gemma_expert):
            head = getattr(owner, "lm_head", None)
            if head is None or head.weight.data_ptr() == embed.data_ptr():
                continue
            freed += head.weight.numel() * head.weight.element_size()
            owner.lm_head = None
        if freed and self._device.type == "cuda":
            torch.cuda.empty_cache()  # hand the freed blocks back so other processes can use them
        self._log.info(f"dropped unused lm_heads: {freed / 2**20:.0f} MiB")

    def _build_state(self, obs):
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in self._state_keys])

    def _warmup(self, shapes):
        """shapes: image feature -> (H, W). Both encodings produce (H, W, 3) uint8."""
        frame = {key: np.zeros((*hw, 3), np.uint8) for key, hw in shapes.items()}
        frame[OBS_STATE] = np.zeros(sum(_STATE_SEGMENTS[k] for k in self._state_keys), np.float32)
        self._infer_chunk(frame)
        self.reset()

    def _infer_chunk(self, frame):
        """frame -> (n_action_steps, ACTION_DIM) unnormalized actions on CPU."""
        started = time.perf_counter()
        autocast = torch.autocast(device_type=self._device.type) if self._use_amp else nullcontext()
        with torch.inference_mode(), autocast:
            batch = prepare_observation_for_inference(frame, self._device, self._task, self._robot_type)
            chunk = self._policy.predict_action_chunk(self._preprocessor(batch))[:, : self._n_action_steps]
            chunk = self._postprocessor(chunk)
        self._log.info(f"denoised new chunk in {(time.perf_counter() - started) * 1e3:.0f} ms")
        return chunk.squeeze(0).float().cpu().numpy()
