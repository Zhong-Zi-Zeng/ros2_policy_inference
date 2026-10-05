"""Diffusion Policy backend: wraps a lerobot checkpoint trained on the fan-grasp dataset.

The checkpoint predicts 7-D per-step deltas [dx, dy, dz, drx, dry, drz, gripper], where the
position delta is in metres in the WORLD frame, the rotation delta is an axis-angle (rotation
vector) in radians applied by LEFT multiplication (R_next = R(drot) @ R_now), and the gripper is
0/1 with 1 = open. One action corresponds to one camera period, so the node must tick at the rate
the training data was recorded at: this class declares inference_rate_hz (30 Hz, from
policies/diffusion/config.yaml), which the node uses in place of config/params.yaml's rate.

Chunking: lerobot's DiffusionPolicy keeps its own queues. select_action() runs the diffusion
sampler only when its action queue is empty (generating `horizon` actions and keeping
`n_action_steps` of them) and otherwise pops the next queued action. So this wrapper calls
select_action() exactly once per predict() and does no chunking of its own. Measured on an
RTX 4070 Ti SUPER with DDIM/10: the sampler tick costs ~62 ms (p95 65 ms) and the other seven
ticks ~0.02 ms each, so one tick in every n_action_steps overruns a 33 ms control period. That is
accepted here; move inference to its own callback group if the jitter matters.

Image handling note: obs["rgb"] arrives as HWC uint8 RGB at image_size x image_size, produced by
ros_bridge._compressed_msg_to_array (cv2.imdecode + BGR->RGB) and preprocessing.resize_image
(cv2.INTER_AREA). The recorder that produced the training data uses byte-for-byte the same two
functions, duplicated in ros2_data_collector/ros_bridge.py and ros2_data_collector/recorder.py.
Editing either copy silently breaks the match between training and deployment images — change both
or neither.
"""
from __future__ import annotations

import os

import numpy as np
import yaml

from ..base_policy import ACTION_DIM, PolicyModel

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH) as _f:
    _CFG = yaml.safe_load(_f)


class DiffusionPolicyModel(PolicyModel):
    """lerobot DiffusionPolicy checkpoint, driven one action per predict() call."""

    def __init__(self, checkpoint_path: str = _CFG["checkpoint_dir"], device: str = _CFG["device"],
                 task: str = _CFG["task"], warmup_iters: int = _CFG["warmup_iters"], logger=None):
        # Imported lazily: torch/lerobot are only needed when this policy is selected.
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self._torch = torch
        self.inference_rate_hz = float(_CFG["inference_rate_hz"])
        self.task = task
        self.device = device
        self._log = logger

        cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
        cfg.device = device
        self.cfg = cfg
        self.policy = DiffusionPolicy.from_pretrained(checkpoint_path, config=cfg).to(device).eval()
        # Normalization statistics saved with the checkpoint (QUANTILES for the action, MIN_MAX for
        # the state, MEAN_STD for the image). Loading them from `pretrained_path` rather than
        # recomputing is what keeps deployment identical to training; the post-processor maps the
        # network's [-1, 1] output back into metres/radians.
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint_path)

        self._image_key = next(iter(cfg.image_features)) if cfg.image_features else "observation.images.wrist"
        self._state_dim = cfg.input_features["observation.state"].shape[0]
        self._info(
            f"loaded {cfg.type} from {checkpoint_path} on {device}: "
            f"horizon={cfg.horizon}, n_action_steps={cfg.n_action_steps}, n_obs_steps={cfg.n_obs_steps}, "
            f"{cfg.noise_scheduler_type}/{cfg.num_inference_steps} denoising steps, image key {self._image_key}"
        )
        self._warmup(warmup_iters)

    # ---------- PolicyModel interface ----------

    def reset(self):
        """Clear the policy's observation/action queues between episodes.

        Without this the first actions of a new episode are leftovers from the previous episode's
        chunk, and the n_obs_steps history starts with stale frames.
        """
        self.policy.reset()

    def predict(self, obs: dict) -> np.ndarray:
        batch = self._build_batch(obs)
        with self._torch.no_grad():
            action = self.post(self.policy.select_action(self.pre(batch)))
        action = np.asarray(action.squeeze(0).float().cpu().numpy(), dtype=np.float64)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"policy returned {action.shape}, expected ({ACTION_DIM},)")
        return action

    # ---------- internals ----------

    def _build_batch(self, obs: dict) -> dict:
        """obs (ROS node format) -> the batch the checkpoint expects.

        Only two of the node's observation keys are used: the policy was trained on the wrist image
        and a 7-D state of [joint_0..joint_5, gripper]. depth, joint_vel, ee_pos, ee_quat and
        ft_wrench are deliberately ignored.
        """
        torch = self._torch
        rgb = obs["rgb"]                                   # HWC uint8 RGB, already image_size square
        if rgb.dtype != np.uint8:
            raise ValueError(f"obs['rgb'] must be uint8, got {rgb.dtype}")
        img = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        img = img.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)   # HWC uint8 -> 1CHW float [0,1]

        joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32).reshape(-1)
        state = np.concatenate([joint_pos, [np.float32(obs["gripper_state"])]]).astype(np.float32)
        if state.shape[0] != self._state_dim:
            raise ValueError(
                f"state has {state.shape[0]} dims (joint_pos {joint_pos.shape[0]} + gripper), "
                f"checkpoint expects {self._state_dim}"
            )
        return {
            self._image_key: img,
            "observation.state": torch.from_numpy(state).to(self.device).unsqueeze(0),
            # Carried through for parity with training batches; the diffusion policy itself does not
            # condition on language, but the processor pipeline passes the field along.
            "task": [self.task],
        }

    def _warmup(self, iters: int):
        """Spend the first-call cost (CUDA context, cuDNN autotune, lazy kernels) before the robot
        moves: measured ~226 ms on the first sampler call versus ~62 ms steady state."""
        if iters <= 0:
            return
        torch = self._torch
        dummy = {
            "rgb": np.zeros((self.cfg.image_features[self._image_key].shape[1] if self.cfg.image_features else 256,
                             self.cfg.image_features[self._image_key].shape[2] if self.cfg.image_features else 256,
                             3), dtype=np.uint8),
            "joint_pos": np.zeros(self._state_dim - 1, dtype=np.float32),
            "gripper_state": np.float32(1.0),
        }
        with torch.no_grad():
            for _ in range(iters):
                self.policy.reset()
                self.post(self.policy.select_action(self.pre(self._build_batch(dummy))))
        self.policy.reset()          # drop the dummy observations from the queues
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._info(f"warmed up with {iters} dummy inferences")

    def _info(self, msg: str):
        if self._log is not None:
            self._log.info(msg)
        else:
            print(f"[diffusion_policy] {msg}")
