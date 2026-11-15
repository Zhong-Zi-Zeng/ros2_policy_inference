#!/usr/bin/env python3
"""Check that policies/omnireset matches rl-template for the given checkpoints:

- the observation the wrapper builds from robot-encoded inputs (rgb uint8, depth 16UC1 mm or 32FC1 m,
  0 = no return) vs OmniResetEnv._get_observations' own code (run on a fake env, augmentation off);
- the wrapper's actor vs rl_games' image_vec network loaded with the same weights.

The camera setup and depth constants come from the checkpoint / its .env.yaml and rl-template's
omnireset_env_cfg.py, not from the wrapper, so a change on either side shows up as a FAIL.

Runs on the training PC in the rl-template Python env (torch + rl_games), no ROS needed:
  conda activate mis_isaaclab
  python tools/check_omnireset_parity.py --rl-template ~/Desktop/delta_robotic/rl-template checkpoints/omnireset/*.pth
"""
import argparse
import ast
import contextlib
import importlib.util
import io
import logging
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU only: leave a training GPU alone

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "source/RLTemplate/RLTemplate/tasks/direct/rltemplate"
TOL = 1e-4
CLIP_OBS = 5.0
DEPTH_KEYS = ("depth_near", "depth_far", "depth_mean", "depth_scale")


class _EnvYamlLoader(yaml.SafeLoader):
    """params/env.yaml has python tags (tuples, class names); only plain values are needed."""


_EnvYamlLoader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def _load_wrapper():
    for n in ("rclpy", "rclpy.logging"):
        sys.modules.setdefault(n, types.ModuleType(n))
    sys.modules["rclpy.logging"].get_logger = logging.getLogger
    sys.path.insert(0, REPO)
    from ros2_policy_inference.policies.omnireset import omnireset_policy
    return omnireset_policy


def _env_get_observations(task_dir):
    """OmniResetEnv._get_observations, verbatim from rl-template."""
    path = os.path.join(task_dir, "omnireset_env.py")
    tree = ast.parse(open(path).read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniResetEnv")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_observations")
    fn.returns = None
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), ns)
    return ns["_get_observations"]


def _cfg_defaults(task_dir):
    """Depth clipping / normalization defaults of OmniResetEnvCfg (omnireset_env_cfg.py)."""
    tree = ast.parse(open(os.path.join(task_dir, "omnireset_env_cfg.py")).read())
    vals = {n.target.id: ast.literal_eval(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id in DEPTH_KEYS}
    missing = set(DEPTH_KEYS) - set(vals)
    if missing:
        raise KeyError(f"omnireset_env_cfg.py no longer defines {sorted(missing)}")
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rl-template", required=True, help="rl-template checkout the checkpoints were trained with")
    ap.add_argument("ckpts", nargs="+")
    args = ap.parse_args()
    task_dir = os.path.join(os.path.expanduser(args.rl_template), TASK)

    op = _load_wrapper()
    ivn_spec = importlib.util.spec_from_file_location("image_vec_network", os.path.join(task_dir, "agents", "image_vec_network.py"))
    ivn = importlib.util.module_from_spec(ivn_spec)
    ivn_spec.loader.exec_module(ivn)   # registers the image_vec network with rl_games
    from rl_games.algos_torch import model_builder
    env_get_obs = _env_get_observations(task_dir)
    defaults = _cfg_defaults(task_dir)
    agent_yaml = os.path.join(task_dir, "agents", "rl_games_omnireset.yaml")

    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:256, 0:256]
    rgb = np.clip(np.stack([(xx + 50 * c + yy // 3) % 256 for c in range(3)], -1) + rng.integers(-15, 16, (256, 256, 3)), 0, 255).astype(np.uint8)
    depth_mm = np.round(rng.uniform(5, 1000, (256, 256))).astype(np.uint16)
    depth_mm[rng.random((256, 256)) < 0.6] = 0   # no return, like most of the real wrist depth image
    depth_m = depth_mm.astype(np.float32) / 1000.0
    q = np.array([0.1796, 1.2752, 0.1759, 1.1603, 3.2942, -1.5708], np.float32)
    prev = rng.uniform(-1, 1, 7).astype(np.float32)

    failed = 0
    for path in args.ckpts:
        name = os.path.basename(path)
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
            if "student" in raw:
                meta, consts = raw, defaults
            else:
                with open(os.path.splitext(path)[0] + ".env.yaml") as f:
                    meta = yaml.load(f, Loader=_EnvYamlLoader)
                consts = {k: meta[k] for k in DEPTH_KEYS}
            ch, compact, real = meta.get("cam_channels", "rgbd"), bool(meta.get("cam_compact", False)), bool(meta.get("real_cam_model", False))
            cnn, in_c = ivn.image_vec_layout(raw)

            params = yaml.safe_load(open(agent_yaml))["params"]
            params["network"]["image_shape"][2] = in_c
            params["network"]["cnn"]["type"] = cnn
            with contextlib.redirect_stdout(io.StringIO()):   # rl_games prints its layer sizes
                ref = model_builder.ModelBuilder().load(params).build({
                    "actions_num": 7, "input_shape": (256 * 256 * in_c + 14,), "num_seqs": 1, "value_size": 1,
                    "normalize_value": params["config"]["normalize_value"], "normalize_input": False})
            ref.load_state_dict(raw["student"] if "student" in raw else raw["model"])
            ref.eval()

            # sim side: the real camera model gives metres with 0 = no return, the old camera inf for no hit
            fake = types.SimpleNamespace(
                cfg=types.SimpleNamespace(use_camera=True, real_cam_model=real, aug_enable=False, depth_stale_prob=0.0, cam_h=False,
                                          rgb_depth_mask=bool(meta.get("rgb_depth_mask", False)),
                                          rgb_mask_value=float(meta.get("rgb_mask_value", 0.5)),
                                          cam_channels=ch, cam_compact=compact, obs_history=1, **consts),
                num_envs=1, _task_errors=lambda: None, _maybe_save_images=lambda *a: None,
                _critic_state=lambda: None, _log_outcomes=lambda: None,
                _real_camera=lambda out: (torch.from_numpy(rgb.astype(np.float32) / 255.0)[None], torch.from_numpy(depth_m)[None, ..., None]),
                _proprio=lambda: torch.cat([torch.from_numpy(q)[None], torch.zeros(1, 1), torch.from_numpy(prev)[None]], dim=-1),
                tiled_camera=types.SimpleNamespace(data=types.SimpleNamespace(output={
                    "rgb": torch.from_numpy(rgb)[None],
                    "depth": torch.from_numpy(np.where(depth_m > 0, depth_m, np.inf).astype(np.float32))[None, ..., None]})))
            sim_obs = torch.clamp(env_get_obs(fake)["policy"], -CLIP_OBS, CLIP_OBS)
            with torch.no_grad():
                mu_ref = ref({"is_train": False, "obs": sim_obs})["mus"]

            policy = op.OmniResetPolicyModel(path)
            worst = 0.0
            for depth in (depth_mm, depth_m):   # 16UC1 mm and 32FC1 m encodings
                policy._prev_action, policy._gripper_state = prev.copy(), 0.0
                img, vec = policy._preprocess(rgb, depth), policy._proprio(q)
                ours = torch.cat([img.permute(0, 2, 3, 1).reshape(1, -1), vec], dim=1)
                if ours.shape != sim_obs.shape:
                    raise ValueError(f"observation shape {tuple(ours.shape)} != env {tuple(sim_obs.shape)}")
                with torch.no_grad():
                    mu = policy._net(img, vec)
                worst = max(worst, (ours - sim_obs).abs().max().item(), (mu - mu_ref).abs().max().item())
            ok = worst < TOL
            print(f"{'PASS' if ok else 'FAIL'}  {name}: {cnn}, input {ch} ({in_c} ch), real_cam_model {real} | "
                  f"max |diff| obs and action {worst:.1e} (tol {TOL:g})")
            failed += not ok
        except Exception as e:   # noqa: BLE001  report every checkpoint, not just the first bad one
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"{len(args.ckpts) - failed}/{len(args.ckpts)} checkpoints match rl-template")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
