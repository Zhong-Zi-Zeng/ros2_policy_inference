#!/usr/bin/env python3
"""Load every omnireset checkpoint in a folder with the ROS wrapper, run it on fake inputs and time it.

Inside the omnireset image (no ROS topics needed):
  ./run_docker.sh --policy-name omnireset python3 /workspace/src/ros2_policy_inference/tools/check_omnireset_ckpts.py \\
      [/checkpoints/omnireset]
"""
import glob
import os
import sys
import time

import numpy as np
import torch

from ros2_policy_inference.policies.omnireset.omnireset_policy import OmniResetPolicyModel

RATE_HZ = 30.0


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "/checkpoints/omnireset"
    ckpts = sorted(glob.glob(os.path.join(folder, "*.pth")))
    if not ckpts:
        print(f"FAIL  no .pth files in {folder}")
        return 1
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads, cuda available: {torch.cuda.is_available()}")
    rng = np.random.default_rng(0)
    obs = {
        "rgb": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "depth": rng.integers(0, 800, (256, 256), dtype=np.uint16),   # mm, 0 = no return
        "joint_pos": np.array([0.18, 1.28, 0.18, 1.16, 3.29, -1.57], np.float32),
        "gripper_state": np.float32(0.0),
        "ft_wrench": np.zeros(6, np.float32),
    }
    failed = 0
    for path in ckpts:
        name = os.path.basename(path)
        try:
            policy = OmniResetPolicyModel(path)
            times = []
            for _ in range(60):
                t = time.perf_counter()
                action = policy.predict(obs)
                times.append(time.perf_counter() - t)
            times = np.array(times[10:]) * 1000
            ok = action.shape == (7,) and bool(np.isfinite(action).all())
            # predict shares the node's single loop with image decoding: keep it well under one period
            fast = np.percentile(times, 99) < 500.0 / RATE_HZ
            print(f"{'PASS' if ok and fast else 'FAIL'}  {name}: cnn {policy._net.cnn_type}, input {policy._cam_channels} "
                  f"({policy._net.in_c} ch), real_cam_model {policy._real_cam} | predict {np.median(times):.1f} ms "
                  f"(p99 {np.percentile(times, 99):.1f}) | action {np.round(action, 4)}")
            failed += not (ok and fast)
        except Exception as e:   # noqa: BLE001  report every checkpoint, not just the first bad one
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"{len(ckpts) - failed}/{len(ckpts)} checkpoints OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
