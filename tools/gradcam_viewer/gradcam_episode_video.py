#!/usr/bin/env python3
"""
Grad-CAM video of one recorded episode: runs the policy's Grad-CAM (ros2_policy_inference/gradcam.py)
on every frame of a LeRobot dataset episode and writes an H.264 .mp4 that any player opens.

Offline, no ROS: the frames are fed straight to the same policy wrapper and GradCAM class the node
uses. Depth is taken from the dataset's stored training image (already encoded) and turned back into
millimetres, so the policy re-encodes it to the same image. The state is the recorded one
(joint_0..5 + gripper); the recorded action of each frame is printed under the map for comparison.

Usage (conda env lerobot312, from ros2_policy_inference/):
    python tools/gradcam_viewer/gradcam_episode_video.py --policy depth_act --episode 1
    python tools/gradcam_viewer/gradcam_episode_video.py --policy depth_inference \
        --checkpoint checkpoints/diffusion_depth/checkpoint_diffusion_50 --episode 5 --target gripper
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ros2_policy_inference.gradcam import GradCAM, TARGETS, _caption, render  # noqa: E402
from ros2_policy_inference.policies import create_policy  # noqa: E402

RGB_KEY = "observation.images.wrist"
DEPTH_KEY = "observation.images.wrist_depth"
DEFAULT_CKPT = {"act": "act/checkpoint_act_375", "depth_act": "act_depth/checkpoint_act_375",
                "diffusion": "diffusion/checkpoint_diffusion_375",
                "depth_inference": "diffusion_depth/checkpoint_diffusion_375"}


def load_episode(dataset: Path, idx: int):
    info = json.loads((dataset / "meta/info.json").read_text())
    eps = pd.concat(pd.read_parquet(f) for f in sorted((dataset / "meta/episodes").rglob("*.parquet")))
    ep = eps.set_index("episode_index").loc[idx]
    size = info["features"][RGB_KEY]["shape"][0]
    video = dataset / f"videos/{RGB_KEY}/chunk-{ep[f'videos/{RGB_KEY}/chunk_index']:03d}/" \
                      f"file-{ep[f'videos/{RGB_KEY}/file_index']:03d}.mp4"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ep[f'videos/{RGB_KEY}/from_timestamp']:.6f}",
                          "-i", str(video), "-frames:v", str(ep["length"]), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    rgb = np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)
    cols = ["frame_index", "observation.state", "action"] + ([DEPTH_KEY] if DEPTH_KEY in info["features"] else [])
    rows = pq.read_table(dataset / f"data/chunk-{ep['data/chunk_index']:03d}/file-{ep['data/file_index']:03d}.parquet",
                         columns=cols, filters=[("episode_index", "=", idx)]).to_pylist()
    rows.sort(key=lambda r: r["frame_index"])
    if len(rows) != len(rgb):
        raise RuntimeError(f"episode {idx}: {len(rgb)} video frames but {len(rows)} data rows")
    enc_path = dataset / "meta/depth_encoding.json"
    enc = json.loads(enc_path.read_text()) if enc_path.exists() else {"depth_min_m": 0.0, "depth_max_m": 1.0}
    return info["fps"], rgb, rows, enc


def to_obs(rgb, row, enc) -> dict:
    obs = {"rgb": np.array(rgb), "joint_pos": np.asarray(row["observation.state"][:6], np.float32),
           "gripper_state": np.float32(row["observation.state"][6])}
    if DEPTH_KEY in row:
        v = cv2.imdecode(np.frombuffer(row[DEPTH_KEY]["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        d_min, d_max = enc["depth_min_m"], enc["depth_max_m"]
        mm = 1000 * (d_min + (255 - v) / 254 * (d_max - d_min))     # inverse of encode_depth
        obs["depth"] = np.where(v == 0, 0, np.rint(mm)).astype(np.uint16)
    return obs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="depth_act", choices=list(DEFAULT_CKPT))
    ap.add_argument("--checkpoint", help="checkpoint folder (default: the policy's 375k one under checkpoints/)")
    ap.add_argument("--dataset", default=str(REPO.parents[1] / "Robot_Friday/data/lerobot_data_depth"))
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--target", default="motion", choices=list(TARGETS))
    ap.add_argument("--out", help="output .mp4 (default: recordings/gradcam_<policy>_ep<N>_<target>.mp4)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint or REPO / "checkpoints" / DEFAULT_CKPT[args.policy]).resolve()
    out = Path(args.out or REPO / "recordings" / f"gradcam_{args.policy}_ep{args.episode:03d}_{args.target}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    fps, rgb, rows, enc = load_episode(Path(args.dataset), args.episode)
    policy = create_policy(args.policy, str(ckpt))
    gc = GradCAM(policy, target=args.target)
    if any(k == DEPTH_KEY for k in gc.cameras) and DEPTH_KEY not in rows[0]:
        raise SystemExit(f"{args.policy} needs depth, but {args.dataset} has no {DEPTH_KEY}")
    print(f"episode {args.episode}: {len(rgb)} frames at {fps} fps, {args.policy} from {ckpt}, target {args.target}")

    enc_proc, shares, t0 = None, [], time.time()
    frames = []
    for i, (img, row) in enumerate(zip(rgb, rows)):
        frames = (frames + [to_obs(img, row, enc)])[-gc.n_frames:]
        res = gc.compute(frames)
        shares.append([res["share"][k] for k in gc.cameras])
        pic = render(res, args.target)
        rec = np.asarray(row["action"], np.float64)
        _caption(pic, f"frame {i}/{len(rgb) - 1}   recorded: d=({rec[0] * 1e3:+.2f},{rec[1] * 1e3:+.2f},"
                      f"{rec[2] * 1e3:+.2f}) mm/step gripper {rec[6]:.0f}", 2)
        if enc_proc is None:
            h, w = pic.shape[:2]
            enc_proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                         "-pix_fmt", "yuv420p", "-crf", "20", str(out)], stdin=subprocess.PIPE)
        enc_proc.stdin.write(np.ascontiguousarray(pic).tobytes())
        if i % 150 == 0:
            print(f"  frame {i}/{len(rgb)}  ({time.time() - t0:.0f} s)", flush=True)
    enc_proc.stdin.close()
    if enc_proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed writing {out}")
    mean = np.mean(shares, axis=0)
    print(f"wrote {out}  ({len(rgb)} frames, {len(rgb) / fps:.1f} s, made in {time.time() - t0:.0f} s)")
    print("mean share per camera: " + ", ".join(f"{k.rsplit('.', 1)[-1]} {100 * s:.0f}%" for k, s in zip(gc.cameras, mean)))


if __name__ == "__main__":
    main()
