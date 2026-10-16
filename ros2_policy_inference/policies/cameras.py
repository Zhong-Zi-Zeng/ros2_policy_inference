"""Maps a checkpoint's image features to the node's obs, as declared by each policy's config.yaml.

A checkpoint states WHICH image inputs it takes (config.json's image_features); config.yaml's
cameras block states where each one comes from and how to encode it:

    cameras:
      observation.images.wrist:
        source: rgb
        encoding: rgb
      observation.images.depth:
        source: depth
        encoding: depth
        info_path: /checkpoints/<ckpt>/depth_info.json

A new camera configuration (rgb only, depth only, rgb+depth, two rgb+depth cameras) is then a
config.yaml change, not a code change.
"""
import json

import numpy as np


def encode_depth(depth, d_min, d_max):
    """Same encoding as the dataset conversion: clip to [d_min, d_max] mm, near=bright,
    valid -> 1..255, invalid (0/NaN) -> 0, grey replicated to 3 channels."""
    raw = np.asarray(depth, dtype=np.float32)
    if np.issubdtype(np.asarray(depth).dtype, np.floating):
        raw = raw * 1000.0  # 32FC1 is metres, the range is mm
    valid = np.isfinite(raw) & (raw > 0)
    x = np.clip((raw - d_min) / (d_max - d_min), 0.0, 1.0)
    gray = np.where(valid, 1.0 + np.round((1.0 - x) * 254.0), 0.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _make_encoder(spec):
    """One cameras entry -> f(obs value) -> the (H, W, 3) uint8 the model expects. A function
    rather than an inline lambda so each camera's parameters are captured by value."""
    encoding = spec["encoding"]
    if encoding == "rgb":
        return lambda rgb: np.ascontiguousarray(rgb, dtype=np.uint8)
    if encoding == "depth":
        with open(spec["info_path"]) as f:
            info = json.load(f)
        # two info schemas in the wild: depth_min/depth_max (mm) or depth_min_m/depth_max_m (metres)
        if "depth_min_m" in info:
            d_min, d_max = float(info["depth_min_m"]) * 1000.0, float(info["depth_max_m"]) * 1000.0
        else:
            d_min, d_max = float(info["depth_min"]), float(info["depth_max"])
        return lambda depth: encode_depth(depth, d_min, d_max)
    raise ValueError(f"unknown camera encoding '{encoding}' (known: rgb, depth)")


def build_cameras(cameras, declared, ckpt_path):
    """config.yaml's cameras -> {image feature: (obs key, encoder)}.

    declared is the set of image features the caller expects to feed (pi05 leaves out its
    empty_camera_* placeholders, which it fills itself). The two must match exactly, so a
    checkpoint whose inputs changed fails here rather than at the first inference.
    """
    if set(cameras) != set(declared):
        raise ValueError(
            f"config.yaml cameras {sorted(cameras)} don't match the image features "
            f"{sorted(declared)} of {ckpt_path}"
        )
    return {key: (spec["source"], _make_encoder(spec)) for key, spec in cameras.items()}
