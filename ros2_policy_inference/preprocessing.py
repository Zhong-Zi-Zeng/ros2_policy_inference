"""
Observation preprocessing shared by PolicyInferenceNode: image resize,
matching what ros2_data_collector applies before recording so the policy
sees observations in the same format it was trained on.
"""
import numpy as np


def resize_image(img, size, is_depth):
    """Resize an (H,W) / (H,W,C) image to size x size. Depth uses
    nearest-neighbour interpolation so real depth values are never blended."""
    if img is None or getattr(img, "ndim", 0) < 2:
        return img
    if img.shape[0] == size and img.shape[1] == size:
        return img
    import cv2
    interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_AREA
    return np.ascontiguousarray(cv2.resize(img, (size, size), interpolation=interp))
