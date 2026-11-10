"""The single preprocessing function shared by training and serving.

Keeping one implementation prevents training-serving skew: the model always sees
inputs prepared in exactly the same way.
"""

from __future__ import annotations

import numpy as np

# The encoder was pretrained on ImageNet, so inputs are normalised with ImageNet statistics.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB image (H, W, 3) -> normalised float32 array (3, H, W)."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected uint8 array of shape (H, W, 3), got {rgb.dtype} {rgb.shape}")
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))
