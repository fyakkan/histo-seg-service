"""H&E stain features and stain augmentation via colour deconvolution (rgb2hed).

The same functions serve three purposes: stain augmentation during training,
reference stain statistics, and (later) the simulated stain drift in the monitoring demo.
"""

from __future__ import annotations

import numpy as np
from skimage.color import hed_from_rgb, rgb2hed, rgb_from_hed

# Glass background is nearly white; tissue is darker. Mean RGB below this counts as tissue.
TISSUE_MAX_BRIGHTNESS = 220

# skimage's optical-density scaling: log(intensity) / log(1e-6).
_LOG_MIN = np.log(1e-6)


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean HxW mask of tissue pixels in a uint8 RGB image."""
    return rgb.mean(axis=-1) < TISSUE_MAX_BRIGHTNESS


def stain_features(rgb: np.ndarray) -> np.ndarray:
    """Mean haematoxylin and eosin optical density over tissue pixels, as [H, E].

    Returns NaNs if the image has no tissue, so blank inputs never pull the average to zero.
    """
    mask = tissue_mask(rgb)
    if not mask.any():
        return np.array([np.nan, np.nan])
    hed = rgb2hed(rgb)
    return hed[mask][:, :2].mean(axis=0)


def shift_stain(
    rgb: np.ndarray, scale: tuple[float, float], shift: tuple[float, float]
) -> np.ndarray:
    """Change H and E stain amounts: new = old * scale + shift, per channel.

    The third (DAB) channel is left unchanged because H&E slides contain no DAB.
    Uses skimage's stain matrices directly: rgb2hed clips negative values to zero, so
    rgb2hed -> hed2rgb changes some pixels even with no shift. Without clipping, a neutral
    shift returns the original image exactly.
    """
    od = np.log(np.maximum(rgb / 255.0, 1e-6)) / _LOG_MIN
    hed = od @ hed_from_rgb
    for channel in (0, 1):
        hed[..., channel] = hed[..., channel] * scale[channel] + shift[channel]
    out = np.exp((hed @ rgb_from_hed) * _LOG_MIN)
    return (np.clip(out, 0.0, 1.0) * 255).round().astype(np.uint8)


def random_stain_augment(
    rgb: np.ndarray, rng: np.random.Generator, max_scale: float, max_shift: float
) -> np.ndarray:
    """Random H&E jitter for training, so the model is less sensitive to staining differences."""
    scale = tuple(rng.uniform(1 - max_scale, 1 + max_scale, size=2))
    shift = tuple(rng.uniform(-max_shift, max_shift, size=2))
    return shift_stain(rgb, scale, shift)
