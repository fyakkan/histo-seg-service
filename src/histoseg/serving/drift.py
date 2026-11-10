"""Stain-drift score: how far recent inputs' H&E staining is from the training data.

For each request we take the mean haematoxylin and eosin optical density of its tissue
pixels (histoseg.stain.stain_features). Over the last `window` requests:

    score = max over H and E of |window mean - reference mean| / reference std

The reference mean and std come from the training split (model card). A score of 1 means
recent inputs are, on average, one training-set standard deviation away in H or E.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class StainDriftMonitor:
    def __init__(self, reference: dict, window: int = 50) -> None:
        self.ref_mean = np.array([reference["h_mean"], reference["e_mean"]])
        self.ref_std = np.array([reference["h_std"], reference["e_std"]])
        self.recent: deque[np.ndarray] = deque(maxlen=window)
        # Requests are handled in parallel threads; the lock keeps the window consistent.
        self._lock = threading.Lock()

    def update(self, features: np.ndarray) -> float:
        """Add one request's [H, E] features (ignored if NaN, i.e. no tissue); return the score."""
        with self._lock:
            if not np.isnan(features).any():
                self.recent.append(features)
            return self._score()

    def score(self) -> float:
        """Current drift score; 0 before any request with tissue has been seen."""
        with self._lock:
            return self._score()

    def _score(self) -> float:
        if not self.recent:
            return 0.0
        window_mean = np.mean(self.recent, axis=0)
        return float(np.max(np.abs(window_mean - self.ref_mean) / self.ref_std))
