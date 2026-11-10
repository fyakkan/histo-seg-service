"""Device selection and seeding."""

from __future__ import annotations

import random

import numpy as np
import torch


def select_device() -> torch.device:
    """Prefer an NVIDIA GPU, then Apple Silicon (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed every random number generator the project uses."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
