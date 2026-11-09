"""Load YAML configs and resolve data paths from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_dir() -> Path:
    """Root folder for downloaded and derived data (override with HISTOSEG_DATA_DIR)."""
    return Path(os.environ.get("HISTOSEG_DATA_DIR", "./data"))
