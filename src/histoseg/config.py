"""Load YAML configs and resolve paths from environment variables."""

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


def tracking_uri() -> str:
    """MLflow tracking store (override with MLFLOW_TRACKING_URI)."""
    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """{"a": {"b": 1}} -> {"a.b": 1}, the shape MLflow expects for parameters."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat
