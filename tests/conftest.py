"""Shared test helpers: the fixture tiles in tests/fixtures/."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_rows() -> list[dict]:
    with open(FIXTURES / "tiles.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def tile(fixture_rows) -> np.ndarray:
    """One fixture tile as a uint8 RGB array."""
    return np.asarray(
        Image.open(FIXTURES / "images" / f"{fixture_rows[0]['tile_id']}.png").convert("RGB")
    )
