"""Dice of the ONNX model on the committed test fixtures (numpy only, no PyTorch).

Used at export time to record a reference score in the model card, and in CI as a
regression guard: a changed model or preprocessing that hurts predictions fails the test.
It is a sanity check on 8 tiles, not an evaluation (see reports/model_report.md for that).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from histoseg.serving.inference import SegmentationModel

FIXTURES = Path("tests/fixtures")


def fixture_dice(
    model: SegmentationModel, target_code: int, outside_code: int, fixtures: Path = FIXTURES
) -> float:
    """Pooled Dice over all fixture tiles, ignoring outside_roi pixels."""
    with open(fixtures / "tiles.csv", encoding="utf-8") as f:
        tile_ids = [row["tile_id"] for row in csv.DictReader(f)]
    tp = fp = fn = 0
    for tile_id in tile_ids:
        rgb = np.asarray(Image.open(fixtures / "images" / f"{tile_id}.png").convert("RGB"))
        mask = np.asarray(Image.open(fixtures / "masks" / f"{tile_id}.png"))
        pred, _ = model.predict(rgb)
        valid = mask != outside_code
        truth = (mask == target_code) & valid
        pred = pred & valid
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
