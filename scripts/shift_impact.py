"""Measure how the demo stain shift affects the champion model on the test patients.

Usage (needs the local tiles from `make data` and the MLflow champion from `make train`):
    python scripts/shift_impact.py

Runs the champion on every test-split tile twice: unchanged, and with the stain shift that
scripts/demo_drift.py uses (configs/serve.yaml drift.demo_scale / demo_shift). Reports Dice
for both sets and the drift score of each whole set, and writes them into
reports/drift_calibration.md (section "Effect of the shift on the model").
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import torch

from histoseg.config import data_dir, load_config, tracking_uri
from histoseg.data.dataset import TileDataset, load_tile, read_tiles
from histoseg.device import select_device
from histoseg.evaluate import evaluate_patients, summarise
from histoseg.preprocess import preprocess
from histoseg.serving.drift import StainDriftMonitor
from histoseg.stain import shift_stain, stain_features

REPORT = Path("reports/drift_calibration.md")
SECTION = "## Effect of the shift on the model"


class ShiftedTileDataset(TileDataset):
    """Test tiles with a fixed stain shift applied before preprocessing."""

    def __init__(self, *args, scale, shift, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale, self.shift = scale, shift

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Same as TileDataset.__getitem__ (no augmentation), plus the stain shift.
        image, mask = load_tile(self.tiles_dir, self.rows[index]["tile_id"])
        x = preprocess(shift_stain(image, self.scale, self.shift))
        target = (mask == self.target_code).astype(np.float32)[None]
        valid = (mask != self.outside_code).astype(np.float32)[None]
        return torch.from_numpy(x), torch.from_numpy(target), torch.from_numpy(valid)


def set_drift_score(dataset: TileDataset, reference: dict, shifted: bool, scale, shift) -> float:
    """Drift score (same formula as the service) with the whole set as one window."""
    monitor = StainDriftMonitor(reference, window=len(dataset.rows))
    for row in dataset.rows:
        image, _ = load_tile(dataset.tiles_dir, row["tile_id"])
        if shifted:
            image = shift_stain(image, scale, shift)
        monitor.update(stain_features(image))
    return monitor.score()


def main() -> None:
    cfg = load_config("configs/train.yaml")
    data_cfg = load_config(cfg["data_config"])
    drift = load_config("configs/serve.yaml")["drift"]
    scale, shift = drift["demo_scale"], drift["demo_shift"]

    tiles_dir = data_dir() / data_cfg["tiling"]["subdir"]
    reference = json.loads((tiles_dir / "stain_reference.json").read_text(encoding="utf-8"))
    rows = read_tiles(tiles_dir, "test")
    target_code, outside = data_cfg["target"]["code"], data_cfg["mask_codes"]["outside_roi"]

    mlflow.set_tracking_uri(tracking_uri())
    name, alias = cfg["mlflow"]["registered_model"], cfg["mlflow"]["alias"]
    version = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    device = select_device()
    model = mlflow.pytorch.load_model(f"models:/{name}@{alias}", map_location="cpu").to(device)
    threshold, batch = cfg["train"]["threshold"], cfg["train"]["batch_size"]

    results = {}
    for label, dataset in [
        ("Unchanged", TileDataset(rows, tiles_dir, target_code, outside)),
        (
            "Shifted",
            ShiftedTileDataset(rows, tiles_dir, target_code, outside, scale=scale, shift=shift),
        ),
    ]:
        metrics = summarise(evaluate_patients(model, rows, dataset, device, threshold, batch))
        score = set_drift_score(dataset, reference, label == "Shifted", scale, shift)
        results[label] = (metrics, score)
        print(
            f"{label}: pooled Dice {metrics['test_dice']:.4f}, "
            f"per-patient Dice {metrics['test_dice_patient_mean']:.4f}, drift score {score:.2f}"
        )

    def row(label: str) -> str:
        m, score = results[label]
        return (
            f"| {label} | {m['test_dice']:.4f} | {m['test_iou']:.4f} | "
            f"{m['test_dice_patient_mean']:.4f} ± {m['test_dice_patient_std']:.4f} | "
            f"{m['test_dice_patient_median']:.4f} | {score:.2f} |"
        )

    change = results["Shifted"][0]["test_dice"] - results["Unchanged"][0]["test_dice"]
    section = f"""{SECTION}

Champion `{name}` version {version.version} on all {len(rows)} test-split tiles
({results["Unchanged"][0]["test_patients"]} patients), unchanged and with the demo stain shift
(scale {scale}, shift {shift}). Drift score: the service's formula with the whole set as one
window. Segmentation probability threshold {cfg["train"]["threshold"]}; outside_roi pixels ignored.

| Test tiles | Dice (pooled) | IoU (pooled) | Dice per patient, mean ± std | Median | Drift score |
|---|---:|---:|---:|---:|---:|
{row("Unchanged")}
{row("Shifted")}

Change in pooled Dice under the shift: {change:+.4f}.

Reproduce: `python scripts/shift_impact.py` (needs the local tiles and the MLflow champion).
"""
    text = REPORT.read_text(encoding="utf-8")
    head = text.split(SECTION)[0].rstrip() + "\n\n"
    REPORT.write_text(head + section, encoding="utf-8")
    print(section)


if __name__ == "__main__":
    main()
