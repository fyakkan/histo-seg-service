"""Evaluate the champion model on the test patients and write reports/model_report.md.

Usage:
    python -m histoseg.evaluate --config configs/train.yaml

Dice and IoU ignore outside_roi pixels. Scores are reported pooled over all test pixels
and per patient (mean and spread), because one good or bad patient can dominate a pool.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from PIL import Image
from torch.utils.data import DataLoader

from histoseg.config import data_dir, load_config, tracking_uri
from histoseg.data.dataset import TileDataset, read_tiles
from histoseg.device import select_device

REPORT_PATH = Path("reports/model_report.md")


def confusion_counts(
    probs: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, threshold: float
) -> tuple[int, int, int]:
    """True positives, false positives and false negatives over valid pixels."""
    pred = (probs >= threshold) & (valid > 0)
    truth = (target > 0) & (valid > 0)
    tp = int((pred & truth).sum())
    return tp, int(pred.sum()) - tp, int(truth.sum()) - tp


def dice_iou(tp: int, fp: int, fn: int) -> tuple[float, float]:
    """Dice and IoU; NaN when there is no target and no prediction (nothing to score)."""
    if tp + fp + fn == 0:
        return math.nan, math.nan
    return 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Sigmoid probabilities for a batch, returned on the CPU."""
    return torch.sigmoid(model(x.to(device))).cpu()


def overlay_panel(
    image: np.ndarray, target: np.ndarray, probs: np.ndarray, threshold: float
) -> Image.Image:
    """Side-by-side picture: tile | ground truth in green | prediction in magenta."""

    def tint(mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
        out = image.astype(np.float32)
        out[mask] = 0.5 * out[mask] + 0.5 * np.array(colour)
        return out.astype(np.uint8)

    panels = [image, tint(target > 0, (0, 255, 0)), tint(probs >= threshold, (255, 0, 255))]
    return Image.fromarray(np.concatenate(panels, axis=1))


def evaluate_patients(
    model: torch.nn.Module,
    rows: list[dict],
    dataset: TileDataset,
    device: torch.device,
    threshold: float,
    batch_size: int,
) -> dict[str, list[int]]:
    """Accumulate [tp, fp, fn] per patient over all of that patient's tiles."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    offset = 0
    for x, target, valid in loader:
        probs = predict(model, x, device)
        for i in range(len(x)):
            tp, fp, fn = confusion_counts(probs[i], target[i], valid[i], threshold)
            c = counts[rows[offset + i]["patient_id"]]
            c[0], c[1], c[2] = c[0] + tp, c[1] + fp, c[2] + fn
        offset += len(x)
    return dict(counts)


def summarise(counts: dict[str, list[int]]) -> dict[str, float]:
    """Pooled and per-patient Dice/IoU."""
    tp, fp, fn = (sum(c[k] for c in counts.values()) for k in range(3))
    pooled_dice, pooled_iou = dice_iou(tp, fp, fn)
    per_patient = np.array([dice_iou(*c) for c in counts.values()])
    scored = per_patient[~np.isnan(per_patient[:, 0])]
    return {
        "test_dice": pooled_dice,
        "test_iou": pooled_iou,
        "test_dice_patient_mean": float(scored[:, 0].mean()),
        "test_dice_patient_std": float(scored[:, 0].std()),
        "test_dice_patient_median": float(np.median(scored[:, 0])),
        "test_dice_patient_min": float(scored[:, 0].min()),
        "test_dice_patient_max": float(scored[:, 0].max()),
        "test_iou_patient_mean": float(scored[:, 1].mean()),
        "test_iou_patient_std": float(scored[:, 1].std()),
        "test_patients": len(counts),
        "test_patients_scored": len(scored),
        "test_patients_with_target": sum(1 for c in counts.values() if c[0] + c[2] > 0),
    }


def write_report(cfg: dict, data_cfg: dict, version, run, metrics: dict, n_tiles: int) -> None:
    m, rm = metrics, run.data.metrics
    target = data_cfg["target"]["name"]
    text = f"""# Model report

Model `{cfg["mlflow"]["registered_model"]}` version {version.version} (alias `{cfg["mlflow"]["alias"]}`),
MLflow run `{run.info.run_id}`. Target: `{target}` vs everything else; outside_roi pixels ignored.
U-Net with a {cfg["model"]["encoder"]} encoder ({cfg["model"]["encoder_weights"]} weights),
threshold {cfg["train"]["threshold"]}.

## Training

| | |
|---|---:|
| Epochs run | {int(rm["epochs_run"])} |
| Best epoch | {int(rm["best_epoch"])} |
| Best validation Dice (pooled) | {rm["best_val_dice"]:.4f} |
| Training time | {rm["train_minutes"]:.1f} min |

## Test set ({m["test_patients"]} held-out patients, {n_tiles} tiles)

| Metric | Value |
|---|---:|
| Dice, pooled over all test pixels | {m["test_dice"]:.4f} |
| IoU, pooled over all test pixels | {m["test_iou"]:.4f} |
| Dice per patient, mean ± std | {m["test_dice_patient_mean"]:.4f} ± {m["test_dice_patient_std"]:.4f} |
| Dice per patient, median [min, max] | {m["test_dice_patient_median"]:.4f} [{m["test_dice_patient_min"]:.4f}, {m["test_dice_patient_max"]:.4f}] |
| IoU per patient, mean ± std | {m["test_iou_patient_mean"]:.4f} ± {m["test_iou_patient_std"]:.4f} |

Each patient has one ROI. {m["test_patients_with_target"]} of {m["test_patients"]} test patients have
`{target}` pixels; per-patient scores use the {m["test_patients_scored"]} patients where a score
is defined (a patient with no `{target}` pixels and no predicted pixels has nothing to score;
one with no `{target}` pixels but some predicted pixels scores 0).

## Reproduce

```
make data       # download, split, tiles, data report
make train      # trains, logs to MLflow, registers, promotes if better
make evaluate   # this report
```
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = load_config(cfg["data_config"])
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    name, alias = cfg["mlflow"]["registered_model"], cfg["mlflow"]["alias"]
    version = client.get_model_version_by_alias(name, alias)
    run = client.get_run(version.run_id)

    device = select_device()
    model = mlflow.pytorch.load_model(f"models:/{name}@{alias}", map_location="cpu").to(device)

    tiles_dir = data_dir() / data_cfg["tiling"]["subdir"]
    rows = read_tiles(tiles_dir, "test")
    codes = data_cfg["mask_codes"]
    dataset = TileDataset(rows, tiles_dir, data_cfg["target"]["code"], codes["outside_roi"])
    counts = evaluate_patients(
        model, rows, dataset, device, cfg["train"]["threshold"], cfg["train"]["batch_size"]
    )
    metrics = summarise(counts)

    for key, value in metrics.items():
        client.log_metric(run.info.run_id, key, value)
    write_report(cfg, data_cfg, version, run, metrics, len(rows))
    print(f"Wrote {REPORT_PATH}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
