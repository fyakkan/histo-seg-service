"""Class statistics, the target decision rule, and reference stain statistics.

Usage:
    python -m histoseg.data.report --config configs/data.yaml

Writes reports/data_report.md and <data>/tiles/stain_reference.json.
Exits with an error if configs/data.yaml names a different target than the rule picks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from histoseg.config import data_dir, load_config
from histoseg.data.split import SPLITS
from histoseg.stain import stain_features

REPORT_PATH = Path("reports/data_report.md")


def class_stats(rows: list[dict], name: str) -> dict:
    """Tiles containing a class and its share of labelled (not outside_roi) pixels."""
    labelled = sum(int(r["n_pixels"]) - int(r["n_outside_roi"]) for r in rows)
    pixels = sum(int(r[f"n_{name}"]) for r in rows)
    return {
        "tiles_with": sum(int(r[f"n_{name}"]) > 0 for r in rows),
        "pixel_fraction": pixels / labelled if labelled else 0.0,
    }


def decide_target(train_rows: list[dict], target: dict) -> tuple[str, list[str]]:
    """Apply the decision rule from the spec; return the chosen class and the reasons."""
    stats = class_stats(train_rows, target["name"])
    reasons = []
    if stats["tiles_with"] < target["min_train_tiles"]:
        reasons.append(f"{stats['tiles_with']} training tiles < {target['min_train_tiles']}")
    if stats["pixel_fraction"] < target["min_pixel_fraction"]:
        reasons.append(
            f"{stats['pixel_fraction']:.3%} of labelled pixels < {target['min_pixel_fraction']:.1%}"
        )
    return (target["fallback"] if reasons else target["name"]), reasons


def stain_reference(image_paths: list[Path]) -> dict:
    """Mean and std (across tiles) of the per-tile mean H and E optical density."""
    features = np.array(
        [stain_features(np.asarray(Image.open(p).convert("RGB"))) for p in image_paths]
    )
    features = features[~np.isnan(features).any(axis=1)]  # tiles without tissue have no stain
    mean, std = features.mean(axis=0), features.std(axis=0)
    return {
        "n_tiles": int(len(features)),
        "h_mean": float(mean[0]),
        "h_std": float(std[0]),
        "e_mean": float(mean[1]),
        "e_std": float(std[1]),
    }


def build_report(
    cfg: dict, rows: list[dict], summary: dict, chosen: str, reasons: list[str], ref: dict
) -> str:
    target, tiling = cfg["target"], cfg["tiling"]
    names = [target["name"], target["fallback"]]
    lines = [
        "# Data report",
        "",
        f"Dataset: BCSS ({cfg['dataset']['license']}), Hugging Face `{cfg['dataset']['repo_id']}`"
        f" at revision `{cfg['dataset']['revision']}`.",
        f"ROIs are resampled from {cfg['dataset']['source_mpp']} to {tiling['target_mpp']} µm/px"
        f" and cut into non-overlapping {tiling['tile_size']}×{tiling['tile_size']} tiles;"
        f" tiles with more than {tiling['max_outside_roi']:.0%} outside_roi pixels are dropped."
        " Pixel shares below are of labelled pixels (outside_roi excluded).",
        "",
        "## Splits (grouped by patient)",
        "",
        "| Split | Patients | Candidate tiles | Kept tiles | "
        + " | ".join(f"Tiles with {n} | {n} pixel share" for n in names)
        + " |",
        "|---|---:|---:|---:|" + "---:|---:|" * len(names),
    ]
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        cells = []
        for n in names:
            s = class_stats(part, n)
            cells += [str(s["tiles_with"]), f"{s['pixel_fraction']:.2%}"]
        patients = len({r["patient_id"] for r in part})
        candidates = summary["candidate_tiles"][split]
        lines.append(
            f"| {split} | {patients} | {candidates} | {len(part)} | " + " | ".join(cells) + " |"
        )

    train = [r for r in rows if r["split"] == "train"]
    s = class_stats(train, target["name"])
    lines += [
        "",
        "## Target decision",
        "",
        f"Rule: keep `{target['name']}` unless the training split has fewer than"
        f" {target['min_train_tiles']} tiles containing it, or it makes up less than"
        f" {target['min_pixel_fraction']:.1%} of labelled pixels; otherwise switch to"
        f" `{target['fallback']}`.",
        "",
        f"Training split: {s['tiles_with']} tiles contain `{target['name']}`,"
        f" {s['pixel_fraction']:.3%} of labelled pixels.",
        "",
        f"**Decision: `{chosen}`**"
        + (f" (rule triggered: {'; '.join(reasons)})." if reasons else " (both thresholds met)."),
        "",
        "## Reference stain statistics (training split)",
        "",
        "Per-tile mean optical density from colour deconvolution (`skimage.color.rgb2hed`),"
        " tissue pixels only (mean RGB < 220); then mean and standard deviation across tiles.",
        "",
        "| Channel | Mean | Std |",
        "|---|---:|---:|",
        f"| Haematoxylin (H) | {ref['h_mean']:.5f} | {ref['h_std']:.5f} |",
        f"| Eosin (E) | {ref['e_mean']:.5f} | {ref['e_std']:.5f} |",
        "",
        f"Computed over {ref['n_tiles']} training tiles. Note: the mirror's images were"
        " colour-normalised upstream, so natural stain variation is smaller than in raw slides.",
        "",
        "## Reproduce",
        "",
        "```",
        "make data   # download (resumable) -> split -> tiling -> this report",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tiles_dir = data_dir() / cfg["tiling"]["subdir"]
    with open(tiles_dir / "tiles.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary = json.loads((tiles_dir / "summary.json").read_text(encoding="utf-8"))

    train = [r for r in rows if r["split"] == "train"]
    chosen, reasons = decide_target(train, cfg["target"])
    print(f"Computing stain reference over {len(train)} training tiles ...")
    ref = stain_reference([tiles_dir / "images" / f"{r['tile_id']}.png" for r in train])
    (tiles_dir / "stain_reference.json").write_text(
        json.dumps(ref, indent=2) + "\n", encoding="utf-8"
    )

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(build_report(cfg, rows, summary, chosen, reasons, ref), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}. Decision: {chosen} {reasons or ''}")
    if chosen != cfg["target"]["name"]:
        sys.exit(f"The decision rule picks '{chosen}': update target in {args.config}.")


if __name__ == "__main__":
    main()
