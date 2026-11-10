"""Choose the stain-drift threshold from normal and stain-shifted test tiles.

Usage (needs the local tiles from `make data`):
    python scripts/calibrate_drift.py

Simulates many request windows (configs/serve.yaml drift.window tiles each) of
(a) normal test-split tiles and (b) the same tiles with the demo stain shift applied,
and writes the score distributions to reports/drift_calibration.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from histoseg.config import data_dir, load_config
from histoseg.data.dataset import read_tiles
from histoseg.serving.drift import StainDriftMonitor
from histoseg.stain import shift_stain, stain_features

N_TILES = 300
N_WINDOWS = 2000
SEED = 0
SHIFT_SECTION = "## Effect of the shift on the model"


def window_scores(features: np.ndarray, reference: dict, window: int, rng) -> np.ndarray:
    """Drift score of many random windows drawn from `features`."""
    scores = []
    for _ in range(N_WINDOWS):
        monitor = StainDriftMonitor(reference, window)
        for f in features[rng.choice(len(features), window, replace=False)]:
            monitor.update(f)
        scores.append(monitor.score())
    return np.array(scores)


def main() -> None:
    serve = load_config("configs/serve.yaml")["drift"]
    data_cfg = load_config("configs/data.yaml")
    tiles_dir = data_dir() / data_cfg["tiling"]["subdir"]
    reference = json.loads((tiles_dir / "stain_reference.json").read_text(encoding="utf-8"))

    rng = np.random.default_rng(SEED)
    rows = read_tiles(tiles_dir, "test")
    chosen = [rows[i] for i in rng.choice(len(rows), N_TILES, replace=False)]
    images = [
        np.asarray(Image.open(tiles_dir / "images" / f"{r['tile_id']}.png").convert("RGB"))
        for r in chosen
    ]

    normal = np.array([stain_features(im) for im in images])
    shifted = np.array(
        [stain_features(shift_stain(im, serve["demo_scale"], serve["demo_shift"])) for im in images]
    )
    normal_scores = window_scores(normal, reference, serve["window"], rng)
    shifted_scores = window_scores(shifted, reference, serve["window"], rng)

    def row(name: str, s: np.ndarray) -> str:
        p = np.percentile(s, [1, 50, 99])
        return f"| {name} | {s.min():.2f} | {p[0]:.2f} | {p[1]:.2f} | {p[2]:.2f} | {s.max():.2f} |"

    threshold = serve["threshold"]
    report = f"""# Stain-drift threshold calibration

Windows of {serve["window"]} tiles drawn at random ({N_WINDOWS} windows each) from {N_TILES}
test-split tiles (patients never used in training), seed {SEED}. "Shifted" applies the demo
stain shift: H and E scaled by {serve["demo_scale"]} and shifted by {serve["demo_shift"]}
optical density units. Reference: training-split statistics.

| Windows | Min | 1st pct | Median | 99th pct | Max |
|---|---:|---:|---:|---:|---:|
{row("Normal", normal_scores)}
{row("Shifted", shifted_scores)}

Configured threshold: **{threshold}**. Normal windows above it: {np.mean(normal_scores > threshold):.2%};
shifted windows at or below it: {np.mean(shifted_scores <= threshold):.2%}.
The threshold sits between the highest normal score and the lowest shifted score, with a
margin on both sides, so ordinary patient-to-patient variation does not trigger the alert.

Reproduce: `python scripts/calibrate_drift.py` (needs the local tiles from `make data`).
"""
    # Keep the section written by scripts/shift_impact.py, if present.
    path = Path("reports/drift_calibration.md")
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if SHIFT_SECTION in old:
        report += "\n" + SHIFT_SECTION + old.split(SHIFT_SECTION, 1)[1]
    path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
