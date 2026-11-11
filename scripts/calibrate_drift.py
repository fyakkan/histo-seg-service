"""Choose the stain-drift threshold from normal and stain-shifted test tiles.

Usage (needs the local tiles from `make data`):
    python scripts/calibrate_drift.py

Simulates many request windows of (a) normal test-split tiles and (b) the same tiles with
the demo stain shift applied, and writes the score distributions to
reports/drift_calibration.md. Window sizes: the full window (configs/serve.yaml drift.window)
and the smallest window the alert accepts (drift.min_window).
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
    # Drawn after the full windows so the full-window rows stay the same as before.
    normal_min = window_scores(normal, reference, serve["min_window"], rng)
    shifted_min = window_scores(shifted, reference, serve["min_window"], rng)

    def row(name: str, s: np.ndarray) -> str:
        p = np.percentile(s, [1, 50, 99])
        return f"| {name} | {s.min():.2f} | {p[0]:.2f} | {p[1]:.2f} | {p[2]:.2f} | {s.max():.2f} |"

    threshold = serve["threshold"]
    w, m = serve["window"], serve["min_window"]
    report = f"""# Stain-drift threshold calibration

Windows of {serve["window"]} tiles drawn at random ({N_WINDOWS} windows each) from {N_TILES}
test-split tiles (patients never used in training), seed {SEED}. "Shifted" applies the demo
stain shift: H and E scaled by {serve["demo_scale"]} and shifted by {serve["demo_shift"]}
optical density units. Reference: training-split statistics.

| Windows | Min | 1st pct | Median | 99th pct | Max |
|---|---:|---:|---:|---:|---:|
{row("Normal", normal_scores)}
{row("Shifted", shifted_scores)}
{row(f"Normal, {m} tiles", normal_min)}
{row(f"Shifted, {m} tiles", shifted_min)}

Configured threshold: **{threshold}**. Normal windows above it: {np.mean(normal_scores > threshold):.2%}
({w} tiles), {np.mean(normal_min > threshold):.2%} ({m} tiles); shifted windows at or below it:
{np.mean(shifted_scores <= threshold):.2%} ({w} tiles), {np.mean(shifted_min <= threshold):.2%} ({m} tiles).
With full windows of {w} tiles the threshold sits between the highest normal score and the
lowest shifted score, with a margin on both sides, so ordinary patient-to-patient variation
does not trigger the alert. Windows of only {m} tiles are noisier: the highest normal score is
{normal_min.max():.2f}, so for the first requests after a restart a false alarm is rare but still
possible (the rule must also stay true for 1 minute, while the window keeps filling up to {w}).

Minimum window: the alert waits until the window holds {m} requests (drift.min_window).
Bug fixed on 11 Nov 2026: right after a restart, one /predict request gave a drift score of
1.64 and StainDriftHigh fired, because the window held a single image; the service now exposes
the window size and the rule requires at least {m} requests (tested in
monitoring/alert_rules_test.yml).

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
