# Stain-drift threshold calibration

Windows of 50 tiles drawn at random (2000 windows each) from 300
test-split tiles (patients never used in training), seed 0. "Shifted" applies the demo
stain shift: H and E scaled by [1.0, 1.0] and shifted by [0.03, -0.015]
optical density units. Reference: training-split statistics.

| Windows | Min | 1st pct | Median | 99th pct | Max |
|---|---:|---:|---:|---:|---:|
| Normal | 0.01 | 0.02 | 0.18 | 0.48 | 0.58 |
| Shifted | 1.72 | 1.80 | 2.08 | 2.39 | 2.53 |

Configured threshold: **0.8**. Normal windows above it: 0.00%;
shifted windows at or below it: 0.00%.
The threshold sits between the highest normal score and the lowest shifted score, with a
margin on both sides, so ordinary patient-to-patient variation does not trigger the alert.

Reproduce: `python scripts/calibrate_drift.py` (needs the local tiles from `make data`).

## Effect of the shift on the model

Champion `histo-seg` version 1 on all 1231 test-split tiles
(22 patients), unchanged and with the demo stain shift
(scale [1.0, 1.0], shift [0.03, -0.015]). Drift score: the service's formula with the whole set as one
window. Segmentation probability threshold 0.5; outside_roi pixels ignored.

| Test tiles | Dice (pooled) | IoU (pooled) | Dice per patient, mean ± std | Median | Drift score |
|---|---:|---:|---:|---:|---:|
| Unchanged | 0.3327 | 0.1995 | 0.2318 ± 0.2231 | 0.1630 | 0.05 |
| Shifted | 0.1392 | 0.0748 | 0.1912 ± 0.2290 | 0.1520 | 1.97 |

Change in pooled Dice under the shift: -0.1935.

Reproduce: `python scripts/shift_impact.py` (needs the local tiles and the MLflow champion).
