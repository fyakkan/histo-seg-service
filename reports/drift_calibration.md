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
