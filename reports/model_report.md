# Model report

Model `histo-seg` version 1 (alias `champion`),
MLflow run `321facae9bdf4ea3a786c3b9d12272f7`. Target: `blood_vessel` vs everything else; outside_roi pixels ignored.
U-Net with a resnet34 encoder (imagenet weights),
threshold 0.5.

## Context

`blood_vessel` is a rare class: 0.52 % of labelled training pixels (see `reports/data_report.md`).
Thin vessels and small lumens make pixel-level overlap hard, so a modest Dice is expected.
All numbers below are reported exactly as measured.

## Training

| | |
|---|---:|
| Epochs run | 12 |
| Best epoch | 10 |
| Best validation Dice (pooled) | 0.3223 |
| Training time | 39.9 min |

## Test set (22 held-out patients, 1231 tiles)

| Metric | Value |
|---|---:|
| Dice, pooled over all test pixels | 0.3327 |
| IoU, pooled over all test pixels | 0.1995 |
| Dice per patient, mean ± std | 0.2318 ± 0.2231 |
| Dice per patient, median [min, max] | 0.1630 [0.0000, 0.6858] |
| IoU per patient, mean ± std | 0.1508 ± 0.1571 |

Each patient has one ROI. 19 of 22 test patients have
`blood_vessel` pixels; per-patient scores use the 22 patients where a score
is defined (a patient with no `blood_vessel` pixels and no predicted pixels has nothing to score;
one with no `blood_vessel` pixels but some predicted pixels scores 0).

## Reproduce

```
make data       # download, split, tiles, data report
make train      # trains, logs to MLflow, registers, promotes if better
make evaluate   # this report
```
