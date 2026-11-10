# Data report

Dataset: BCSS (CC0-1.0), Hugging Face `MedOtter/BCSS` at revision `502d4a3fbc77dbaca6f4664c19e2379ff077d418`.
ROIs are resampled from 0.25 to 0.5 µm/px and cut into non-overlapping 256×256 tiles; tiles with more than 50% outside_roi pixels are dropped. Pixel shares below are of labelled pixels (outside_roi excluded).

## Splits (grouped by patient)

| Split | Patients | Candidate tiles | Kept tiles | Tiles with blood_vessel | blood_vessel pixel share | Tiles with tumor | tumor pixel share |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 106 | 6275 | 5055 | 463 | 0.52% | 3779 | 39.53% |
| val | 23 | 924 | 752 | 72 | 0.68% | 619 | 39.89% |
| test | 22 | 1579 | 1231 | 108 | 0.67% | 871 | 40.16% |

## Target decision

Rule: keep `blood_vessel` unless the training split has fewer than 200 tiles containing it, or it makes up less than 0.5% of labelled pixels; otherwise switch to `tumor`.

Training split: 463 tiles contain `blood_vessel`, 0.521% of labelled pixels.

**Decision: `blood_vessel`** (both thresholds met).

## Reference stain statistics (training split)

Per-tile mean optical density from colour deconvolution (`skimage.color.rgb2hed`), tissue pixels only (mean RGB < 220); then mean and standard deviation across tiles.

| Channel | Mean | Std |
|---|---:|---:|
| Haematoxylin (H) | 0.03274 | 0.01351 |
| Eosin (E) | 0.02934 | 0.00959 |

Computed over 5055 training tiles. Note: the mirror's images were colour-normalised upstream, so natural stain variation is smaller than in raw slides.

## Reproduce

```
make data   # download (resumable) -> split -> tiling -> this report
```
