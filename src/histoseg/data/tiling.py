"""Resample each ROI to the target resolution and cut it into non-overlapping tiles.

Usage:
    python -m histoseg.data.tiling --config configs/data.yaml

Writes <data>/tiles/images/*.png, <data>/tiles/masks/*.png (original class codes),
and <data>/tiles/tiles.csv with the split and per-class pixel counts of every kept tile.
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
from PIL import Image

from histoseg.config import data_dir, load_config
from histoseg.data.split import patient_to_split

# The largest BCSS ROI is ~91 megapixels, just above PIL's default bomb limit (~89 MP).
# These are trusted files from a pinned dataset revision, so a slightly higher limit is safe.
Image.MAX_IMAGE_PIXELS = 100_000_000


def resample(image: Image.Image, mask: Image.Image, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Resize image (area averaging) and mask (nearest neighbour, so no new class codes)."""
    size = (round(image.width * scale), round(image.height * scale))
    image_small = image.resize(size, Image.Resampling.BOX)
    mask_small = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image_small), np.asarray(mask_small)


def tile_grid(height: int, width: int, tile_size: int) -> list[tuple[int, int]]:
    """Top-left corners of full, non-overlapping tiles; the leftover border is dropped."""
    return [
        (y, x)
        for y in range(0, height - tile_size + 1, tile_size)
        for x in range(0, width - tile_size + 1, tile_size)
    ]


def tile_roi(
    image: np.ndarray, mask: np.ndarray, tile_size: int, max_outside: float, outside_code: int
) -> tuple[list[tuple[int, int, np.ndarray, np.ndarray]], int]:
    """Cut one ROI into tiles and drop mostly-unannotated ones.

    Returns the kept (y, x, image_tile, mask_tile) tuples and the number of candidate tiles.
    """
    corners = tile_grid(mask.shape[0], mask.shape[1], tile_size)
    kept = []
    for y, x in corners:
        mask_tile = mask[y : y + tile_size, x : x + tile_size]
        if np.mean(mask_tile == outside_code) > max_outside:
            continue
        kept.append((y, x, image[y : y + tile_size, x : x + tile_size], mask_tile))
    return kept, len(corners)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tiling, codes = cfg["tiling"], cfg["mask_codes"]
    scale = cfg["dataset"]["source_mpp"] / tiling["target_mpp"]
    src_dir = data_dir() / cfg["download"]["subdir"]
    out_dir = data_dir() / tiling["subdir"]
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    with open(cfg["split"]["output"], encoding="utf-8") as f:
        split_of = patient_to_split(json.load(f))
    with open(src_dir / "metadata.csv", encoding="utf-8") as f:
        rois = sorted(csv.DictReader(f), key=lambda r: r["image_id"])

    rows, candidates = [], {"train": 0, "val": 0, "test": 0}
    start = time.time()
    for i, roi in enumerate(rois, start=1):
        image_id, split = roi["image_id"], split_of[roi["patient_id"]]
        with Image.open(src_dir / "images" / f"{image_id}.png") as im:
            with Image.open(src_dir / "masks" / f"{image_id}.png") as mk:
                image, mask = resample(im.convert("RGB"), mk, scale)

        tiles, n_candidates = tile_roi(
            image, mask, tiling["tile_size"], tiling["max_outside_roi"], codes["outside_roi"]
        )
        candidates[split] += n_candidates
        for y, x, image_tile, mask_tile in tiles:
            tile_id = f"{image_id}_y{y:05d}_x{x:05d}"
            Image.fromarray(image_tile).save(out_dir / "images" / f"{tile_id}.png")
            Image.fromarray(mask_tile).save(out_dir / "masks" / f"{tile_id}.png")
            counts = np.bincount(mask_tile.ravel(), minlength=256)
            rows.append(
                {
                    "tile_id": tile_id,
                    "image_id": image_id,
                    "patient_id": roi["patient_id"],
                    "split": split,
                    "y": y,
                    "x": x,
                    "n_pixels": mask_tile.size,
                    **{f"n_{name}": int(counts[code]) for name, code in codes.items()},
                }
            )
        print(f"[{i:3d}/{len(rois)}] {image_id}: {len(tiles)}/{n_candidates} tiles kept ({split})")

    with open(out_dir / "tiles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"candidate_tiles": candidates, "scale": scale, "tile_size": tiling["tile_size"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Done: {len(rows)} tiles in {time.time() - start:.0f} s -> {out_dir / 'tiles.csv'}")


if __name__ == "__main__":
    main()
