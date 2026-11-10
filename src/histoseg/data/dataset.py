"""PyTorch dataset over the tiles written by histoseg.data.tiling."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from histoseg.preprocess import preprocess
from histoseg.stain import random_stain_augment


def read_tiles(tiles_dir: Path, split: str | None = None) -> list[dict]:
    """Rows of tiles.csv, optionally only one split."""
    with open(tiles_dir / "tiles.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"] == split]


def load_tile(tiles_dir: Path, tile_id: str) -> tuple[np.ndarray, np.ndarray]:
    """uint8 RGB image (H, W, 3) and uint8 class-code mask (H, W) of one tile."""
    image = np.asarray(Image.open(tiles_dir / "images" / f"{tile_id}.png").convert("RGB"))
    mask = np.asarray(Image.open(tiles_dir / "masks" / f"{tile_id}.png"))
    return image, mask


class TileDataset(Dataset):
    """Returns (input, target, valid) tensors, each with a leading channel dimension.

    target is 1 where the mask equals the target class; valid is 0 on outside_roi pixels,
    so those pixels are ignored by the loss and the metrics.
    """

    def __init__(
        self,
        rows: list[dict],
        tiles_dir: Path,
        target_code: int,
        outside_code: int,
        augment: dict | None = None,
    ) -> None:
        self.rows = rows
        self.tiles_dir = tiles_dir
        self.target_code = target_code
        self.outside_code = outside_code
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, mask = load_tile(self.tiles_dir, self.rows[index]["tile_id"])
        if self.augment:
            image, mask = self._augment(image, mask)
        x = preprocess(image)
        target = (mask == self.target_code).astype(np.float32)[None]
        valid = (mask != self.outside_code).astype(np.float32)[None]
        return torch.from_numpy(x), torch.from_numpy(target), torch.from_numpy(valid)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Draw the seed from torch: DataLoader seeds torch differently in each worker,
        # so workers do not repeat each other's augmentations.
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        k = int(rng.integers(4))
        image, mask = np.rot90(image, k), np.rot90(mask, k)
        if rng.random() < 0.5:
            image, mask = np.fliplr(image), np.fliplr(mask)
        image = np.ascontiguousarray(image)
        if rng.random() < self.augment["stain_p"]:
            image = random_stain_augment(
                image, rng, self.augment["stain_scale"], self.augment["stain_shift"]
            )
        return image, np.ascontiguousarray(mask)
