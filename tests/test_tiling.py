import numpy as np
from PIL import Image

from histoseg.data.tiling import resample, tile_grid, tile_roi

OUTSIDE = 0


def make_roi(height=600, width=700, seed=0):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    mask = rng.choice(np.array([1, 2, 18], dtype=np.uint8), size=(height, width))
    return image, mask


def test_tiles_do_not_overlap_and_stay_inside_the_roi():
    height, width, size = 600, 700, 256
    coverage = np.zeros((height, width), dtype=int)
    for y, x in tile_grid(height, width, size):
        assert y + size <= height and x + size <= width
        coverage[y : y + size, x : x + size] += 1
    assert coverage.max() == 1
    assert len(tile_grid(height, width, size)) == 2 * 2


def test_tiles_with_mostly_outside_roi_are_dropped():
    image, mask = make_roi(512, 512)
    mask[:256, :256] = OUTSIDE  # top-left tile fully outside
    mask[256:, :256][:100] = OUTSIDE  # bottom-left tile ~39 % outside: kept
    tiles, candidates = tile_roi(image, mask, 256, 0.5, OUTSIDE)
    assert candidates == 4
    assert sorted((y, x) for y, x, _, _ in tiles) == [(0, 256), (256, 0), (256, 256)]


def test_tiling_is_deterministic():
    image, mask = make_roi()
    first, _ = tile_roi(image, mask, 256, 0.5, OUTSIDE)
    second, _ = tile_roi(image, mask, 256, 0.5, OUTSIDE)
    assert len(first) == len(second)
    for (y1, x1, i1, m1), (y2, x2, i2, m2) in zip(first, second, strict=True):
        assert (y1, x1) == (y2, x2)
        assert np.array_equal(i1, i2) and np.array_equal(m1, m2)


def test_resample_halves_size_and_keeps_only_existing_mask_codes():
    image, mask = make_roi(600, 700)
    small_image, small_mask = resample(Image.fromarray(image), Image.fromarray(mask), 0.5)
    assert small_image.shape == (300, 350, 3)
    assert small_mask.shape == (300, 350)
    assert set(np.unique(small_mask)) <= {1, 2, 18}
