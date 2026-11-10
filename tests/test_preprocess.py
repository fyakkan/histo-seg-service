import numpy as np
import pytest

from histoseg.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess

LOW = float(((0 - IMAGENET_MEAN) / IMAGENET_STD).min())
HIGH = float(((1 - IMAGENET_MEAN) / IMAGENET_STD).max())


def test_output_shape_dtype_and_range(tile):
    x = preprocess(tile)
    assert x.shape == (3, 256, 256)
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]
    assert LOW - 1e-6 <= x.min() and x.max() <= HIGH + 1e-6


def test_works_for_any_image_size():
    rgb = np.full((100, 60, 3), 255, dtype=np.uint8)
    assert preprocess(rgb).shape == (3, 100, 60)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((8, 8, 3), dtype=np.float32),
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
    ],
)
def test_rejects_wrong_input(bad):
    with pytest.raises(ValueError):
        preprocess(bad)


@pytest.mark.train
def test_training_dataset_uses_the_shared_preprocess(fixture_rows):
    pytest.importorskip("torch")
    from conftest import FIXTURES
    from histoseg.data.dataset import TileDataset, load_tile

    dataset = TileDataset(fixture_rows, FIXTURES, target_code=18, outside_code=0)
    for i, row in enumerate(fixture_rows):
        x, target, valid = dataset[i]
        image, mask = load_tile(FIXTURES, row["tile_id"])
        assert np.array_equal(x.numpy(), preprocess(image))
        assert target.sum().item() == int(row["n_blood_vessel"])
        assert valid.sum().item() == mask.size - int(row["n_outside_roi"])
