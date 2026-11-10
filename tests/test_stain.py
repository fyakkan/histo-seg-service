import numpy as np

from histoseg.stain import random_stain_augment, shift_stain, stain_features, tissue_mask

WHITE = 255


def test_augmentation_keeps_shape_and_dtype_and_changes_colour(tile):
    out = random_stain_augment(tile, np.random.default_rng(0), max_scale=0.05, max_shift=0.01)
    assert out.shape == tile.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - tile).mean() > 1.0


def test_augmentation_is_reproducible_with_the_same_seed(tile):
    a = random_stain_augment(tile, np.random.default_rng(7), 0.05, 0.01)
    b = random_stain_augment(tile, np.random.default_rng(7), 0.05, 0.01)
    assert np.array_equal(a, b)


def test_neutral_shift_returns_the_original_image(fixture_rows):
    # Guards against clipping in the colour conversion, which would distort every augmented tile.
    from PIL import Image

    from conftest import FIXTURES

    for row in fixture_rows:
        rgb = np.asarray(Image.open(FIXTURES / "images" / f"{row['tile_id']}.png").convert("RGB"))
        out = shift_stain(rgb, scale=(1.0, 1.0), shift=(0.0, 0.0))
        assert np.abs(out.astype(int) - rgb).max() <= 1


def test_more_haematoxylin_raises_the_h_feature(tile):
    before = stain_features(tile)
    after = stain_features(shift_stain(tile, scale=(1.0, 1.0), shift=(0.03, 0.0)))
    assert after[0] > before[0] + 0.02


def test_stain_features_ignore_background(tile):
    # Surround the tile with white glass: the features must not change.
    padded = np.full((512, 512, 3), WHITE, dtype=np.uint8)
    padded[128:384, 128:384] = tile
    assert not tissue_mask(padded)[:128].any()
    np.testing.assert_allclose(stain_features(padded), stain_features(tile), rtol=1e-6)


def test_blank_image_has_no_stain_features():
    blank = np.full((64, 64, 3), WHITE, dtype=np.uint8)
    assert np.isnan(stain_features(blank)).all()


def test_numpy_rgb2hed_matches_skimage(fixture_rows):
    # Serving uses the NumPy version so the image needs no scikit-image/SciPy;
    # it must give the same stain values the reference statistics were computed with.
    from PIL import Image
    from skimage.color import hed_from_rgb as sk_hed_from_rgb
    from skimage.color import rgb2hed as sk_rgb2hed

    from conftest import FIXTURES
    from histoseg.stain import hed_from_rgb, rgb2hed

    np.testing.assert_allclose(hed_from_rgb, sk_hed_from_rgb, rtol=0, atol=1e-12)
    extremes = np.array([[[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 255]]], np.uint8)
    images = [extremes] + [
        np.asarray(Image.open(FIXTURES / "images" / f"{r['tile_id']}.png").convert("RGB"))
        for r in fixture_rows
    ]
    for rgb in images:
        np.testing.assert_allclose(rgb2hed(rgb), sk_rgb2hed(rgb), rtol=0, atol=1e-6)
