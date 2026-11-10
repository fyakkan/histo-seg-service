"""Training and serving must prepare inputs with the very same function (no skew)."""

import numpy as np

import histoseg.preprocess
from histoseg.serving import inference
from histoseg.serving.inference import SegmentationModel, pad_to_multiple
from serving_helpers import make_settings


def test_serving_imports_the_shared_function():
    assert inference.preprocess is histoseg.preprocess.preprocess


def test_serving_sends_exactly_the_preprocessed_image_to_the_model(tmp_path, tile):
    settings = make_settings(tmp_path)
    model = SegmentationModel.load(settings.model_path, settings.expected_sha256, 0.5, 32)
    seen = {}
    real_run = model.session.run

    def recording_run(outputs, feeds):
        seen.update(feeds)
        return real_run(outputs, feeds)

    model.session.run = recording_run
    rgb = tile[:100, :70]
    model.predict(rgb)
    expected = histoseg.preprocess.preprocess(pad_to_multiple(rgb, 32))[None]
    assert np.array_equal(seen[model.input_name], expected)


def test_padding_only_adds_white_bottom_right():
    rgb = np.zeros((33, 64, 3), dtype=np.uint8)
    padded = pad_to_multiple(rgb, 32)
    assert padded.shape == (64, 64, 3)
    assert (padded[:33] == 0).all() and (padded[33:] == 255).all()
