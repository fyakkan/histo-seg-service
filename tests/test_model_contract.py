"""Checks on the real released model (artifacts/model.onnx).

Skipped when the model is not present (download it from the model-v1 release or run make export).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from histoseg.config import load_config
from histoseg.quality import fixture_dice
from histoseg.serving.inference import SegmentationModel

SERVE = load_config("configs/serve.yaml")
MODEL_PATH = Path(SERVE["model"]["path"])
CARD_PATH = Path(SERVE["model"]["card"])

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and CARD_PATH.exists()), reason="released model not downloaded"
)


@pytest.fixture(scope="module")
def card() -> dict:
    return json.loads(CARD_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model(card) -> SegmentationModel:
    # Also checks the file against the sha256 recorded in configs/serve.yaml.
    return SegmentationModel.load(MODEL_PATH, SERVE["model"]["sha256"], card["threshold"], 32)


@pytest.mark.parametrize("shape", [(1, 3, 256, 256), (2, 3, 320, 288)])
def test_output_shape_dtype_and_range(model, shape):
    x = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    y = model.session.run(None, {model.input_name: x})[0]
    assert y.shape == (shape[0], 1, shape[2], shape[3])
    assert y.dtype == np.float32
    assert 0.0 <= y.min() and y.max() <= 1.0


def test_quality_gate_on_fixtures(model, card):
    # Regression guard, not an evaluation: the score must not drop below the export-time value.
    data = load_config("configs/data.yaml")
    dice = fixture_dice(model, data["target"]["code"], data["mask_codes"]["outside_roi"])
    assert dice >= card["fixture_dice"] - 0.02


@pytest.mark.train
def test_onnx_matches_pytorch_champion(model):
    torch = pytest.importorskip("torch")
    import mlflow

    from histoseg.config import tracking_uri
    from histoseg.export_onnx import WithSigmoid, fixture_inputs

    mlflow.set_tracking_uri(tracking_uri())
    torch_model = WithSigmoid(
        mlflow.pytorch.load_model("models:/histo-seg@champion", map_location="cpu")
    ).eval()
    with torch.no_grad():
        for x in fixture_inputs():
            expected = torch_model(torch.from_numpy(x)).numpy()
            actual = model.session.run(None, {model.input_name: x})[0]
            assert np.abs(expected - actual).max() <= 1e-4
