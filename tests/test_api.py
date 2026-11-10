import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from histoseg.serving.app import create_app
from histoseg.serving.inference import ChecksumError
from serving_helpers import make_settings

RESPONSE_KEYS = {"target", "target_fraction", "width", "height", "mask_png_base64", "model_version"}


def encode(rgb: np.ndarray, fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path):
    # The `with` block runs the app's startup, which loads the model.
    with TestClient(create_app(make_settings(tmp_path))) as c:
        yield c


def post(client, data: bytes, content_type: str):
    return client.post("/predict", files={"file": ("tile", data, content_type)})


def test_health_reports_model_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test"}


def test_health_is_503_before_the_model_is_loaded(tmp_path):
    # Without `with`, startup never runs, so no model is loaded.
    response = TestClient(create_app(make_settings(tmp_path))).get("/health")
    assert response.status_code == 503


def test_predict_png_returns_mask_and_fraction(client, tile):
    response = post(client, encode(tile, "PNG"), "image/png")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_KEYS
    mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["mask_png_base64"]))))
    assert mask.shape == tile.shape[:2]
    assert set(np.unique(mask)) <= {0, 255}
    assert body["target_fraction"] == pytest.approx((mask == 255).mean())


def test_predict_jpeg_with_size_not_divisible_by_32(client, tile):
    response = post(client, encode(tile[:100, :70], "JPEG"), "image/jpeg")
    assert response.status_code == 200
    assert (response.json()["height"], response.json()["width"]) == (100, 70)


def test_wrong_content_type_is_rejected(client):
    assert post(client, b"hello", "text/plain").status_code == 415


def test_other_format_disguised_as_png_is_rejected(client, tile):
    assert post(client, encode(tile, "GIF"), "image/png").status_code == 415


def test_garbage_bytes_are_rejected(client):
    assert post(client, b"\x89PNG not really an image", "image/png").status_code == 415


def test_upload_over_5_mb_is_rejected(client):
    assert post(client, b"0" * (5 * 1024 * 1024 + 1), "image/png").status_code == 413


def test_image_over_1024_pixels_is_rejected(client):
    wide = np.zeros((8, 1025, 3), dtype=np.uint8)
    assert post(client, encode(wide, "PNG"), "image/png").status_code == 413


def test_wrong_checksum_stops_startup(tmp_path):
    app = create_app(make_settings(tmp_path, sha256="0" * 64))
    with pytest.raises(ChecksumError), TestClient(app):
        pass


def test_missing_checksum_stops_startup(tmp_path):
    with pytest.raises(ChecksumError), TestClient(create_app(make_settings(tmp_path, sha256=""))):
        pass


def test_metrics_are_exposed(client, tile):
    post(client, encode(tile, "PNG"), "image/png")
    text = client.get("/metrics").text
    assert 'histoseg_requests_total{endpoint="/predict",status="200"} 1.0' in text
    assert "histoseg_request_latency_seconds_bucket" in text
    assert "histoseg_inference_latency_seconds_count 1.0" in text
    assert "histoseg_predicted_fraction_bucket" in text
    assert 'histoseg_model_info{sha256="' in text and 'version="test"' in text
