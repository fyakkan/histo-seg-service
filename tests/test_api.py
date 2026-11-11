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


def gauge(client, name: str) -> float:
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[1])
    raise AssertionError(f"{name} metric missing")


def drift_score(client) -> float:
    return gauge(client, "histoseg_stain_drift_score")


def drift_alert_condition(client) -> bool:
    """The StainDriftHigh expression (monitoring/alert_rules.yml) on the current metrics.

    The rule itself is tested with promtool (monitoring/alert_rules_test.yml); this checks
    that the service exposes values that make it behave as intended.
    """
    return drift_score(client) > gauge(client, "histoseg_stain_drift_threshold") and gauge(
        client, "histoseg_stain_drift_window_size"
    ) >= gauge(client, "histoseg_stain_drift_min_window_size")


def test_drift_metrics_are_exposed(client, tile):
    post(client, encode(tile, "PNG"), "image/png")
    text = client.get("/metrics").text
    assert "histoseg_stain_drift_score " in text
    assert "histoseg_stain_drift_threshold 0.8" in text
    assert "histoseg_stain_drift_window_size 1.0" in text
    assert "histoseg_stain_drift_min_window_size 20.0" in text
    assert 'histoseg_stain_od_count{stain="h"} 1.0' in text
    assert 'histoseg_stain_od_count{stain="e"} 1.0' in text


def test_stain_shifted_uploads_raise_drift_above_threshold(client, fixture_rows):
    from conftest import FIXTURES
    from histoseg.stain import shift_stain

    tiles = [
        np.asarray(Image.open(FIXTURES / "images" / f"{r['tile_id']}.png").convert("RGB"))
        for r in fixture_rows
    ]
    for rgb in tiles:
        post(client, encode(rgb, "PNG"), "image/png")
    assert drift_score(client) < 0.8
    for _ in range(7):  # 56 shifted requests: more than the 50-request window
        for rgb in tiles:
            post(client, encode(shift_stain(rgb, (1.0, 1.0), (0.03, -0.015)), "PNG"), "image/png")
    assert drift_score(client) > 0.8
    assert drift_alert_condition(client)


def test_one_request_after_restart_does_not_meet_the_drift_alert_condition(client, fixture_rows):
    # The false alarm: right after a restart, one shifted image gave a score above the
    # threshold and StainDriftHigh fired. A window of one image must not count.
    from conftest import FIXTURES
    from histoseg.stain import shift_stain

    rgb = np.asarray(
        Image.open(FIXTURES / "images" / f"{fixture_rows[0]['tile_id']}.png").convert("RGB")
    )
    post(client, encode(shift_stain(rgb, (1.0, 1.0), (0.03, -0.015)), "PNG"), "image/png")
    assert drift_score(client) > 0.8  # the score alone still looks like drift ...
    assert gauge(client, "histoseg_stain_drift_window_size") == 1.0
    assert not drift_alert_condition(client)  # ... but the window is too small to alert
