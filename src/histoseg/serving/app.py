"""HTTP API: /health, /predict and /metrics.

Run locally with:
    uvicorn histoseg.serving.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from histoseg.config import load_config
from histoseg.serving.drift import StainDriftMonitor
from histoseg.serving.inference import SegmentationModel
from histoseg.serving.metrics import ServiceMetrics
from histoseg.stain import stain_features

ALLOWED_TYPES = {"image/png": "PNG", "image/jpeg": "JPEG"}
ENDPOINTS = {"/health", "/predict", "/metrics"}
# Room for the multipart headers around the file itself.
MULTIPART_OVERHEAD = 64 * 1024


@dataclass(frozen=True)
class ServeSettings:
    model_path: Path
    card_path: Path
    expected_sha256: str
    pad_multiple: int
    max_upload_bytes: int
    max_side: int
    drift_window: int
    drift_min_window: int
    drift_threshold: float

    @classmethod
    def from_config(cls, path: str | Path) -> ServeSettings:
        cfg = load_config(path)
        return cls(
            model_path=Path(cfg["model"]["path"]),
            card_path=Path(cfg["model"]["card"]),
            expected_sha256=cfg["model"]["sha256"],
            pad_multiple=cfg["inference"]["pad_multiple"],
            max_upload_bytes=cfg["limits"]["max_upload_bytes"],
            max_side=cfg["limits"]["max_side"],
            drift_window=cfg["drift"]["window"],
            drift_min_window=cfg["drift"]["min_window"],
            drift_threshold=cfg["drift"]["threshold"],
        )


def decode_image(data: bytes, max_side: int) -> np.ndarray:
    """Bytes -> uint8 RGB array, checking the real format and size before decoding pixels."""
    try:
        image = Image.open(io.BytesIO(data))  # reads only the header
    except UnidentifiedImageError as exc:
        raise HTTPException(415, "file is not a PNG or JPEG image") from exc
    if image.format not in ALLOWED_TYPES.values():
        raise HTTPException(415, "file is not a PNG or JPEG image")
    if image.width > max_side or image.height > max_side:
        raise HTTPException(413, f"image larger than {max_side}x{max_side} pixels")
    try:
        return np.asarray(image.convert("RGB"))
    except (OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, "image could not be decoded") from exc


def encode_mask(mask: np.ndarray) -> str:
    """Boolean mask -> base64 PNG (white = target)."""
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_app(settings: ServeSettings | None = None) -> FastAPI:
    settings = settings or ServeSettings.from_config(
        os.environ.get("HISTOSEG_SERVE_CONFIG", "configs/serve.yaml")
    )
    # PIL refuses images far above this many pixels (decompression-bomb protection);
    # nothing larger than max_side x max_side is ever accepted anyway.
    Image.MAX_IMAGE_PIXELS = settings.max_side * settings.max_side
    metrics = ServiceMetrics()
    too_large = f"upload larger than {settings.max_upload_bytes / 1e6:.1f} MB"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Loaded once at startup; a checksum mismatch stops the service from starting.
        card = json.loads(settings.card_path.read_text(encoding="utf-8"))
        app.state.model = SegmentationModel.load(
            settings.model_path, settings.expected_sha256, card["threshold"], settings.pad_multiple
        )
        app.state.card = card
        app.state.drift = StainDriftMonitor(card["reference_stain"], settings.drift_window)
        metrics.model_info.labels(str(card["model_version"]), settings.expected_sha256).set(1)
        metrics.drift_threshold.set(settings.drift_threshold)
        metrics.drift_min_window_size.set(settings.drift_min_window)
        yield

    app = FastAPI(title="histo-seg-service", lifespan=lifespan)
    app.state.model = None
    app.state.card = None

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        endpoint = request.url.path if request.url.path in ENDPOINTS else "other"
        start = time.perf_counter()
        length = request.headers.get("content-length")
        # Reject oversized uploads before the body is read and parsed.
        if (
            endpoint == "/predict"
            and length is not None
            and length.isdigit()
            and int(length) > settings.max_upload_bytes + MULTIPART_OVERHEAD
        ):
            response = JSONResponse({"detail": too_large}, status_code=413)
        else:
            response = await call_next(request)
        metrics.requests.labels(endpoint, str(response.status_code)).inc()
        metrics.request_latency.labels(endpoint).observe(time.perf_counter() - start)
        return response

    @app.get("/health")
    def health() -> JSONResponse:
        if app.state.model is None:
            return JSONResponse({"status": "loading"}, status_code=503)
        return JSONResponse({"status": "ok", "model_version": str(app.state.card["model_version"])})

    @app.post("/predict")
    def predict(file: UploadFile) -> dict:
        if app.state.model is None:
            raise HTTPException(503, "model not loaded")
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, "only image/png and image/jpeg are accepted")
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, too_large)

        rgb = decode_image(data, settings.max_side)
        mask, seconds = app.state.model.predict(rgb)
        fraction = float(mask.mean())
        metrics.inference_latency.observe(seconds)
        metrics.predicted_fraction.observe(fraction)

        h_od, e_od = stain_features(rgb)
        if not np.isnan(h_od):  # NaN means no tissue in the image: nothing to measure
            metrics.stain_od.labels("h").observe(h_od)
            metrics.stain_od.labels("e").observe(e_od)
        metrics.drift_score.set(app.state.drift.update(np.array([h_od, e_od])))
        metrics.drift_window_size.set(app.state.drift.size())
        return {
            "target": app.state.card["target"],
            "target_fraction": fraction,
            "width": rgb.shape[1],
            "height": rgb.shape[0],
            "mask_png_base64": encode_mask(mask),
            "model_version": str(app.state.card["model_version"]),
        }

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
