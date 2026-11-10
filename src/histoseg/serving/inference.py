"""Load the ONNX model (after checking its checksum) and predict masks for RGB images."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from histoseg.preprocess import preprocess


class ChecksumError(RuntimeError):
    """The model file is not the one the configuration expects."""


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks so large files do not fill memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pad_to_multiple(rgb: np.ndarray, multiple: int) -> np.ndarray:
    """Pad bottom and right with white (empty glass) so both sides divide by `multiple`."""
    height, width = rgb.shape[:2]
    pad_h, pad_w = -height % multiple, -width % multiple
    return np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=255)


class SegmentationModel:
    """An ONNX Runtime session plus the pre- and post-processing around it."""

    def __init__(self, session: ort.InferenceSession, threshold: float, pad_multiple: int) -> None:
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.threshold = threshold
        self.pad_multiple = pad_multiple

    @classmethod
    def load(
        cls, path: Path, expected_sha256: str, threshold: float, pad_multiple: int
    ) -> SegmentationModel:
        """Verify the checksum, then create a CPU inference session."""
        if not expected_sha256:
            raise ChecksumError(
                "no expected sha256 configured; refusing to load an unverified model"
            )
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ChecksumError(
                f"{path}: sha256 {actual} does not match expected {expected_sha256}"
            )
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return cls(session, threshold, pad_multiple)

    def predict_probs(self, rgb: np.ndarray) -> np.ndarray:
        """Target probability per pixel, same height and width as the input."""
        height, width = rgb.shape[:2]
        x = preprocess(pad_to_multiple(rgb, self.pad_multiple))[None]
        probs = self.session.run(None, {self.input_name: x})[0]
        return probs[0, 0, :height, :width]

    def predict(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """Boolean target mask and the inference time in seconds."""
        start = time.perf_counter()
        mask = self.predict_probs(rgb) >= self.threshold
        return mask, time.perf_counter() - start
