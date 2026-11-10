"""Builds a tiny random-weight ONNX model so API tests run without the real (released) model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from histoseg.serving.app import ServeSettings
from histoseg.serving.inference import sha256_file


def make_random_onnx(path: Path, seed: int = 0) -> None:
    """1x1 convolution + sigmoid with the same input/output contract as the real model."""
    rng = np.random.default_rng(seed)
    weight = numpy_helper.from_array(rng.normal(0, 0.5, (1, 3, 1, 1)).astype(np.float32), "w")
    bias = numpy_helper.from_array(np.zeros(1, dtype=np.float32), "b")
    nodes = [
        helper.make_node("Conv", ["image", "w", "b"], ["logit"]),
        helper.make_node("Sigmoid", ["logit"], ["probability"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny-random",
        [
            helper.make_tensor_value_info(
                "image", TensorProto.FLOAT, ["batch", 3, "height", "width"]
            )
        ],
        [
            helper.make_tensor_value_info(
                "probability", TensorProto.FLOAT, ["batch", 1, "height", "width"]
            )
        ],
        initializer=[weight, bias],
    )
    # IR version 8 is the format that goes with opset 17; newer onnx releases default to an IR
    # version that ONNX Runtime may not read yet.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_settings(tmp_path: Path, sha256: str | None = None) -> ServeSettings:
    """Random model + minimal model card in tmp_path; sha256 defaults to the real checksum."""
    model_path, card_path = tmp_path / "model.onnx", tmp_path / "model_card.json"
    make_random_onnx(model_path)
    card = {"model_version": "test", "threshold": 0.5, "target": "blood_vessel"}
    card_path.write_text(json.dumps(card), encoding="utf-8")
    return ServeSettings(
        model_path=model_path,
        card_path=card_path,
        expected_sha256=sha256 if sha256 is not None else sha256_file(model_path),
        pad_multiple=32,
        max_upload_bytes=5 * 1024 * 1024,
        max_side=1024,
    )
