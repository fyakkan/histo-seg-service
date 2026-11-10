"""Export the champion model to ONNX, check it against PyTorch, and write the model card.

Usage:
    python -m histoseg.export_onnx --config configs/train.yaml

Writes artifacts/model.onnx, artifacts/model.onnx.sha256 and artifacts/model_card.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import onnx
import torch
from mlflow import MlflowClient
from PIL import Image

from histoseg.config import load_config, tracking_uri
from histoseg.preprocess import preprocess
from histoseg.quality import FIXTURES, fixture_dice
from histoseg.serving.inference import SegmentationModel, sha256_file

ARTIFACTS = Path("artifacts")
OPSET = 17
MAX_ABS_DIFF = 1e-4


class WithSigmoid(torch.nn.Module):
    """Wraps the model so the exported graph returns probabilities, not logits."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def fixture_inputs() -> list[np.ndarray]:
    """Preprocessed fixture tiles, each with a batch dimension."""
    paths = sorted((FIXTURES / "images").glob("*.png"))
    return [preprocess(np.asarray(Image.open(p).convert("RGB")))[None] for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = load_config(cfg["data_config"])
    name, alias = cfg["mlflow"]["registered_model"], cfg["mlflow"]["alias"]
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    version = client.get_model_version_by_alias(name, alias)
    run = client.get_run(version.run_id)

    model = mlflow.pytorch.load_model(f"models:/{name}@{alias}", map_location="cpu")
    wrapped = WithSigmoid(model).eval()
    inputs = fixture_inputs()

    ARTIFACTS.mkdir(exist_ok=True)
    onnx_path = ARTIFACTS / "model.onnx"
    dynamic = {0: "batch", 2: "height", 3: "width"}
    torch.onnx.export(
        wrapped,
        (torch.from_numpy(inputs[0]),),
        str(onnx_path),
        input_names=["image"],
        output_names=["probability"],
        dynamic_axes={"image": dynamic, "probability": dynamic},
        opset_version=OPSET,
        # The classic exporter writes one self-contained file with the weights inside.
        dynamo=False,
    )
    onnx.checker.check_model(str(onnx_path))

    # Parity: ONNX Runtime must give the same probabilities as PyTorch on every fixture.
    checksum = sha256_file(onnx_path)
    threshold = cfg["train"]["threshold"]
    onnx_model = SegmentationModel.load(onnx_path, checksum, threshold, pad_multiple=32)
    max_diff = 0.0
    with torch.no_grad():
        for x in inputs:
            expected = wrapped(torch.from_numpy(x)).numpy()
            actual = onnx_model.session.run(None, {onnx_model.input_name: x})[0]
            max_diff = max(max_diff, float(np.abs(expected - actual).max()))
    print(f"Parity check: max abs difference {max_diff:.2e} (limit {MAX_ABS_DIFF:.0e})")
    if max_diff > MAX_ABS_DIFF:
        raise SystemExit("ONNX output differs from PyTorch; not writing the model card")

    target, codes = data_cfg["target"], data_cfg["mask_codes"]
    dice = fixture_dice(onnx_model, target["code"], codes["outside_roi"])
    stain_ref = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/stain_reference.json")
    test_metrics = {k: v for k, v in run.data.metrics.items() if k.startswith("test_")}
    card = {
        "model_name": name,
        "model_version": int(version.version),
        "mlflow_run_id": run.info.run_id,
        "target": target["name"],
        "threshold": threshold,
        "tile_size": data_cfg["tiling"]["tile_size"],
        "mpp": data_cfg["tiling"]["target_mpp"],
        "input": "uint8 RGB, preprocessed by histoseg.preprocess.preprocess (ImageNet normalisation)",
        "onnx_opset": OPSET,
        "sha256": checksum,
        "parity_max_abs_diff": max_diff,
        "fixture_dice": dice,
        "val_dice": run.data.metrics["best_val_dice"],
        "test_metrics": test_metrics,
        "reference_stain": stain_ref,
        "data": {
            "source": "BCSS (Amgad et al., Bioinformatics 2019)",
            "repo_id": data_cfg["dataset"]["repo_id"],
            "revision": data_cfg["dataset"]["revision"],
            "license": data_cfg["dataset"]["license"],
        },
        "intended_use": "Portfolio/demo only. Not a medical device; not for clinical use.",
    }
    (ARTIFACTS / "model_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    (ARTIFACTS / "model.onnx.sha256").write_text(f"{checksum}  model.onnx\n", encoding="utf-8")
    print(f"Wrote {onnx_path} (sha256 {checksum}), fixture Dice {dice:.4f}")


if __name__ == "__main__":
    main()
