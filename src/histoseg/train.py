"""Train the U-Net, track the run in MLflow, register the model and promote it if it is best.

Usage:
    python -m histoseg.train --config configs/train.yaml [--epochs N] [--no-register]

--epochs overrides max_epochs (used to time a single epoch); --no-register skips the registry.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import segmentation_models_pytorch as smp
import torch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler

from histoseg.config import data_dir, flatten, load_config, tracking_uri
from histoseg.data.dataset import TileDataset, load_tile, read_tiles
from histoseg.device import select_device, set_seed
from histoseg.evaluate import dice_iou, evaluate_patients, overlay_panel, predict
from histoseg.model import build_model, masked_dice_bce_loss
from histoseg.preprocess import preprocess


def target_sampler(
    rows: list[dict], target: str, fraction: float, generator: torch.Generator
) -> WeightedRandomSampler:
    """Oversample tiles containing the target so they make up about `fraction` of an epoch."""
    has_target = np.array([int(r[f"n_{target}"]) > 0 for r in rows])
    weights = np.where(
        has_target, fraction / has_target.sum(), (1 - fraction) / (~has_target).sum()
    )
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(rows),
        replacement=True,
        generator=generator,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One pass over the sampled training tiles; returns the mean loss."""
    model.train()
    total, batches = 0.0, 0
    for x, target, valid in loader:
        x, target, valid = x.to(device), target.to(device), valid.to(device)
        optimizer.zero_grad()
        loss = masked_dice_bce_loss(model(x), target, valid)
        loss.backward()
        optimizer.step()
        total += loss.item()
        batches += 1
    return total / batches


def validate(model, rows, dataset, device, threshold, batch_size) -> tuple[float, float]:
    """Pooled Dice and IoU over all validation pixels."""
    counts = evaluate_patients(model, rows, dataset, device, threshold, batch_size)
    tp, fp, fn = (sum(c[k] for c in counts.values()) for k in range(3))
    return dice_iou(tp, fp, fn)


def setup_mlflow(experiment: str) -> None:
    """Use the SQLite tracking store and keep artifacts in ./mlartifacts."""
    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=Path("mlartifacts").resolve().as_uri()
        )
    mlflow.set_experiment(experiment)


def log_overlays(model, rows, tiles_dir, target, device, threshold, n) -> None:
    """Save prediction pictures for the validation tiles with the most target pixels."""
    chosen = sorted(rows, key=lambda r: (-int(r[f"n_{target['name']}"]), r["tile_id"]))[:n]
    panels = []
    for row in chosen:
        image, mask = load_tile(tiles_dir, row["tile_id"])
        x = torch.from_numpy(preprocess(image))[None]
        probs = predict(model, x, device)[0, 0].numpy()
        panels.append(np.asarray(overlay_panel(image, mask == target["code"], probs, threshold)))
    mlflow.log_image(Image.fromarray(np.concatenate(panels, axis=0)), "overlays/val_examples.png")


def promote_if_better(
    client: MlflowClient, name: str, alias: str, version: str, score: float
) -> bool:
    """Point the alias at the new version if it beats the current holder (or there is none)."""
    try:
        current = client.get_model_version_by_alias(name, alias)
        best = client.get_run(current.run_id).data.metrics.get("best_val_dice", -1.0)
    except MlflowException:
        current, best = None, -1.0
    if score > best:
        client.set_registered_model_alias(name, alias, version)
        return True
    print(f"Not promoted: val Dice {score:.4f} <= {alias} {best:.4f} (version {current.version})")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override max_epochs")
    parser.add_argument("--no-register", action="store_true", help="skip the model registry")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = load_config(cfg["data_config"])
    tcfg, target, codes = cfg["train"], data_cfg["target"], data_cfg["mask_codes"]
    max_epochs = args.epochs or tcfg["max_epochs"]
    set_seed(cfg["seed"])
    device = select_device()

    tiles_dir = data_dir() / data_cfg["tiling"]["subdir"]
    train_rows, val_rows = read_tiles(tiles_dir, "train"), read_tiles(tiles_dir, "val")
    train_set = TileDataset(
        train_rows, tiles_dir, target["code"], codes["outside_roi"], cfg["augment"]
    )
    val_set = TileDataset(val_rows, tiles_dir, target["code"], codes["outside_roi"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_set,
        batch_size=tcfg["batch_size"],
        sampler=target_sampler(
            train_rows, target["name"], tcfg["oversample_target_fraction"], generator
        ),
        num_workers=tcfg["num_workers"],
        persistent_workers=tcfg["num_workers"] > 0,
        generator=generator,
    )

    model = build_model(cfg["model"]["encoder"], cfg["model"]["encoder_weights"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    stain_ref = json.loads((tiles_dir / "stain_reference.json").read_text(encoding="utf-8"))

    setup_mlflow(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name="timing" if args.epochs else None) as run:
        mlflow.log_params(flatten({**cfg, "data": data_cfg, "stain_reference": stain_ref}))
        mlflow.log_params(
            {"device": device.type, "train_tiles": len(train_rows), "val_tiles": len(val_rows)}
        )
        for path in (
            args.config,
            cfg["data_config"],
            data_cfg["split"]["output"],
            "reports/data_report.md",
        ):
            mlflow.log_artifact(path, "inputs")
        mlflow.log_dict(stain_ref, "stain_reference.json")

        best_dice, best_epoch, best_state, epoch_seconds = -1.0, 0, None, 0.0
        epochs_run = 0  # counted separately: after a budget stop, `epoch` is one past the last run
        start = time.time()
        for epoch in range(1, max_epochs + 1):
            elapsed = time.time() - start
            if epoch > 1 and elapsed + epoch_seconds > tcfg["max_minutes"] * 60:
                print(f"Stopping: the next epoch would exceed the {tcfg['max_minutes']} min budget")
                break
            epoch_start = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_dice, val_iou = validate(
                model, val_rows, val_set, device, tcfg["threshold"], tcfg["batch_size"]
            )
            epoch_seconds = time.time() - epoch_start
            epochs_run = epoch
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "epoch_seconds": epoch_seconds,
                },
                step=epoch,
            )
            print(
                f"epoch {epoch:2d}: loss {train_loss:.4f}  val Dice {val_dice:.4f}  "
                f"val IoU {val_iou:.4f}  ({epoch_seconds:.0f} s)"
            )
            if val_dice > best_dice:
                best_dice, best_epoch = val_dice, epoch
                best_state = copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= tcfg["patience"]:
                print(f"Early stopping: no improvement for {tcfg['patience']} epochs")
                break

        model.load_state_dict(best_state)
        mlflow.log_metrics(
            {
                "best_val_dice": best_dice,
                "best_epoch": best_epoch,
                "epochs_run": epochs_run,
                "train_minutes": (time.time() - start) / 60,
            }
        )
        log_overlays(
            model,
            val_rows,
            tiles_dir,
            target,
            device,
            tcfg["threshold"],
            cfg["mlflow"]["num_overlays"],
        )

        # Saved from the CPU so the model loads on any machine.
        model_info = mlflow.pytorch.log_model(
            model.cpu(),
            name="model",
            # Classic torch pickle: loads as a normal module with any input size (needed for ONNX).
            serialization_format="pickle",
            pip_requirements=[
                f"torch=={torch.__version__}",
                f"segmentation-models-pytorch=={smp.__version__}",
            ],
        )
        print(f"Run {run.info.run_id}: best val Dice {best_dice:.4f} at epoch {best_epoch}")

    if args.no_register:
        return
    name, alias = cfg["mlflow"]["registered_model"], cfg["mlflow"]["alias"]
    version = mlflow.register_model(model_info.model_uri, name)
    promoted = promote_if_better(MlflowClient(), name, alias, version.version, best_dice)
    print(f"Registered {name} version {version.version}" + (f", now @{alias}" if promoted else ""))


if __name__ == "__main__":
    main()
