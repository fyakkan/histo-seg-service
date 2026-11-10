"""Split patients into train/val/test so that no patient appears in two splits.

Usage:
    python -m histoseg.data.split --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from histoseg.config import data_dir, load_config

SPLITS = ("train", "val", "test")


def split_patients(
    patients: list[str], fractions: dict[str, float], seed: int
) -> dict[str, list[str]]:
    """Shuffle unique patient IDs with a fixed seed and cut them into three groups."""
    # Sorting first makes the result independent of the order the IDs arrive in.
    unique = sorted(set(patients))
    order = np.random.default_rng(seed).permutation(len(unique))
    shuffled = [unique[i] for i in order]

    n_train = round(fractions["train"] * len(unique))
    n_val = round(fractions["val"] * len(unique))
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def patient_to_split(splits: dict[str, list[str]]) -> dict[str, str]:
    """Invert {split: [patients]} into {patient: split}."""
    return {p: name for name in SPLITS for p in splits[name]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metadata = data_dir() / cfg["download"]["subdir"] / "metadata.csv"
    with open(metadata, encoding="utf-8") as f:
        patients = [row["patient_id"] for row in csv.DictReader(f)]

    split_cfg = cfg["split"]
    splits = split_patients(patients, split_cfg["fractions"], split_cfg["seed"])
    result = {
        "dataset": cfg["dataset"]["repo_id"],
        "revision": cfg["dataset"]["revision"],
        "seed": split_cfg["seed"],
        "counts": {name: len(splits[name]) for name in SPLITS},
        **splits,
    }
    Path(split_cfg["output"]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {split_cfg['output']}: {result['counts']}")


if __name__ == "__main__":
    main()
