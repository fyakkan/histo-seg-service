"""Download BCSS from a pinned Hugging Face revision and unpack it into one PNG per ROI.

Usage:
    python -m histoseg.data.download --config configs/data.yaml [--limit N]

The run is resumable: parquet shards that are already complete are not downloaded
again, and ROIs whose image and mask PNGs already exist are skipped.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from histoseg.config import data_dir, load_config

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
METADATA_FIELDS = ["image_id", "patient_id", "xmin", "ymin"]


def list_shards(repo_id: str, revision: str) -> list[str]:
    """Parquet shard paths in the pinned revision, sorted so the order never changes."""
    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    return sorted(f for f in files if f.endswith(".parquet"))


def write_png(path: Path, data: bytes) -> None:
    """Write PNG bytes via a temporary file.

    Renaming at the end means an interrupted run never leaves a half-written PNG
    that the next run would wrongly treat as done.
    """
    # We store the original bytes unchanged, so they must already be PNG.
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path.name}: expected PNG bytes from the dataset")
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def download(config_path: str, limit: int | None = None) -> Path:
    """Download shards and unpack ROIs; return the output folder."""
    cfg = load_config(config_path)
    ds = cfg["dataset"]
    out_dir = data_dir() / cfg["download"]["subdir"]
    images_dir, masks_dir = out_dir / "images", out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    total = min(limit, ds["num_rois"]) if limit else ds["num_rois"]
    done = 0
    metadata: list[dict] = []
    print(f"Dataset {ds['repo_id']} @ {ds['revision'][:7]} -> {out_dir} ({total} ROIs)")

    for shard in list_shards(ds["repo_id"], ds["revision"]):
        if done >= total:
            break
        # hf_hub_download skips complete files and resumes partial ones; shows a byte progress bar.
        print(f"Shard {shard}")
        local_shard = hf_hub_download(
            ds["repo_id"],
            shard,
            repo_type="dataset",
            revision=ds["revision"],
            local_dir=out_dir / "parquet",
        )

        # One row at a time keeps memory use low (each row holds a multi-megabyte image).
        for batch in pq.ParquetFile(local_shard).iter_batches(batch_size=1):
            if done >= total:
                break
            row = batch.to_pylist()[0]
            image_id = row["image_id"]
            image_path = images_dir / f"{image_id}.png"
            mask_path = masks_dir / f"{image_id}.png"

            if image_path.exists() and mask_path.exists():
                status = "skip (already there)"
            else:
                write_png(image_path, row["image"]["bytes"])
                write_png(mask_path, row["mask"]["bytes"])
                status = "saved"

            metadata.append({field: row[field] for field in METADATA_FIELDS})
            done += 1
            print(f"  [{done:3d}/{total}] {status}: {image_id}")

    metadata_path = out_dir / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(metadata)
    print(f"Done: {done} ROIs, metadata in {metadata_path}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--limit", type=int, default=None, help="only the first N ROIs (testing)")
    args = parser.parse_args()
    download(args.config, args.limit)


if __name__ == "__main__":
    main()
