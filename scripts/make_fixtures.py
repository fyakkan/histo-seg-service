"""Copy a few test-split tiles and masks into tests/fixtures/ for fast tests.

Usage:
    python scripts/make_fixtures.py --config configs/data.yaml

Picks the 5 tiles with the most target pixels and 3 tiles without target, each from a
different test patient, so the fixtures cover both cases. Test-split tiles are used so the
fixtures never overlap with what the model was trained on.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from histoseg.config import data_dir, load_config
from histoseg.data.dataset import read_tiles

FIXTURES = Path("tests/fixtures")
MAX_BYTES = 2_000_000


def pick(rows: list[dict], key, n: int, used: set[str]) -> list[dict]:
    """First n rows by `key`, at most one per patient."""
    chosen = []
    for row in sorted(rows, key=key):
        if row["patient_id"] not in used and len(chosen) < n:
            chosen.append(row)
            used.add(row["patient_id"])
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tiles_dir = data_dir() / cfg["tiling"]["subdir"]
    column = f"n_{cfg['target']['name']}"
    rows = read_tiles(tiles_dir, "test")

    used: set[str] = set()
    with_target = pick(
        [r for r in rows if int(r[column]) > 0], lambda r: (-int(r[column]), r["tile_id"]), 5, used
    )
    # Without target: prefer fully annotated tiles (fewest outside_roi pixels).
    without = pick(
        [r for r in rows if int(r[column]) == 0],
        lambda r: (int(r["n_outside_roi"]), r["tile_id"]),
        3,
        used,
    )

    for sub in ("images", "masks"):
        (FIXTURES / sub).mkdir(parents=True, exist_ok=True)
    chosen = with_target + without
    for row in chosen:
        for sub in ("images", "masks"):
            shutil.copy(tiles_dir / sub / f"{row['tile_id']}.png", FIXTURES / sub)
    with open(FIXTURES / "tiles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(chosen[0]))
        writer.writeheader()
        writer.writerows(chosen)

    total = sum(p.stat().st_size for p in FIXTURES.rglob("*") if p.is_file())
    print(f"{len(chosen)} fixture tiles, {total / 1e6:.2f} MB in {FIXTURES}")
    if total > MAX_BYTES:
        raise SystemExit(f"fixtures are larger than {MAX_BYTES / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
