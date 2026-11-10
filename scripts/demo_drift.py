"""Simulate stain drift against the running stack and watch the StainDriftHigh alert fire.

Usage (after `make up`):
    python scripts/demo_drift.py

Phase 1 sends the fixture tiles unchanged; phase 2 sends the same tiles with the simulated
stain shift from configs/serve.yaml until Prometheus reports the alert as firing.
The drift is simulated: no real scanner or staining change is involved.
"""

from __future__ import annotations

import argparse
import io
import itertools
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from histoseg.config import load_config
from histoseg.stain import shift_stain

FIXTURES = Path("tests/fixtures/images")
STATUS_EVERY = 5.0  # seconds between status lines


def png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def drift_score(client: httpx.Client, api: str) -> float:
    for line in client.get(f"{api}/metrics").text.splitlines():
        if line.startswith("histoseg_stain_drift_score "):
            return float(line.split()[1])
    return float("nan")


def alert_state(client: httpx.Client, prometheus: str) -> str:
    """'inactive', 'pending' or 'firing' for StainDriftHigh."""
    alerts = client.get(f"{prometheus}/api/v1/alerts").json()["data"]["alerts"]
    states = [a["state"] for a in alerts if a["labels"]["alertname"] == "StainDriftHigh"]
    return states[0] if states else "inactive"


def run_phase(client, args, name, tiles, seconds, stop_when_firing=False) -> str:
    """Send tiles in a loop for `seconds`; print the score and alert state now and then."""
    start = last_status = time.time()
    state = "inactive"
    for rgb in itertools.cycle(tiles):
        files = {"file": ("tile.png", png_bytes(rgb), "image/png")}
        client.post(f"{args.api}/predict", files=files).raise_for_status()
        now = time.time()
        if now - last_status >= STATUS_EVERY:
            state = alert_state(client, args.prometheus)
            print(
                f"[{name:7s} {now - start:5.0f} s] drift score {drift_score(client, args.api):.2f}  alert: {state}"
            )
            last_status = now
            if stop_when_firing and state == "firing":
                break
        if now - start >= seconds:
            break
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--prometheus", default="http://127.0.0.1:9090")
    parser.add_argument("--normal-seconds", type=float, default=60)
    parser.add_argument("--max-shifted-seconds", type=float, default=240)
    args = parser.parse_args()

    drift = load_config("configs/serve.yaml")["drift"]
    tiles = [np.asarray(Image.open(p).convert("RGB")) for p in sorted(FIXTURES.glob("*.png"))]
    shifted = [shift_stain(t, drift["demo_scale"], drift["demo_shift"]) for t in tiles]
    print(
        f"Threshold {drift['threshold']}, window {drift['window']} requests, {len(tiles)} fixture tiles"
    )

    with httpx.Client(timeout=30) as client:
        run_phase(client, args, "normal", tiles, args.normal_seconds)
        print(
            f"Applying simulated stain shift: scale {drift['demo_scale']}, shift {drift['demo_shift']}"
        )
        state = run_phase(
            client, args, "shifted", shifted, args.max_shifted_seconds, stop_when_firing=True
        )

    print(
        "StainDriftHigh is firing."
        if state == "firing"
        else f"Alert did not fire (state: {state})."
    )
    sys.exit(0 if state == "firing" else 1)


if __name__ == "__main__":
    main()
