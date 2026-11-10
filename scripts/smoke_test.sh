#!/usr/bin/env bash
# Start the container, wait for /health, send one fixture tile to /predict, check the answer, stop.
# Usage: scripts/smoke_test.sh [image]   (default: histo-seg-service:local)
set -euo pipefail

IMAGE="${1:-histo-seg-service:local}"
PORT="${PORT:-8000}"
NAME="histoseg-smoke-$$"
URL="http://127.0.0.1:${PORT}"
TILE="$(ls tests/fixtures/images/*.png | head -n 1)"

# Always remove the container, also when a check fails.
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:8000" "$IMAGE" >/dev/null

echo "Waiting for ${URL}/health ..."
for attempt in $(seq 1 30); do
    if curl -fsS "${URL}/health" >/dev/null 2>&1; then
        break
    fi
    if [ "$attempt" -eq 30 ]; then
        echo "FAIL: service did not become healthy" >&2
        docker logs "$NAME" >&2
        exit 1
    fi
    sleep 1
done
echo "health: $(curl -fsS "${URL}/health")"

response="$(curl -fsS -F "file=@${TILE};type=image/png" "${URL}/predict")"
echo "$response" | python3 -c '
import json, sys
body = json.load(sys.stdin)
expected = {"target", "target_fraction", "width", "height", "mask_png_base64", "model_version"}
missing = expected - body.keys()
if missing:
    sys.exit(f"FAIL: response is missing {sorted(missing)}")
print("predict:", body["target"], "fraction", round(body["target_fraction"], 4),
      "size", body["width"], "x", body["height"], "model", body["model_version"])
'

echo "PASS: smoke test"
