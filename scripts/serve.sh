#!/usr/bin/env bash
# Serve the jev System One API. Configuration via environment:
#   JEV_BACKEND (mock|vlm), JEV_HOST, JEV_PORT, JEV_MODEL, JEV_DEVICE
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m jev serve \
  --backend "${JEV_BACKEND:-mock}" \
  --host "${JEV_HOST:-127.0.0.1}" \
  --port "${JEV_PORT:-8080}"
