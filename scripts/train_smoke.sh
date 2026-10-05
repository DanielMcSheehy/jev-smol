#!/usr/bin/env bash
# End-to-end smoke train: synth data (if missing) -> wait for train deps ->
# 3-step LoRA SFT on data/synth with configs/train_smoke.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

# 1. Synthetic data (idempotent)
if [ ! -f data/synth/train.jsonl ] || [ ! -f data/synth/val.jsonl ]; then
  echo "[smoke] generating synthetic data..."
  "$PY" scripts/make_synth_data.py
fi

# 2. Wait for the (possibly still installing) training stack
echo "[smoke] waiting for torch/transformers/trl..."
ready=0
for _ in $(seq 1 40); do
  if "$PY" -c "import trl, transformers, torch" 2>/dev/null; then ready=1; break; fi
  sleep 30
done
if [ "$ready" -ne 1 ]; then
  echo "[smoke] ERROR: training deps never became importable" >&2
  exit 1
fi

# 3. Train
exec "$PY" -m jev.train.sft --config configs/train_smoke.yaml "$@"
