# GPU training handoff — jev

You are picking up the `jev` project on a CUDA machine to run real training.
This file is self-contained: read it, then read `README.md` and
`docs/research/smolvlm2-finetuning.md` for depth.

## What this project is

`jev` is a small embedded text+image **decision model**: it implements the
Jev/CLEF "System One" API (`POST /v1/systemone` — a `state` plus typed
questions `noul`/`choice`/`score`, answered with per-option probabilities)
on top of **SmolVLM2-500M-Video-Instruct**. Inference scores options by
teacher-forcing each answer value and reading token log-probs — no free-form
generation. Everything is wired: data pipelines, LoRA SFT, eval harness, API
server, 89 passing tests.

Current state (measured 2026-10-04, text-only synthetic val, 165 questions):
base 41.2% → LoRA 2 epochs **58.2%** overall (noul 81.3%, choice 39.6%,
score 47.2%; Brier 0.457). Choice accuracy is the weak spot — improving it is
a data problem (more/better distractors, more real data), not just steps.

## Setup (Linux + CUDA)

```bash
git clone https://github.com/DanielMcSheehy/jev-smol.git jev && cd jev
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[api,data,dev]" -r requirements-train.txt
# CUDA torch wheels install automatically on Linux; verify:
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

The training script auto-selects device/dtype: **cuda → bf16**, and enables
`bf16=True` in SFTConfig only on CUDA (Apple Silicon must stay fp32).

## Data

Two sources, both rebuilt locally — nothing large is stored in git:

```bash
# 1. Synthetic (deterministic, CC0, unlimited volume; text + PIL images)
.venv/bin/python -m jev.data.synthgen.build --domains all \
    --per-domain 2000 --val-per-domain 200 --out data/synthetic
# 2. Real scraped mix (~20k; per-source fault isolation, licenses in manifest)
.venv/bin/python -m jev.data.build --config configs/data_starter.yaml --out data/processed
```

Optional combined training set (recommended for the real run):

```bash
.venv/bin/python - <<'PY'
import json, pathlib, random
rows = []
for p in ("data/synthetic/train.jsonl", "data/processed/train.jsonl"):
    if pathlib.Path(p).exists():
        rows += [l for l in pathlib.Path(p).read_text().splitlines() if l.strip()]
random.Random(0).shuffle(rows)
pathlib.Path("data/combined").mkdir(parents=True, exist_ok=True)
pathlib.Path("data/combined/train.jsonl").write_text("\n".join(rows) + "\n")
print(len(rows), "combined train examples")
PY
```

Validate any split with `.venv/bin/python -m jev.data.validate <file> --root <dir>`.

## Training

```bash
.venv/bin/python -m jev.train.sft --config configs/train_synth.yaml       # synthetic 3000-mix
.venv/bin/python -m jev.train.sft --config configs/train_lora.yaml        # real mix (data/processed)
# or copy a config and point train_data/val_data at data/combined/
```

- LoRA is regex-scoped to the language tower and asserted at startup
  (`lora_matches_vision` must be 0 in `<out>/train_summary.json`).
- Loss is completion-only (TRL prompt-completion); `max_length=None` so image
  tokens are never truncated; a separate `max_token_len` guard drops
  over-long examples (2600 for image rows).
- Epoch-driven configs use `max_steps: null` (the script converts to -1).
- **Cost lever**: the stock processor expands every image to 1088 tokens
  (`do_image_splitting: true`). If you flip `do_image_splitting` to false in
  the processor for BOTH training and `engine_vlm.py`, images cost ~128
  tokens (~8× faster image epochs). Keep train/inference consistent.
- Merge an adapter for serving:
  `.venv/bin/python -m jev.train.export --adapter <out>/adapter --out <out>/merged`

## Evaluate (do not skip)

```bash
.venv/bin/python -m jev.eval --data data/synthetic/val.jsonl --root data/synthetic \
    --model <out>/merged --limit 200 --out runs/evals/gpu_run.json
```

Reports accuracy/Brier/confidence per question type against gold labels. If
scoring accuracy diverges from generation accuracy, run
`scripts/diag_scoring.py` — the usual cause is train/inference prompt or
tokenization drift (see `option_token_spans` in `engine_vlm.py`; never
hand-roll the scoring format).

## Publishing results

- Base weights are already on HF (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`),
  so the **LoRA adapter alone (~35 MB) is sufficient** to reproduce the model.
- Push adapters / merged checkpoints to the Hugging Face Hub (private repo is
  fine), not into git:

```bash
.venv/bin/huggingface-cli login   # or hf token
.venv/bin/python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
repo = api.create_repo("<org>/jev-mini-500m-lora", private=True, exist_ok=True)
api.upload_folder(folder_path="runs/<run>/adapter", repo_id=repo, path_in_repo="adapter")
PY
```

- The Mac that serves the API then only needs:
  `JEV_MODEL=<org>/jev-mini-500m-lora` with a `VlmEngine` pointed at the
  adapter-merged model, or download the merged folder and pass its path.

## Suggested first run on GPU

1. Build synthetic (2000/domain) + real mix; build combined train set.
2. Train: r=16 LoRA, 2–3 epochs, batch 8–16, lr 1e-4 cosine, eval every 250
   steps (`eval: true`). Expect <1 h/epoch on an L4/A10-class card.
3. Eval on BOTH `data/synthetic/val.jsonl` and `data/processed/val.jsonl`.
4. Iterate where choice accuracy lags: inspect confusions per question id
   (`runs/evals/*.json`), sharpen generator distractors, upweight real data.
5. Push adapter + a merged copy to HF; record the eval JSONs alongside.

## Gotchas already solved (do not reintroduce)

- The engine MUST send the system prompt (`prompting.SYSTEM_PROMPT`) —
  training includes it.
- Option scoring MUST go through `option_token_spans` (joint BPE
  tokenization of prefix+value; discriminative span past the LCP). Scoring
  bare option ids without JSON quotes misaligns by one token.
- transformers v5: `max_steps` must be numeric (-1 for epochs); LoRA regex
  must exclude `model.vision_model`/`model.connector`; load via
  `AutoModelForImageTextToText` + `AutoProcessor`.
- MPS (if you ever fall back): fp32 only, sdpa/eager attention.
