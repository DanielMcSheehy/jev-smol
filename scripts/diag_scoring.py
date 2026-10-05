#!/usr/bin/env python
"""Diagnose engine-scoring vs model-generation alignment.

Greedy-generates the answers JSON from a model on val examples (with and
without the training system message), scores the parsed decisions against
gold, and prints samples. If generation accuracy is high while the
logit-scoring engine is poor, the engine's scoring path is misaligned with
training (system message / tokenization boundary), not the model.

Usage:
    .venv/bin/python scripts/diag_scoring.py [--model runs/textmini/merged] \
        [--data data/synthetic/val_text.jsonl] [--n 24]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from jev import prompting  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/textmini/merged")
    ap.add_argument("--data", default="data/synthetic/val_text.jsonl")
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    examples = [json.loads(l) for l in open(args.data)][: args.n]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.model)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    tok = proc.tokenizer

    def messages_for(ex: dict, with_system: bool) -> list:
        from jev.schema import JevRequest

        request = JevRequest.model_validate(ex["request"])
        user_text = prompting.render_user_text(request.state, request.questions)
        msgs = []
        if with_system:
            msgs.append(
                {"role": "system", "content": [{"type": "text", "text": prompting.SYSTEM_PROMPT}]}
            )
        msgs.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        return msgs

    def generate(ex: dict, with_system: bool) -> str:
        msgs = messages_for(ex, with_system)
        out = proc.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        )
        ids = out.input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=200, do_sample=False)
        return tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)

    for with_system in (False, True):
        correct = total = parse_fail = 0
        per_type: dict = {}
        for i, ex in enumerate(examples):
            text = generate(ex, with_system)
            if i < 2:
                print(f"--- with_system={with_system} sample {ex['id']} ---")
                print(text[:300])
            try:
                answers = json.loads(text)
            except Exception:
                parse_fail += 1
                total += len(ex["target"])
                continue
            for qid, gt in ex["target"].items():
                t = gt["type"]
                gold = gt[t]
                pred = (answers.get(qid) or {}).get(t)
                if t == "noul":
                    pred = bool(pred)
                elif t == "score":
                    pred = int(pred) if pred is not None else None
                ok = pred == gold
                agg = per_type.setdefault(t, [0, 0])
                agg[0] += int(ok)
                agg[1] += 1
                correct += int(ok)
                total += 1
        print(
            f"== with_system={with_system}: accuracy {correct}/{total} = {correct/max(total,1):.4f}"
            f" (parse failures: {parse_fail}) per-type "
            + ", ".join(f"{t} {c}/{n}" for t, (c, n) in sorted(per_type.items()))
        )


if __name__ == "__main__":
    main()
