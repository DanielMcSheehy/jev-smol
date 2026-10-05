"""Offline evaluation of a jev engine against gold-labeled decision data.

Scores every question of every example against the gold target:

* accuracy  — noul: P(true) >= 0.5 vs gold; choice: argmax option vs gold;
              score: round(expected level) vs gold int level
* Brier     — per-question squared error of the full probability distribution
              (0 = perfect, 2 = maximally wrong for multiway)
* mean confidence of the predicted answer

Usage::

    .venv/bin/python -m jev.eval --data data/synthetic/val_text.jsonl \
        --root data/synthetic [--model <hf-id-or-path>] [--limit 48] \
        [--text-only] [--device auto] [--out runs/evals/result.json]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from jev.schema import JevRequest

MIME_BY_SUFFIX = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def load_examples(path: Path, limit: Optional[int], text_only: bool) -> List[Dict[str, Any]]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if text_only and ex.get("image"):
                continue
            examples.append(ex)
            if limit is not None and len(examples) >= limit:
                break
    return examples


def attach_image(request_dict: Dict[str, Any], root: Path, rel: str) -> None:
    p = root / rel
    mime = MIME_BY_SUFFIX.get(p.suffix.lower(), "image/jpeg")
    data_url = "data:" + mime + ";base64," + base64.b64encode(p.read_bytes()).decode()
    request_dict.setdefault("images", []).append(data_url)


def score_answer(qtype: str, gold: Any, ans) -> Dict[str, float]:
    """One question's record: correctness, Brier, confidence of the prediction."""
    if qtype == "noul":
        p_true = float(ans.noul)
        pred = p_true >= 0.5
        y = 1.0 if gold else 0.0
        return {
            "correct": float(pred == bool(gold)),
            "brier": (p_true - y) ** 2,
            "confidence": max(p_true, 1.0 - p_true),
        }
    if qtype == "choice":
        y = 1.0 if ans.choice == gold else 0.0
        return {
            "correct": float(ans.choice == gold),
            "brier": sum((p - (1.0 if oid == gold else 0.0)) ** 2 for oid, p in ans.probabilities.items()),
            "confidence": float(ans.confidence),
        }
    # score
    pred_level = int(round(float(ans.score)))
    return {
        "correct": float(pred_level == int(gold)),
        "brier": sum(
            (p - (1.0 if int(k) == int(gold) else 0.0)) ** 2 for k, p in ans.probabilities.items()
        ),
        "confidence": float(ans.confidence),
    }


def evaluate(engine, examples: List[Dict[str, Any]], root: Path) -> Dict[str, Any]:
    by_type: Dict[str, Dict[str, float]] = {}
    errors = 0
    for i, ex in enumerate(examples):
        try:
            request_dict = dict(ex["request"])
            if ex.get("image"):
                attach_image(request_dict, root, ex["image"])
            request = JevRequest.model_validate(request_dict)
            response = engine.decide(request)
            for qid, gold_typed in ex["target"].items():
                qtype = gold_typed["type"]
                gold = gold_typed[qtype]
                rec = score_answer(qtype, gold, response.answers[qid])
                agg = by_type.setdefault(qtype, {"n": 0, "correct": 0.0, "brier": 0.0, "confidence": 0.0})
                agg["n"] += 1
                agg["correct"] += rec["correct"]
                agg["brier"] += rec["brier"]
                agg["confidence"] += rec["confidence"]
        except Exception as e:  # one bad example must not kill the run
            errors += 1
            print(f"  [{i}] error on {ex.get('id')}: {type(e).__name__}: {e}", file=sys.stderr)

    summary: Dict[str, Any] = {"examples": len(examples), "example_errors": errors, "types": {}}
    total_n = total_correct = 0.0
    total_brier = 0.0
    for qtype, agg in sorted(by_type.items()):
        n = agg["n"]
        acc = agg["correct"] / n
        summary["types"][qtype] = {
            "questions": n,
            "accuracy": round(acc, 4),
            "brier": round(agg["brier"] / n, 4),
            "mean_confidence": round(agg["confidence"] / n, 4),
        }
        total_n += n
        total_correct += agg["correct"]
        total_brier += agg["brier"]
    if total_n:
        summary["overall"] = {
            "questions": int(total_n),
            "accuracy": round(total_correct / total_n, 4),
            "brier": round(total_brier / total_n, 4),
        }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a jev engine against gold labels")
    parser.add_argument("--data", required=True, help="val/train JSONL in the unified format")
    parser.add_argument("--root", default=None, help="root dir for relative image paths (default: data dir's parent)")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--out", default=None, help="write the summary JSON here")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    root = Path(args.root) if args.root else data_path.parent
    examples = load_examples(data_path, args.limit, args.text_only)
    print(f"evaluating {len(examples)} examples from {data_path} (model={args.model})")

    from jev.engine import get_engine  # heavy imports only when actually evaluating

    engine = get_engine("vlm", model=args.model, device=args.device)

    summary = evaluate(engine, examples, root)
    summary["model"] = args.model
    summary["data"] = str(data_path)

    print("\n== results ==")
    print(f"{'type':<10} {'questions':>9} {'accuracy':>9} {'brier':>8} {'mean_conf':>10}")
    for qtype, s in summary["types"].items():
        print(f"{qtype:<10} {s['questions']:>9} {s['accuracy']:>9.4f} {s['brier']:>8.4f} {s['mean_confidence']:>10.4f}")
    if "overall" in summary:
        o = summary["overall"]
        print(f"{'OVERALL':<10} {o['questions']:>9} {o['accuracy']:>9.4f} {o['brier']:>8.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
