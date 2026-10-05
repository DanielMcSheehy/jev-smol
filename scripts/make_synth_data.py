#!/usr/bin/env python
"""Generate deterministic synthetic SFT data for jev in the unified JSONL format.

Output (default ``data/synth``):
  train.jsonl / val.jsonl  - unified records:
      {"id", "source", "license",
       "image": null | "images/<name>.png"   (path relative to the split dir),
       "request": {"state": <str>, "questions": {<id>: <question dict>}},
       "target": {<qid>: <gold value>},
       "assistant_text": <compact answers JSON>}
  images/                  - 96x96 solid-color PNG swatches

Design: every example's state is assembled from seeded phrase templates whose
class determines the gold answer:
  * sentiment words   -> "sentiment" choice gold (positive/negative)
  * urgency keywords  -> "urgent" noul gold (true/false)
  * severity keywords -> "priority" score gold (level 0..3)
  * attached palette swatch -> "color" choice gold (image-class matched)

Examples carry 2-4 questions spanning all three schema types (noul, choice,
score); image examples additionally get the "color" choice question whose gold
matches the generated image's palette color. The assistant text is produced
exclusively through ``jev.prompting`` helpers and each request is validated
against ``jev.schema.JevRequest`` before writing. Deterministic under --seed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from jev import prompting  # noqa: E402
from jev.schema import JevRequest  # noqa: E402

# --------------------------------------------------------------------------- #
# Phrase banks: key == gold class
# --------------------------------------------------------------------------- #

SENTIMENT_PHRASES: Dict[str, List[str]] = {
    "positive": [
        "The customer is happy with the fix",
        "The user praised the support team",
        "The client is satisfied with the outcome",
        "Feedback about the release is very positive",
    ],
    "negative": [
        "The customer is frustrated by the outage",
        "The user complains about repeated failures",
        "The client is unhappy with the delay",
        "Feedback about the incident is strongly negative",
    ],
}

URGENCY_PHRASES: Dict[str, List[str]] = {
    "true": [
        "It must be handled ASAP",
        "This is urgent and blocks a release",
        "Leadership asked for an immediate response",
    ],
    "false": [
        "It can wait until next week",
        "There is no rush on this one",
        "It belongs in the normal backlog",
    ],
}

PRIORITY_PHRASES: Dict[int, List[str]] = {
    3: [
        "A full outage affects every customer",
        "Production is down with data loss",
    ],
    2: [
        "A major feature is degraded for many users",
        "A serious regression blocks part of the workflow",
    ],
    1: [
        "A minor cosmetic bug affects a few users",
        "There is a small delay in report generation",
    ],
    0: [
        "A documentation typo was reported",
        "An informational question needs an answer",
    ],
}

# Fixed palette: solid 96x96 swatches; gold of the "color" question.
PALETTE: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 60, 50),
    "green": (60, 180, 75),
    "blue": (55, 95, 220),
    "yellow": (240, 200, 60),
}

IMAGE_SWATCH_SIZE = (96, 96)


# --------------------------------------------------------------------------- #
# Question templates
# --------------------------------------------------------------------------- #

def sentiment_question() -> dict:
    return {
        "type": "choice",
        "instructions": "Classify the overall customer sentiment of the state.",
        "criteria": {
            "negative": "The customer is dissatisfied or frustrated",
            "positive": "The customer is satisfied or happy",
        },
    }


def urgent_question() -> dict:
    return {
        "type": "noul",
        "instructions": "Does the state require immediate action?",
        "criteria": {
            "true": "Needs to be handled immediately",
            "false": "Can follow the normal queue",
        },
    }


def priority_question() -> dict:
    return {
        "type": "score",
        "instructions": "Rate the operational impact of the state.",
        "criteria": ["No impact", "Minor", "Major", "Critical"],
    }


def color_question() -> dict:
    return {
        "type": "choice",
        "instructions": "Which color is shown in the attached image swatch?",
        "criteria": {
            "blue": "The swatch is blue",
            "green": "The swatch is green",
            "red": "The swatch is red",
            "yellow": "The swatch is yellow",
        },
    }


# --------------------------------------------------------------------------- #
# Example construction
# --------------------------------------------------------------------------- #

def make_example(index: int, split: str, seed: int, with_image: bool) -> Dict[str, Any]:
    """Build one unified record (no I/O). Deterministic given (split, index, seed)."""
    rng = random.Random(f"{seed}|{split}|{index}")

    sentiment_gold = rng.choice(sorted(SENTIMENT_PHRASES))
    urgent_gold = rng.choice(sorted(URGENCY_PHRASES))
    priority_gold = rng.choice(sorted(PRIORITY_PHRASES))

    phrases = [
        rng.choice(SENTIMENT_PHRASES[sentiment_gold]),
        rng.choice(URGENCY_PHRASES[urgent_gold]),
        rng.choice(PRIORITY_PHRASES[priority_gold]),
    ]
    rng.shuffle(phrases)
    state = ". ".join(phrases) + "."

    color_gold: Optional[str] = None
    if with_image:
        color_gold = rng.choice(sorted(PALETTE))
        state += " A solid-color swatch image is attached."

    # Question set: image examples carry all three base types + color (4).
    # Text examples rotate between all three (3) and dropping one (2), so every
    # type appears across the dataset and per-example counts span 2-4.
    questions: Dict[str, dict] = {"sentiment": sentiment_question()}
    drop_idx = index % 3
    if with_image or index % 2 == 0 or drop_idx != 0:
        questions["urgent"] = urgent_question()
    if with_image or index % 2 == 0 or drop_idx != 1:
        questions["priority"] = priority_question()
    if color_gold is not None:
        questions["color"] = color_question()

    # Validate against the wire schema before anything else.
    request = {"state": state, "questions": questions}
    validated = JevRequest(**request)

    target: Dict[str, Any] = {"sentiment": sentiment_gold}
    if "urgent" in questions:
        target["urgent"] = urgent_gold == "true"
    if "priority" in questions:
        target["priority"] = priority_gold
    if color_gold is not None:
        target["color"] = color_gold

    # Assistant text via the canonical serializer, in schema order.
    answers = [(qid, questions[qid]["type"], target[qid]) for qid in questions]
    assistant_text = prompting.assistant_answers_text(answers)

    # Roundtrip check: parsed assistant text must equal the target exactly.
    parsed = json.loads(assistant_text)
    assert parsed == {qid: {"type": questions[qid]["type"], questions[qid]["type"]: target[qid]}
                      for qid in questions}, f"assistant roundtrip mismatch at {split}/{index}"
    assert validated.state == state  # schema accepted the request

    return {
        "id": f"{split}-{index:04d}",
        "source": "synthetic",
        "license": "CC0-1.0",
        "image": None,
        "request": request,
        "target": target,
        "assistant_text": assistant_text,
    }


def write_swatch(path: Path, color: Tuple[int, int, int]) -> None:
    Image.new("RGB", IMAGE_SWATCH_SIZE, color).save(path, format="PNG")


def generate_split(
    out_dir: Path,
    split: str,
    n: int,
    seed: int,
    image_every: int = 5,
) -> List[Dict[str, Any]]:
    """Generate one split: n records + swatch images (every ``image_every``-th)."""
    records: List[Dict[str, Any]] = []
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    n_with_image = 0
    for i in range(n):
        with_image = i % image_every == 0
        record = make_example(i, split, seed, with_image=with_image)
        if with_image:
            color = PALETTE[record["target"]["color"]]
            filename = f"{record['id']}.png"
            write_swatch(images_dir / filename, color)
            record["image"] = f"images/{filename}"  # relative to the split dir
            n_with_image += 1
        records.append(record)

    jsonl_path = out_dir / f"{split}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {jsonl_path}: {len(records)} records ({n_with_image} with images)")
    return records


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "synth",
                        help="output directory (default: data/synth)")
    parser.add_argument("--train", type=int, default=64, help="number of train examples")
    parser.add_argument("--val", type=int, default=8, help="number of val examples")
    parser.add_argument("--seed", type=int, default=42, help="deterministic seed")
    args = parser.parse_args(argv)

    generate_split(args.out, "train", args.train, args.seed)
    generate_split(args.out, "val", args.val, args.seed)


if __name__ == "__main__":
    main()
