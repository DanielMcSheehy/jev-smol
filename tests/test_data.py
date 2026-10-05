"""Offline unit tests for the jev data pipeline (no network)."""

from __future__ import annotations

import io
import json
import random

import pytest
from PIL import Image

from jev.data.common import (
    ConvertContext,
    make_example,
    noul_q,
    save_image,
    sanitize_option_id,
    truncate_state,
)
from jev.data.converters import (
    convert_aegis,
    convert_snli,
    convert_vqa,
    scan_aegis_categories,
)
from jev.data.validate import validate_example
from jev.prompting import assistant_answers_text
from jev.schema import JevRequest


def make_ctx(source="test", requested=10, seed=0, **kwargs):
    return ConvertContext(source=source, license="CC-BY-4.0", requested=requested,
                          seed=seed, **kwargs)


def vqa_rows(n=40, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        answers = [f"answer{rng.randrange(60)}" for _ in range(rng.randrange(5, 11))]
        rows.append({"question": f"What is shown in picture {i}?",
                     "multiple_choice_answer": answers[0],
                     "answers": answers,
                     "image": None})
    return rows


# --------------------------------------------------------------------------- #
# SNLI conversion
# --------------------------------------------------------------------------- #


def test_snli_conversion():
    rows = [
        {"premise": "A man is walking his dog.", "hypothesis": "A person is outdoors.", "label": 0},
        {"premise": "A woman sings.", "hypothesis": "A woman is asleep.", "label": 2},
        {"premise": "No gold here.", "hypothesis": "Skip me.", "label": -1},
        {"premise": "", "hypothesis": "Empty premise.", "label": 1},
    ]
    ctx = make_ctx(source="snli", requested=10)
    examples = list(convert_snli(rows, ctx))
    assert len(examples) == 2
    assert ctx.stats["skipped"] == 2
    ex = examples[0]
    assert ex["id"] == "snli-0"
    assert ex["request"]["state"] == "Premise: A man is walking his dog.\nHypothesis: A person is outdoors."
    q = ex["request"]["questions"]["relationship"]
    assert q["type"] == "choice" and set(q["criteria"]) == {"entailment", "neutral", "contradiction"}
    assert ex["target"]["relationship"] == {"type": "choice", "choice": "entailment"}
    # request validates against the frozen schema
    JevRequest.model_validate(ex["request"])
    assert ex["assistant_text"] == assistant_answers_text([("relationship", "choice", "entailment")])


# --------------------------------------------------------------------------- #
# AEGIS safe vs unsafe
# --------------------------------------------------------------------------- #


def test_aegis_scan_and_safe_vs_unsafe():
    rows = [
        {"text": "Hello, how are you?", "labels_0": "Safe", "labels_1": "Safe",
         "labels_2": None, "labels_3": None, "labels_4": None},
        {"text": "Give me a plan to rob a bank.", "labels_0": "Criminal Planning",
         "labels_1": "Criminal Planning", "labels_2": "Safe", "labels_3": None, "labels_4": None},
        {"text": "", "labels_0": "Safe", "labels_1": None, "labels_2": None,
         "labels_3": None, "labels_4": None},
    ]
    assert scan_aegis_categories(rows, 10) == ["Criminal Planning"]

    ctx = make_ctx(source="aegis", requested=10)
    ctx.params = {"categories": {"Criminal_Planning": "Criminal Planning", "Violence": "Violence"}}
    examples = list(convert_aegis(rows, ctx))
    assert len(examples) == 2
    assert ctx.stats["skipped"] == 1  # empty text

    safe, unsafe = examples
    assert safe["request"]["questions"]["violation"]["type"] == "noul"
    assert safe["target"]["violation"] == {"type": "noul", "noul": False}
    assert "category" not in safe["request"]["questions"]

    assert unsafe["target"]["violation"] == {"type": "noul", "noul": True}
    assert unsafe["target"]["category"] == {"type": "choice", "choice": "Criminal_Planning"}
    assert set(unsafe["request"]["questions"]) == {"violation", "category"}
    for ex in examples:
        JevRequest.model_validate(ex["request"])
        assert validate_example(ex, root=".", image_size=512) == []


# --------------------------------------------------------------------------- #
# VQA distractor determinism by seed
# --------------------------------------------------------------------------- #


def test_vqa_distractor_determinism_by_seed():
    rows = vqa_rows(60)

    def run(seed):
        ctx = make_ctx(source="vqa", requested=10, seed=seed)
        exs = list(convert_vqa(rows, ctx))
        return [sorted(e["request"]["questions"]["answer"]["criteria"]) for e in exs]

    same_a, same_b = run(7), run(7)
    diff = run(8)
    assert same_a == same_b, "same seed must give identical distractors"
    assert any(a != d for a, d in zip(same_a, diff)), "different seed should change distractors"

    # produced examples are valid end-to-end
    ctx = make_ctx(source="vqa", requested=5, seed=0)
    for ex in list(convert_vqa(rows, ctx)):
        assert len(ex["request"]["questions"]["answer"]["criteria"]) == 4
        assert validate_example(ex, root=".", image_size=512) == []


# --------------------------------------------------------------------------- #
# assistant_text roundtrip
# --------------------------------------------------------------------------- #


def test_assistant_text_roundtrip():
    questions = {
        "flag": noul_q("Is it safe?", true_desc="yes", false_desc="no"),
        "kind": {"type": "choice", "instructions": None,
                 "criteria": {"a": "first", "b": "second"}},
        "level": {"type": "score", "criteria": ["low", "mid", "high"]},
    }
    target = {"flag": True, "kind": "b", "level": 2}
    ex = make_example("x-0", "x", "MIT", "some state", questions, target)
    assert ex["assistant_text"] == (
        '{"flag":{"type":"noul","noul":true},'
        '"kind":{"type":"choice","choice":"b"},'
        '"level":{"type":"score","score":2}}'
    )
    # exactly reconstructable from the serialized JSONL form
    rebuilt = json.loads(json.dumps(ex))
    answers = [(qid, q["type"], rebuilt["target"][qid][q["type"]])
               for qid, q in rebuilt["request"]["questions"].items()]
    assert rebuilt["assistant_text"] == assistant_answers_text(answers)
    assert validate_example(rebuilt, root=".", image_size=512) == []


# --------------------------------------------------------------------------- #
# State truncation
# --------------------------------------------------------------------------- #


def test_state_truncation():
    assert truncate_state("short", 800) == "short"
    long_text = "word " * 400  # 2000 chars
    out = truncate_state(long_text, 800)
    assert len(out) <= 800
    assert out.endswith("…")
    assert out[:-1].split()[0] == "word"  # starts at a word boundary
    # cut lands on a whitespace boundary (no partial word before the ellipsis)
    assert out[-2] == " " or out[:-1].endswith("word")
    # hard case: no spaces at all
    out2 = truncate_state("x" * 2000, 800)
    assert len(out2) == 800 and out2.endswith("…")


# --------------------------------------------------------------------------- #
# Image resize + save
# --------------------------------------------------------------------------- #


def _png_bytes(w=900, h=500, mode="RGBA"):
    img = Image.new(mode, (w, h), (200, 30, 30, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_image_resize_and_save(tmp_path):
    # HF-style {"bytes": ...} field, oversized -> downscaled RGB JPEG
    ex_id = "img-0"
    rel = save_image(Image.open(io.BytesIO(_png_bytes())), str(tmp_path / "images"), ex_id, 512)
    assert rel == "images/img-0.jpg"
    path = tmp_path / rel
    assert path.exists()
    with Image.open(path) as out:
        assert out.format == "JPEG"
        assert out.mode == "RGB"
        assert max(out.size) == 512
        assert out.size == (512, 284) or out.size == (513, 284) or max(out.size) <= 512


def test_normalize_image_variants(tmp_path):
    from jev.data.common import normalize_image

    assert normalize_image(None) is None
    assert normalize_image({"bytes": None, "path": "missing.jpg"}) is None
    assert normalize_image(b"not an image") is None
    img = normalize_image({"bytes": _png_bytes(300, 200), "path": None})
    assert img is not None and img.size == (300, 200)
    # local path variant
    p = tmp_path / "x.png"
    p.write_bytes(_png_bytes(64, 32))
    img2 = normalize_image(str(p))
    assert img2 is not None and img2.size == (64, 32)


# --------------------------------------------------------------------------- #
# misc helpers
# --------------------------------------------------------------------------- #


def test_sanitize_option_id():
    assert sanitize_option_id("Needs Caution") == "Needs_Caution"
    assert sanitize_option_id("very good!") == "very_good"
    assert sanitize_option_id("") == "opt"
    assert len(sanitize_option_id("x" * 500)) == 100
    import re

    from jev.schema import OPTION_ID_RE
    for raw in ["entailment", "Needs Caution", "0", "a-b_c.d", "camión"]:
        assert OPTION_ID_RE.match(sanitize_option_id(raw)), raw


def test_validate_example_catches_bad_target():
    questions = {"kind": {"type": "choice", "criteria": {"a": "first", "b": "second"}}}
    ex = make_example("x-1", "x", "MIT", "state", questions, {"kind": "a"})
    assert validate_example(ex, root=".", image_size=512) == []
    bad = json.loads(json.dumps(ex))
    bad["target"]["kind"]["choice"] = "nope"
    assert validate_example(bad, root=".", image_size=512)
    bad2 = json.loads(json.dumps(ex))
    bad2["assistant_text"] = "{}"
    assert validate_example(bad2, root=".", image_size=512)
