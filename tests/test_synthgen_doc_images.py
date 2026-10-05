"""Offline tests for the synthetic doc_images domain generator."""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict

import pytest
from PIL import Image

from jev.data.synthgen import doc_images as di
from jev.data.validate import validate_example
from jev.prompting import assistant_answers_text
from jev.schema import JevRequest


def _answers(ex):
    return [
        (qid, q["type"], ex["target"][qid][q["type"]])
        for qid, q in ex["request"]["questions"].items()
    ]


def test_generate_shape_schema_and_images(tmp_path):
    images_dir = str(tmp_path / "images")
    exs = di.generate(40, seed=7, images_dir=images_dir, prefix="doc_images")
    assert len(exs) == 40
    seen_ids = set()
    for i, ex in enumerate(exs):
        assert ex["id"] == f"doc_images-{i:05d}"
        assert ex["source"] == "synthetic/doc_images"
        assert ex["license"] == "CC0-1.0"
        assert ex["image"] == f"images/{ex['id']}.jpg"
        assert os.path.exists(os.path.join(images_dir, f"{ex['id']}.jpg"))
        seen_ids.add(ex["id"])

        request = JevRequest.model_validate(ex["request"])  # raises if invalid
        qids = set(request.questions)
        assert 1 <= len(qids) <= 4
        assert set(ex["target"]) == qids
        for qid, q in request.questions.items():
            entry = ex["target"][qid]
            assert entry["type"] == q.type
            if q.type == "noul":
                assert isinstance(entry["noul"], bool)
            elif q.type == "choice":
                assert entry["choice"] in q.criteria
            else:
                assert isinstance(entry["score"], int)
                assert 0 <= entry["score"] < len(q.criteria)
        # assistant_text roundtrips through the prompting serializer
        assert ex["assistant_text"] == assistant_answers_text(_answers(ex))
        # image opens and respects the 512 cap
        img = Image.open(os.path.join(images_dir, f"{ex['id']}.jpg"))
        img.load()
        assert max(img.size) <= 512
        assert img.mode == "RGB"
    assert len(seen_ids) == 40


def test_validate_example_clean(tmp_path):
    images_dir = str(tmp_path / "images")
    exs = di.generate(10, seed=13, images_dir=images_dir, prefix="doc_images")
    for ex in exs:
        assert validate_example(ex, root=str(tmp_path), image_size=512) == []


def test_determinism_same_seed_identical_bytes(tmp_path):
    dirs = [str(tmp_path / "a"), str(tmp_path / "b")]
    runs = [di.generate(12, seed=9, images_dir=d, prefix="doc_images") for d in dirs]
    for ex_a, ex_b in zip(*runs):
        assert json.dumps(ex_a, sort_keys=True) == json.dumps(ex_b, sort_keys=True)
        a = open(os.path.join(dirs[0], f"{ex_a['id']}.jpg"), "rb").read()
        b = open(os.path.join(dirs[1], f"{ex_b['id']}.jpg"), "rb").read()
        assert a == b
    # a different seed produces different images
    dir_c = str(tmp_path / "c")
    exs_c = di.generate(12, seed=10, images_dir=dir_c, prefix="doc_images")
    assert any(
        open(os.path.join(dir_c, f"{ex['id']}.jpg"), "rb").read()
        != open(os.path.join(dirs[0], f"{exs[0]['id']}.jpg"), "rb").read()
        for exs, ex in zip(runs, exs_c)
    )


def test_states_unique_and_varied(tmp_path):
    images_dir = str(tmp_path / "images")
    exs = di.generate(150, seed=21, images_dir=images_dir, prefix="doc_images")
    keys = [json.dumps(ex["request"]["state"], sort_keys=True) for ex in exs]
    assert len(set(keys)) == len(keys)
    str_states = [ex["request"]["state"] for ex in exs if isinstance(ex["request"]["state"], str)]
    dict_states = [ex["request"]["state"] for ex in exs if isinstance(ex["request"]["state"], dict)]
    assert len(str_states) + len(dict_states) == 150
    frac_dict = len(dict_states) / 150
    assert 0.05 <= frac_dict <= 0.30  # ~15% dict states
    assert len({s.split()[0] for s in str_states}) >= 5  # varied phrasings


def _slot_of(ex, qid):
    """Map a question onto its generator slot by its instructions/criteria."""
    q = ex["request"]["questions"][qid]
    instr = q.get("instructions") or ""
    if q["type"] == "choice":
        return "vendor"
    if q["type"] == "noul":
        return "total_over" if "$" in instr else "is_paid"
    crit = " ".join(q["criteria"])
    return "payment_urgency" if "due" in crit else "line_item_count"


def test_label_balance(tmp_path):
    images_dir = str(tmp_path / "images")
    exs = di.generate(240, seed=11, images_dir=images_dir, prefix="doc_images")
    golds: dict = defaultdict(Counter)
    thresholds = Counter()
    for ex in exs:
        for qid in ex["request"]["questions"]:
            q = ex["request"]["questions"][qid]
            slot = _slot_of(ex, qid)
            golds[slot][ex["target"][qid][q["type"]]] += 1
            if slot == "total_over":
                thresholds[q["instructions"]] += 1
    for slot, counter in golds.items():
        total = sum(counter.values())
        assert total > 20, slot
        assert max(counter.values()) / total <= 0.55, (slot, counter)
    # all four thresholds actually used
    assert len(thresholds) >= 4


def test_drawn_total_matches_gold_params():
    rng = random.Random(3)
    counters = defaultdict(Counter)
    checked = 0
    for _ in range(60):
        slots = di._pick_slots(rng)
        p = di._build_params(rng, slots, counters)
        _img, meta = di.render_receipt(p)
        # the TOTAL text drawn on the image is exactly the computed total
        assert f"{p['total']:.2f}" in meta["total_text"]
        assert meta["total"] == p["total"]
        if "total_over" in slots:
            assert p["total_over_gold"] == (p["total"] > p["threshold"])
            checked += 1
    assert checked > 0


def test_image_content_legible(tmp_path):
    images_dir = str(tmp_path / "images")
    exs = di.generate(8, seed=5, images_dir=images_dir, prefix="doc_images")
    for ex in exs:
        img = Image.open(os.path.join(images_dir, f"{ex['id']}.jpg")).convert("RGB")
        px = list(img.getdata())
        dark = sum(1 for r, g, b in px if r < 80 and g < 80 and b < 80)
        red = sum(1 for r, g, b in px if r > 130 and g < 90 and b < 90)
        assert dark > 500, "expected substantial black text pixels"
        assert red > 50, "expected a red stamp or due line"


def test_generate_matches_registry_contract():
    assert di.generate.__code__.co_varnames[:4] == ("n", "seed", "images_dir", "prefix")
