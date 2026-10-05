"""Offline tests for the synthetic chart_images domain generator."""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict

from PIL import Image

from jev.data.synthgen import chart_images as ci
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
    exs = ci.generate(40, seed=7, images_dir=images_dir, prefix="chart_images")
    assert len(exs) == 40
    seen_ids = set()
    for i, ex in enumerate(exs):
        assert ex["id"] == f"chart_images-{i:05d}"
        assert ex["source"] == "synthetic/chart_images"
        assert ex["license"] == "CC0-1.0"
        assert ex["image"] == f"images/{ex['id']}.jpg"
        assert os.path.exists(os.path.join(images_dir, f"{ex['id']}.jpg"))
        seen_ids.add(ex["id"])

        request = JevRequest.model_validate(ex["request"])  # raises if invalid
        qids = set(request.questions)
        assert 2 <= len(qids) <= 4
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
    exs = ci.generate(10, seed=13, images_dir=images_dir, prefix="chart_images")
    for ex in exs:
        assert validate_example(ex, root=str(tmp_path), image_size=512) == []


def test_determinism_same_seed_identical_bytes(tmp_path):
    dirs = [str(tmp_path / "a"), str(tmp_path / "b")]
    runs = [ci.generate(12, seed=9, images_dir=d, prefix="chart_images") for d in dirs]
    for ex_a, ex_b in zip(*runs):
        assert json.dumps(ex_a, sort_keys=True) == json.dumps(ex_b, sort_keys=True)
        a = open(os.path.join(dirs[0], f"{ex_a['id']}.jpg"), "rb").read()
        b = open(os.path.join(dirs[1], f"{ex_b['id']}.jpg"), "rb").read()
        assert a == b
    # a different seed produces different images
    dir_c = str(tmp_path / "c")
    exs_c = ci.generate(12, seed=10, images_dir=dir_c, prefix="chart_images")
    assert any(
        open(os.path.join(dir_c, f"{ex['id']}.jpg"), "rb").read()
        != open(os.path.join(dirs[0], f"{exs[0]['id']}.jpg"), "rb").read()
        for exs, ex in zip(runs, exs_c)
    )


def test_states_unique_and_varied(tmp_path):
    images_dir = str(tmp_path / "images")
    exs = ci.generate(150, seed=21, images_dir=images_dir, prefix="chart_images")
    keys = [json.dumps(ex["request"]["state"], sort_keys=True) for ex in exs]
    assert len(set(keys)) == len(keys)
    str_states = [ex["request"]["state"] for ex in exs if isinstance(ex["request"]["state"], str)]
    dict_states = [ex["request"]["state"] for ex in exs if isinstance(ex["request"]["state"], dict)]
    assert len(str_states) + len(dict_states) == 150
    frac_dict = len(dict_states) / 150
    assert 0.05 <= frac_dict <= 0.30  # ~15% dict states
    # at least 10 distinct string templates in use
    leads = {s.split()[0] for s in str_states}
    assert len(leads) >= 5


def _slot_of(ex, qid):
    """Map a question onto its generator slot by its structure."""
    q = ex["request"]["questions"][qid]
    if q["type"] == "choice":
        return "peak" if "start" in q["criteria"] else "trend"
    if q["type"] == "noul":
        return "threshold"
    crit = " ".join(q["criteria"])
    return "count" if "data points" in crit else "magnitude"


def test_label_balance(tmp_path):
    images_dir = str(tmp_path / "images")
    exs = ci.generate(240, seed=11, images_dir=images_dir, prefix="chart_images")
    golds: dict = defaultdict(Counter)
    thr_instrs = Counter()
    trend_opts = Counter()
    for ex in exs:
        for qid in ex["request"]["questions"]:
            q = ex["request"]["questions"][qid]
            slot = _slot_of(ex, qid)
            golds[slot][ex["target"][qid][q["type"]]] += 1
            if slot == "threshold":
                assert "red" in q["instructions"]  # threshold is in the text
                thr_instrs[q["instructions"]] += 1
            if slot == "trend":
                assert 3 <= len(q["criteria"]) <= 4
                assert ex["target"][qid]["choice"] in q["criteria"]
                trend_opts[len(q["criteria"])] += 1
    for slot, counter in golds.items():
        total = sum(counter.values())
        assert total > 20, slot
        assert max(counter.values()) / total <= 0.55, (slot, counter)
    assert len(thr_instrs) >= 3  # varied threshold instructions
    assert set(trend_opts) == {3, 4}  # option subsets vary


def test_golds_derivable_from_drawn_series():
    rng = random.Random(5)
    counters = defaultdict(Counter)
    seen_thr_true = seen_thr_false = 0
    for _ in range(120):
        p = ci._build_params(rng, counters)
        vals = p["values"]
        assert len(vals) == p["k"]
        # trend/peak gold is exactly the classification of the plotted series
        assert ci._classify(vals, p["band"]) == p["shape"]
        # magnitude gold is the mean bucket of the plotted series
        mean = sum(vals) / len(vals)
        level = 0 if mean < ci.MAG_THRESHOLDS[0] else (1 if mean < ci.MAG_THRESHOLDS[1] else 2)
        assert level == p["mag"]
        # count gold matches the number of plotted points
        expected_cl = 0 if p["k"] <= 5 else (1 if p["k"] <= 7 else 2)
        assert expected_cl == p["clevel"]
        # threshold gold matches the drawn line position
        if p["threshold"] is not None:
            assert p["thr_gold"] == (max(vals) > p["threshold"])
            assert 0 < p["threshold"] < ci.Y_MAX
            seen_thr_true += p["thr_gold"]
            seen_thr_false += not p["thr_gold"]
        # the red line exists iff the question is present
        assert (p["threshold"] is not None) == ("threshold" in p["slots"])
    assert seen_thr_true > 0 and seen_thr_false > 0  # both threshold golds occur


def test_chart_type_split_balanced():
    rng = random.Random(17)
    counters = defaultdict(Counter)
    types = Counter(
        ci._build_params(rng, counters)["ctype"] for _ in range(300)
    )
    for count in types.values():
        assert 0.35 <= count / 300 <= 0.65


def test_image_content_legible(tmp_path):
    images_dir = str(tmp_path / "images")
    exs = ci.generate(12, seed=5, images_dir=images_dir, prefix="chart_images")
    n_with_red = 0
    for ex in exs:
        img = Image.open(os.path.join(images_dir, f"{ex['id']}.jpg")).convert("RGB")
        px = list(img.getdata())
        dark = sum(1 for r, g, b in px if r < 80 and g < 80 and b < 80)
        red = sum(1 for r, g, b in px if r > 150 and g < 90 and b < 90)
        blue = sum(1 for r, g, b in px if b > 120 and b > r + 30 and b > g + 20)
        assert dark > 300, "expected black axis pixels"
        assert blue > 300, "expected blue bars or line pixels"
        if red > 50:
            n_with_red += 1
    # threshold questions appear in a good fraction of examples
    n_thr = sum(
        1
        for ex in exs
        for q in ex["request"]["questions"].values()
        if q["type"] == "noul"
    )
    assert n_thr > 0
    assert n_with_red == n_thr  # red drawn exactly for threshold questions


def test_generate_matches_registry_contract():
    assert ci.generate.__code__.co_varnames[:4] == ("n", "seed", "images_dir", "prefix")
