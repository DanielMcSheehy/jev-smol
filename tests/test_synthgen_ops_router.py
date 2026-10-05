"""Offline tests for the synthetic ops_router domain generator."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from jev.data.synthgen import REGISTRY
from jev.data.synthgen.ops_router import generate
from jev.data.validate import validate_example
from jev.prompting import assistant_answers_text
from jev.schema import JevRequest

TMP_IMAGES = "/tmp/jev-test-ops-router-images"  # never used (text-only domain)


def _gold_labels(exs, qid):
    vals = []
    for ex in exs:
        t = ex["target"].get(qid)
        if t is None:
            continue
        vals.append(t[t["type"]])
    return vals


# --------------------------------------------------------------------------- #
# Contract basics
# --------------------------------------------------------------------------- #


def test_registered_in_synthgen_registry():
    assert REGISTRY["ops_router"] == "jev.data.synthgen.ops_router"


def test_exact_count_and_ids():
    exs = generate(n=120, seed=7, images_dir=TMP_IMAGES, prefix="opr")
    assert len(exs) == 120
    assert [ex["id"] for ex in exs] == [f"opr-{i:05d}" for i in range(120)]


def test_deterministic_and_seed_sensitive():
    a = generate(n=80, seed=3, images_dir=TMP_IMAGES, prefix="opr")
    b = generate(n=80, seed=3, images_dir=TMP_IMAGES, prefix="opr")
    c = generate(n=80, seed=4, images_dir=TMP_IMAGES, prefix="opr")
    assert a == b
    assert a != c


def test_text_only_and_metadata():
    for ex in generate(n=50, seed=11, images_dir=TMP_IMAGES, prefix="opr"):
        assert ex["image"] is None
        assert ex["source"] == "synthetic/ops_router"
        assert ex["license"] == "CC0-1.0"


# --------------------------------------------------------------------------- #
# Schema validity + target/question consistency (reuse the pipeline validator)
# --------------------------------------------------------------------------- #


def test_all_examples_pass_pipeline_validator():
    exs = generate(n=200, seed=5, images_dir=TMP_IMAGES, prefix="opr")
    for ex in exs:
        assert validate_example(ex, root=".", image_size=512) == []
        JevRequest.model_validate(ex["request"])


def test_question_count_and_types():
    for ex in generate(n=200, seed=5, images_dir=TMP_IMAGES, prefix="opr"):
        qs = ex["request"]["questions"]
        assert 1 <= len(qs) <= 4
        for q in qs.values():
            assert q["type"] in {"noul", "choice", "score"}


def test_state_shapes_and_no_duplicates():
    exs = generate(n=400, seed=9, images_dir=TMP_IMAGES, prefix="opr")
    json_states = 0
    seen = set()
    for ex in exs:
        state = ex["request"]["state"]
        if isinstance(state, str):
            assert 0 < len(state) <= 900
            assert 1 <= state.count(".") + state.count("!") + state.count("?")
        else:
            assert isinstance(state, dict) and state
            json_states += 1
        key = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
        assert key not in seen, "duplicate state"
        seen.add(key)
    # ~20% dict-valued JSON states
    assert 0.10 <= json_states / len(exs) <= 0.30


def test_assistant_text_roundtrip():
    for ex in generate(n=120, seed=13, images_dir=TMP_IMAGES, prefix="opr"):
        parsed = json.loads(ex["assistant_text"])
        assert parsed == ex["target"]


# --------------------------------------------------------------------------- #
# Sub-family structure and gold balance
# --------------------------------------------------------------------------- #


def _question_pool(exs, qid):
    return [ex["request"]["questions"][qid] for ex in exs if qid in ex["request"]["questions"]]


def test_subfamily_split_and_question_families():
    exs = generate(n=600, seed=0, images_dir=TMP_IMAGES, prefix="ops_router")
    incident_qids = {"priority", "incident_priority", "severity", "urgency"}
    routing_qids = {"route_to", "target_model", "model_choice", "best_model"}
    n_inc = sum(1 for ex in exs if incident_qids & set(ex["request"]["questions"]))
    n_rt = sum(1 for ex in exs if routing_qids & set(ex["request"]["questions"]))
    assert n_inc + n_rt == len(exs)
    assert 0.45 <= n_inc / len(exs) <= 0.55
    assert 0.45 <= n_rt / len(exs) <= 0.55


def test_incident_question_pools_and_options():
    exs = generate(n=600, seed=0, images_dir=TMP_IMAGES, prefix="ops_router")
    for qid in ("production_affecting", "prod_impact", "affects_production",
                "production_incident", "customer_facing", "customers_affected",
                "user_facing", "customer_impact"):
        for q in _question_pool(exs, qid):
            assert q["type"] == "noul"
    for qid in ("priority", "incident_priority", "severity", "urgency"):
        for q in _question_pool(exs, qid):
            assert q["type"] == "score"
            assert q["criteria"] == ["P5 informational", "P4 low", "P3 moderate",
                                     "P2 high", "P1 critical"]
    teams = {"on_call", "networking", "database", "frontend", "security", "vendor"}
    for qid in ("team", "assign_to", "owning_team", "route_team"):
        pool = _question_pool(exs, qid)
        assert pool, f"question id {qid} never generated"
        for q in pool:
            assert q["type"] == "choice"
            assert 3 <= len(q["criteria"]) <= 6
            assert set(q["criteria"]) <= teams


def test_routing_question_pools_and_options():
    exs = generate(n=600, seed=0, images_dir=TMP_IMAGES, prefix="ops_router")
    models = {"fast_model", "reasoning_model", "code_model", "vision_model"}
    for qid in ("route_to", "target_model", "model_choice", "best_model"):
        pool = _question_pool(exs, qid)
        assert pool
        for q in pool:
            assert q["type"] == "choice"
            assert 3 <= len(q["criteria"]) <= 4
            assert set(q["criteria"]) <= models
    for qid in ("complexity", "task_complexity", "effort_level"):
        for q in _question_pool(exs, qid):
            assert q["type"] == "score"
            assert 3 <= len(q["criteria"]) <= 4


def test_no_gold_class_above_55_percent():
    exs = generate(n=1000, seed=0, images_dir=TMP_IMAGES, prefix="ops_router")
    qids = set()
    for ex in exs:
        qids |= set(ex["request"]["questions"])
    worst = {}
    for qid in sorted(qids):
        golds = _gold_labels(exs, qid)
        assert golds, qid
        top, count = Counter(golds).most_common(1)[0]
        share = count / len(golds)
        worst[qid] = (top, round(share, 3))
        assert share <= 0.55, f"{qid}: {top} at {share:.2%}"
    # vision_model stays rare wherever it appears as a route gold
    for qid in ("route_to", "target_model", "model_choice", "best_model"):
        golds = _gold_labels(exs, qid)
        if golds:
            vision = sum(1 for g in golds if g == "vision_model")
            assert vision / len(golds) <= 0.12, qid


def test_noul_golds_are_bools_and_scores_in_range():
    exs = generate(n=300, seed=21, images_dir=TMP_IMAGES, prefix="opr")
    for ex in exs:
        for qid, q in ex["request"]["questions"].items():
            val = ex["target"][qid][q["type"]]
            if q["type"] == "noul":
                assert isinstance(val, bool)
            elif q["type"] == "choice":
                assert val in q["criteria"]
            else:
                assert isinstance(val, int) and 0 <= val < len(q["criteria"])


# --------------------------------------------------------------------------- #
# Diversity
# --------------------------------------------------------------------------- #


def test_template_diversity_via_vocabulary():
    exs = generate(n=400, seed=17, images_dir=TMP_IMAGES, prefix="opr")
    vocab = Counter()
    for ex in exs:
        state = ex["request"]["state"]
        text = json.dumps(state, ensure_ascii=False) if isinstance(state, dict) else state
        for word in text.lower().replace(".", " ").replace(",", " ").split():
            vocab[word] += 1
    # rich vocab pools feed the templates; a degenerate generator would collapse
    assert len(vocab) > 250


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_generate_respects_exact_n(seed):
    assert len(generate(n=37, seed=seed, images_dir=TMP_IMAGES, prefix="opr")) == 37
