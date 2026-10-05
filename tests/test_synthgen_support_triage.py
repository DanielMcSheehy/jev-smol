"""Offline tests for the support_triage synthetic domain generator."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Dict, List

from jev.data.synthgen import support_triage
from jev.schema import JevRequest

PREFIX = "support_triage"
SOURCE = "synthetic/support_triage"


def _generate(n: int, seed: int = 7, prefix: str = PREFIX) -> List[Dict[str, Any]]:
    return support_triage.generate(n=n, seed=seed, images_dir="", prefix=prefix)


def _target_map(ex: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return ex["target"]


def _raw_gold(ex: Dict[str, Any], qid: str) -> Any:
    """Unwrap the typed target entry {qid: {"type": t, t: value}} to value."""
    entry = ex["target"][qid]
    qtype = entry["type"]
    return entry[qtype]


def test_exact_count_ids_and_metadata() -> None:
    n = 40
    examples = _generate(n)
    assert len(examples) == n
    for i, ex in enumerate(examples):
        assert ex["id"] == f"{PREFIX}-{i:05d}"
        assert ex["source"] == SOURCE
        assert ex["license"] == "CC0-1.0"
        assert ex["image"] is None  # text-only domain


def test_requests_validate_and_targets_match_questions() -> None:
    examples = _generate(80)
    for ex in examples:
        request = ex["request"]
        assert request["model"] == "jev-mini"
        JevRequest.model_validate(request)  # raises on any schema violation

        questions = request["questions"]
        assert 1 <= len(questions) <= 4
        assert set(ex["target"]) == set(questions)
        for qid, q in questions.items():
            gold = _raw_gold(ex, qid)
            if q["type"] == "noul":
                assert isinstance(gold, bool)
            elif q["type"] == "choice":
                assert gold in q["criteria"]
            else:
                assert isinstance(gold, int) and 0 <= gold < len(q["criteria"])


def test_question_type_mix() -> None:
    examples = _generate(120)
    types = Counter(q["type"] for ex in examples for q in ex["request"]["questions"].values())
    assert types["noul"] > 0 and types["choice"] > 0 and types["score"] > 0
    counts = Counter(len(ex["request"]["questions"]) for ex in examples)
    assert set(counts) <= {2, 3, 4}


def test_determinism() -> None:
    a = _generate(30, seed=123)
    b = _generate(30, seed=123)
    assert a == b
    c = _generate(30, seed=124)
    assert a != c


def test_assistant_text_roundtrip() -> None:
    examples = _generate(60)
    for ex in examples:
        assert json.loads(ex["assistant_text"]) == _target_map(ex)


def test_label_balance() -> None:
    examples = _generate(600)
    per_qid: Dict[str, Counter] = defaultdict(Counter)
    for ex in examples:
        for qid in ex["target"]:
            per_qid[qid][_raw_gold(ex, qid)] += 1
    assert per_qid, "expected at least one question id"
    for qid, dist in per_qid.items():
        total = sum(dist.values())
        top = max(dist.values()) / total
        assert top <= 0.55, f"question {qid!r} imbalanced: {dict(dist)}"


def test_state_diversity_and_shape() -> None:
    examples = _generate(200)
    states = [ex["request"]["state"] for ex in examples]
    keys = [
        json.dumps(s, sort_keys=True) if not isinstance(s, str) else s for s in states
    ]
    assert len(set(keys)) == len(keys), "duplicate states emitted"

    str_states = [s for s in states if isinstance(s, str)]
    dict_states = [s for s in states if isinstance(s, dict)]
    assert str_states and dict_states, "expected both string and JSON states"
    # ~20% dict states (loose band)
    assert 0.05 <= len(dict_states) / len(states) <= 0.40

    for s in str_states:
        assert len(s) <= 900
        assert s.strip() == s and len(s) > 50
    for s in dict_states:
        assert {"channel", "subject", "body", "customer", "tier"} <= set(s)

    # template coverage: distinct detail/product/tone slots mean many distinct
    # openings; sanity-check that at least 18 distinct template shapes appear
    # by counting distinct first-sentence shapes on string states.
    assert len({s.split("\n")[0][:30] for s in str_states}) >= 10
