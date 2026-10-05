"""Offline tests for the moderation_policy synthetic domain generator."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

from jev.data.synthgen import moderation_policy
from jev.schema import JevRequest

PREFIX = "moderation_policy"
SOURCE = "synthetic/moderation_policy"

VIOLATION_CATEGORIES = {"harassment", "spam", "pii"}


def _generate(n: int, seed: int = 7, prefix: str = PREFIX) -> List[Dict[str, Any]]:
    return moderation_policy.generate(n=n, seed=seed, images_dir="", prefix=prefix)


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


def test_family_semantics_consistent() -> None:
    """noul gold, presence of the category question and severity gold must
    agree: violating rows carry a real category; safe rows are severity 0."""
    examples = _generate(120)
    seen_noul, seen_choice, seen_score = False, False, False
    for ex in examples:
        violating = None
        for qid, q in ex["request"]["questions"].items():
            gold = _raw_gold(ex, qid)
            if q["type"] == "noul":
                seen_noul = True
                violating = gold
            elif q["type"] == "choice":
                seen_choice = True
                assert gold in VIOLATION_CATEGORIES, (
                    f"category gold must be a real violation category, got {gold!r}"
                )
            else:
                seen_score = True
        assert violating is not None, "every example has the violates noul question"
        if violating:
            assert any(
                ex["request"]["questions"][qid]["type"] == "choice"
                for qid in ex["target"]
            ), "violating rows must include the category choice question"
            for qid, q in ex["request"]["questions"].items():
                if q["type"] == "score":
                    assert _raw_gold(ex, qid) >= 1
        else:
            for qid, q in ex["request"]["questions"].items():
                if q["type"] == "choice":
                    raise AssertionError("safe rows must not get the category question")
                if q["type"] == "score":
                    assert _raw_gold(ex, qid) == 0
    assert seen_noul and seen_choice and seen_score


def test_severity_levels_fixed() -> None:
    for ex in _generate(60):
        for q in ex["request"]["questions"].values():
            if q["type"] == "score":
                assert q["criteria"] == [
                    "benign", "borderline", "clear_violation", "egregious"
                ]


def test_question_type_mix() -> None:
    examples = _generate(120)
    types = Counter(
        q["type"] for ex in examples for q in ex["request"]["questions"].values()
    )
    assert types["noul"] > 0 and types["choice"] > 0 and types["score"] > 0
    counts = Counter(len(ex["request"]["questions"]) for ex in examples)
    assert set(counts) <= {1, 2, 3}
    # violating rows always carry a category question; verify both families occur
    noul_golds = Counter(
        _raw_gold(ex, qid)
        for ex in examples
        for qid, q in ex["request"]["questions"].items()
        if q["type"] == "noul"
    )
    assert noul_golds[True] > 0 and noul_golds[False] > 0


def test_determinism() -> None:
    a = _generate(30, seed=123)
    b = _generate(30, seed=123)
    assert a == b
    c = _generate(30, seed=124)
    assert a != c


def test_assistant_text_roundtrip() -> None:
    examples = _generate(60)
    for ex in examples:
        assert json.loads(ex["assistant_text"]) == ex["target"]


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
        json.dumps(s, sort_keys=True) if not isinstance(s, str) else s
        for s in states
    ]
    assert len(set(keys)) == len(keys), "duplicate states emitted"

    str_states = [s for s in states if isinstance(s, str)]
    dict_states = [s for s in states if isinstance(s, dict)]
    assert str_states and dict_states, "expected both string and JSON states"
    # ~15% dict states (loose band)
    assert 0.05 <= len(dict_states) / len(states) <= 0.30

    for s in str_states:
        assert len(s) <= 900
        assert s.strip() == s and len(s) > 20
        # 1-6 content sentences (headers/lead-ins excluded loosely)
        assert len([p for p in re.split(r"[.!?]\s", s) if p.strip()]) <= 8
    for s in dict_states:
        assert {"platform", "author_tier", "content"} <= set(s)

    # template coverage: many distinct openings on string states
    assert len({s.split("\n")[0][:30] for s in str_states}) >= 10


def test_all_contact_data_clearly_fake() -> None:
    """Mildness / safety check: any phone number is a reserved 555 number, any
    email or linked domain is a fictional example/sample domain."""
    examples = _generate(300)
    for ex in examples:
        state = ex["request"]["state"]
        text = state if isinstance(state, str) else json.dumps(state)
        for phone in re.findall(r"(?:\d{3}[-. ])?\d{3}[-. ]\d{4}", text):
            assert "555" in phone, f"non-555 phone in state: {phone!r}"
        for domain in re.findall(r"[A-Za-z0-9.-]+\.(?:com|net|org|example)\b", text):
            assert (
                domain.endswith((".example", "example.com", "example.net", "example.org"))
                or domain == "sample-mail.org"
            ), f"non-fictional domain in state: {domain!r}"
