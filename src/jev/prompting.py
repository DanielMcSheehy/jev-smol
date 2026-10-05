"""Prompt rendering for jev.

Mirrors the CLEF/Jev "System One" joint-schema prompt:

* system: a fixed instruction to decide every field jointly.
* user:   ``STATE:`` (state rendered as compact, key-sorted JSON when it is not
          already a string) followed by ``SCHEMA FIELDS:`` with one block per
          question listing id, type, instructions and the allowed options.
* assistant: a compact JSON object mapping each question id (in schema order)
  to ``{"type":"<t>","<t>":<value>}`` where value is

      noul   -> JSON true / false
      choice -> the option id (JSON string)
      score  -> the level index (JSON integer)

  Example::

      {"urgent":{"type":"noul","noul":true},
       "team":{"type":"choice","choice":"billing"},
       "severity":{"type":"score","score":2}}

The assistant text is produced by :class:`AnswerJSONBuilder` /
:func:`assistant_answers_text` — never hand-format it. Inference scores each
option by teacher-forcing :func:`answer_prefix_for_next` (the exact prefix the
model was trained on) and reading the logits at the value position, which is
the logit-per-option scheme CLEF uses.

Images are attached as separate ``{"type": "image"}`` content parts *before*
the text part (CLEF puts media tokens ahead of the state); that happens in the
engine/training code, not here — :func:`render_user_text` only produces text.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .schema import Question

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)


def render_state(state: Any) -> str:
    """Render the request state: strings pass through, JSON values are dumped
    compactly with sorted keys."""
    if isinstance(state, str):
        return state
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def question_instructions(question_id: str, question: Question) -> str:
    return question.instructions or question_id


def question_options(question: Question) -> List[Tuple[str, str]]:
    """Canonical (option_value, description) pairs for a question.

    noul   -> fixed order [true, false]
    choice -> option ids sorted by id (matching the CLEF prompt rendering)
    score  -> level indices "0".."N-1" in criteria order
    """
    if question.type == "noul":
        crit = question.criteria
        return [
            ("true", (crit.true if crit is not None and crit.true else "Yes")),
            ("false", (crit.false if crit is not None and crit.false else "No")),
        ]
    if question.type == "choice":
        return sorted(question.criteria.items(), key=lambda kv: kv[0])
    return [(str(i), desc) for i, desc in enumerate(question.criteria)]


def canonical_option_values(question: Question) -> List[Any]:
    """Gold/scoring values corresponding to :func:`question_options`:
    [True, False] for noul, option ids for choice, level ints for score."""
    if question.type == "noul":
        return [True, False]
    if question.type == "choice":
        return [oid for oid, _ in sorted(question.criteria.items(), key=lambda kv: kv[0])]
    return list(range(len(question.criteria)))


def option_label_text(qtype: str, value: Any) -> str:
    """The text whose tokens are scored for one option at the value position:
    "true"/"false", the bare option id, or the level digit."""
    if qtype == "noul":
        return "true" if value else "false"
    if qtype == "score":
        return str(int(value))
    return str(value)


def value_json(qtype: str, value: Any) -> str:
    """JSON literal for an answer value inside the assistant answers object."""
    if qtype == "noul":
        return "true" if value else "false"
    if qtype == "score":
        return str(int(value))
    return json.dumps(str(value), ensure_ascii=False)


def _field_block(n: int, question_id: str, question: Question) -> str:
    lines = [
        f"\nFIELD {n}",
        f"ID: {question_id}",
        f"TYPE: {question.type}",
        f"INSTRUCTION: {question_instructions(question_id, question)}",
        "ALLOWED OPTIONS:",
    ]
    for m, (option_id, description) in enumerate(question_options(question)):
        lines.append(
            f"OPTION {m}: "
            + json.dumps(
                {"option_id": option_id, "description": description}, ensure_ascii=False
            )
        )
    lines.append("END FIELD\n")
    return "\n".join(lines)


def render_user_text(state: Any, questions: Mapping[str, Question]) -> str:
    """Render the full user-side text: STATE + SCHEMA FIELDS blocks."""
    text = "STATE:\n" + render_state(state) + "\n\nSCHEMA FIELDS:\n"
    for n, (qid, question) in enumerate(questions.items(), start=1):
        text += _field_block(n, qid, question)
    return text


# --------------------------------------------------------------------------- #
# Assistant answers JSON
# --------------------------------------------------------------------------- #


def _answer_object(qtype: str, value: Any) -> str:
    return f'{{"type":"{qtype}","{qtype}":{value_json(qtype, value)}}}'


class AnswerJSONBuilder:
    """Builds the assistant answers JSON incrementally, question by question.

    ``prefix_for_next`` returns the exact string that precedes an answer value
    in the fully serialized object, e.g.::

        {"team":{"type":"choice","choice":

    Inference teacher-forces this prefix and scores option labels at its end;
    training serializes complete objects via ``text()``.
    """

    def __init__(self) -> None:
        self._items: List[Tuple[str, str, Any]] = []  # (question_id, qtype, value)

    def add(self, question_id: str, qtype: str, value: Any) -> None:
        self._items.append((question_id, qtype, value))

    def prefix_for_next(self, question_id: str, qtype: str) -> str:
        segments = [
            json.dumps(q, ensure_ascii=False) + ":" + _answer_object(t, v)
            for q, t, v in self._items
        ]
        s = "{" + ",".join(segments)
        if segments:
            s += ","
        s += json.dumps(question_id, ensure_ascii=False) + ":" + _answer_object_open(qtype)
        return s

    def text(self) -> str:
        segments = [
            json.dumps(q, ensure_ascii=False) + ":" + _answer_object(t, v)
            for q, t, v in self._items
        ]
        return "{" + ",".join(segments) + "}"


def _answer_object_open(qtype: str) -> str:
    return f'{{"type":"{qtype}","{qtype}":'


def assistant_answers_text(answers: Iterable[Tuple[str, str, Any]]) -> str:
    """Serialize completed answers ``(question_id, qtype, value)`` in order."""
    builder = AnswerJSONBuilder()
    for question_id, qtype, value in answers:
        builder.add(question_id, qtype, value)
    return builder.text()
