"""Engine interface, mock engine, and registry.

An engine turns a validated :class:`jev.schema.JevRequest` into a
:class:`jev.schema.JevResponse`. The real model lives in
``jev.engine_vlm`` (imported lazily because it pulls in torch/transformers).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from random import Random
from typing import Any, Callable, Dict

from . import prompting
from .schema import JevRequest, JevResponse, Question, Usage, build_answer


class JevEngine(ABC):
    model_id: str = "jev"

    @abstractmethod
    def decide(self, request: JevRequest) -> JevResponse:
        raise NotImplementedError

    def _usage(self, request: JevRequest) -> Usage:
        # Cheap character-based estimate; the VLM engine reports real token counts.
        text = prompting.render_user_text(request.state, request.questions)
        return Usage(input_tokens=max(1, len(text) // 4), output_tokens=0)


class MockEngine(JevEngine):
    """Deterministic, hash-seeded engine for tests, dry runs, and API demos.

    Probabilities are pseudo-random but stable for a given (state, question id)
    pair, and always conform to the response schema.
    """

    def __init__(self, model_id: str = "jev-mock") -> None:
        self.model_id = model_id

    def decide(self, request: JevRequest) -> JevResponse:
        answers: Dict[str, Any] = {}
        for qid, question in request.questions.items():
            rng = self._rng(request, qid)
            if question.type == "noul":
                weights = {"true": rng.random(), "false": rng.random()}
            elif question.type == "choice":
                weights = {
                    oid: rng.random() ** 2 + 0.01
                    for oid, _ in prompting.question_options(question)
                }
            else:  # score
                weights = {
                    str(i): rng.random() ** 2 + 0.01 for i in range(len(question.criteria))
                }
            answers[qid] = build_answer(question, weights)
        return JevResponse(model=self.model_id, answers=answers, usage=self._usage(request))

    def _rng(self, request: JevRequest, question_id: str) -> Random:
        material = prompting.render_state(request.state) + "|" + question_id
        seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
        return Random(seed)


_ENGINES: Dict[str, Callable[..., JevEngine]] = {"mock": MockEngine}


def register_engine(name: str, factory: Callable[..., JevEngine]) -> None:
    _ENGINES[name] = factory


def get_engine(name: str = "mock", **kwargs: Any) -> JevEngine:
    if name == "vlm":
        from .engine_vlm import VlmEngine  # heavy deps, imported lazily

        return VlmEngine(**kwargs)
    if name not in _ENGINES:
        raise ValueError(f"unknown engine {name!r}; available: {sorted(_ENGINES)}")
    return _ENGINES[name](**kwargs)
