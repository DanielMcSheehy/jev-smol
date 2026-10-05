"""jev — a small, embedded text+image decision model.

Implements a Jev/CLEF ("System One") compatible decision API on top of a
fine-tuned SmolVLM2-500M-Video-Instruct backbone.

Core modules:
    schema    — pydantic wire models for requests/responses (+ answer assembly)
    prompting — the joint-schema prompt renderer (STATE + SCHEMA FIELDS) and the
                assistant-side answers-JSON builder shared by training + inference
    engine    — engine interface, mock engine, and the engine registry
"""

__version__ = "0.1.0"

from . import prompting
from .engine import JevEngine, MockEngine, get_engine
from .schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    ImageData,
    JevRequest,
    JevResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
    build_answer,
)

__all__ = [
    "__version__",
    "prompting",
    "JevEngine",
    "MockEngine",
    "get_engine",
    "JevRequest",
    "JevResponse",
    "Question",
    "NoulQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "NoulAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "ImageData",
    "Usage",
    "build_answer",
]
