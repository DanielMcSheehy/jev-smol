"""Pydantic wire models for the Jev / CLEF "System One" decision API.

Request (POST /v1/systemone)::

    {
      "model": "jev-mini",
      "state": "any string or JSON value describing the situation",
      "questions": {
        "urgent":   {"type": "noul",   "instructions": "...",
                     "criteria": {"true": "desc", "false": "desc"}},
        "team":     {"type": "choice", "instructions": "...",
                     "criteria": {"billing": "desc", "technical": "desc"}},
        "severity": {"type": "score",  "instructions": "...",
                     "criteria": ["No impact", "Minor", "Major", "Critical"]}
      },
      "images": [ "data:image/png;base64,..." | {"content_type": "image/png", "base64": "..."} ]
    }

Response::

    {
      "model": "jev-mini",
      "answers": {
        "urgent":   {"type": "noul", "noul": 0.95},
        "team":     {"type": "choice", "choice": "billing",
                     "probabilities": {"billing": 0.88, "technical": 0.12},
                     "confidence": 0.88},
        "severity": {"type": "score", "score": 1.05,
                     "legend": {"0": "No impact", "1": "Minor", "2": "Major", "3": "Critical"},
                     "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05, "3": 0.0},
                     "confidence": 0.95}
      },
      "usage": {"input_tokens": 296, "output_tokens": 0}
    }

Rules (mirroring the published Jev/SystemOne + CLEF schemas):
  * 1..64 questions; question ids match [A-Za-z0-9_.-]{1,100}.
  * noul   — optional criteria {true, false} descriptions.
  * choice — criteria maps option-id -> description, 2..255 options,
             option ids match [A-Za-z0-9_.-]{1,100}.
  * score  — criteria is an ordered list of 2..10 level descriptions (indexed from 0).
  * images — CLEF extension: up to 4 PNG/JPEG/WebP inputs as data URLs or
             {content_type, base64} objects. Byte/pixel limits are enforced at
             decode time (see jev.api / jev.engine_vlm), not here.
  * videos — accepted at the schema level so clients get a clean 400 instead of
             a 422; this model does not process them.
  * score answers: `score` is the level expectation sum(i * p_i); `legend` echoes
    the level descriptions; probabilities are keyed "0".."N-1".
  * floats are rounded to 4 decimals.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QUESTION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
OPTION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
DATA_URL_RE = re.compile(r"^data:(image/(?:png|jpeg|webp));base64,[A-Za-z0-9+/=\s]+$")

MAX_QUESTIONS = 64
MIN_CHOICE_OPTIONS = 2
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
MAX_IMAGES = 4
ALLOWED_IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp")


def round4(x: float) -> float:
    return round(float(x), 4)


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: Optional[str] = None
    false: Optional[str] = None


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    instructions: Optional[str] = Field(default=None, max_length=2000)
    criteria: Optional[NoulCriteria] = None


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    instructions: Optional[str] = Field(default=None, max_length=2000)
    criteria: Dict[str, str]

    @field_validator("criteria")
    @classmethod
    def _validate_criteria(cls, v: Dict[str, str]) -> Dict[str, str]:
        if not (MIN_CHOICE_OPTIONS <= len(v) <= MAX_CHOICE_OPTIONS):
            raise ValueError(
                f"choice criteria must have between {MIN_CHOICE_OPTIONS} and "
                f"{MAX_CHOICE_OPTIONS} options, got {len(v)}"
            )
        for option_id, description in v.items():
            if not OPTION_ID_RE.match(option_id):
                raise ValueError(
                    f"invalid choice option id {option_id!r}: must match "
                    f"{OPTION_ID_RE.pattern}"
                )
            if not description or not description.strip():
                raise ValueError(f"choice option {option_id!r} needs a non-empty description")
        return v


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"]
    instructions: Optional[str] = Field(default=None, max_length=2000)
    criteria: List[str] = Field(min_length=MIN_SCORE_LEVELS, max_length=MAX_SCORE_LEVELS)

    @field_validator("criteria")
    @classmethod
    def _validate_levels(cls, v: List[str]) -> List[str]:
        for i, level in enumerate(v):
            if not level or not level.strip():
                raise ValueError(f"score level {i} must be a non-empty description")
        return v


Question = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")
]


class ImageData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_type: Literal["image/png", "image/jpeg", "image/webp"]
    base64: str


ImageInput = Union[str, ImageData]


class JevRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = None
    state: Any  # required; any string or JSON value
    questions: Dict[str, Question]
    images: Optional[List[ImageInput]] = None
    videos: Optional[List[Any]] = None

    @field_validator("state")
    @classmethod
    def _state_required(cls, v: Any) -> Any:
        if v is None:
            raise ValueError("state is required")
        return v

    @field_validator("images", mode="before")
    @classmethod
    def _validate_images(cls, v):
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("images must be a list")
        if len(v) > MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images are allowed, got {len(v)}")
        for item in v:
            if isinstance(item, str) and not DATA_URL_RE.match(item):
                raise ValueError(
                    "image strings must be data URLs of the form "
                    "data:image/(png|jpeg|webp);base64,..."
                )
        return v

    @model_validator(mode="after")
    def _validate_questions(self) -> "JevRequest":
        if not (1 <= len(self.questions) <= MAX_QUESTIONS):
            raise ValueError(
                f"questions must contain between 1 and {MAX_QUESTIONS} entries, "
                f"got {len(self.questions)}"
            )
        for qid in self.questions:
            if not QUESTION_ID_RE.match(qid):
                raise ValueError(
                    f"invalid question id {qid!r}: must match {QUESTION_ID_RE.pattern}"
                )
        return self


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0


class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    noul: float = Field(ge=0.0, le=1.0)  # P(true)


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    choice: str
    probabilities: Dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate(self) -> "ChoiceAnswer":
        if self.choice not in self.probabilities:
            raise ValueError("chosen option must appear in probabilities")
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"probabilities must sum to ~1.0, got {total:.4f}")
        return self


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"]
    score: float  # level expectation: sum(i * p_i)
    legend: Dict[str, str]
    probabilities: Dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate(self) -> "ScoreAnswer":
        n = len(self.probabilities)
        keys = set(self.probabilities)
        expected = {str(i) for i in range(n)}
        if n < MIN_SCORE_LEVELS or keys != expected:
            raise ValueError(
                'score probabilities must be keyed "0".."N-1" with N >= 2'
            )
        if set(self.legend) != expected:
            raise ValueError("legend must echo one description per probability level")
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"probabilities must sum to ~1.0, got {total:.4f}")
        if not (0.0 <= self.score <= n - 1 + 1e-6):
            raise ValueError(f"score must be within [0, {n - 1}], got {self.score}")
        return self


Answer = Annotated[
    Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")
]


class JevResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    answers: Dict[str, Answer]
    usage: Usage = Usage()


# --------------------------------------------------------------------------- #
# Answer assembly (shared by every engine)
# --------------------------------------------------------------------------- #


def build_answer(question: Question, probs: Dict[str, float]) -> Answer:
    """Assemble the API answer for one question from an option distribution.

    Keys of ``probs`` are canonical option labels:
      noul   -> "true" / "false"
      choice -> option ids
      score  -> "0" .. "N-1"

    The distribution is normalized (defensively) and all emitted floats are
    rounded to 4 decimals, matching the CLEF reference behaviour.
    """
    if question.type == "noul":
        p_true = max(0.0, float(probs.get("true", 0.0)))
        p_false = max(0.0, float(probs.get("false", 0.0)))
        total = p_true + p_false
        noul = p_true / total if total > 0 else 0.5
        return NoulAnswer(type="noul", noul=round4(noul))

    p = {k: round4(v) for k, v in probs.items()}
    total = sum(p.values())
    if total <= 0:
        n = len(p) or 1
        p = {k: round4(1.0 / n) for k in p}
    else:
        p = {k: round4(v / total) for k, v in p.items()}

    if question.type == "choice":
        chosen = max(p.items(), key=lambda kv: kv[1])[0]
        return ChoiceAnswer(
            type="choice", choice=chosen, probabilities=p, confidence=p[chosen]
        )

    # score
    score = round4(sum(int(k) * v for k, v in p.items()))
    legend = {str(i): desc for i, desc in enumerate(question.criteria)}
    confidence = max(p.values()) if p else 0.0
    return ScoreAnswer(
        type="score",
        score=score,
        legend=legend,
        probabilities=p,
        confidence=round4(confidence),
    )
