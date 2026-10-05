"""VLM-backed engine: SmolVLM2-500M-Video-Instruct scoring options by log-prob.

CLEF-style option scoring, adapted to multi-token labels — NO free-form
generation. For every question we teacher-force the exact assistant prefix
produced by :class:`jev.prompting.AnswerJSONBuilder` and score each allowed
option label by the mean log-probability of the label's tokens at the value
position. Scores are softmaxed per question and passed to
:func:`jev.schema.build_answer`.

Chat-template finding (verified against the installed SmolVLM2 chat template,
transformers 5.18.0): the Jinja template renders ``...<end_of_utterance>\\n`` for
a user turn and only appends the assistant header when
``add_generation_prompt=True``, in which case it appends the literal
``Assistant:`` — tokenized as ``Ass``, ``istant``, ``:`` with NO trailing
space/newline. Without that flag the sequence ends inside the user turn and any
continuation would be scored as user text, so ``add_generation_prompt=True`` is
required and the result DOES end with the assistant header (no extra tokens are
appended here).

Also verified: in transformers 5.18 the SmolVLM processor raises
``TypeError: got multiple values for keyword argument 'images'`` if PIL images
are passed both as the ``images=`` kwarg and as ``{"type": "image"}`` content
parts, so this engine embeds the PIL images directly in the message content
parts (``{"type": "image", "image": pil_image}``), which returns
``input_ids``/``attention_mask``/``pixel_values``/``pixel_attention_mask`` with
the ``<image>`` placeholder already expanded (image_seq_len=64 per tile).
"""

from __future__ import annotations

import base64
import binascii
import io
import threading
from typing import Any, Dict, List, Optional, Tuple

from . import prompting
from .api.errors import ImageError, UnsupportedError
from .schema import DATA_URL_RE, JevRequest, JevResponse, Usage, build_answer

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MiB decoded per image
MAX_IMAGE_PIXELS = 16_000_000  # 16 Mpx per image
MAX_LONGEST_SIDE = 512  # resize before processing
_PIL_FORMATS = {"PNG", "JPEG", "WEBP"}


def _resolve_device(device: str):
    import torch

    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def decode_image(item: Any):
    """Decode one request image (data URL or {content_type, base64}) to PIL.

    Enforces: png/jpeg/webp only, <= 4 MiB decoded bytes, <= 16 Mpx, and
    resizes the longest side to <= 512 px. Raises :class:`ImageError` on any
    violation.
    """
    from PIL import Image, UnidentifiedImageError

    if isinstance(item, str):
        match = DATA_URL_RE.match(item)
        if not match:
            raise ImageError(
                "image strings must be data URLs of the form data:image/(png|jpeg|webp);base64,..."
            )
        content_type = match.group(1)
        payload = item.split(",", 1)[1] if "," in item else ""
    else:
        content_type = getattr(item, "content_type", None)
        payload = getattr(item, "base64", None)
        if content_type not in ("image/png", "image/jpeg", "image/webp"):
            raise ImageError(f"unsupported image content_type {content_type!r}; use png, jpeg, or webp")
        if not isinstance(payload, str):
            raise ImageError("image base64 payload must be a string")

    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ImageError(f"invalid base64 image payload: {exc}") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError(f"decoded image is {len(raw)} bytes; limit is {MAX_IMAGE_BYTES} bytes")

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageError(f"could not decode image: {exc}") from exc

    if img.format not in _PIL_FORMATS:
        raise ImageError(f"decoded image format {img.format!r} is not png/jpeg/webp")
    if content_type == "image/png" and img.format != "PNG":
        raise ImageError("declared image/png but decoded image is not a PNG")
    if content_type == "image/jpeg" and img.format != "JPEG":
        raise ImageError("declared image/jpeg but decoded image is not a JPEG")
    if content_type == "image/webp" and img.format != "WEBP":
        raise ImageError("declared image/webp but decoded image is not a WebP")

    width, height = img.size
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageError(
            f"image is {width}x{height} ({width * height} px); limit is {MAX_IMAGE_PIXELS} px"
        )

    longest = max(width, height)
    if longest > MAX_LONGEST_SIDE:
        scale = MAX_LONGEST_SIDE / longest
        img = img.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS
        )
    return img


def option_token_spans(tokenizer, cont_text: str, value_texts: List[str]):
    """Tokenize each option's full continuation jointly with the shared prefix.

    Training tokenized the whole completion string at once, so within-answer
    BPE merges (e.g. ``"choice":"billing"`` keeping the JSON quotes fused to
    the id) only appear when ``cont_text + value_text`` is tokenized as one
    string. The discriminative span for each option is everything past the
    longest common prefix across all options. Returns a list of
    ``(seq_ids, span_ids)``.
    """
    seqs = [
        tokenizer(cont_text + vt, add_special_tokens=False)["input_ids"]
        for vt in value_texts
    ]
    lcp = 0
    for i in range(min(len(s) for s in seqs)):
        if all(s[i] == seqs[0][i] for s in seqs):
            lcp = i + 1
        else:
            break
    spans = [s[lcp:] for s in seqs]
    if any(not span for span in spans):
        raise ValueError(
            f"degenerate option tokenization (empty discriminative span) for {value_texts!r}"
        )
    return list(zip(seqs, spans))


class VlmEngine:
    """Decision engine on SmolVLM2; model + processor load lazily.

    One singleton instance per (model, device, model_id) config, guarded by a
    class-level threading.Lock so concurrent requests trigger a single load.
    """

    _instances: Dict[Tuple[str, str, Optional[str]], "VlmEngine"] = {}
    _load_lock = threading.Lock()

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        model_id: Optional[str] = None,
    ) -> None:
        self.model = model
        self.device_name = device
        self.model_id = model_id  # reported response name; None -> per-request default
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._loaded = False
        VlmEngine._instances[(model, device, model_id)] = self

    @classmethod
    def instance(
        cls,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        model_id: Optional[str] = None,
    ) -> "VlmEngine":
        return cls._instances.get((model, device, model_id)) or cls(model, device, model_id)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with VlmEngine._load_lock:
            if self._loaded:
                return
            import torch
            from transformers import AutoProcessor

            try:
                from transformers import AutoModelForImageTextToText as ModelCls
            except ImportError:  # older transformers
                from transformers import AutoModelForVision2Seq as ModelCls

            device = _resolve_device(self.device_name)
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            processor = AutoProcessor.from_pretrained(self.model)
            try:
                model = ModelCls.from_pretrained(self.model, dtype=dtype, attn_implementation="sdpa")
            except Exception:
                model = ModelCls.from_pretrained(self.model, dtype=dtype, attn_implementation="eager")
            self._model = model.to(device).eval()
            self._processor = processor
            self._tokenizer = processor.tokenizer
            self._device = device
            self._loaded = True

    # ------------------------------------------------------------------ #
    # Decision
    # ------------------------------------------------------------------ #

    def decide(self, request: JevRequest) -> JevResponse:
        import torch

        if request.videos:
            raise UnsupportedError("this model does not process videos; send images instead")

        pil_images = [decode_image(item) for item in (request.images or [])]

        self._ensure_loaded()
        device = self._device

        # 1. Build the prompt once: system + user text + (expanded) image
        #    tokens + assistant header. Ends with "Assistant:" (see module
        #    docstring). The system message MUST match training, which used
        #    {"role": "system", SYSTEM_PROMPT} (verified: omitting it shifts
        #    the decision distribution measurably).
        user_text = prompting.render_user_text(request.state, request.questions)
        content: List[Dict[str, Any]] = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": user_text})
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompting.SYSTEM_PROMPT}],
            },
            {"role": "user", "content": content},
        ]
        prefix = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        prefix_ids = prefix["input_ids"][0].to(device)
        prefix_mask = prefix["attention_mask"][0].to(device)
        pixel_values = prefix.get("pixel_values")
        pixel_attention_mask = prefix.get("pixel_attention_mask")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        if pixel_attention_mask is not None:
            pixel_attention_mask = pixel_attention_mask.to(device)

        # 2. Sequential per-question scoring of every option. Each option's
        #    continuation text is tokenized JOINTLY with the shared prefix
        #    (cont + value_json) so the token boundaries match training exactly
        #    (e.g. choice values keep their JSON quotes); the discriminative
        #    span is everything past the longest common prefix.
        builder = prompting.AnswerJSONBuilder()
        answers: Dict[str, Any] = {}
        for qid, question in request.questions.items():
            cont = builder.prefix_for_next(qid, question.type)
            values = prompting.canonical_option_values(question)
            value_texts = [prompting.value_json(question.type, v) for v in values]
            spans = option_token_spans(self._tokenizer, cont, value_texts)

            scores: List[torch.Tensor] = []
            for seq_ids, span_ids in spans:
                seq_t = torch.tensor(seq_ids, dtype=torch.long, device=device)
                span_t = torch.tensor(span_ids, dtype=torch.long, device=device)
                scores.append(
                    self._score_span(
                        prefix_ids, prefix_mask, seq_t, span_t,
                        pixel_values, pixel_attention_mask,
                    )
                )

            probs = torch.softmax(torch.stack(scores), dim=-1).tolist()
            keys = [prompting.option_label_text(question.type, v) for v in values]
            answers[qid] = build_answer(question, dict(zip(keys, probs)))
            builder.add(qid, question.type, values[int(torch.tensor(probs).argmax())])

        return JevResponse(
            model=self.model_id or request.model or "jev-mini",
            answers=answers,
            usage=Usage(input_tokens=int(prefix_ids.shape[0]), output_tokens=0),
        )

    def _score_span(self, prefix_ids, prefix_mask, seq_ids, span_ids,
                    pixel_values=None, pixel_attention_mask=None):
        """Mean log-prob of ``span_ids`` teacher-forced after prefix+seq."""
        import torch

        full_ids = torch.cat([prefix_ids, seq_ids]).unsqueeze(0)
        tail_len = seq_ids.shape[0]
        full_mask = torch.cat(
            [prefix_mask, torch.ones(tail_len, dtype=prefix_mask.dtype, device=prefix_mask.device)]
        ).unsqueeze(0)
        forward: Dict[str, Any] = {"input_ids": full_ids, "attention_mask": full_mask}
        if pixel_values is not None:
            forward["pixel_values"] = pixel_values
            forward["pixel_attention_mask"] = pixel_attention_mask
        with torch.no_grad():
            logits = self._model(**forward).logits
        n = span_ids.shape[0]
        # logits[:, i] predicts token i+1; the span occupies the last n slots.
        predicting = logits[0, -(n + 1):-1, :].float()
        logprobs = torch.log_softmax(predicting, dim=-1)
        return logprobs[torch.arange(n, device=logits.device), span_ids].mean()
