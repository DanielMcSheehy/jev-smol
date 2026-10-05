"""Config-driven LoRA SFT for jev on SmolVLM2-500M-Video-Instruct.

Implementation decisions (verified against the INSTALLED stack:
trl==1.14.1, transformers==5.18.0, peft==0.21.2, torch==2.14.1):

* Assistant-loss masking: the SmolVLM2 chat template has NO ``{% generation %}``
  markers (verified: ``"{% generation %}" not in template``), so TRL's
  ``assistant_only_loss`` cannot produce assistant masks. TRL 1.14.1 instead
  natively supports the conversational *prompt-completion* format for VLMs:
  rows of ``{"prompt": [...], "completion": [...], "images": [...]}`` are
  handled by ``DataCollatorForVisionLanguageModeling._collate_prompt_completion``,
  which tokenizes prompt and completion separately (injecting pixel values)
  and builds a ``completion_mask``; SFTTrainer auto-sets
  ``completion_only_loss=True`` for prompt-completion datasets, so the loss is
  computed on assistant tokens (including the ``<end_of_utterance>`` eos) only.
  We therefore use prompt-completion format. Verified end-to-end: supervised
  tokens for a sample row are exactly `` <assistant json><end_of_utterance>``.

* System role: the installed SmolVLM2 template renders system messages as
  ``<|im_start|>System: ...<end_of_utterance>\n`` (verified), so the system
  prompt is passed as a proper ``system`` message rather than prepended to the
  user text. (TRL's prompt-completion path requires the last prompt message to
  be a ``user`` turn, which holds here.)

* Image tokens: with the stock SmolVLM2 processor, ANY image (even 96x96) is
  split into 16 patches + 1 global tile => 17 x image_seq_len(64) = 1088 image
  tokens, not ~128. The token-length guard therefore counts the EXACT expanded
  length via ``processor.apply_chat_template``; the default threshold is 1600
  but ``configs/train_smoke.yaml`` raises it to 2600 so image examples survive.
  ``SFTConfig(max_length=None)`` is kept so image tokens are never truncated
  (the tokenizer's truncation_side is "left" and would eat image tokens).

* LoRA scoping: transformers 5.18.0 names the language tower
  ``model.text_model`` (older docs say ``language_model``), the vision tower
  ``model.vision_model`` (SigLIP, which ALSO has q/k/v_proj) and the pixel
  shuffle bridge ``model.connector``. The default target regex matches
  projection names under either ``text_model``/``language_model``; real match
  names are printed once at startup and an assertion guarantees zero matches
  inside vision/connector modules.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from jev import prompting  # noqa: E402
from jev.schema import JevRequest  # noqa: E402

DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# Language-model projections only. PEFT applies ``re.fullmatch`` per module
# name, so this must match the FULL module path. Covers both the transformers
# v5 spelling (model.text_model.*) and the older one (model.language_model.*).
LORA_TARGET_MODULES = re.compile(
    r"model\.(?:text_model|language_model)\."
    r".*(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
# Anything matched by the target regex that also touches vision/connector is a
# scoping bug (the SigLIP tower shares q/k/v_proj names).
_VISION_MARKER = re.compile(r"vision|connector|visual", re.IGNORECASE)

DEFAULT_MAX_TOKEN_LEN = 1600


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Data: unified JSONL -> TRL prompt-completion VLM rows
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_to_prompt_completion(record: Dict[str, Any], data_dir: Path) -> Dict[str, Any]:
    """Convert one unified JSONL record into a TRL conversational
    prompt-completion row: image content parts come BEFORE the text part
    (per jev.prompting's CLEF media-token convention), the system prompt is a
    system message, and the completion is the pre-serialized assistant JSON.
    """
    request = JevRequest(**record["request"])
    user_text = prompting.render_user_text(request.state, request.questions)

    content: List[Dict[str, str]] = []
    images = []
    if record.get("image"):
        from PIL import Image

        image_path = Path(data_dir) / record["image"]
        images.append(Image.open(image_path).convert("RGB"))
        content.append({"type": "image"})
    content.append({"type": "text", "text": user_text})

    prompt = [
        {"role": "system", "content": [{"type": "text", "text": prompting.SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    completion = [
        {"role": "assistant", "content": [{"type": "text", "text": record["assistant_text"]}]}
    ]
    return {"prompt": prompt, "completion": completion, "images": images}


def build_dataset(
    jsonl_path: Path,
    processor: Any,
    max_token_len: int = DEFAULT_MAX_TOKEN_LEN,
    max_examples: Optional[int] = None,
) -> Dataset:
    """Load JSONL, map to prompt-completion rows, drop over-length examples
    (exact expanded token count) and return a ``datasets.Dataset``."""
    records = load_jsonl(jsonl_path)
    if max_examples is not None:
        records = records[:max_examples]
    data_dir = Path(jsonl_path).resolve().parent

    rows = [record_to_prompt_completion(r, data_dir) for r in records]

    kept, dropped = [], 0
    for row in rows:
        if count_tokens(row, processor) <= max_token_len:
            kept.append(row)
        else:
            dropped += 1
    if dropped:
        print(f"[data] dropped {dropped}/{len(rows)} examples over {max_token_len} tokens")

    return Dataset.from_list(kept)


def count_tokens(row: Dict[str, Any], processor: Any) -> int:
    """Exact expanded token count: template the full prompt+completion with the
    processor so the single <image> placeholder expands to its real token count."""
    messages = row["prompt"] + row["completion"]
    if row["images"]:
        # Embed PIL images directly in the content so apply_chat_template
        # expands image tokens during tokenization.
        messages = _embed_images(messages, row["images"])
    out = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    ids = out["input_ids"] if isinstance(out, dict) else out
    while isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def _embed_images(messages: List[Dict[str, Any]], images: List[Any]) -> List[Dict[str, Any]]:
    it = iter(images)
    out = []
    for message in messages:
        content = []
        for part in message["content"]:
            if part.get("type") == "image":
                content.append({"type": "image", "image": next(it)})
            else:
                content.append(part)
        out.append({**message, "content": content})
    return out


# --------------------------------------------------------------------------- #
# Model / LoRA
# --------------------------------------------------------------------------- #

def resolve_device(choice: str) -> str:
    if choice != "auto":
        if choice not in ("cpu", "mps", "cuda"):
            raise ValueError(f"unknown device {choice!r}; expected auto|cpu|mps|cuda")
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(base_model: str, device: str):
    """Load the VLM with the right dtype per device (cuda -> bf16, else fp32)
    and sdpa attention with an eager fallback."""
    from transformers import AutoModelForImageTextToText

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    kwargs: Dict[str, Any] = {"dtype": dtype}
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            base_model, attn_implementation="sdpa", **kwargs
        )
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"[model] sdpa unavailable ({exc}); falling back to eager attention")
        model = AutoModelForImageTextToText.from_pretrained(
            base_model, attn_implementation="eager", **kwargs
        )
    model.to(device)
    model.config.use_cache = False  # correct for training
    return model


def apply_lora(model: Any, lora_cfg: Dict[str, Any]):
    """Wrap the model with LoRA scoped to language-model projections.

    Prints the matched module names once and asserts that nothing inside the
    vision tower / connector was touched (the SigLIP tower shares q/k/v_proj
    names with the LLM).
    """
    from peft import LoraConfig, get_peft_model

    named = [name for name, _ in model.named_modules()]
    matches = sorted(n for n in named if LORA_TARGET_MODULES.fullmatch(n))
    vision_hits = [n for n in matches if _VISION_MARKER.search(n)]
    if not matches:
        raise RuntimeError(
            "LoRA target regex matched no modules; inspect "
            "[n for n, _ in model.named_modules()] and fix LORA_TARGET_MODULES"
        )
    if vision_hits:
        raise AssertionError(
            f"LoRA regex leaked into vision/connector modules: {vision_hits[:5]}"
        )
    print(f"[lora] {len(matches)} language-module matches, 0 vision/connector matches")
    print(f"[lora] first matches: {matches[:4]} ... last: {matches[-2:]}")

    pattern = lora_cfg.get("target_modules", LORA_TARGET_MODULES.pattern)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 8)),
            lora_dropout=float(lora_cfg.get("dropout", 0.1)),
            target_modules=pattern,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    peft_model.print_trainable_parameters()
    return peft_model, matches


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def build_sft_config(
    cfg: Dict[str, Any],
    out_dir: Path,
    max_steps: Optional[int],
    device: str,
    num_train_examples: int = 0,
):
    from trl import SFTConfig

    eval_enabled = bool(cfg.get("eval", False))
    steps = max_steps if max_steps is not None else cfg.get("max_steps")
    epochs = cfg.get("num_train_epochs")

    # transformers v5 removed warmup_ratio; convert it to warmup_steps from the
    # expected total optimizer steps.
    effective_batch = int(cfg.get("per_device_train_batch_size", 1)) * int(
        cfg.get("gradient_accumulation_steps", 1)
    )
    if steps is not None:
        total_steps = steps
    elif epochs is not None and num_train_examples:
        import math

        total_steps = math.ceil(num_train_examples / effective_batch * float(epochs))
    else:
        total_steps = 0
    warmup_ratio = float(cfg.get("warmup_ratio", 0.0))
    warmup_steps = int(round(warmup_ratio * total_steps)) if total_steps else 0

    return SFTConfig(
        output_dir=str(out_dir),
        max_length=None,  # never truncate: image tokens must survive
        max_steps=(int(steps) if steps else -1),  # -1 = epoch-driven (v5 rejects None)
        # transformers v5 requires a numeric value here; it is ignored when
        # max_steps > 0 (default 3.0 matches TrainingArguments' own default).
        num_train_epochs=(float(epochs) if epochs is not None else 3.0),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        warmup_steps=warmup_steps,
        logging_steps=int(cfg.get("logging_steps", 1)),
        save_strategy=cfg.get("save_strategy", "no"),
        save_steps=int(cfg["save_steps"]) if cfg.get("save_steps") else 500,
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=int(cfg["eval_steps"]) if cfg.get("eval_steps") else 500,
        bf16=(device == "cuda"),
        seed=int(cfg.get("seed", 42)),
        report_to=[],
    )


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Config-driven LoRA SFT for jev")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="cap the number of training examples")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override max_steps from the config")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--out", type=Path, default=None,
                        help="override the output directory from the config")
    args = parser.parse_args(argv)

    from transformers import AutoProcessor
    from trl import SFTTrainer

    cfg = load_config(args.config)
    base_model = cfg.get("base_model", DEFAULT_BASE_MODEL)
    device = resolve_device(args.device)
    out_dir = args.out or Path(cfg.get("out", "runs/jev-sft"))
    out_dir.mkdir(parents=True, exist_ok=True)
    max_token_len = int(cfg.get("max_token_len", DEFAULT_MAX_TOKEN_LEN))

    print(f"[sft] base={base_model} device={device} out={out_dir}")

    processor = AutoProcessor.from_pretrained(base_model)
    model = load_model(base_model, device)
    peft_model, lora_matches = apply_lora(model, cfg.get("lora", {}))

    train_dataset = build_dataset(
        Path(cfg["train_data"]), processor, max_token_len, args.max_examples
    )
    eval_dataset = None
    if cfg.get("eval") and cfg.get("val_data"):
        eval_dataset = build_dataset(Path(cfg["val_data"]), processor, max_token_len)

    sft_config = build_sft_config(
        cfg, out_dir, args.max_steps, device, num_train_examples=len(train_dataset)
    )
    trainer = SFTTrainer(
        model=peft_model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )
    result = trainer.train()

    # Save the final adapter + processor copy so <out>/adapter is loadable
    # standalone (PeftModel.from_pretrained + AutoProcessor).
    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    summary = {
        "steps": int(trainer.state.global_step),
        "examples": len(train_dataset),
        "final_loss": float(result.metrics.get("train_loss", float("nan"))),
        "device": device,
        "dtype": "bfloat16" if device == "cuda" else "float32",
        "lora_matches_language": len(lora_matches),
        "lora_matches_vision": 0,
    }
    summary_path = out_dir / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[sft] wrote {summary_path}: {json.dumps(summary)}")
    return summary


if __name__ == "__main__":
    main()
