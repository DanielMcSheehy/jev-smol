"""Offline tests for jev training utilities.

All tests are offline: no model weights are downloaded and nothing imports the
model classes (only tokenizer-free helpers, the synth generator, and the LoRA
regex). The trl import guard skips the module entirely when the training stack
is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("trl")  # needs torch/transformers/trl installed

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from jev import prompting  # noqa: E402
from jev.schema import JevRequest  # noqa: E402

from jev.train import sft as train_sft  # noqa: E402


def _load_make_synth_data():
    spec = importlib.util.spec_from_file_location(
        "make_synth_data", REPO_ROOT / "scripts" / "make_synth_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Synth generator
# --------------------------------------------------------------------------- #

def test_synth_records_validate_against_schema_and_roundtrip():
    mod = _load_make_synth_data()
    for split in ("train", "val"):
        for i in range(6):
            record = mod.make_example(i, split, seed=42, with_image=(i % 5 == 0))
            # Request validates against the wire schema.
            request = JevRequest(**record["request"])
            questions = record["request"]["questions"]
            # Target keys match questions exactly; value types match qtype.
            assert set(record["target"]) == set(questions)
            for qid, question in questions.items():
                value = record["target"][qid]
                if question["type"] == "noul":
                    assert isinstance(value, bool)
                elif question["type"] == "choice":
                    assert value in question["criteria"]
                else:
                    assert isinstance(value, int)
                    assert 0 <= value < len(question["criteria"])
            # assistant_text is the canonical serialization of the target, in
            # schema order, and parses back to the same answers.
            expected = prompting.assistant_answers_text(
                [(qid, questions[qid]["type"], record["target"][qid]) for qid in questions]
            )
            assert record["assistant_text"] == expected
            assert json.loads(record["assistant_text"]) == {
                qid: {"type": questions[qid]["type"],
                      questions[qid]["type"]: record["target"][qid]}
                for qid in questions
            }
            # 2-4 questions; image records carry a color question + image path.
            assert 2 <= len(questions) <= 4
            if record["image"] is not None:
                assert record["image"].startswith("images/")
                assert "color" in questions
            else:
                assert record["image"] is None


def test_synth_covers_all_question_types_and_is_deterministic():
    mod = _load_make_synth_data()
    records = [mod.make_example(i, "train", seed=42, with_image=False) for i in range(64)]
    types = {q["type"] for r in records for q in r["request"]["questions"].values()}
    assert types == {"noul", "choice", "score"}
    counts = {len(r["request"]["questions"]) for r in records}
    assert 2 <= min(counts) and max(counts) <= 4
    again = [mod.make_example(i, "train", seed=42, with_image=False) for i in range(64)]
    assert records == again
    # A different seed changes the data (checked across the whole split).
    other = [mod.make_example(i, "train", seed=7, with_image=False) for i in range(64)]
    assert any(o["request"]["state"] != r["request"]["state"]
               for o, r in zip(other, records))


def test_synth_main_writes_files(tmp_path):
    mod = _load_make_synth_data()
    mod.main(["--out", str(tmp_path), "--train", "8", "--val", "2", "--seed", "42"])
    for split, n in (("train", 8), ("val", 2)):
        lines = (tmp_path / f"{split}.jsonl").read_text().strip().splitlines()
        assert len(lines) == n
        for line in lines:
            record = json.loads(line)
            assert set(record) == {"id", "source", "license", "image", "request",
                                   "target", "assistant_text"}
            if record["image"] is not None:
                assert (tmp_path / record["image"]).is_file()
    # Image swatches are 96x96 solid palette colors.
    from PIL import Image

    records = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    with_image = [r for r in records if r["image"]]
    assert with_image, "expected at least one image example in 8 records"
    img = Image.open(tmp_path / with_image[0]["image"])
    assert img.size == (96, 96)
    assert img.convert("RGB").getpixel((0, 0)) in set(mod.PALETTE.values())


# --------------------------------------------------------------------------- #
# LoRA scoping
# --------------------------------------------------------------------------- #

def test_lora_regex_matches_language_modules_only():
    rx = train_sft.LORA_TARGET_MODULES
    language_names = [
        "model.text_model.layers.0.self_attn.q_proj",   # transformers 5.18 names
        "model.text_model.layers.31.mlp.gate_proj",
        "model.language_model.layers.0.self_attn.q_proj",  # older naming
        "model.language_model.layers.5.mlp.down_proj",
    ]
    vision_names = [
        "model.vision_encoder.blocks.0.attn.q_proj",
        "model.vision_model.encoder.layers.0.self_attn.q_proj",
        "model.vision_model.encoder.layers.3.mlp.fc1",
        "model.connector.mod_proj.0.linear1",
        "model.visual.blocks.0.attn.q_proj",
    ]
    for name in language_names:
        assert rx.fullmatch(name), name
    for name in vision_names:
        assert rx.fullmatch(name) is None, name
    # And the vision-marker assertion helper agrees.
    assert not any(train_sft._VISION_MARKER.search(n) for n in language_names)


# --------------------------------------------------------------------------- #
# JSONL -> prompt-completion mapping
# --------------------------------------------------------------------------- #

def _text_only_record():
    return {
        "id": "train-0001",
        "source": "synthetic",
        "license": "CC0-1.0",
        "image": None,
        "request": {
            "state": "The customer is happy with the fix.",
            "questions": {
                "sentiment": {
                    "type": "choice",
                    "instructions": "Classify sentiment.",
                    "criteria": {"negative": "Dissatisfied", "positive": "Satisfied"},
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "Immediate?",
                    "criteria": {"true": "Yes", "false": "No"},
                },
            },
        },
        "target": {"sentiment": "positive", "urgent": True},
        "assistant_text": '{"sentiment":{"type":"choice","choice":"positive"},'
                          '"urgent":{"type":"noul","noul":true}}',
    }


def test_record_to_prompt_completion_text_only():
    row = train_sft.record_to_prompt_completion(_text_only_record(), data_dir=REPO_ROOT)
    assert row["images"] == []
    prompt, completion = row["prompt"], row["completion"]
    assert [m["role"] for m in prompt] == ["system", "user"]
    assert prompt[0]["content"][0]["text"] == prompting.SYSTEM_PROMPT
    user_part = prompt[1]["content"]
    assert user_part[0]["type"] == "text"
    expected_text = prompting.render_user_text(
        _text_only_record()["request"]["state"],
        JevRequest(**_text_only_record()["request"]).questions,
    )
    assert user_part[0]["text"] == expected_text
    assert completion == [
        {"role": "assistant",
         "content": [{"type": "text", "text": _text_only_record()["assistant_text"]}]}
    ]


def test_record_to_prompt_completion_image_before_text(tmp_path):
    from PIL import Image

    mod = _load_make_synth_data()
    record = mod.make_example(0, "train", seed=42, with_image=True)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    filename = "train-0000.png"
    mod.write_swatch(images_dir / filename, mod.PALETTE[record["target"]["color"]])
    record["image"] = f"images/{filename}"

    row = train_sft.record_to_prompt_completion(record, data_dir=tmp_path)
    content = row["prompt"][1]["content"]
    # Image part comes BEFORE the text part.
    assert content[0]["type"] == "image"
    assert content[1]["type"] == "text"
    assert content[1]["text"].startswith("STATE:\n")
    assert len(row["images"]) == 1 and row["images"][0].size == (96, 96)
