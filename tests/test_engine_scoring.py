"""Tests for the engine's option-scoring token alignment (no model needed)."""

import pytest

from jev import prompting

torch = pytest.importorskip("torch", reason="torch not installed")
transformers = pytest.importorskip("transformers", reason="transformers not installed")

from jev.engine_vlm import option_token_spans  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")


def test_choice_spans_discriminative_and_training_exact(tokenizer):
    cont = '{"team":{"type":"choice","choice":'
    values = ['"billing"', '"technical"', '"sales"']
    out = option_token_spans(tokenizer, cont, values)
    for (seq, span), vt in zip(out, values):
        # Joint tokenization reproduces the training string exactly; the span
        # is the suffix past the shared LCP (the opening quote may fuse into
        # the shared ':"' token — that is desired, it is constant across
        # options and therefore non-discriminative).
        assert tokenizer.decode(seq) == cont + vt
        assert seq[: len(seq) - len(span)] == out[0][0][: len(seq) - len(span)]
    decoded = [tokenizer.decode(s) for _, s in out]
    assert len(set(decoded)) == len(decoded)


def test_noul_and_score_spans_are_single_aligned_tokens(tokenizer):
    cont = '{"urgent":{"type":"noul","noul":'
    out = option_token_spans(tokenizer, cont, ["true", "false"])
    assert [tokenizer.decode(s) for _, s in out] == ["true", "false"]

    cont = '{"severity":{"type":"score","score":'
    out = option_token_spans(tokenizer, cont, ["0", "1", "2", "3"])
    assert [tokenizer.decode(s) for _, s in out] == ["0", "1", "2", "3"]


def test_joint_tokenization_matches_training_string(tokenizer):
    """Every option sequence must decode to exactly cont + value literal —
    i.e. the tokens the model saw during training for this completion."""
    cont = '{"team":{"type":"choice","choice":'
    out = option_token_spans(tokenizer, cont, ['"billing"', '"technical"'])
    assert tokenizer.decode(out[0][0]) == cont + '"billing"'
    assert tokenizer.decode(out[1][0]) == cont + '"technical"'


def test_prefix_for_next_matches_assistant_text(tokenizer):
    """Engine prefixes must literally prefix the trained completion string."""
    b = prompting.AnswerJSONBuilder()
    b.add("urgent", "noul", True)
    prefix = b.prefix_for_next("team", "choice")
    full = prompting.assistant_answers_text(
        [("urgent", "noul", True), ("team", "choice", "billing")]
    )
    assert full.startswith(prefix)
