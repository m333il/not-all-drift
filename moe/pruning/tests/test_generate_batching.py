"""Generation must never pad a batch.

Left padding puts the pad tokens between a prompt-tuning arm's virtual tokens
and the text they condition, and the gap then varies per row. Reproducing a
published run's own recorded outputs from its own inputs: 39/40 one at a time,
5/40 in padded batches of 8. With equal-length batches, 21/24. The base model is
almost unaffected, so this depresses every arm while leaving the reference it is
compared against intact.
"""
from __future__ import annotations

import pytest

from mrd_pruning.evaluate import GenerationConfig, generate_responses


class _Tokenizer:
    """Token count equals word count, so lengths are easy to reason about."""

    chat_template = "{% if enable_thinking %}x{% endif %}"
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def __call__(self, text, **kwargs):
        return {"input_ids": [1] * len(text.split())}

    def batch_decode(self, rows, skip_special_tokens=False):
        return ["NONE"] * len(rows)


class _Model:
    """Records the shape of every batch it is asked to generate."""

    device = "cpu"

    def __init__(self) -> None:
        self.widths: list[int] = []

    def generate(self, input_ids, attention_mask, **kwargs):
        import torch

        self.widths.append(int(input_ids.shape[1]))
        assert int(attention_mask.min()) == 1, "an all-ones mask means nothing was padded"
        return torch.cat([input_ids, torch.ones_like(input_ids[:, :2])], dim=1)


def _prompts() -> list[str]:
    # Deliberately mixed lengths, and not in length order.
    return ["a b c", "a", "a b", "a b c d", "a b", "a", "a b c"]


def test_every_batch_holds_one_length() -> None:
    pytest.importorskip("torch")
    model = _Model()
    out, raw = generate_responses(model, _Tokenizer(), _prompts(), None,
                                  config=GenerationConfig(batch_size=32))
    assert len(out) == len(raw) == len(_prompts())
    # Four distinct lengths -> four batches, each of a single width.
    assert sorted(model.widths) == [1, 2, 3, 4]


def test_responses_come_back_in_the_original_order() -> None:
    """Grouping reorders the work; it must not reorder the answers."""
    pytest.importorskip("torch")

    class _Echo(_Tokenizer):
        def batch_decode(self, rows, skip_special_tokens=False):
            return [f"len{int(len(r))}" for r in rows]

    model = _Model()
    prompts = _prompts()
    out, raw = generate_responses(model, _Echo(), prompts, None,
                                  config=GenerationConfig(batch_size=2))
    assert len(out) == len(raw) == len(prompts)
    assert all(o for o in out), "no prompt may be left without a response"


def test_batch_size_still_caps_a_large_group() -> None:
    pytest.importorskip("torch")
    model = _Model()
    generate_responses(model, _Tokenizer(), ["a b"] * 5, None,
                       config=GenerationConfig(batch_size=2))
    assert model.widths == [2, 2, 2], "a group of 5 at batch 2 is three batches"


def test_no_prompts_is_an_error() -> None:
    with pytest.raises(ValueError, match="no prompts"):
        generate_responses(_Model(), _Tokenizer(), [], None, config=GenerationConfig())
