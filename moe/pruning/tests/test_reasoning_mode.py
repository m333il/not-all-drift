"""Reasoning can be left on, as the model's native mode.

gpt-oss has no non-reasoning setting: the default run suppresses reasoning by
opening the ``final`` channel in the prompt itself. ``reasoning=True`` gives
that choice back to the model - the prompt stops before any channel marker, and
the template is asked *for* reasoning rather than against it.
"""

from __future__ import annotations

import pytest

from mrd_pruning.task import (
    FINAL_CHANNEL,
    render_chat,
    reasoning_off_kwargs,
    reasoning_on_kwargs,
)


class FakeTokenizer:
    """Minimal stand-in: the code decides by template text, not model name."""

    def __init__(self, template: str) -> None:
        self.chat_template = template
        self.seen_kwargs: dict[str, object] = {}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kw):
        self.seen_kwargs = dict(kw)
        body = "".join(m["content"] for m in messages)
        # The date sits on its own line, as harmony writes it - the pinning
        # regex is line-anchored and would not match it mid-line.
        text = f"<sys>\nCurrent date: 2020-01-01\n{body}<|start|>assistant"
        return list(text.encode()) if tokenize else text

    def __call__(self, text, **kw):
        return {"input_ids": list(text.encode())}


HARMONY = "template with <|channel|> and reasoning_effort"
QWEN = "template with enable_thinking"
PLAIN = "a template with no knobs at all"


def test_harmony_suppressed_prompt_opens_the_final_channel():
    tok = FakeTokenizer(HARMONY)
    text = render_chat(tok, "u", None, tokenize=False)
    assert text.endswith(FINAL_CHANNEL)


def test_harmony_reasoning_prompt_stops_before_the_channel():
    tok = FakeTokenizer(HARMONY)
    text = render_chat(tok, "u", None, tokenize=False, reasoning=True)
    assert FINAL_CHANNEL not in text
    assert text.endswith("<|start|>assistant")


def test_reasoning_asks_the_template_for_effort():
    tok = FakeTokenizer(HARMONY)
    render_chat(tok, "u", None, tokenize=False, reasoning=True)
    assert tok.seen_kwargs == {"reasoning_effort": "medium"}


def test_suppressed_asks_the_template_against_it():
    tok = FakeTokenizer(HARMONY)
    render_chat(tok, "u", None, tokenize=False)
    assert tok.seen_kwargs == {"reasoning_effort": "low"}


def test_qwen_knob_is_flipped_both_ways():
    tok = FakeTokenizer(QWEN)
    assert reasoning_on_kwargs(tok) == {"enable_thinking": True}
    assert reasoning_off_kwargs(tok) == {"enable_thinking": False}


def test_template_without_knobs_gets_nothing():
    assert reasoning_on_kwargs(FakeTokenizer(PLAIN)) == {}


def test_date_is_still_pinned_in_reasoning_mode():
    tok = FakeTokenizer(HARMONY)
    text = render_chat(tok, "u", None, tokenize=False, reasoning=True)
    assert "Current date: 2020-01-01" not in text
    assert "Current date: 2026-09-07" in text


@pytest.mark.parametrize("reasoning", [False, True])
def test_tokenised_and_text_renders_agree(reasoning):
    tok = FakeTokenizer(HARMONY)
    text = render_chat(tok, "u", None, tokenize=False, reasoning=reasoning)
    ids = render_chat(tok, "u", None, tokenize=True, reasoning=reasoning)
    assert ids == list(text.encode())
