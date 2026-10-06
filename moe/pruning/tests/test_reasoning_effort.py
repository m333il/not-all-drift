"""How hard the model may think is a measurement choice, so it has to be one.

The harness asked the harmony template for reasoning and let it pick its own
default, which is `medium`. Nobody chose that. The published runs pin `low`, and
on the same 2000 rows the same base model writes 11037 characters of `analysis`
at medium against their 374 and scores 0.6998 against their 0.6249 - a gap
bigger than most of the pruning effects the study is measuring.

It matters only where the model actually reasons. The six PEFT arms answer in
79 characters at either setting and reproduce the training manifest to four
decimals (prompt-m500: 0.7988 both sides, exactly). So these tests pin three
things: the effort reaches the template, it is recorded beside the number, and
a model that cannot reason is unaffected by it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrd_pruning.evaluate import GenerationConfig  # noqa: E402
from mrd_pruning.task import reasoning_on_kwargs, render_chat  # noqa: E402


class _Harmony:
    """A tokenizer whose template reads `reasoning_effort`, as harmony's does."""

    chat_template = "<|channel|>analysis reasoning_effort"

    def __init__(self):
        self.seen: dict = {}

    def apply_chat_template(self, messages, **kw):
        self.seen = dict(kw)
        return "RENDERED"


class _Plain:
    """A template that knows neither knob - Qwen3-Instruct does not reason."""

    chat_template = "{{ messages }}"

    def __init__(self):
        self.seen: dict = {}

    def apply_chat_template(self, messages, **kw):
        self.seen = dict(kw)
        return "RENDERED"


def test_effort_reaches_the_template():
    tok = _Harmony()
    render_chat(tok, "text", None, tokenize=False,
                reasoning=True, reasoning_effort="low")
    assert tok.seen.get("reasoning_effort") == "low"


def test_the_default_is_still_medium():
    """Unchanged for every call that does not ask, so old runs stay reproducible."""
    tok = _Harmony()
    render_chat(tok, "text", None, tokenize=False, reasoning=True)
    assert tok.seen.get("reasoning_effort") == "medium"


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_every_allowed_effort_is_passed_through(effort):
    tok = _Harmony()
    render_chat(tok, "text", None, tokenize=False,
                reasoning=True, reasoning_effort=effort)
    assert tok.seen.get("reasoning_effort") == effort


def test_a_model_that_cannot_reason_ignores_it():
    """Qwen3-Instruct: asking for an effort must not inject an unknown kwarg."""
    tok = _Plain()
    render_chat(tok, "text", None, tokenize=False,
                reasoning=True, reasoning_effort="low")
    assert "reasoning_effort" not in tok.seen


def test_suppressed_reasoning_ignores_the_argument():
    """Suppression is its own contract and must not be steerable from here.

    `reasoning_off_kwargs` already pins `low` and then opens the `final`
    channel, because harmony has no switch that turns reasoning off - `low`
    only shortens it. Letting this argument through would let a caller ask for
    `high` while believing reasoning was suppressed.
    """
    tok = _Harmony()
    render_chat(tok, "text", None, tokenize=False,
                reasoning=False, reasoning_effort="high")
    assert tok.seen.get("reasoning_effort") == "low"


def test_reasoning_on_kwargs_honours_the_argument():
    assert reasoning_on_kwargs(_Harmony(), "low") == {"reasoning_effort": "low"}


def test_the_config_records_the_effort():
    """A number without its effort is not reproducible - it goes in the summary."""
    cfg = GenerationConfig(reasoning=True, reasoning_effort="low")
    assert cfg.as_dict()["reasoning_effort"] == "low"


def test_a_non_reasoning_config_records_none():
    """`medium` beside `reasoning: false` would read as a setting that applied."""
    cfg = GenerationConfig(reasoning=False, reasoning_effort="medium")
    assert cfg.as_dict()["reasoning_effort"] is None
