"""A damaged channel header must not throw the answer away.

Pruning breaks the scaffolding before it breaks the content. At −75% gpt-oss
writes `<|channel|>final message<|message|>toxicity, insult` - the same final
channel, its name with a word stuck to it - and an exact match on the header
reads that as "this response has no final channel" and scores a correct answer
as silence. On prefix-m100 at −75% that was 143 rows of 2000, and the loss is
systematic: it grows with the pruning level, so it bends the curve downward
exactly where the conclusion is read off it.

The opposite error is worse and these tests pin it too: the `analysis` channel
holds the model reasoning *about the label names*, and reading it as the answer
harvests labels the model never predicted.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrd_pruning.task import answer_text  # noqa: E402


def test_the_plain_final_channel_still_reads():
    raw = "<|channel|>final<|message|>toxicity, insult<|return|>"
    assert answer_text(raw, "x").strip() == "toxicity, insult"


def test_a_channel_named_final_message_is_the_final_channel():
    """The case measured on the pruned runs."""
    raw = "<|channel|>final message<|message|>toxicity, insult<|return|>"
    assert answer_text(raw, "x").strip() == "toxicity, insult"


def test_analysis_is_never_read_as_the_answer():
    """The reasoning lists the roster; harvesting it invents predictions."""
    raw = ("<|channel|>analysis<|message|>Is it toxicity, obscene, threat? "
           "I think not.<|end|><|channel|>final<|message|>NONE<|return|>")
    assert answer_text(raw, "x").strip() == "NONE"


def test_a_response_that_never_reached_final_still_yields_nothing():
    """Truncated mid-analysis: there is no answer, and inventing one is worse."""
    raw = "<|channel|>analysis<|message|>We need to decide whether toxicity"
    assert answer_text(raw, "x") == ""


def test_a_name_that_merely_contains_final_does_not_match():
    """`semifinal` is not the final channel, and the pattern must not take it.

    The match is anchored at the name's start, so a channel whose name happens
    to end in `final` is left alone.
    """
    raw = "<|channel|>semifinal<|message|>toxicity<|return|>"
    assert answer_text(raw, "x") == ""


def test_the_last_final_channel_wins():
    raw = ("<|channel|>final<|message|>obscene<|end|>"
           "<|channel|>final message<|message|>insult<|return|>")
    assert answer_text(raw, "x").strip() == "insult"


def test_a_model_without_channels_is_untouched():
    """Qwen answers in plain text; the channel logic must not touch it."""
    assert answer_text("toxicity, insult<|im_end|>", "toxicity, insult") == "toxicity, insult"
