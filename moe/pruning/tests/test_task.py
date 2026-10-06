"""Parser, metric and prompt rendering, checked against real published responses."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mrd_pruning.task import (
    FINAL_CHANNEL, ParseResult, answer_text, f1_sample, parse_response,
    predicted_labels, render_chat,
    render_user_prompt, score_batch,
)

FIXTURE = Path(__file__).parent / "data" / "published_responses.jsonl"


def load_fixture() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


def test_strict_policy_reproduces_published_scores() -> None:
    """The published *_score fields were produced by the strict NONE rule.

    Reproducing them exactly is what proves the parser and metric here are the
    same ones, so any later difference is a deliberate policy change rather
    than a silent reimplementation drift.
    """
    rows = load_fixture()
    assert rows, "fixture is empty"
    for row in rows:
        parsed = parse_response(row["response"])
        pred = predicted_labels(parsed, "strict")
        assert f1_sample(pred, row["gold"]) == pytest.approx(row["published_score"]), row


def test_lenient_policy_differs_only_on_mixed_none() -> None:
    rows = load_fixture()
    differing = [
        row for row in rows
        if f1_sample(predicted_labels(parse_response(row["response"]), "lenient"), row["gold"])
        != pytest.approx(row["published_score"])
    ]
    assert differing, "fixture must contain the ', NONE' case the policies disagree on"
    for row in differing:
        assert parse_response(row["response"]).mixed_none


def test_parse_flags() -> None:
    assert parse_response("NONE") == ParseResult((), True, False, False, False)
    assert parse_response("toxicity, insult").labels == ("toxicity", "insult")
    assert parse_response("insult, toxicity").labels == ("toxicity", "insult")  # canonical order
    assert parse_response("toxicity, NONE").mixed_none
    assert parse_response("<think> let me").unparsable
    assert parse_response("toxicity\nand also insult").labels == ("toxicity",)  # first line only


def test_f1_empty_aware() -> None:
    assert f1_sample([], []) == 1.0
    assert f1_sample([], ["toxicity"]) == 0.0
    assert f1_sample(["toxicity"], []) == 0.0
    assert f1_sample(["toxicity", "insult"], ["toxicity"]) == pytest.approx(2 / 3)


def test_score_batch_reports_failure_counters() -> None:
    summary, rows = score_batch(
        ["NONE", "toxicity", "<think> hmm", "toxicity, NONE"],
        [[], ["toxicity"], ["insult"], ["toxicity"]],
    )
    assert summary.n == 4
    assert summary.unparsable_rate == 0.25
    assert summary.mixed_none_rate == 0.25
    assert summary.empty_pred_rate == 0.5  # "NONE" and the unparsable one
    assert len(rows) == 4


def test_score_batch_refuses_empty_input() -> None:
    with pytest.raises(ValueError, match="zero examples"):
        score_batch([], [])


class _RecordingTokenizer:
    """A template is matched by its text, so the fake has to carry real text."""

    chat_template = "{% if enable_thinking %}<think>{% endif %}"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return "RENDERED"


class _HarmonyTokenizer(_RecordingTokenizer):
    """gpt-oss: no enable_thinking anywhere, reasoning_effort defaults to medium."""

    chat_template = '{% if reasoning_effort is not defined %}{% set reasoning_effort = "medium" %}{% endif %}'


def test_render_chat_always_disables_thinking() -> None:
    """Qwen3 turns reasoning on by default and it eats the whole token budget."""
    tokenizer = _RecordingTokenizer()
    render_chat(tokenizer, "user text", "system text")
    assert tokenizer.calls[0]["enable_thinking"] is False
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert "reasoning_effort" not in tokenizer.calls[0], (
        "passing a knob this template does not know would raise on strict templates"
    )


def test_render_chat_uses_reasoning_effort_on_harmony_templates() -> None:
    """gpt-oss ignores enable_thinking entirely.

    Passing it and nothing else left the model reasoning at medium effort, which
    spent the whole 24-token budget in the analysis channel and produced 82.5%
    unparsable responses from an unmodified model.
    """
    tokenizer = _HarmonyTokenizer()
    render_chat(tokenizer, "user text", "system text")
    assert tokenizer.calls[0]["reasoning_effort"] == "low"
    assert "enable_thinking" not in tokenizer.calls[0]


def test_render_chat_warns_when_template_knows_neither_knob(caplog) -> None:
    class _Silent(_RecordingTokenizer):
        chat_template = "no knobs here"

    tokenizer = _Silent()
    with caplog.at_level("WARNING"):
        render_chat(tokenizer, "user text", None)
    assert "neither enable_thinking nor reasoning_effort" in caplog.text

    # Once per template. render_chat runs for every one of 2000 prompts, and a
    # per-call warning buries the run log.
    caplog.clear()
    with caplog.at_level("WARNING"):
        render_chat(tokenizer, "another text", None)
    assert caplog.text == ""


def test_render_chat_omits_system_turn_when_none() -> None:
    tokenizer = _RecordingTokenizer()
    render_chat(tokenizer, "user text", None)
    roles = [m["role"] for m in tokenizer.calls[0]["messages"]]
    assert roles == ["user"], "an empty system turn is not the same as no system turn"


def test_render_chat_refuses_without_template() -> None:
    class _Bare:
        chat_template = None

    with pytest.raises(ValueError, match="chat_template"):
        render_chat(_Bare(), "user", "system")


def test_user_prompt_matches_published_format() -> None:
    prompt = render_user_prompt("hello")
    assert prompt.startswith("Text:\nhello\n\nLabels (use exact names):")
    assert prompt.endswith("Answer:")


# --- harmony channels -------------------------------------------------------
# The strings below are verbatim decodes from gpt-oss-20b on the project's own
# test set (11-09-2026), not invented examples.

_ANALYSIS_ONLY_RAW = (
    "<|channel|>analysis<|message|>We need to label. The text: \"Close the mosque "
    "for a year or two.\" This is hateful towards a religious group. It's a call to"
)
_ANALYSIS_ONLY_CLEAN = (
    "analysisWe need to label. The text: \"Close the mosque for a year or two.\" "
    "This is hateful towards a religious group. It's a call to"
)
_FINISHED_RAW = (
    "<|channel|>analysis<|message|>We need to label. The text contains political "
    "content, but no toxicity, obscene, threat, insult, identity_attack. It's "
    "political. So NONE.<|end|><|start|>assistant<|channel|>final<|message|>NONE<|return|>"
)
_FINISHED_CLEAN = (
    "analysisWe need to label. The text contains political content, but no "
    "toxicity, obscene, threat, insult, identity_attack. It's political. So "
    "NONE.assistantfinalNONE"
)


def test_answer_text_passes_plain_responses_through() -> None:
    """A model without channels answers in the clean decode directly."""
    assert answer_text("toxicity, insult", "toxicity, insult") == "toxicity, insult"


def test_answer_text_takes_only_the_final_channel() -> None:
    assert answer_text(_FINISHED_RAW, _FINISHED_CLEAN) == "NONE"


def test_answer_text_is_empty_when_the_budget_ran_out_mid_analysis() -> None:
    """No final channel means no answer - unparsable, not a guess at the prose."""
    assert answer_text(_ANALYSIS_ONLY_RAW, _ANALYSIS_ONLY_CLEAN) == ""


def test_reasoning_prose_is_never_harvested_for_labels() -> None:
    """The failure this whole path exists to prevent.

    The model reasons about the label names in a comma-separated list, so the
    old parser read three labels out of an unfinished thought and reported them
    as a confident prediction.
    """
    naive = parse_response(_FINISHED_CLEAN)
    assert set(naive.labels) == {"obscene", "threat", "insult"}, (
        "guard is pointless if the raw decode no longer reproduces the bug"
    )
    assert not naive.unparsable, "the bug was silent, not a visible parse failure"

    fixed = parse_response(answer_text(_FINISHED_RAW, _FINISHED_CLEAN))
    assert fixed.labels == ()
    assert fixed.saw_none

    cut = parse_response(answer_text(_ANALYSIS_ONLY_RAW, _ANALYSIS_ONLY_CLEAN))
    assert cut.unparsable and cut.labels == ()


def test_answer_prefix_is_stripped() -> None:
    """The prompt ends with "Answer:" and an untuned model echoes it.

    153 of 2000 gpt-oss base responses were exactly this, and every one of them
    was a correct answer being discarded as unreadable.
    """
    assert parse_response("Answer: NONE").saw_none
    assert not parse_response("Answer: NONE").unparsable
    assert parse_response("Answer: insult").labels == ("insult",)
    assert parse_response("answer:toxicity, insult").labels == ("toxicity", "insult")


def test_only_that_one_prefix_is_stripped() -> None:
    """A looser rule would start inventing answers the model never gave."""
    assert parse_response("The answer: insult").unparsable
    assert parse_response("My best guess is insult").unparsable


def test_harmony_prompt_opens_the_final_channel() -> None:
    """gpt-oss must answer immediately, like Qwen does.

    There is no switch that turns harmony's reasoning off - reasoning_effort
    only shortens it - so the generation prompt is left at the point where the
    answer starts. Without this the model reasons first, which needed a
    512-token budget and still truncated 12% of an arm's responses to zero.
    """
    tokenizer = _HarmonyTokenizer()
    tokenizer.chat_template += "<|channel|>"
    out = render_chat(tokenizer, "user text", "system text", tokenize=False)
    assert out.endswith(FINAL_CHANNEL)


def test_plain_prompt_is_left_alone() -> None:
    """Qwen's rendering must stay byte-identical to every earlier measurement."""
    tokenizer = _RecordingTokenizer()
    assert render_chat(tokenizer, "user text", "system text", tokenize=False) == "RENDERED"


def test_an_echoed_label_roster_is_not_an_answer() -> None:
    """The prompt's own roster names every label; reading it awards all five.

    gpt-oss under the GEPA prompt reproduced the instruction in 22% of its
    final-channel responses, and each one was scored as a confident five-label
    prediction: 492 invented `threat` calls against 72 true ones, precision
    0.48. An echo carries no answer and must read as unparsable.
    """
    echo = ("Labels (use exact names): toxicity, obscene, threat, insult, "
            "identity_attack\nText:\nJohnny - sadly ...")
    parsed = parse_response(echo)
    assert parsed.labels == ()
    assert parsed.unparsable is True
    assert parsed.saw_labels is False


def test_a_real_five_label_answer_still_counts() -> None:
    """The guard keys off the roster's wording, not off breadth of prediction."""
    parsed = parse_response("toxicity, obscene, threat, insult, identity_attack")
    assert len(parsed.labels) == 5
    assert parsed.unparsable is False


def test_an_answer_wearing_the_roster_heading_is_still_an_answer() -> None:
    """153 gepa-n500 rows answered like this, none followed by `Text:`."""
    parsed = parse_response("Labels (use exact names): toxicity, insult")
    assert parsed.labels == ("toxicity", "insult")
    assert parsed.unparsable is False


def test_the_guard_needs_the_whole_roster_not_just_its_phrase() -> None:
    """Narrow on purpose: only the full five-name roster reads as an echo."""
    from mrd_pruning.task import _echoes_the_label_list

    roster = ("Labels (use exact names): toxicity, obscene, threat, insult, "
              "identity_attack")
    assert _echoes_the_label_list(roster) is True
    assert _echoes_the_label_list("Labels (use exact names): insult") is False
    assert _echoes_the_label_list("toxicity, insult") is False
    # Order matters: the roster is echoed as written, an answer is not.
    shuffled = ("Labels (use exact names): insult, toxicity, obscene, threat, "
                "identity_attack")
    assert _echoes_the_label_list(shuffled) is False
