from __future__ import annotations

import numpy as np
import pytest

from interpretability_gepa.config import DecisionConfig
from interpretability_gepa.datasets import CIVIL_LABELS
from interpretability_gepa.decisions import classify_world
from interpretability_gepa.errors import ParseError
from interpretability_gepa.evaluation import build_prediction
from interpretability_gepa.metrics import (
    Interval,
    backend_parity,
    evaluate_multilabel,
    paired_bootstrap_difference,
    parse_gain_fraction,
    per_input_effects,
    per_input_recovery_summary,
    recovery,
)
from interpretability_gepa.prompts import SEED_INSTRUCTIONS, parse_labels, render_messages


def test_civil_comments_seed_prompt_matches_frozen_contract_exactly() -> None:
    rendered = render_messages(SEED_INSTRUCTIONS["civil_comments"], CIVIL_LABELS, "{text}")
    assert rendered.messages == (
        {
            "role": "user",
            "content": (
                "You are a text classifier. Classify the text using the provided labels. "
                "More than one label may apply.\n"
                "Labels (use exact names): toxicity, obscene, threat, insult, "
                "identity_attack\n"
                "Text:\n{text}\n"
                "Return every applicable label in the order listed, separated by commas. "
                "If no label applies, return exactly NONE. Return no other text.\n"
                "Answer:"
            ),
        },
    )


def test_civil_comments_parser_accepts_canonical_comma_labels_and_none() -> None:
    assert parse_labels("toxicity, insult, identity_attack", CIVIL_LABELS) == (
        "toxicity",
        "insult",
        "identity_attack",
    )
    assert parse_labels("NONE", CIVIL_LABELS) == ()


@pytest.mark.parametrize(
    "response",
    [
        '["toxicity", "insult"]',
        "insult, toxicity",
        "toxicity, toxicity",
        "toxicity, unknown",
        "none",
        "toxicity because the text is hostile",
        "```toxicity```",
        "<think>analysis</think> toxicity",
    ],
)
def test_civil_comments_parser_rejects_noncanonical_responses(response: str) -> None:
    with pytest.raises(ParseError):
        parse_labels(response, CIVIL_LABELS)


def test_prompt_and_strict_parser() -> None:
    prompt = render_messages("Classify.", ("a", "b"), "hello")
    assert "INPUT_TEXT:\nhello" in prompt.messages[0]["content"]
    assert parse_labels('["b", "a", "a"]', ("a", "b")) == ("a", "b")
    assert parse_labels(" NONE ", ("a", "b")) == ()
    with pytest.raises(ParseError, match="unknown"):
        parse_labels('["c"]', ("a", "b"))
    with pytest.raises(ParseError, match="thinking"):
        parse_labels("<think>x</think> NONE", ("a",))


def test_metrics_include_empty_gold() -> None:
    result = evaluate_multilabel([("a",), (), ("b",)], [("a",), (), ()], ("a", "b"))
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 1
    assert 0 <= result["f1_samples"] <= 1
    assert result["parse_failure_rate"] == 0.0


def test_unparseable_response_is_not_rewarded_on_empty_gold() -> None:
    gold = [(), ()]
    predicted = [(), ()]
    rewarded = evaluate_multilabel(gold, predicted, ("a", "b"), [True, True])
    penalised = evaluate_multilabel(gold, predicted, ("a", "b"), [True, False])
    assert rewarded["f1_samples"] == 1.0
    assert rewarded["no_label_accuracy"] == 1.0
    assert rewarded["subset_accuracy"] == 1.0
    assert penalised["f1_samples"] == 0.5
    assert penalised["no_label_accuracy"] == 0.5
    assert penalised["subset_accuracy"] == 0.5
    assert penalised["parse_failure_rate"] == 0.5


def test_parse_ok_length_is_validated() -> None:
    with pytest.raises(ValueError, match="one flag per example"):
        evaluate_multilabel([()], [()], ("a",), [True, False])


def test_lenient_parsing_accepts_only_the_ordering_violation() -> None:
    assert parse_labels("insult, toxicity", CIVIL_LABELS, enforce_order=False) == (
        "toxicity",
        "insult",
    )
    # Everything else the strict parser rejects stays rejected.
    for response in ("toxicity, toxicity", "toxicity, unknown", "<think>x</think> toxicity"):
        with pytest.raises(ParseError):
            parse_labels(response, CIVIL_LABELS, enforce_order=False)


def test_build_prediction_flags_order_violation_without_losing_the_labels() -> None:
    violation = build_prediction("id", "C_seed", "insult, toxicity", CIVIL_LABELS)
    assert not violation.parse_ok
    assert violation.labels == ()
    assert violation.order_violation
    assert violation.lenient_labels == ("toxicity", "insult")

    clean = build_prediction("id", "C_seed", "toxicity, insult", CIVIL_LABELS)
    assert clean.parse_ok and not clean.order_violation
    assert clean.labels == clean.lenient_labels == ("toxicity", "insult")

    broken = build_prediction("id", "C_seed", "toxicity, unknown", CIVIL_LABELS)
    assert not broken.parse_ok and not broken.order_violation
    assert broken.lenient_labels == ()


def test_parse_gain_fraction_attributes_a_format_only_gain() -> None:
    # The gain disappears once ordering stops being scored, so it was formatting.
    assert parse_gain_fraction(
        seed_strict=0.5, optimized_strict=0.7, seed_lenient=0.7, optimized_lenient=0.7
    ) == pytest.approx(1.0)
    # The gain survives relaxed parsing, so it was the decision.
    assert parse_gain_fraction(
        seed_strict=0.5, optimized_strict=0.7, seed_lenient=0.5, optimized_lenient=0.7
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="too small"):
        parse_gain_fraction(
            seed_strict=0.5, optimized_strict=0.505, seed_lenient=0.5, optimized_lenient=0.5
        )


def test_paired_bootstrap_and_recovery_guard() -> None:
    interval = paired_bootstrap_difference(np.ones(10), np.zeros(10), samples=100, seed=1)
    assert interval == Interval(1.0, 1.0, 1.0)
    assert recovery(0.5, 0.6, 0.7) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        recovery(0.5, 0.51, 0.51)


def test_world_rules() -> None:
    config = DecisionConfig()
    world_b = classify_world(
        probe_change=Interval(0.03, 0.01, 0.05),
        seed_gap=Interval(0, 0, 0),
        target_gap=Interval(0, 0, 0),
        elicitation_fraction=0,
        config=config,
    )
    assert world_b.world == "B"
    world_a = classify_world(
        probe_change=Interval(0, -0.01, 0.01),
        seed_gap=Interval(0.1, 0.07, 0.13),
        target_gap=Interval(0.03, 0, 0.06),
        elicitation_fraction=0.7,
        config=config,
    )
    assert world_a.world == "A"


def test_per_input_recovery_preserves_undefined_cases() -> None:
    values = per_input_effects(np.array([0.0, 1.0]), np.array([0.5, 1.0]), np.array([1.0, 1.0]))
    assert values["recovery"][0] == 0.5
    assert np.isnan(values["recovery"][1])

    summary = per_input_recovery_summary(
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, 0.0, 0.5]),
        np.array([1.0, 0.0, 1.0]),
    )
    assert summary["fixed_recall"] == 1.0
    assert summary["degraded_recall"] == 1.0
    assert summary["new_errors"] == 1


def test_backend_parity_uses_shared_metric() -> None:
    passed, difference = backend_parity(
        [("a",), ("b",)], [("a",), ("b",)], [("a",), ("b",)], ("a", "b")
    )
    assert passed and difference == 0
