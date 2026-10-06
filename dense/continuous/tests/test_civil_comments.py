from __future__ import annotations

import numpy as np
import pytest

from prompt_optimization.civil_comments import (
    BINARY_PROMPT_TEMPLATE,
    BINARY_YES_NO_PROMPT_TEMPLATE,
    CIVIL_COMMENTS_LABELS,
    CIVIL_COMMENTS_PROMPT_TEMPLATE,
    binary_target,
    binary_target_text,
    canonical_target,
    canonicalize_labels,
    compute_binary_metrics,
    compute_multilabel_metrics,
    parse_binary_prediction,
    parse_prediction,
    parse_prediction_strict,
    prepended_virtual_tokens,
    sha256_directory,
    prompt_template,
    split_file_name,
    use_gradient_checkpointing,
    validate_rows,
)


def test_fixed_prompt_is_exact() -> None:
    assert (
        CIVIL_COMMENTS_PROMPT_TEMPLATE
        == """You are a text classifier. Classify the text using the provided labels. More than one label may apply.
Labels (use exact names): toxicity, obscene, threat, insult, identity_attack, sexual_explicit
Text:
{text}
Return every applicable label in the order listed, separated by commas. If no label applies, return exactly NONE. Return no other text.
Answer:"""
    )


def test_targets_and_predictions_use_canonical_order() -> None:
    assert canonicalize_labels(["insult", "toxicity"]) == ("toxicity", "insult")
    assert canonical_target(["insult", "toxicity"]) == "toxicity, insult"
    assert canonical_target([]) == "NONE"
    assert parse_prediction("toxicity, insult\n") == ("toxicity", "insult")
    assert parse_prediction("insult, toxicity") == ("toxicity", "insult")
    assert parse_prediction("toxicity, insult, toxicity") == ("toxicity", "insult")
    assert parse_prediction("NONE") == ()
    assert parse_prediction("none") is None
    assert parse_prediction("toxicity.") is None


def test_strict_parser_enforces_order_and_unique_labels() -> None:
    assert parse_prediction_strict("toxicity, insult\n") == ("toxicity", "insult")
    assert parse_prediction_strict("insult, toxicity") is None
    assert parse_prediction_strict("toxicity, insult, toxicity") is None
    assert parse_prediction_strict("NONE") == ()


def test_v2_label_contract_controls_prompt_parser_and_metrics() -> None:
    labels = CIVIL_COMMENTS_LABELS[:-1]
    rendered = prompt_template(labels)
    assert "identity_attack" in rendered
    assert "sexual_explicit" not in rendered
    assert parse_prediction(
        "insult, toxicity",
        allowed_labels=labels,
    ) == ("toxicity", "insult")
    assert parse_prediction("sexual_explicit", allowed_labels=labels) is None
    metrics = compute_multilabel_metrics(
        [("toxicity",), ()],
        [("toxicity",), ()],
        labels=labels,
    )
    assert metrics["samples_f1"] == 1.0
    assert "f1_sexual_explicit" not in metrics


def test_binary_contract_uses_safe_toxic_targets_and_strict_parser() -> None:
    assert prompt_template(CIVIL_COMMENTS_LABELS[:-1], setup="binary") == BINARY_PROMPT_TEMPLATE
    assert binary_target({"binary_label": "safe"}) == "safe"
    assert binary_target({"binary_label": "toxic"}) == "toxic"
    assert parse_binary_prediction("toxic\n") == "toxic"
    assert parse_binary_prediction("TOXIC") is None
    assert parse_binary_prediction("toxic, safe") is None
    metrics = compute_binary_metrics(
        ["safe", "safe", "toxic", "toxic"],
        ["safe", None, "toxic", "safe"],
    )
    assert metrics["invalid_rate"] == pytest.approx(0.25)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["f1_toxic"] == pytest.approx(2 / 3)


def test_binary_yes_no_contract_maps_to_internal_safe_toxic_labels() -> None:
    assert (
        prompt_template(
            ("toxicity",),
            setup="binary",
            binary_output_format="yes-no",
        )
        == BINARY_YES_NO_PROMPT_TEMPLATE
    )
    assert binary_target_text(
        {"binary_label": "toxic"}, output_format="yes-no"
    ) == "Yes"
    assert binary_target_text(
        {"binary_label": "safe"}, output_format="yes-no"
    ) == "No"
    assert parse_binary_prediction("Yes", output_format="yes-no") == "toxic"
    assert parse_binary_prediction(" No\n", output_format="yes-no") == "safe"
    assert parse_binary_prediction("yes", output_format="yes-no") is None
    assert parse_binary_prediction("Yes.", output_format="yes-no") is None


def test_binary_rows_require_consistent_binary_label() -> None:
    validate_rows(
        [
            {"id": "a", "text": "x", "labels": [], "binary_label": "safe"},
            {
                "id": "b",
                "text": "y",
                "labels": ["toxicity"],
                "binary_label": "toxic",
            },
        ],
        setup="binary",
    )
    with pytest.raises(ValueError, match="conflicts with labels"):
        validate_rows(
            [{"id": "a", "text": "x", "labels": [], "binary_label": "toxic"}],
            setup="binary",
        )


def test_multilabel_metrics_include_format_none_and_each_label() -> None:
    targets = [(), ("toxicity",), ("toxicity", "insult")]
    predictions = [(), ("toxicity",), None]
    metrics = compute_multilabel_metrics(targets, predictions)
    assert metrics["invalid_rate"] == pytest.approx(1 / 3)
    assert metrics["none_f1"] == pytest.approx(2 / 3)
    assert metrics["micro_f1"] == pytest.approx(0.5)
    assert metrics["samples_f1"] == pytest.approx(2 / 3)
    assert {f"f1_{label}" for label in CIVIL_COMMENTS_LABELS} <= set(metrics)
    assert np.isfinite(list(metrics.values())).all()


def test_split_names_and_hidden_state_offsets() -> None:
    assert split_file_name("optimizer_train", split_seed=42, train_samples=200) == (
        "optimizer_train_seed42_n200.jsonl"
    )
    assert split_file_name(
        "optimizer_val", split_seed=42, train_samples=1000
    ) == "optimizer_val_seed42_n1000.jsonl"
    assert split_file_name("probe_train_subset", split_seed=3) == "probe_train_seed3.jsonl"
    assert prepended_virtual_tokens("PROMPT_TUNING", 50) == 50
    assert prepended_virtual_tokens("PREFIX_TUNING", 50) == 0
    assert prepended_virtual_tokens(None, 0) == 0
    assert use_gradient_checkpointing("prompt_tuning") is True
    assert use_gradient_checkpointing("prefix_tuning") is False


def test_validate_rows_rejects_duplicates_and_unknown_labels() -> None:
    validate_rows([{"id": "a", "text": "x", "labels": ["toxicity"]}], expected_size=1)
    with pytest.raises(ValueError, match="Duplicate id"):
        validate_rows(
            [
                {"id": "a", "text": "x", "labels": []},
                {"id": "a", "text": "y", "labels": []},
            ]
        )
    with pytest.raises(ValueError, match="Unknown Civil Comments labels"):
        validate_rows([{"id": "a", "text": "x", "labels": ["severe_toxicity"]}])


def test_directory_hash_includes_names_and_contents(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "adapter.safetensors").write_bytes(b"same")
    (right / "adapter.safetensors").write_bytes(b"same")
    assert sha256_directory(left) == sha256_directory(right)
    (right / "adapter.safetensors").write_bytes(b"changed")
    assert sha256_directory(left) != sha256_directory(right)
