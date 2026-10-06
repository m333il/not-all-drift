from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, roc_auc_score


def multilabel_matrix(rows: Sequence[Sequence[str]], labels: Sequence[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(rows), len(labels)), dtype=np.int8)
    for row_index, row in enumerate(rows):
        for label in row:
            matrix[row_index, index[label]] = 1
    return matrix


def evaluate_multilabel(
    gold: Sequence[Sequence[str]],
    predicted: Sequence[Sequence[str]],
    labels: Sequence[str],
    parse_ok: Sequence[bool] | None = None,
) -> dict[str, float | list[float | None] | None]:
    y_true = multilabel_matrix(gold, labels)
    y_pred = multilabel_matrix(predicted, labels)
    # Unparseable responses score zero instead of counting as an empty prediction.
    if parse_ok is None:
        ok = np.ones(len(y_true), dtype=bool)
    else:
        ok = np.asarray(parse_ok, dtype=bool)
        if ok.shape != (len(y_true),):
            raise ValueError("parse_ok must carry one flag per example")
    aurocs: list[float | None] = []
    for index in range(len(labels)):
        column = y_true[:, index]
        aurocs.append(
            None if len(np.unique(column)) < 2 else float(roc_auc_score(column, y_pred[:, index]))
        )
    empty_gold = y_true.sum(axis=1) == 0
    headline_rows = empty_aware_sample_f1_rows(y_true, y_pred)
    headline_rows[~ok] = 0.0
    subset_match = np.logical_and((y_true == y_pred).all(axis=1), ok)
    return {
        # Empty prediction for an empty gold set counts as a perfect match.
        "f1_samples": float(headline_rows.mean()),
        "f1_samples_legacy": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        # Split the headline into positive rows and empty rows.
        "f1_samples_positive": (
            float(sample_f1_rows(y_true, y_pred)[~empty_gold].mean())
            if np.any(~empty_gold)
            else None
        ),
        "no_label_accuracy": (
            float(np.logical_and(y_pred[empty_gold].sum(axis=1) == 0, ok[empty_gold]).mean())
            if np.any(empty_gold)
            else None
        ),
        "parse_failure_rate": float(1.0 - ok.mean()),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(subset_match.mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "avg_predictions": float(y_pred.sum(axis=1).mean()),
        "false_positives": float(np.logical_and(y_pred == 1, y_true == 0).sum()),
        "false_negatives": float(np.logical_and(y_pred == 0, y_true == 1).sum()),
        "per_label_auroc": aurocs,
    }


def sample_f1_rows(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(y_true, y_pred).sum(axis=1)
    denominator = y_true.sum(axis=1) + y_pred.sum(axis=1)
    return np.divide(
        2 * intersection,
        denominator,
        out=np.zeros_like(intersection, dtype=float),
        where=denominator != 0,
    )


def empty_aware_sample_f1_rows(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return per-example F1 while treating a correct empty prediction as one."""
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("multilabel arrays must have the same [examples, labels] shape")
    scores = sample_f1_rows(y_true, y_pred)
    empty_match = np.logical_and(y_true.sum(axis=1) == 0, y_pred.sum(axis=1) == 0)
    scores[empty_match] = 1.0
    return scores


def evaluate_multilabel_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | list[float | None] | None]:
    """Evaluate aligned multilabel arrays with explicit empty-row semantics."""
    if y_true.shape != y_pred.shape or y_true.shape != probabilities.shape:
        raise ValueError("targets, predictions, and probabilities must share one shape")
    if y_true.ndim != 2:
        raise ValueError("multilabel arrays must have shape [examples, labels]")
    per_label_auroc: list[float | None] = []
    for label in range(y_true.shape[1]):
        column = y_true[:, label]
        per_label_auroc.append(
            None
            if len(np.unique(column)) < 2
            else float(roc_auc_score(column, probabilities[:, label]))
        )
    empty_gold = y_true.sum(axis=1) == 0
    positive_gold = ~empty_gold
    legacy_rows = sample_f1_rows(y_true, y_pred)
    empty_aware_rows = empty_aware_sample_f1_rows(y_true, y_pred)
    return {
        "f1_samples_legacy": float(legacy_rows.mean()),
        "f1_samples_empty_aware": float(empty_aware_rows.mean()),
        "f1_samples_positive": (
            float(legacy_rows[positive_gold].mean()) if np.any(positive_gold) else None
        ),
        "no_label_accuracy": (
            float((y_pred[empty_gold].sum(axis=1) == 0).mean()) if np.any(empty_gold) else None
        ),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "avg_predictions": float(y_pred.sum(axis=1).mean()),
        "false_positives": float(np.logical_and(y_pred == 1, y_true == 0).sum()),
        "false_negatives": float(np.logical_and(y_pred == 0, y_true == 1).sum()),
        "per_label_auroc": per_label_auroc,
    }


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def paired_bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    samples: int = 1000,
    seed: int = 0,
) -> Interval:
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("paired inputs must be equal-length vectors")
    differences = first - second
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(first), size=(samples, len(first)))
    boot = differences[indices].mean(axis=1)
    return Interval(
        float(differences.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    )


def recovery(
    seed: float, intervention: float, target: float, *, minimum_gain: float = 0.02
) -> float:
    denominator = target - seed
    if denominator < minimum_gain:
        raise ValueError("target improvement is too small for a stable recovery fraction")
    return (intervention - seed) / denominator


def parse_gain_fraction(
    *,
    seed_strict: float,
    optimized_strict: float,
    seed_lenient: float,
    optimized_lenient: float,
    minimum_gain: float = 0.02,
) -> float:
    """Share of an optimizer gain that disappears under the set parser."""
    strict_gain = optimized_strict - seed_strict
    if strict_gain < minimum_gain:
        raise ValueError("strict gain is too small to attribute between format and decision")
    return (strict_gain - (optimized_lenient - seed_lenient)) / strict_gain


def per_input_effects(
    seed_scores: np.ndarray, intervention_scores: np.ndarray, target_scores: np.ndarray
) -> dict[str, np.ndarray]:
    if seed_scores.shape != intervention_scores.shape or seed_scores.shape != target_scores.shape:
        raise ValueError("per-input score arrays must align")
    target_gain = target_scores - seed_scores
    intervention_gain = intervention_scores - seed_scores
    ratio = np.divide(
        intervention_gain,
        target_gain,
        out=np.full_like(target_gain, np.nan),
        where=np.abs(target_gain) > 1e-12,
    )
    return {"target_gain": target_gain, "intervention_gain": intervention_gain, "recovery": ratio}


def per_input_recovery_summary(
    seed_scores: np.ndarray,
    intervention_scores: np.ndarray,
    target_scores: np.ndarray,
) -> dict[str, float | int]:
    if seed_scores.shape != intervention_scores.shape or seed_scores.shape != target_scores.shape:
        raise ValueError("per-input score arrays must align")
    target_fixed = target_scores > seed_scores
    target_degraded = target_scores < seed_scores
    intervention_fixed = intervention_scores > seed_scores
    intervention_degraded = intervention_scores < seed_scores
    fixed_count = int(target_fixed.sum())
    degraded_count = int(target_degraded.sum())
    return {
        "target_fixed": fixed_count,
        "target_degraded": degraded_count,
        "fixed_overlap": int(np.logical_and(target_fixed, intervention_fixed).sum()),
        "degraded_overlap": int(np.logical_and(target_degraded, intervention_degraded).sum()),
        "new_errors": int(np.logical_and(~target_degraded, intervention_degraded).sum()),
        "fixed_recall": (
            float(np.logical_and(target_fixed, intervention_fixed).sum() / fixed_count)
            if fixed_count
            else 0.0
        ),
        "degraded_recall": (
            float(np.logical_and(target_degraded, intervention_degraded).sum() / degraded_count)
            if degraded_count
            else 0.0
        ),
    }


def backend_parity(
    gold: Sequence[Sequence[str]],
    hf: Sequence[Sequence[str]],
    vllm: Sequence[Sequence[str]],
    labels: Sequence[str],
    tolerance: float = 0.02,
) -> tuple[bool, float]:
    hf_score = evaluate_multilabel(gold, hf, labels)["f1_samples"]
    vllm_score = evaluate_multilabel(gold, vllm, labels)["f1_samples"]
    assert isinstance(hf_score, float) and isinstance(vllm_score, float)
    difference = abs(hf_score - vllm_score)
    return difference <= tolerance, difference
