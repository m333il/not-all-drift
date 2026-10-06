from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

from .metrics import Interval


def grouped_bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    groups: np.ndarray,
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    if first.shape != second.shape or first.shape != groups.shape or first.ndim != 1:
        raise ValueError("values and group IDs must be aligned one-dimensional arrays")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("grouped bootstrap needs at least two groups")
    differences = first.astype(float) - second.astype(float)
    members = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([members[group] for group in selected])
        estimates[sample] = differences[indices].mean()
    tail = (1 - confidence) / 2
    return Interval(
        float(differences.mean()),
        float(np.quantile(estimates, tail)),
        float(np.quantile(estimates, 1 - tail)),
    )


@dataclass(frozen=True, slots=True)
class McNemarResult:
    discordant_first: int
    discordant_second: int
    p_value: float


def mcnemar_test(first_correct: np.ndarray, second_correct: np.ndarray) -> McNemarResult:
    if first_correct.shape != second_correct.shape or first_correct.ndim != 1:
        raise ValueError("paired correctness arrays must align")
    first = first_correct.astype(bool)
    second = second_correct.astype(bool)
    first_only = int(np.logical_and(first, ~second).sum())
    second_only = int(np.logical_and(~first, second).sum())
    total = first_only + second_only
    p_value = 1.0 if total == 0 else float(binomtest(min(first_only, second_only), total).pvalue)
    return McNemarResult(first_only, second_only, p_value)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a one-dimensional array in [0,1]")
    order = np.argsort(values)
    adjusted_sorted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted_sorted[rank] = running
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted


__all__ = [
    "McNemarResult",
    "grouped_bootstrap_difference",
    "holm_adjust",
    "mcnemar_test",
]
