"""How unevenly a router spreads its tokens over the experts.

Every function here takes a raw count vector (one layer, one stage) and returns
a scalar. They are deliberately pure and free of any file format so the values
can be checked against cases whose answer is known by hand - a uniform load, a
single-expert load - rather than against a previous run of the same code.

Which measure to read depends on the question:

* ``gini`` - the classic inequality summary, 0 uniform, → 1 degenerate. Good
  default, but it says nothing about *how many* experts carry the load.
* ``effective_experts`` - perplexity of the load distribution: the number of
  experts that would produce this entropy if they shared the traffic equally.
  Directly interpretable against the layer's real width.
* ``max_over_mean`` - the imbalance factor MoE papers report; how much the
  busiest expert exceeds a fair share.
* ``top_share`` - the traffic taken by the k busiest experts, the quantity that
  decides how much a frequency-pruner can remove cheaply.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gini",
    "normalised_entropy",
    "effective_experts",
    "max_over_mean",
    "top_share",
    "dead_fraction",
    "summarise",
    "METRIC_NAMES",
]

METRIC_NAMES = (
    "gini",
    "entropy_norm",
    "effective_experts",
    "max_over_mean",
    "top1_share",
    "top4_share",
    "dead_fraction",
)


def _shares(counts: np.ndarray) -> np.ndarray:
    """Counts as a probability vector. An all-zero row yields a uniform one."""
    counts = np.asarray(counts, dtype=np.float64).ravel()
    if (counts < 0).any():
        raise ValueError("counts must be non-negative")
    total = counts.sum()
    if total <= 0:
        return np.full(counts.size, 1.0 / counts.size)
    return counts / total


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of the load: 0 uniform, (n-1)/n when one expert takes all.

    Computed from the sorted cumulative share, which is both exact and O(n log n).
    """
    p = np.sort(_shares(counts))
    n = p.size
    index = np.arange(1, n + 1)
    return float((2 * (index * p).sum()) / (n * p.sum()) - (n + 1) / n)


def normalised_entropy(counts: np.ndarray) -> float:
    """Shannon entropy divided by log(n): 1 uniform, 0 when one expert takes all."""
    p = _shares(counts)
    n = p.size
    if n <= 1:
        return 1.0
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(n))


def effective_experts(counts: np.ndarray) -> float:
    """exp(H): how many equally-loaded experts would give this entropy."""
    p = _shares(counts)
    nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def max_over_mean(counts: np.ndarray) -> float:
    """Busiest expert over a fair share. 1.0 uniform, n when one takes all."""
    p = _shares(counts)
    return float(p.max() * p.size)


def top_share(counts: np.ndarray, k: int = 1) -> float:
    """Share of the traffic taken by the k busiest experts."""
    p = _shares(counts)
    k = min(k, p.size)
    return float(np.sort(p)[-k:].sum())


def dead_fraction(counts: np.ndarray, threshold: float = 0.0) -> float:
    """Fraction of experts at or below ``threshold`` share - the pruner's easy prey."""
    p = _shares(counts)
    return float((p <= threshold).mean())


def summarise(counts: np.ndarray) -> dict[str, float]:
    """Every metric at once, for one count vector."""
    return {
        "gini": gini(counts),
        "entropy_norm": normalised_entropy(counts),
        "effective_experts": effective_experts(counts),
        "max_over_mean": max_over_mean(counts),
        "top1_share": top_share(counts, 1),
        "top4_share": top_share(counts, 4),
        "dead_fraction": dead_fraction(counts),
    }
