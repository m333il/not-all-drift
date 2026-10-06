"""Load-concentration statistics for MoE routing.

One implementation, shared by every consumer. Two copies of a Gini function is
how two tables end up disagreeing without either of them being obviously wrong.

**Conventions here match the project's earlier measurements**, so numbers stay
comparable with what is already written down:

* shares are normalised **per layer** - each layer routes independently and has
  its own hot experts, so pooling the counts across layers first averages the
  peaks away entirely (on Qwen it turns a Gini of 0.730 into 0.234);
* ``n_eff`` is the inverse Simpson index ``1/sum(p^2)``, not ``exp(H)``: the two
  differ substantially on the same data (32.6 against 45.9);
* ``frac_dead`` counts experts below ``1e-4`` of a layer's assignments.

The Gini here is the ordinary Lorenz-curve one: with the curve lying below the
diagonal and ``A`` the area between them, ``G = A / (A + B) = 2A`` since
``A + B = 1/2``. For a finite set of ``n`` experts the maximum is ``(n-1)/n``
rather than 1, which is the bound the project's metric reference states.
"""
from __future__ import annotations

import numpy as np

DEAD_THRESHOLD = 1e-4


def gini(p: np.ndarray) -> float:
    """Gini coefficient of a non-negative load vector.

    Equivalent to twice the area between the Lorenz curve and the line of
    equality; the closed form below avoids building the curve.
    """
    v = np.sort(np.asarray(p, dtype=float))
    if v.min() < 0:
        raise ValueError("load must be non-negative")
    total = v.sum()
    if total <= 0:
        return float("nan")
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * total))


def gini_from_lorenz(p: np.ndarray) -> float:
    """The same quantity, built from the Lorenz curve.

    Kept as an independent path so a test can hold the closed form to the
    definition rather than to itself.
    """
    v = np.sort(np.asarray(p, dtype=float))
    total = v.sum()
    if total <= 0:
        return float("nan")
    n = len(v)
    curve = np.concatenate([[0.0], np.cumsum(v) / total])
    below = np.trapezoid(curve, np.arange(n + 1) / n)
    return float(2 * (0.5 - below))


def layer_shares(counts: np.ndarray) -> np.ndarray:
    """Per-layer usage shares. Layers that saw nothing stay all zero."""
    total = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, total, out=np.zeros_like(counts, dtype=float),
                     where=total > 0)


def layer_stats(p: np.ndarray) -> dict[str, float]:
    """Concentration of one layer's load."""
    n_experts = len(p)
    return {
        "gini": gini(p),
        "n_eff": float(1.0 / np.square(p).sum()),
        "top1_share": float(p.max()),
        "hottest_over_uniform": float(p.max() * n_experts),
        "frac_dead": float((p < DEAD_THRESHOLD).mean()),
        "untouched": float((p <= 0).mean()),
    }
