"""Pure pieces of the router retraining in ``scripts/train_router.py``.

Kept apart from the script so the selection rule and the loss weighting can be
tested without a model.
"""
from __future__ import annotations

import math


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup to 1, then linear decay to 0 at ``total``."""
    if warmup and step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def token_weighted_windows(order, answer_lengths, accumulation):
    """Split ``order`` into accumulation windows of ``(index, weight)``.

    The model returns a per-sequence mean over its supervised tokens. Weighting
    each sequence by its share of the window's supervised tokens makes the
    window's gradient a per-token mean, as in the PEFT campaign's training loop.
    """
    for start in range(0, len(order), accumulation):
        window = order[start:start + accumulation]
        tokens = sum(answer_lengths[i] for i in window)
        yield [(i, answer_lengths[i] / tokens) for i in window]


def select_epochs(val_scores):
    """``(must_train, may_decline)`` epochs by validation score, earliest on ties.

    ``must_train`` ignores epoch 0 and is ``None`` when no epoch was trained.
    """
    def best(epochs):
        epochs = sorted(epochs)
        if not epochs:
            return None
        top = max(val_scores[e] for e in epochs)
        return next(e for e in epochs if val_scores[e] == top)

    return best([e for e in val_scores if e >= 1]), best(list(val_scores))


def paired(left, right):
    """Mean of ``left - right`` over paired examples, its standard error and t."""
    if len(left) != len(right) or not left:
        raise ValueError("paired scores need equal nonzero lengths")
    diffs = [a - b for a, b in zip(left, right)]
    n = len(diffs)
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(variance / n)
    return {"n": n, "mean": mean, "se": se, "t": mean / se if se else None,
            "changed": sum(d != 0 for d in diffs)}
