"""Row allocation under exact-mask floors and soft distributional targets.

Floors are stated on exact label masks, so each floor is a count that is allocated
directly and different subtypes never share rows.
"""

from __future__ import annotations

import logging

import numpy as np

from .contract import (
    CIVIL_LABELS_V2,
    MASK_COUNT,
    SplitContractV2,
    SplitGenerationError,
)
from .corpus import mask_features, mask_histogram

logger = logging.getLogger(__name__)


def mask_floors(
    contract: SplitContractV2, distribution: np.ndarray, positive_size: int
) -> np.ndarray:
    """Per-mask lower bounds for a split with ``positive_size`` positives.

    The floor is a share of the enriched distribution; the absolute term only matters
    at the smallest ladder step.
    """
    floors = np.zeros(MASK_COUNT, dtype=np.int64)
    if contract.mask_floor_fraction <= 0.0 and contract.mask_floor_absolute <= 0:
        return floors
    for mask in contract.quota_masks():
        expected = contract.mask_floor_fraction * float(distribution[mask]) * positive_size
        floors[mask] = max(contract.mask_floor_absolute, int(np.floor(expected)))
    total = int(floors.sum())
    if total > positive_size:
        raise SplitGenerationError(
            f"mask floors need {total} rows but the positive half holds {positive_size}"
        )
    return floors


def _greedy_fill(
    counts: np.ndarray,
    available: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    remaining: int,
) -> None:
    """Add ``remaining`` rows one mask at a time, minimising weighted L1 drift."""
    stats = counts @ features
    for _ in range(remaining):
        room = available - counts
        choices = np.flatnonzero(room > 0)
        if not len(choices):
            raise SplitGenerationError("positive pool exhausted during greedy fill")
        candidate_stats = stats + features[choices]
        losses = np.abs(candidate_stats - target) @ weights
        # Ties resolve to the lower mask index, matching the v1 ordering.
        best = choices[int(np.argmin(losses))]
        counts[best] += 1
        stats = stats + features[best]


def allocate_positive_counts(
    contract: SplitContractV2,
    distribution: np.ndarray,
    available: np.ndarray,
    current_counts: np.ndarray,
    positive_size: int,
) -> np.ndarray:
    """Per-mask counts to add on top of ``current_counts`` to reach ``positive_size``."""
    needed = positive_size - int(current_counts.sum())
    if needed < 0:
        raise SplitGenerationError("nested positive split is larger than its target")
    if int(available.sum()) < needed:
        raise SplitGenerationError(
            f"positive pool has {int(available.sum())} rows, but {needed} more are required"
        )

    floors = mask_floors(contract, distribution, positive_size)
    deficits = np.maximum(floors - current_counts, 0)
    infeasible = np.flatnonzero(deficits > available)
    if len(infeasible):
        details = {
            int(mask): (int(deficits[mask]), int(available[mask])) for mask in infeasible[:5]
        }
        raise SplitGenerationError(
            f"exact-mask floor exceeds supply (mask: needed, available) {details}"
        )
    if int(deficits.sum()) > needed:
        raise SplitGenerationError(
            f"exact-mask floors need {int(deficits.sum())} rows but only {needed} remain"
        )
    counts = deficits.copy()

    features = mask_features()
    # Targets are in feature space: label indicators, then pair indicators.
    target = (distribution @ features) * positive_size
    # Inverse-target weights keep rare labels visible; the pair block is down-weighted.
    weights = 1.0 / np.maximum(target, 1.0)
    weights[len(CIVIL_LABELS_V2) :] *= contract.pairwise_weight

    # Proportional pre-allocation keeps the greedy loop short.
    ideal = distribution * positive_size
    already = current_counts + counts
    remaining = needed - int(counts.sum())
    if remaining > 0:
        room = available - counts
        want = np.clip(np.floor(ideal - already).astype(np.int64), 0, room)
        want[0] = 0
        total_want = int(want.sum())
        if total_want > remaining:
            want = np.floor(want * (remaining / total_want)).astype(np.int64)
        counts += want
        remaining = needed - int(counts.sum())
    if remaining > 0:
        _greedy_fill(counts, available, features, target, weights, remaining)
    if int(counts.sum()) != needed:
        raise SplitGenerationError("mask allocation did not reach the requested size")
    return counts


def select_positive_indices(
    contract: SplitContractV2,
    masks: np.ndarray,
    eligible: np.ndarray,
    distribution: np.ndarray,
    positive_size: int,
    seed: int,
    current: np.ndarray | None = None,
) -> np.ndarray:
    """Draw the positive half, extending ``current`` so ladders stay nested."""
    current = np.asarray([], dtype=np.int64) if current is None else current
    candidates = eligible[masks[eligible] > 0]
    if len(current):
        candidates = candidates[~np.isin(candidates, current)]
    available = mask_histogram(masks, candidates)
    current_counts = mask_histogram(masks, current)
    counts = allocate_positive_counts(
        contract, distribution, available, current_counts, positive_size
    )
    rng = np.random.default_rng(seed)
    selected = [int(index) for index in current]
    for mask in range(1, MASK_COUNT):
        if not counts[mask]:
            continue
        pool = candidates[masks[candidates] == mask].copy()
        rng.shuffle(pool)
        selected.extend(int(index) for index in pool[: counts[mask]])
    result = np.asarray(selected, dtype=np.int64)
    rng.shuffle(result)
    return result


def select_split_indices(
    contract: SplitContractV2,
    masks: np.ndarray,
    eligible: np.ndarray,
    distribution: np.ndarray,
    size: int,
    seed: int,
    current: np.ndarray | None = None,
) -> np.ndarray:
    """Draw one split at the contract's empty fraction, nested on ``current``."""
    current = np.asarray([], dtype=np.int64) if current is None else current
    current_empty = current[masks[current] == 0]
    current_positive = current[masks[current] > 0]
    empty_needed = contract.empty_size(size) - len(current_empty)
    empty_pool = eligible[masks[eligible] == 0]
    if len(current_empty):
        empty_pool = empty_pool[~np.isin(empty_pool, current_empty)]
    empty_pool = empty_pool.copy()
    if empty_needed < 0:
        raise SplitGenerationError("nested empty half is larger than its target")
    if len(empty_pool) < empty_needed:
        raise SplitGenerationError(
            f"empty pool has {len(empty_pool)} rows, but {empty_needed} more are required"
        )
    rng = np.random.default_rng(seed ^ 0x5F3759DF)
    rng.shuffle(empty_pool)
    positives = select_positive_indices(
        contract,
        masks,
        eligible,
        distribution,
        contract.positive_size(size),
        seed,
        current_positive,
    )
    result = np.concatenate((current_empty, empty_pool[:empty_needed], positives))
    rng.shuffle(result)
    return result
