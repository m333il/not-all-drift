"""Corpus loading, deduplication, and the enriched mask distribution."""

from __future__ import annotations

import hashlib
import logging
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads

from .contract import (
    CIVIL_LABELS_V2,
    MASK_COUNT,
    SplitGenerationError,
)

logger = logging.getLogger(__name__)

IPF_MAX_ITERATIONS = 500
IPF_TOLERANCE = 1e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_dataset(paths: Sequence[Path]) -> pads.Dataset:
    if not paths:
        raise SplitGenerationError("at least one parquet path is required")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SplitGenerationError(f"missing parquet files: {missing}")
    dataset = pads.dataset([str(path) for path in paths], format="parquet")
    absent = sorted({"text", *CIVIL_LABELS_V2} - set(dataset.schema.names))
    if absent:
        raise SplitGenerationError(f"missing Civil Comments columns: {absent}")
    return dataset


def load_masks(dataset: pads.Dataset, threshold: float) -> np.ndarray:
    """Binarise the five v2 labels into one integer mask per row."""
    table = dataset.to_table(columns=list(CIVIL_LABELS_V2))
    scores = np.column_stack(
        [table.column(label).to_numpy(zero_copy_only=False) for label in CIVIL_LABELS_V2]
    ).astype(np.float64, copy=False)
    if not np.isfinite(scores).all():
        raise SplitGenerationError("Civil Comments scores contain NaN or infinite values")
    bit_values = 1 << np.arange(len(CIVIL_LABELS_V2), dtype=np.uint8)
    return ((scores >= threshold) * bit_values).sum(axis=1, dtype=np.uint8)


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def unique_text_indices(
    dataset: pads.Dataset, forbidden_hashes: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Keep one canonical row per normalized text, optionally excluding a hash set.

    Reproduces the v1 preprocessing exactly so the two manifests stay comparable.
    """
    hashes = np.empty(dataset.count_rows(), dtype="V16")
    offset = 0
    for batch in dataset.scanner(columns=["text"], batch_size=65_536).to_batches():
        batch_hashes = [
            hashlib.blake2b(normalized_text(str(text)).encode(), digest_size=16).digest()
            for text in batch.column("text").to_pylist()
        ]
        hashes[offset : offset + len(batch_hashes)] = batch_hashes
        offset += len(batch_hashes)
    _, first_indices = np.unique(hashes, return_index=True)
    keep = np.sort(first_indices.astype(np.int64))
    kept_hashes = hashes[keep]
    if forbidden_hashes is not None:
        allowed = ~np.isin(kept_hashes, forbidden_hashes)
        keep = keep[allowed]
        kept_hashes = kept_hashes[allowed]
    return keep, kept_hashes


def mask_features() -> np.ndarray:
    """``[mask, feature]`` design matrix of label indicators followed by pair indicators."""
    label_bits = 1 << np.arange(len(CIVIL_LABELS_V2), dtype=np.uint16)
    labels = np.asarray(
        [[bool(mask & bit) for bit in label_bits] for mask in range(MASK_COUNT)],
        dtype=np.float64,
    )
    pairs = np.column_stack(
        [
            labels[:, left] * labels[:, right]
            for left in range(len(CIVIL_LABELS_V2))
            for right in range(left + 1, len(CIVIL_LABELS_V2))
        ]
    )
    return np.column_stack((labels, pairs))


def mask_histogram(masks: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.bincount(masks[indices], minlength=MASK_COUNT).astype(np.int64)


def natural_mask_distribution(masks: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Distribution over the ``MASK_COUNT - 1`` positive masks, natural rates."""
    counts = mask_histogram(masks, indices).astype(np.float64)
    counts[0] = 0.0
    total = counts.sum()
    if total <= 0:
        raise SplitGenerationError("source has no positive examples at the configured threshold")
    return counts / total


def enriched_mask_distribution(
    natural: np.ndarray, enrichment: Mapping[str, float]
) -> np.ndarray:
    """Reweight the mask distribution so the named labels reach their target share.

    Iterative proportional fitting over masks that carry each label; this keeps the
    conditional structure among the other labels as far as possible.
    """
    if not enrichment:
        return natural.copy()
    features = mask_features()
    distribution = natural.copy()
    targets = {
        CIVIL_LABELS_V2.index(label): float(share) for label, share in enrichment.items()
    }
    for _ in range(IPF_MAX_ITERATIONS):
        largest_gap = 0.0
        for label_index, target in targets.items():
            carrier = features[:, label_index] > 0
            current = float(distribution[carrier].sum())
            if current <= 0.0:
                raise SplitGenerationError(
                    f"cannot enrich {CIVIL_LABELS_V2[label_index]}: no rows carry it"
                )
            largest_gap = max(largest_gap, abs(current - target))
            # Solve f * c / (f * c + 1 - c) = target, renormalise, repeat over labels.
            scale = target * (1.0 - current) / (current * (1.0 - target))
            distribution[carrier] *= scale
            distribution /= distribution.sum()
        if largest_gap < IPF_TOLERANCE:
            break
    else:
        achieved = {
            CIVIL_LABELS_V2[index]: float(distribution[features[:, index] > 0].sum())
            for index in targets
        }
        raise SplitGenerationError(f"enrichment did not converge, reached {achieved}")
    logger.info(
        "enriched mask distribution: %s",
        {
            CIVIL_LABELS_V2[index]: round(float(distribution[features[:, index] > 0].sum()), 5)
            for index in targets
        },
    )
    return distribution


def label_marginals(distribution: np.ndarray) -> dict[str, float]:
    features = mask_features()
    return {
        label: float(distribution[features[:, index] > 0].sum())
        for index, label in enumerate(CIVIL_LABELS_V2)
    }
