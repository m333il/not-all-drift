"""Orchestration and manifest for the v2 Civil Comments splits."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads

from ..schemas import stable_example_id
from .allocation import mask_floors, select_split_indices
from .contract import (
    CIVIL_LABELS_V2,
    MASK_COUNT,
    SplitContractV2,
    SplitGenerationError,
)
from .corpus import (
    enriched_mask_distribution,
    label_marginals,
    load_masks,
    mask_histogram,
    natural_mask_distribution,
    normalized_text,
    open_dataset,
    sha256_file,
    unique_text_indices,
)

logger = logging.getLogger(__name__)


def labels_from_scores(scores: Mapping[str, float], threshold: float = 0.5) -> tuple[str, ...]:
    return tuple(label for label in CIVIL_LABELS_V2 if float(scores.get(label, 0.0)) >= threshold)


def _stratified_subset(
    masks: np.ndarray, master: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    counts = mask_histogram(masks, master)
    expected = counts * fraction
    quotas = np.floor(expected).astype(np.int64)
    target_size = int(len(master) * fraction)
    rng = np.random.default_rng(seed)
    tie_breaks = rng.random(MASK_COUNT)
    order = sorted(
        range(MASK_COUNT), key=lambda mask: (expected[mask] % 1, tie_breaks[mask]), reverse=True
    )
    for mask in order[: target_size - int(quotas.sum())]:
        quotas[mask] += 1
    selected: list[int] = []
    for mask in range(MASK_COUNT):
        pool = master[masks[master] == mask].copy()
        rng.shuffle(pool)
        selected.extend(int(index) for index in pool[: quotas[mask]])
    result = np.asarray(selected, dtype=np.int64)
    rng.shuffle(result)
    return result


def _records(
    dataset: pads.Dataset, indices: np.ndarray, partition: str, contract: SplitContractV2
) -> list[dict[str, Any]]:
    table = dataset.take(pa.array(indices, type=pa.int64()), columns=["text", *CIVIL_LABELS_V2])
    result: list[dict[str, Any]] = []
    for offset, index in enumerate(indices):
        text = str(table.column("text")[offset].as_py())
        scores = {label: float(table.column(label)[offset].as_py()) for label in CIVIL_LABELS_V2}
        labels = labels_from_scores(scores, contract.threshold)
        source_id = f"{partition}:{int(index)}"
        result.append(
            {
                "id": stable_example_id("civil_comments", text, source_id),
                "dataset": "civil_comments",
                "text": text,
                "labels": list(labels),
                # Used by the binary setup.
                "binary_label": "toxic" if labels else "safe",
                "source_id": source_id,
                "group_id": None,
                "scores": scores,
            }
        )
    return result


def _pairwise_audit(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Jaccard, conditionals and exclusive support for every label pair."""
    sets = {
        label: {index for index, record in enumerate(records) if label in record["labels"]}
        for label in CIVIL_LABELS_V2
    }
    audit: dict[str, Any] = {}
    for left_index, left in enumerate(CIVIL_LABELS_V2):
        for right in CIVIL_LABELS_V2[left_index + 1 :]:
            first, second = sets[left], sets[right]
            union = len(first | second)
            audit[f"{left}+{right}"] = {
                "jaccard": (len(first & second) / union) if union else 0.0,
                "p_left_given_right": (len(first & second) / len(second)) if second else 0.0,
                "p_right_given_left": (len(first & second) / len(first)) if first else 0.0,
                "left_without_right": len(first - second),
                "right_without_left": len(second - first),
            }
    return audit


def _split_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(label for record in records for label in record["labels"])
    mask_counts: Counter[str] = Counter()
    for record in records:
        mask_counts["|".join(record["labels"]) or "NONE"] += 1
    ids = [str(record["id"]) for record in records]
    positive_count = sum(bool(record["labels"]) for record in records)
    return {
        "size": len(records),
        "empty_count": len(records) - positive_count,
        "positive_count": positive_count,
        "label_counts": {label: label_counts[label] for label in CIVIL_LABELS_V2},
        "label_share_of_positives": {
            label: (label_counts[label] / positive_count) if positive_count else 0.0
            for label in CIVIL_LABELS_V2
        },
        "exact_mask_counts": dict(sorted(mask_counts.items())),
        "mean_label_cardinality": sum(len(record["labels"]) for record in records) / len(records),
        "pairwise_audit": _pairwise_audit(records),
        "id_hash": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "example_ids": ids,
    }


def _audit_base_partitions(
    records: Mapping[str, Sequence[dict[str, Any]]], base_names: Sequence[str]
) -> dict[str, Any]:
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    for name in base_names:
        ids = {str(record["source_id"]) for record in records[name]}
        overlap = ids & seen_ids
        if overlap:
            raise SplitGenerationError(f"source-id leakage in {name}: {sorted(overlap)[:3]}")
        seen_ids |= ids
        for record in records[name]:
            normalized = normalized_text(str(record["text"]))
            previous = seen_text.get(normalized)
            if previous is not None and previous != name:
                raise SplitGenerationError(f"normalized-text leakage between {previous} and {name}")
            seen_text[normalized] = name
    return {"base_partitions": list(base_names), "source_id_overlap": 0, "text_overlap": 0}


def generate_splits_v2(
    train_paths: Sequence[Path],
    test_paths: Sequence[Path],
    output: Path,
    contract: SplitContractV2,
) -> dict[str, Any]:
    contract.validate()
    if output.exists():
        raise SplitGenerationError(f"output already exists: {output}")
    salt = contract.seed_salt

    train_dataset = open_dataset(train_paths)
    test_dataset = open_dataset(test_paths)
    train_masks = load_masks(train_dataset, contract.threshold)
    test_masks = load_masks(test_dataset, contract.threshold)
    train_unique, train_hashes = unique_text_indices(train_dataset)
    test_unique, _ = unique_text_indices(test_dataset, train_hashes)
    logger.info(
        "corpus: %d unique train rows, %d unique test rows absent from train",
        len(train_unique),
        len(test_unique),
    )

    train_natural = natural_mask_distribution(train_masks, train_unique)
    test_natural = natural_mask_distribution(test_masks, test_unique)
    # Test is drawn at the same label shares as train.
    train_distribution = enriched_mask_distribution(train_natural, contract.enrichment)
    test_distribution = enriched_mask_distribution(test_natural, contract.enrichment)

    available = train_unique
    selected: dict[str, tuple[str, np.ndarray]] = {}
    base_names: list[str] = []

    for seed in contract.optimizer_seeds:
        nested_val: np.ndarray | None = None
        for size in contract.optimizer_val_ladder:
            nested_val = select_split_indices(
                contract,
                train_masks,
                available,
                train_distribution,
                size,
                seed + 100_000 + salt,
                nested_val,
            )
            selected[f"optimizer_val_seed{seed}_n{size}"] = ("train", nested_val.copy())
        assert nested_val is not None
        base_names.append(f"optimizer_val_seed{seed}_n{contract.optimizer_val_ladder[-1]}")
        available = available[~np.isin(available, nested_val)]

        nested: np.ndarray | None = None
        for size in contract.ladder:
            nested = select_split_indices(
                contract,
                train_masks,
                available,
                train_distribution,
                size,
                seed + salt,
                nested,
            )
            selected[f"optimizer_train_seed{seed}_n{size}"] = ("train", nested.copy())
        assert nested is not None
        base_names.append(f"optimizer_train_seed{seed}_n{contract.ladder[-1]}")
        available = available[~np.isin(available, nested)]

    auxiliary = (
        ("probe_train", contract.probe_train_size),
        ("probe_val", contract.probe_val_size),
        ("intervention_val", contract.intervention_val_size),
    )
    for offset, (name, size) in enumerate(auxiliary):
        indices = select_split_indices(
            contract,
            train_masks,
            available,
            train_distribution,
            size,
            contract.auxiliary_seed + offset + salt,
        )
        selected[name] = ("train", indices)
        base_names.append(name)
        available = available[~np.isin(available, indices)]

    probe_master = selected["probe_train"][1]
    subset_size = int(contract.probe_train_size * contract.probe_fraction)
    for seed in contract.probe_seeds:
        subset = _stratified_subset(
            train_masks, probe_master, contract.probe_fraction, seed + 200_000 + salt
        )
        if len(subset) != subset_size:
            raise SplitGenerationError("probe subset allocation produced the wrong size")
        selected[f"probe_train_seed{seed}"] = ("train", subset)

    test = select_split_indices(
        contract,
        test_masks,
        test_unique,
        test_distribution,
        contract.test_size,
        contract.test_seed + salt,
    )
    selected["test"] = ("test", test)
    base_names.append("test")

    records = {
        name: _records(
            train_dataset if partition == "train" else test_dataset,
            indices,
            partition,
            contract,
        )
        for name, (partition, indices) in selected.items()
    }
    overlap_audit = _audit_base_partitions(records, base_names)

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "dataset": "google/civil_comments",
        "dataset_revision": contract.dataset_revision,
        "setup": contract.setup,
        "labels": list(CIVIL_LABELS_V2),
        "contract": asdict(contract) | {"enrichment": dict(contract.enrichment)},
        "deduplication_audit": {
            "method": "NFKC + casefold + whitespace collapse; BLAKE2b-128 identity",
            "train_source_rows": len(train_masks),
            "train_unique_rows": len(train_unique),
            "test_source_rows": len(test_masks),
            "test_unique_and_absent_from_train_rows": len(test_unique),
        },
        "sources": [
            {
                "partition": partition,
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for partition, paths in (("train", train_paths), ("test", test_paths))
            for path in paths
        ],
        "mask_distribution": {
            "train_natural_label_shares": label_marginals(train_natural),
            "train_target_label_shares": label_marginals(train_distribution),
            "test_natural_label_shares": label_marginals(test_natural),
            "test_target_label_shares": label_marginals(test_distribution),
            "enrichment_factors": {
                label: (
                    label_marginals(train_distribution)[label]
                    / label_marginals(train_natural)[label]
                )
                for label in CIVIL_LABELS_V2
            },
        },
        "mask_floors": {
            f"n{size}": {
                "|".join(
                    label
                    for index, label in enumerate(CIVIL_LABELS_V2)
                    if mask & (1 << index)
                ): int(floor)
                for mask, floor in enumerate(
                    mask_floors(contract, train_distribution, contract.positive_size(size))
                )
                if floor
            }
            for size in contract.ladder
        },
        "splits": {name: _split_stats(rows) for name, rows in records.items()},
        "overlap_audit": overlap_audit,
        "nesting_audit": {
            str(seed): {
                f"n{small}_subset_n{large}": bool(
                    set(selected[f"optimizer_train_seed{seed}_n{small}"][1])
                    < set(selected[f"optimizer_train_seed{seed}_n{large}"][1])
                )
                for small, large in zip(contract.ladder, contract.ladder[1:], strict=False)
            }
            for seed in contract.optimizer_seeds
        },
        "optimizer_val_nesting_audit": {
            str(seed): {
                f"n{small}_subset_n{large}": bool(
                    set(selected[f"optimizer_val_seed{seed}_n{small}"][1])
                    < set(selected[f"optimizer_val_seed{seed}_n{large}"][1])
                )
                for small, large in zip(
                    contract.optimizer_val_ladder, contract.optimizer_val_ladder[1:], strict=False
                )
            }
            for seed in contract.optimizer_seeds
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, rows in records.items():
            (temporary / f"{name}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            pd.DataFrame(rows).to_parquet(temporary / f"{name}.parquet", index=False)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest
