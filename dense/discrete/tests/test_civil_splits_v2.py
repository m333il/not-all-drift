from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from interpretability_gepa.civil_splits_v2 import (
    CIVIL_LABELS_V2,
    SplitContractV2,
    SplitGenerationError,
    enriched_mask_distribution,
    generate_splits_v2,
    label_marginals,
    mask_floors,
    natural_mask_distribution,
)
from interpretability_gepa.civil_splits_v2.corpus import mask_features

UMBRELLA = "toxicity"
SUBTYPES = tuple(label for label in CIVIL_LABELS_V2 if label != UMBRELLA)


def _write_source(path: Path, size: int, *, prefix: str) -> None:
    """Synthesise a corpus with the real hierarchy: subtypes ride on toxicity."""
    rows: list[dict[str, object]] = []
    for index in range(size):
        scores = {label: 0.0 for label in CIVIL_LABELS_V2}
        if index % 3:
            scores[UMBRELLA] = 0.9
            # One toxic row in four carries no subtype, mirroring the real
            # `{toxicity}` mask that holds 24 019 corpus rows.
            if index % 4:
                subtype = SUBTYPES[(index // 3) % len(SUBTYPES)]
                scores[subtype] = 0.8
                if index % 11 == 0:
                    scores[SUBTYPES[((index // 3) + 1) % len(SUBTYPES)]] = 0.7
        rows.append(
            {
                "id": f"{prefix}-{index}",
                "text": f"Unique {prefix} comment {index}",
                "severe_toxicity": 1.0,
                "sexual_explicit": 1.0,
                **scores,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _small_contract(**overrides: object) -> SplitContractV2:
    base = {
        "ladder": (20, 40, 80),
        "optimizer_val_ladder": (10, 20),
        "optimizer_seeds": (11, 12),
        "test_size": 20,
        "probe_train_size": 40,
        "probe_val_size": 20,
        "intervention_val_size": 20,
        "probe_seeds": (0, 1),
        "enrichment": {},
        "mask_floor_absolute": 1,
        "mask_floor_fraction": 0.3,
    }
    base.update(overrides)
    return SplitContractV2(**base)  # type: ignore[arg-type]


def _records(output: Path, split: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]


def test_label_schema_drops_the_two_removed_labels() -> None:
    assert CIVIL_LABELS_V2 == (
        "toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
    )
    assert "sexual_explicit" not in CIVIL_LABELS_V2
    assert "severe_toxicity" not in CIVIL_LABELS_V2


def test_enrichment_moves_only_the_named_label_to_its_target() -> None:
    natural = np.zeros(1 << len(CIVIL_LABELS_V2))
    # 90% {toxicity, insult}, 10% {toxicity, threat}
    natural[1 | (1 << 3)] = 0.9
    natural[1 | (1 << 2)] = 0.1
    enriched = enriched_mask_distribution(natural, {"threat": 0.25})
    marginals = label_marginals(enriched)
    assert marginals["threat"] == pytest.approx(0.25, abs=1e-6)
    assert marginals["toxicity"] == pytest.approx(1.0, abs=1e-6)
    assert marginals["insult"] == pytest.approx(0.75, abs=1e-6)


def test_enrichment_is_identity_without_targets() -> None:
    natural = np.zeros(1 << len(CIVIL_LABELS_V2))
    natural[3] = 0.4
    natural[5] = 0.6
    assert np.allclose(enriched_mask_distribution(natural, {}), natural)


def test_mask_floors_scale_with_the_split_and_respect_the_absolute_floor() -> None:
    contract = SplitContractV2()
    distribution = np.zeros(1 << len(CIVIL_LABELS_V2))
    for mask in contract.quota_masks():
        distribution[mask] = 1.0 / len(contract.quota_masks())
    large = mask_floors(contract, distribution, 6700)
    small = mask_floors(contract, distribution, 134)
    for mask in contract.quota_masks():
        assert large[mask] > small[mask]
        assert small[mask] >= contract.mask_floor_absolute
    assert int(large.sum()) <= 6700


def test_mask_floors_cannot_exceed_the_positive_half() -> None:
    contract = SplitContractV2(mask_floor_absolute=100)
    distribution = np.zeros(1 << len(CIVIL_LABELS_V2))
    with pytest.raises(SplitGenerationError, match="mask floors need"):
        mask_floors(contract, distribution, 50)


def test_generated_splits_respect_the_empty_fraction(tmp_path: Path) -> None:
    _write_source(tmp_path / "train.parquet", 4000, prefix="train")
    _write_source(tmp_path / "test.parquet", 900, prefix="test")
    contract = _small_contract()
    manifest = generate_splits_v2(
        (tmp_path / "train.parquet",), (tmp_path / "test.parquet",), tmp_path / "out", contract
    )
    for name, stats in manifest["splits"].items():
        if name.startswith("probe_train_seed"):
            continue
        assert stats["empty_count"] == contract.empty_size(stats["size"]), name
        assert stats["positive_count"] == contract.positive_size(stats["size"]), name


def test_binary_setup_uses_a_half_empty_split(tmp_path: Path) -> None:
    _write_source(tmp_path / "train.parquet", 4000, prefix="train")
    _write_source(tmp_path / "test.parquet", 900, prefix="test")
    contract = _small_contract(
        setup="binary", empty_fraction=0.5, mask_floor_fraction=0.0, mask_floor_absolute=0
    )
    manifest = generate_splits_v2(
        (tmp_path / "train.parquet",), (tmp_path / "test.parquet",), tmp_path / "out", contract
    )
    test_stats = manifest["splits"]["test"]
    assert test_stats["empty_count"] == test_stats["positive_count"]
    records = _records(tmp_path / "out", "test")
    assert {str(record["binary_label"]) for record in records} == {"toxic", "safe"}
    for record in records:
        assert record["binary_label"] == ("toxic" if record["labels"] else "safe")


def test_binary_setup_rejects_enrichment() -> None:
    with pytest.raises(SplitGenerationError, match="no per-label claim"):
        SplitContractV2(setup="binary", empty_fraction=0.5).validate()


def test_ladders_are_nested_and_the_anchor_is_shared(tmp_path: Path) -> None:
    _write_source(tmp_path / "train.parquet", 4000, prefix="train")
    _write_source(tmp_path / "test.parquet", 900, prefix="test")
    contract = _small_contract()
    manifest = generate_splits_v2(
        (tmp_path / "train.parquet",), (tmp_path / "test.parquet",), tmp_path / "out", contract
    )
    for audit in manifest["nesting_audit"].values():
        assert all(audit.values()), audit
    for audit in manifest["optimizer_val_nesting_audit"].values():
        assert all(audit.values()), audit
    for seed in contract.optimizer_seeds:
        out = tmp_path / "out"
        small = {r["source_id"] for r in _records(out, f"optimizer_train_seed{seed}_n20")}
        large = {r["source_id"] for r in _records(out, f"optimizer_train_seed{seed}_n80")}
        assert small < large


def test_no_subtype_pair_draws_identical_rows(tmp_path: Path) -> None:
    """The v1 defect: coverage minimisation made rare pairs coincide."""
    _write_source(tmp_path / "train.parquet", 4000, prefix="train")
    _write_source(tmp_path / "test.parquet", 900, prefix="test")
    manifest = generate_splits_v2(
        (tmp_path / "train.parquet",),
        (tmp_path / "test.parquet",),
        tmp_path / "out",
        _small_contract(),
    )
    audit = manifest["splits"]["probe_train"]["pairwise_audit"]
    for pair, stats in audit.items():
        left, right = pair.split("+")
        if UMBRELLA in (left, right):
            continue
        assert stats["jaccard"] < 1.0, pair
        assert stats["left_without_right"] > 0, pair
        assert stats["right_without_left"] > 0, pair


def test_splits_do_not_leak_between_partitions(tmp_path: Path) -> None:
    _write_source(tmp_path / "train.parquet", 4000, prefix="train")
    _write_source(tmp_path / "test.parquet", 900, prefix="test")
    manifest = generate_splits_v2(
        (tmp_path / "train.parquet",),
        (tmp_path / "test.parquet",),
        tmp_path / "out",
        _small_contract(),
    )
    assert manifest["overlap_audit"]["source_id_overlap"] == 0
    probe = {r["source_id"] for r in _records(tmp_path / "out", "probe_train")}
    test = {r["source_id"] for r in _records(tmp_path / "out", "test")}
    val = {r["source_id"] for r in _records(tmp_path / "out", "probe_val")}
    assert not probe & val
    assert not probe & test


def test_generation_is_deterministic(tmp_path: Path) -> None:
    _write_source(tmp_path / "train.parquet", 4000, prefix="train")
    _write_source(tmp_path / "test.parquet", 900, prefix="test")
    contract = _small_contract()
    first = generate_splits_v2(
        (tmp_path / "train.parquet",), (tmp_path / "test.parquet",), tmp_path / "a", contract
    )
    second = generate_splits_v2(
        (tmp_path / "train.parquet",), (tmp_path / "test.parquet",), tmp_path / "b", contract
    )
    for name in first["splits"]:
        assert first["splits"][name]["id_hash"] == second["splits"][name]["id_hash"], name


def test_output_refuses_to_overwrite(tmp_path: Path) -> None:
    _write_source(tmp_path / "train.parquet", 4000, prefix="train")
    _write_source(tmp_path / "test.parquet", 900, prefix="test")
    (tmp_path / "out").mkdir()
    with pytest.raises(SplitGenerationError, match="output already exists"):
        generate_splits_v2(
            (tmp_path / "train.parquet",),
            (tmp_path / "test.parquet",),
            tmp_path / "out",
            _small_contract(),
        )


def test_mask_features_cover_labels_and_pairs() -> None:
    features = mask_features()
    assert features.shape == (1 << len(CIVIL_LABELS_V2), len(CIVIL_LABELS_V2) + 10)
    assert features[0].sum() == 0.0
    everything = (1 << len(CIVIL_LABELS_V2)) - 1
    assert features[everything].sum() == len(CIVIL_LABELS_V2) + 10


def test_natural_distribution_ignores_the_empty_mask() -> None:
    masks = np.array([0, 0, 3, 5, 3], dtype=np.uint8)
    distribution = natural_mask_distribution(masks, np.arange(5))
    assert distribution[0] == 0.0
    assert distribution.sum() == pytest.approx(1.0)
    assert distribution[3] == pytest.approx(2 / 3)
