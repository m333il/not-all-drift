from __future__ import annotations

import json
from pathlib import Path

import pytest

from interpretability_gepa.config import DatasetConfig
from interpretability_gepa.datasets import (
    CIVIL_LABELS,
    CivilCommentsSource,
    choose_training_size,
    civil_labels,
    clean_goemotions_rows,
    hallmark_labels,
    optimizer_train_split_path,
    prepare_from_rows,
)
from interpretability_gepa.errors import ConfigurationError


def test_civil_mapping_and_empty_set() -> None:
    assert civil_labels({"toxicity": 0.49, "threat": 0.2}) == ()
    # ``sexual_explicit`` left the schema with v2, so a raw score for it is ignored.
    assert civil_labels({"toxicity": 0.5, "identity_attack": 0.9, "sexual_explicit": 0.8}) == (
        "toxicity",
        "identity_attack",
    )


def test_optimizer_train_split_path_uses_frozen_civil_size_suffix() -> None:
    root = Path("/splits")

    assert optimizer_train_split_path(root, "civil_comments", seed=42, train_size=100) == (
        root / "optimizer_train_seed42_n100.jsonl"
    )
    assert optimizer_train_split_path(root, "goemotions", seed=42, train_size=100) == (
        root / "optimizer_train_seed42.jsonl"
    )
    assert CIVIL_LABELS == (
        "toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
    )


def test_runtime_labels_track_the_v2_split_contract() -> None:
    # The two constants are kept in step by hand so the generators stay independent
    # of the serving path; drift would silently mislabel every v2 split.
    from interpretability_gepa.civil_splits_v2.contract import CIVIL_LABELS_V2

    assert CIVIL_LABELS == CIVIL_LABELS_V2


def test_civil_source_trusts_only_the_pinned_dataset_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "61864fdf071fdb5a8fd8c74f5d23271c6a9fc65c"
    calls: list[dict[str, object]] = []

    def fake_load_dataset(_name: str, **kwargs: object) -> list[dict[str, object]]:
        calls.append(kwargs)
        return [{"id": "1", "text": "example", "toxicity": 0.7}]

    monkeypatch.setattr(
        "interpretability_gepa.datasets._load_dataset_dependency",
        lambda: fake_load_dataset,
    )
    source = CivilCommentsSource(DatasetConfig(id="civil_comments", revision=revision))

    train, test, labels = source.load()

    assert train == test
    assert labels == CIVIL_LABELS
    assert calls == [
        {"split": "train", "revision": revision, "trust_remote_code": True},
        {"split": "test", "revision": revision, "trust_remote_code": True},
    ]


def test_civil_source_rejects_mutable_remote_code_revision() -> None:
    source = CivilCommentsSource(DatasetConfig(id="civil_comments", revision="main"))

    with pytest.raises(ConfigurationError, match="full commit SHA"):
        source.load()


def test_prepared_splits_are_deterministic_and_disjoint() -> None:
    labels = ("a", "b")
    train = [
        {
            "id": i,
            "text": f"train {i}",
            "labels": (["a"] if i % 3 == 0 else ["b"] if i % 3 == 1 else []),
        }
        for i in range(100)
    ]
    test = [
        {
            "id": i,
            "text": f"test {i}",
            "labels": (["a"] if i % 3 == 0 else ["b"] if i % 3 == 1 else []),
        }
        for i in range(80)
    ]
    kwargs = dict(
        dataset="civil_comments",
        labels=labels,
        train_rows=train,
        test_rows=test,
        seeds=(42, 43),
        n_train=5,
        n_optimizer_val=3,
        n_probe_train=10,
        n_probe_val=5,
        n_intervention_val=5,
        n_mechanistic_eval=10,
        n_natural_eval=20,
        revision="fixed",
    )
    first = prepare_from_rows(**kwargs)
    second = prepare_from_rows(**kwargs)
    assert first.metadata == second.metadata
    assert first.splits == second.splits
    first.assert_disjoint(*first.splits)
    assert any(not x.labels for x in first.splits["eval_natural"])
    assert sum(not x.labels for x in first.splits["eval_mechanistic"]) == 5


def test_probe_train_preserves_rare_label_coverage() -> None:
    labels = ("a", "b", "rare")
    train = [
        {
            "id": i,
            "text": f"train {i}",
            "labels": ["rare"] if i % 8 == 0 else ["a"] if i % 2 == 0 else ["b"],
        }
        for i in range(120)
    ]
    test = [
        {"id": f"test-{i}", "text": f"test {i}", "labels": [labels[i % len(labels)]]}
        for i in range(60)
    ]

    prepared = prepare_from_rows(
        dataset="civil_comments",
        labels=labels,
        train_rows=train,
        test_rows=test,
        seeds=(42,),
        n_train=5,
        n_optimizer_val=3,
        n_probe_train=10,
        n_probe_val=5,
        n_intervention_val=5,
        n_mechanistic_eval=10,
        n_natural_eval=20,
        revision="fixed",
    )

    present = {label for example in prepared.splits["probe_train"] for label in example.labels}
    assert present == set(labels)
    prepared.assert_disjoint(*prepared.splits)


def test_each_optimizer_split_preserves_label_coverage_across_seeds() -> None:
    labels = ("a", "b", "severe_toxic")
    train = [{"id": i, "text": f"train {i}", "labels": [labels[i % 3]]} for i in range(21)]
    test = [{"id": f"test-{i}", "text": f"test {i}", "labels": [labels[i % 3]]} for i in range(9)]
    optimizer_splits = tuple(
        f"optimizer_{part}_seed{seed}" for seed in (42, 43) for part in ("train", "val")
    )

    prepared = prepare_from_rows(
        dataset="toy",
        labels=labels,
        train_rows=train,
        test_rows=test,
        seeds=(42, 43),
        n_train=3,
        n_optimizer_val=3,
        n_probe_train=3,
        n_probe_val=3,
        n_intervention_val=3,
        n_mechanistic_eval=3,
        n_natural_eval=9,
        revision="fixed",
    )

    assert {
        name: {label for example in prepared.splits[name] for label in example.labels}
        for name in optimizer_splits
    } == {name: set(labels) for name in optimizer_splits}
    assert {name: len(prepared.splits[name]) for name in optimizer_splits} == {
        name: 3 for name in optimizer_splits
    }
    prepared.assert_disjoint(*optimizer_splits)


def test_each_auxiliary_split_preserves_scarce_label_coverage() -> None:
    labels = ("a", "b", "rare")
    train_labels = ["a"] * 8 + ["b"] * 8 + ["rare"] * 5
    train = [
        {"id": i, "text": f"train {i}", "labels": [label]} for i, label in enumerate(train_labels)
    ]
    test = [{"id": f"test-{i}", "text": f"test {i}", "labels": [labels[i % 3]]} for i in range(9)]
    auxiliary_sizes = {
        "probe_train": 9,
        "probe_val": 3,
        "intervention_val": 3,
    }

    prepared = prepare_from_rows(
        dataset="toy",
        labels=labels,
        train_rows=train,
        test_rows=test,
        seeds=(42,),
        n_train=3,
        n_optimizer_val=3,
        n_probe_train=auxiliary_sizes["probe_train"],
        n_probe_val=auxiliary_sizes["probe_val"],
        n_intervention_val=auxiliary_sizes["intervention_val"],
        n_mechanistic_eval=3,
        n_natural_eval=9,
        revision="fixed",
    )

    assert {
        name: {label for example in prepared.splits[name] for label in example.labels}
        for name in auxiliary_sizes
    } == {name: set(labels) for name in auxiliary_sizes}
    assert {name: len(prepared.splits[name]) for name in auxiliary_sizes} == auxiliary_sizes
    prepared.assert_disjoint(*auxiliary_sizes)


def test_adaptive_n_uses_validation_only() -> None:
    rows = []
    for size, gain in ((100, 0.01), (300, 0.05)):
        for branch in ("gepa", "prefix"):
            rows.extend(
                {"n": size, "branch": branch, "val_gain": gain + offset, "collapsed": False}
                for offset in (-0.002, 0, 0.002)
            )
    coverage = {
        100: {"a": (5, 3), "b": (2, 1)},
        300: {"a": (8, 4), "b": (5, 3)},
    }
    selected, reasons = choose_training_size(rows, coverage, ("a", "b"), candidates=(100, 300))
    assert selected == 300
    assert reasons == ()


def test_goemotions_clean_removes_normalized_conflicts_and_duplicates() -> None:
    rows = [
        {"id": "1", "text": " Same  text ", "labels": ["joy"]},
        {"id": "2", "text": "same text", "labels": ["joy"]},
        {"id": "3", "text": "Conflict", "labels": ["joy"]},
        {"id": "4", "text": " conflict ", "labels": ["anger"]},
        {"id": "5", "text": "Unique", "labels": []},
    ]

    cleaned, audit = clean_goemotions_rows(rows)

    assert [row["id"] for row in cleaned] == ["1", "5"]
    assert audit == {"input_rows": 5, "deduplicated_rows": 1, "conflicting_rows": 2}


def test_hallmarks_source_none_becomes_empty_and_indices_are_mapped() -> None:
    assert hallmark_labels([7]) == ()
    assert hallmark_labels([9, 0, 7]) == (
        "evading_growth_suppressors",
        "sustaining_proliferative_signaling",
    )


def test_hallmarks_splits_never_cross_pmid_groups() -> None:
    labels = ("a", "b")
    train = [
        {
            "id": f"train-{group}-{index}",
            "text": f"train {group} {index}",
            "labels": [labels[group % 2]],
            "pmid": str(group),
        }
        for group in range(20)
        for index in range(2)
    ]
    test = [
        {
            "id": f"test-{group}-{index}",
            "text": f"test {group} {index}",
            "labels": [labels[group % 2]],
            "pmid": f"t{group}",
        }
        for group in range(10)
        for index in range(2)
    ]

    prepared = prepare_from_rows(
        dataset="hallmarks_of_cancer",
        labels=labels,
        train_rows=train,
        test_rows=test,
        seeds=(42,),
        n_train=4,
        n_optimizer_val=2,
        n_probe_train=6,
        n_probe_val=2,
        n_intervention_val=2,
        n_mechanistic_eval=4,
        n_natural_eval=6,
        revision="fixed",
        group_field="pmid",
    )

    prepared.assert_group_disjoint(*prepared.splits)
    assert prepared.metadata["group_field"] == "pmid"


def test_jsonl_split_survives_unicode_line_separators(tmp_path: Path) -> None:
    # str.splitlines breaks on U+2028 and friends, which appear inside Civil Comments
    # text; splitting there tears a JSON record in half and the load fails.
    from interpretability_gepa.datasets import load_jsonl_split

    row = {
        "id": "abc",
        "dataset": "civil_comments",
        "text": "first second third\x0bfourth\x85fifth",
        "labels": ["toxicity"],
        "source_id": "1",
    }
    path = tmp_path / "split.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = load_jsonl_split(path)

    assert len(loaded) == 1
    assert loaded[0].text == row["text"]
