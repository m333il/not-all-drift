from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np

from .config import DatasetConfig
from .errors import ConfigurationError
from .schemas import Example, stable_example_id

GOEMOTIONS_LABELS = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)
# Schema of the v2 splits (no ``sexual_explicit``). Kept in sync with
# ``civil_splits_v2.contract.CIVIL_LABELS_V2`` by a test.
CIVIL_LABELS = (
    "toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
)
CIVIL_COLUMNS = {
    "toxicity": "toxicity",
    "obscene": "obscene",
    "threat": "threat",
    "insult": "insult",
    "identity_attack": "identity_attack",
}
HALLMARK_LABELS = (
    "activating_invasion_and_metastasis",
    "avoiding_immune_destruction",
    "cellular_energetics",
    "enabling_replicative_immortality",
    "evading_growth_suppressors",
    "genomic_instability_and_mutation",
    "inducing_angiogenesis",
    "resisting_cell_death",
    "sustaining_proliferative_signaling",
    "tumor_promoting_inflammation",
)
HALLMARK_SOURCE_LABELS = (
    "evading growth suppressors",
    "tumor promoting inflammation",
    "enabling replicative immortality",
    "cellular energetics",
    "resisting cell death",
    "activating invasion and metastasis",
    "genomic instability and mutation",
    "none",
    "inducing angiogenesis",
    "sustaining proliferative signaling",
    "avoiding immune destruction",
)
HALLMARK_SOURCE_TO_CANONICAL = {
    "evading growth suppressors": "evading_growth_suppressors",
    "tumor promoting inflammation": "tumor_promoting_inflammation",
    "enabling replicative immortality": "enabling_replicative_immortality",
    "cellular energetics": "cellular_energetics",
    "resisting cell death": "resisting_cell_death",
    "activating invasion and metastasis": "activating_invasion_and_metastasis",
    "genomic instability and mutation": "genomic_instability_and_mutation",
    "inducing angiogenesis": "inducing_angiogenesis",
    "sustaining proliferative signaling": "sustaining_proliferative_signaling",
    "avoiding immune destruction": "avoiding_immune_destruction",
}

DatasetRows = tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[str, ...]]


class DatasetSource:
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg

    def load(self) -> DatasetRows:
        raise NotImplementedError


DatasetSourceType = TypeVar("DatasetSourceType", bound=type[DatasetSource])
DATASET_FACTORY: dict[str, type[DatasetSource]] = {}


def register_dataset(
    name: str,
) -> Callable[[DatasetSourceType], DatasetSourceType]:
    def decorator(cls: DatasetSourceType) -> DatasetSourceType:
        if name in DATASET_FACTORY:
            raise RuntimeError(f"dataset already registered: {name}")
        DATASET_FACTORY[name] = cls
        return cls

    return decorator


def dataset_factory(name: str) -> type[DatasetSource]:
    try:
        return DATASET_FACTORY[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_FACTORY))
        raise ConfigurationError(f"unknown dataset {name!r}; available: {available}") from exc


def labels_for_dataset(dataset: str) -> tuple[str, ...]:
    if dataset == "goemotions":
        return GOEMOTIONS_LABELS
    if dataset == "civil_comments":
        return CIVIL_LABELS
    if dataset == "hallmarks_of_cancer":
        return HALLMARK_LABELS
    raise ConfigurationError(f"no fixed label schema for {dataset}")


@dataclass(frozen=True)
class PreparedSplits:
    labels: tuple[str, ...]
    splits: dict[str, tuple[Example, ...]]
    metadata: dict[str, Any]

    def assert_disjoint(self, *names: str) -> None:
        seen: set[str] = set()
        for name in names:
            ids = {item.id for item in self.splits[name]}
            overlap = ids & seen
            if overlap:
                raise ConfigurationError(f"split leakage in {name}: {sorted(overlap)[:3]}")
            seen |= ids

    def assert_group_disjoint(self, *names: str) -> None:
        seen: set[str] = set()
        for name in names:
            groups = {item.group_id for item in self.splits[name] if item.group_id is not None}
            overlap = groups & seen
            if overlap:
                raise ConfigurationError(f"group leakage in {name}: {sorted(overlap)[:3]}")
            seen |= groups

    def write(self, directory: Path) -> None:
        import pandas as pd

        directory.mkdir(parents=True, exist_ok=False)
        for name, examples in self.splits.items():
            path = directory / f"{name}.jsonl"
            records = [x.to_dict() for x in examples]
            path.write_text(
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8"
            )
            pd.DataFrame(records).to_parquet(directory / f"{name}.parquet", index=False)
        split_hashes = {
            name: hashlib.sha256("\n".join(x.id for x in examples).encode()).hexdigest()
            for name, examples in self.splits.items()
        }
        manifest = {
            "labels": list(self.labels),
            "metadata": self.metadata,
            "split_hashes": split_hashes,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )


def load_jsonl_split(path: Path) -> list[Example]:
    examples = []
    # Split on "\n" only: Civil Comments text contains U+2028 and similar separators.
    with path.open(encoding="utf-8", newline="\n") as handle:
        lines = [line for line in handle.read().split("\n") if line]
    for line in lines:
        row = json.loads(line)
        examples.append(
            Example(
                row["id"],
                row["dataset"],
                row["text"],
                tuple(row["labels"]),
                str(row["source_id"]),
                None if row.get("group_id") is None else str(row["group_id"]),
                tuple((str(key), float(value)) for key, value in row.get("scores", {}).items()),
            )
        )
    return examples


def optimizer_train_split_path(
    directory: Path, dataset: str, *, seed: int, train_size: int
) -> Path:
    name = _optimizer_train_split_name(dataset, seed=seed, train_size=train_size)
    return directory / f"{name}.jsonl"


def optimizer_val_split_path(directory: Path, *, seed: int, val_size: int) -> Path:
    """Resolve the validation split; v2 names carry the size, v1 names do not."""
    sized = directory / f"optimizer_val_seed{seed}_n{val_size}.jsonl"
    return sized if sized.exists() else directory / f"optimizer_val_seed{seed}.jsonl"


def _optimizer_train_split_name(dataset: str, *, seed: int, train_size: int) -> str:
    if dataset == "civil_comments":
        return f"optimizer_train_seed{seed}_n{train_size}"
    return f"optimizer_train_seed{seed}"


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def clean_goemotions_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deduplicate normalized texts and remove every conflicting-label group."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_normalize_text(str(row["text"])), []).append(row)
    cleaned: list[dict[str, Any]] = []
    conflicting_rows = 0
    deduplicated_rows = 0
    for copies in grouped.values():
        label_sets = {tuple(sorted(set(copy["labels"]))) for copy in copies}
        if len(label_sets) > 1:
            conflicting_rows += len(copies)
            continue
        cleaned.append(copies[0])
        deduplicated_rows += len(copies) - 1
    return cleaned, {
        "input_rows": len(rows),
        "deduplicated_rows": deduplicated_rows,
        "conflicting_rows": conflicting_rows,
    }


def choose_training_size(
    pilot_rows: Sequence[dict[str, Any]],
    label_counts: dict[int, dict[str, tuple[int, int]]],
    labels: Sequence[str],
    *,
    candidates: Sequence[int] = (100, 300, 500),
) -> tuple[int, tuple[str, ...]]:
    """Apply the preregistered N escalation rule using validation data only."""
    for size in candidates:
        reasons: list[str] = []
        coverage = label_counts.get(size, {})
        missing = [
            label
            for label in labels
            if coverage.get(label, (0, 0))[0] < 3 or coverage.get(label, (0, 0))[1] < 2
        ]
        if missing:
            reasons.append(f"insufficient label coverage: {','.join(missing)}")
        current = [row for row in pilot_rows if row["n"] == size]
        for branch in ("gepa", "prefix"):
            gains = np.asarray(
                [float(row["val_gain"]) for row in current if row["branch"] == branch]
            )
            if len(gains) < 3:
                reasons.append(f"{branch} has fewer than three pilot seeds")
            elif gains.mean() < 0.02 or gains.std(ddof=1) >= abs(gains.mean()):
                reasons.append(f"{branch} improvement is small or unstable")
            if any(bool(row.get("collapsed")) for row in current if row["branch"] == branch):
                reasons.append(f"{branch} collapsed")
        if not reasons or size == candidates[-1]:
            return size, tuple(reasons)
    raise AssertionError("candidate sizes cannot be empty")


def civil_labels(row: dict[str, Any], threshold: float = 0.5) -> tuple[str, ...]:
    return tuple(
        label for label in CIVIL_LABELS if float(row.get(CIVIL_COLUMNS[label], 0.0)) >= threshold
    )


def hallmark_labels(
    indices: Sequence[int], source_names: Sequence[str] = HALLMARK_SOURCE_LABELS
) -> tuple[str, ...]:
    result = []
    for index in indices:
        try:
            source_name = source_names[index]
        except IndexError as exc:
            raise ConfigurationError(f"Hallmarks label index out of range: {index}") from exc
        if source_name == "none":
            continue
        try:
            result.append(HALLMARK_SOURCE_TO_CANONICAL[source_name])
        except KeyError as exc:
            raise ConfigurationError(f"unknown Hallmarks source label: {source_name}") from exc
    order = {label: index for index, label in enumerate(HALLMARK_LABELS)}
    return tuple(sorted(set(result), key=order.__getitem__))


def _as_examples(
    dataset: str,
    rows: Iterable[dict[str, Any]],
    labels: Sequence[str],
    group_field: str | None = None,
) -> list[Example]:
    order = {label: index for index, label in enumerate(labels)}
    result: list[Example] = []
    for index, row in enumerate(rows):
        text = str(row["text"])
        source_id = str(row.get("id", index))
        raw_labels = tuple(row["labels"])
        unknown = set(raw_labels) - set(labels)
        if unknown:
            raise ConfigurationError(f"unknown {dataset} labels: {sorted(unknown)}")
        canonical = tuple(sorted(set(raw_labels), key=order.__getitem__))
        result.append(
            Example(
                stable_example_id(dataset, text, source_id),
                dataset,
                text,
                canonical,
                source_id,
                None if group_field is None else str(row[group_field]),
                tuple(
                    sorted(
                        (str(key), float(value))
                        for key, value in dict(row.get("scores", {})).items()
                    )
                ),
            )
        )
    return result


def _balanced_sample(
    pool: Sequence[Example], size: int, labels: Sequence[str], seed: int, *, include_empty: bool
) -> list[Example]:
    if size > len(pool):
        raise ConfigurationError(f"requested {size} examples from a pool of {len(pool)}")
    rng = random.Random(seed)
    targets = list(labels) + (["__none__"] if include_empty else [])
    buckets: dict[str, list[int]] = {target: [] for target in targets}
    order = list(range(len(pool)))
    rng.shuffle(order)
    for index in order:
        keys = pool[index].labels or (("__none__",) if include_empty else ())
        for key in keys:
            if key in buckets:
                buckets[key].append(index)
    selected: set[int] = set()
    counts = Counter[str]()
    while len(selected) < size:
        available_targets = [
            target
            for target, bucket in buckets.items()
            if any(index not in selected for index in bucket)
        ]
        if not available_targets:
            break
        target = min(available_targets, key=lambda key: (counts[key], targets.index(key)))
        bucket = buckets[target]
        while bucket and bucket[-1] in selected:
            bucket.pop()
        if not bucket:
            continue
        winner = bucket.pop()
        selected.add(winner)
        counts.update(pool[winner].labels or ("__none__",))
    if len(selected) < size:
        selected.update(index for index in order if index not in selected and len(selected) < size)
    return [pool[index] for index in order if index in selected]


def _take_random(pool: Sequence[Example], size: int, seed: int) -> list[Example]:
    if size > len(pool):
        size = len(pool)
    return random.Random(seed).sample(list(pool), size)


def _take_label_cover(
    pool: Sequence[Example], size: int, labels: Sequence[str], seed: int
) -> list[Example]:
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    missing = set(labels)
    selected: list[Example] = []
    while missing and len(selected) < size:
        candidates = [index for index in order if missing.intersection(pool[index].labels)]
        if not candidates:
            break
        winner = max(candidates, key=lambda index: len(missing.intersection(pool[index].labels)))
        selected.append(pool[winner])
        order.remove(winner)
        missing.difference_update(pool[winner].labels)
    if missing:
        raise ConfigurationError(
            f"cannot reserve label coverage within {size} examples; missing {sorted(missing)}"
        )
    return selected


def _allocate_covered_splits(
    pool: Sequence[Example],
    specs: Sequence[tuple[str, int, int]],
    labels: Sequence[str],
) -> tuple[dict[str, list[Example]], list[Example]]:
    available = list(pool)
    reserved: dict[str, list[Example]] = {}
    for name, size, seed in specs:
        cover = _take_label_cover(available, size, labels, seed)
        reserved[name] = cover
        cover_ids = {x.id for x in cover}
        available = [x for x in available if x.id not in cover_ids]

    allocated: dict[str, list[Example]] = {}
    for name, size, seed in specs:
        cover = reserved[name]
        selected = cover + _balanced_sample(
            available, size - len(cover), labels, seed, include_empty=True
        )
        random.Random(seed + 1).shuffle(selected)
        allocated[name] = selected
        selected_ids = {x.id for x in selected}
        available = [x for x in available if x.id not in selected_ids]
    return allocated, available


def _take_grouped(
    pool: Sequence[Example], size: int, seed: int, labels: Sequence[str] = ()
) -> list[Example]:
    groups: dict[str, list[Example]] = {}
    for example in pool:
        if example.group_id is None:
            raise ConfigurationError("grouped sampling requires group IDs")
        groups.setdefault(example.group_id, []).append(example)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    selected: list[Example] = []
    remaining = keys.copy()
    missing = set(labels)
    while remaining and len(selected) < size:
        key = max(
            remaining,
            key=lambda candidate: len(
                missing & {label for item in groups[candidate] for label in item.labels}
            ),
        )
        remaining.remove(key)
        selected.extend(groups[key])
        missing -= {label for item in groups[key] for label in item.labels}
    if len(selected) < size:
        raise ConfigurationError(f"requested {size} grouped examples from a pool of {len(pool)}")
    return selected


def prepare_from_rows(
    *,
    dataset: str,
    labels: Sequence[str],
    train_rows: Iterable[dict[str, Any]],
    test_rows: Iterable[dict[str, Any]],
    seeds: Sequence[int],
    n_train: int,
    n_optimizer_val: int,
    n_probe_train: int,
    n_probe_val: int,
    n_intervention_val: int,
    n_mechanistic_eval: int,
    n_natural_eval: int,
    revision: str,
    group_field: str | None = None,
) -> PreparedSplits:
    train_source = list(train_rows)
    test_source = list(test_rows)
    cleaning_audit: dict[str, int] | None = None
    train_source, train_audit = clean_goemotions_rows(train_source)
    test_source, test_audit = clean_goemotions_rows(test_source)
    train_norms = {_normalize_text(str(row["text"])) for row in train_source}
    before_cross_split = len(test_source)
    test_source = [
        row for row in test_source if _normalize_text(str(row["text"])) not in train_norms
    ]
    cleaning_audit = {
        key: train_audit[key] + test_audit[key]
        for key in ("input_rows", "deduplicated_rows", "conflicting_rows")
    }
    cleaning_audit["cross_split_rows"] = before_cross_split - len(test_source)
    train = _as_examples(dataset, train_source, labels, group_field)
    test = _as_examples(dataset, test_source, labels, group_field)
    splits: dict[str, tuple[Example, ...]] = {}
    if group_field:
        excluded: set[str] = set()
        for seed in seeds:
            available = [x for x in train if x.id not in excluded]
            optimizer_train = _take_grouped(available, n_train, seed, labels)
            excluded.update(x.id for x in optimizer_train)
            available = [x for x in train if x.id not in excluded]
            optimizer_val = _take_grouped(available, n_optimizer_val, seed + 1000, labels)
            train_name = _optimizer_train_split_name(dataset, seed=seed, train_size=n_train)
            splits[train_name] = tuple(optimizer_train)
            splits[f"optimizer_val_seed{seed}"] = tuple(optimizer_val)
            excluded.update(x.id for x in optimizer_val)
        available = [x for x in train if x.id not in excluded]
        for name, size, seed in (
            ("probe_train", n_probe_train, 9917),
            ("probe_val", n_probe_val, 9919),
            ("intervention_val", n_intervention_val, 9923),
        ):
            selected = _take_grouped(available, size, seed, labels)
            splits[name] = tuple(selected)
            selected_groups = {x.group_id for x in selected}
            available = [x for x in available if x.group_id not in selected_groups]
    else:
        optimizer_specs = tuple(
            spec
            for seed in seeds
            for spec in (
                (
                    _optimizer_train_split_name(dataset, seed=seed, train_size=n_train),
                    n_train,
                    seed,
                ),
                (f"optimizer_val_seed{seed}", n_optimizer_val, seed + 1000),
            )
        )
        optimizer_splits, available = _allocate_covered_splits(train, optimizer_specs, labels)
        splits.update({name: tuple(items) for name, items in optimizer_splits.items()})
        auxiliary_specs = (
            ("probe_train", n_probe_train, 9917),
            ("probe_val", n_probe_val, 9919),
            ("intervention_val", n_intervention_val, 9923),
        )
        auxiliary_splits, available = _allocate_covered_splits(available, auxiliary_specs, labels)
        splits.update({name: tuple(items) for name, items in auxiliary_splits.items()})

    natural = (
        _take_grouped(test, n_natural_eval, 1729, labels)
        if group_field
        else _take_random(test, n_natural_eval, 1729)
    )
    if dataset == "civil_comments":
        remaining = [x for x in test if x.id not in {n.id for n in natural}]
        n_empty = n_mechanistic_eval // 2
        empty = _take_random([x for x in remaining if not x.labels], n_empty, 1733)
        positive = _balanced_sample(
            [x for x in remaining if x.labels],
            n_mechanistic_eval - len(empty),
            labels,
            1739,
            include_empty=False,
        )
        mechanistic = empty + positive
        random.Random(1741).shuffle(mechanistic)
    elif group_field:
        natural_groups = {x.group_id for x in natural}
        remaining = [x for x in test if x.group_id not in natural_groups]
        mechanistic = _take_grouped(remaining, n_mechanistic_eval, 1733, labels)
    else:
        mechanistic = _balanced_sample(
            natural, min(n_mechanistic_eval, len(natural)), labels, 1733, include_empty=False
        )
    splits["eval_natural"] = tuple(natural)
    splits["eval_mechanistic"] = tuple(mechanistic)

    required_splits = ["probe_train", "eval_mechanistic"]
    if group_field is None:
        required_splits.extend(name for name in splits if name.startswith("optimizer_"))
        required_splits.extend(("probe_val", "intervention_val"))
    for required_split in required_splits:
        present = {label for example in splits[required_split] for label in example.labels}
        missing = set(labels) - present
        if missing:
            raise ConfigurationError(
                f"{required_split} lacks positive examples for labels: {sorted(missing)}"
            )

    all_disjoint = [
        name for name in splits if not (dataset != "civil_comments" and name == "eval_mechanistic")
    ]
    prepared = PreparedSplits(
        tuple(labels),
        splits,
        {
            "dataset": dataset,
            "revision": revision,
            "group_field": group_field,
            "cleaning_audit": cleaning_audit,
            "source_hash": hashlib.sha256(
                "".join(sorted(x.id for x in train + test)).encode()
            ).hexdigest(),
            "counts": {name: len(items) for name, items in splits.items()},
            "label_counts": {
                name: dict(Counter(label for x in items for label in x.labels))
                for name, items in splits.items()
            },
        },
    )
    prepared.assert_disjoint(*all_disjoint)
    if group_field:
        prepared.assert_group_disjoint(*splits)
    return prepared


def _load_dataset_dependency() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ConfigurationError("install the 'data' extra to download datasets") from exc
    return load_dataset


@register_dataset("goemotions")
class GoEmotionsSource(DatasetSource):
    def load(self) -> DatasetRows:
        load_dataset = _load_dataset_dependency()
        train = load_dataset(
            "google-research-datasets/go_emotions",
            "simplified",
            split="train",
            revision=self.cfg.revision,
        )
        test = load_dataset(
            "google-research-datasets/go_emotions",
            "simplified",
            split="test",
            revision=self.cfg.revision,
        )
        names = tuple(train.features["labels"].feature.names)
        if names != GOEMOTIONS_LABELS:
            raise ConfigurationError("GoEmotions label schema changed; pin a known revision")

        def convert(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "id": row.get("id"),
                "text": row["text"],
                "labels": [names[i] for i in row["labels"]],
            }

        return [convert(x) for x in train], [convert(x) for x in test], names


@register_dataset("civil_comments")
class CivilCommentsSource(DatasetSource):
    def load(self) -> DatasetRows:
        if re.fullmatch(r"[0-9a-f]{40}", self.cfg.revision) is None:
            raise ConfigurationError(
                "Civil Comments remote dataset code requires a full commit SHA revision"
            )
        load_dataset = _load_dataset_dependency()
        train = load_dataset(
            "google/civil_comments",
            split="train",
            revision=self.cfg.revision,
            trust_remote_code=True,
        )
        test = load_dataset(
            "google/civil_comments",
            split="test",
            revision=self.cfg.revision,
            trust_remote_code=True,
        )

        def convert(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "id": row.get("id"),
                "text": row["text"],
                "labels": list(civil_labels(row, self.cfg.binarization_threshold)),
                "scores": {
                    label: float(row.get(column, 0.0)) for label, column in CIVIL_COLUMNS.items()
                },
            }

        return [convert(x) for x in train], [convert(x) for x in test], CIVIL_LABELS


@register_dataset("hallmarks_of_cancer")
class HallmarksSource(DatasetSource):
    def load(self) -> DatasetRows:
        load_dataset = _load_dataset_dependency()
        source = load_dataset("qanastek/HoC", revision=self.cfg.revision)
        train_split = list(source["train"]) + list(source.get("validation", []))
        test_split = list(source["test"])
        names = tuple(source["train"].features["label"].feature.names)
        if names != HALLMARK_SOURCE_LABELS:
            raise ConfigurationError("Hallmarks of Cancer label schema changed")

        def convert_hallmarks(row: dict[str, Any]) -> dict[str, Any]:
            document_id = str(row["document_id"])
            pmid = document_id.split("_", 1)[0]
            return {
                "id": document_id,
                "text": row["text"],
                "labels": list(hallmark_labels(row["label"], names)),
                "pmid": pmid,
            }

        return (
            [convert_hallmarks(x) for x in train_split],
            [convert_hallmarks(x) for x in test_split],
            HALLMARK_LABELS,
        )


def load_huggingface_rows(
    dataset: Literal["goemotions", "civil_comments", "hallmarks_of_cancer"],
    revision: str,
    *,
    binarization_threshold: float = 0.5,
) -> DatasetRows:
    cfg = DatasetConfig(
        id=dataset,
        revision=revision,
        group_field="pmid" if dataset == "hallmarks_of_cancer" else None,
        binarization_threshold=binarization_threshold,
    )
    return dataset_factory(dataset)(cfg).load()
