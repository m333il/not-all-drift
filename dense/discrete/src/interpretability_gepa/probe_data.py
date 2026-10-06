from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .activations import load_activation_store
from .artifacts import sha256_path
from .datasets import load_jsonl_split
from .errors import ArtifactError
from .metrics import multilabel_matrix

ProbePosition = Literal["last_prompt", "mean_text"]


def ordered_id_hash(example_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(example_ids).encode()).hexdigest()


def _require_unique(example_ids: Sequence[str], *, source: str) -> None:
    if len(example_ids) != len(set(example_ids)):
        raise ArtifactError(f"duplicate example IDs in {source}")


@dataclass(frozen=True, slots=True)
class ProbeDataset:
    example_ids: tuple[str, ...]
    activations: np.ndarray
    targets: np.ndarray
    text_lengths: np.ndarray
    label_counts: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        rows = len(self.example_ids)
        if self.activations.ndim != 3 or self.activations.shape[0] != rows:
            raise ArtifactError("probe activations must have shape [examples, layers, hidden]")
        if self.targets.ndim != 2 or self.targets.shape[0] != rows:
            raise ArtifactError("probe targets must have shape [examples, labels]")
        if self.text_lengths.shape != (rows,) or self.label_counts.shape != (rows,):
            raise ArtifactError("probe covariates must align with examples")
        _require_unique(self.example_ids, source="probe dataset")


@dataclass(frozen=True, slots=True)
class FrozenProbeSubsets:
    seeds: tuple[int, ...]
    indices: tuple[np.ndarray, ...]
    example_ids: tuple[tuple[str, ...], ...]
    id_hashes: dict[int, str]
    master_id_hash: str

    def validate(self, *, master_size: int, expected_size: int | None = None) -> None:
        if len(self.seeds) != len(set(self.seeds)):
            raise ArtifactError("duplicate frozen probe seeds")
        if not (
            len(self.seeds)
            == len(self.indices)
            == len(self.example_ids)
            == len(self.id_hashes)
        ):
            raise ArtifactError("frozen probe subset records are misaligned")
        for seed, indices, ids in zip(
            self.seeds, self.indices, self.example_ids, strict=True
        ):
            if indices.ndim != 1 or len(indices) != len(ids):
                raise ArtifactError(f"invalid frozen subset shape for seed {seed}")
            if expected_size is not None and len(indices) != expected_size:
                raise ArtifactError(
                    f"frozen subset seed {seed} has {len(indices)} rows, expected {expected_size}"
                )
            if len(set(int(index) for index in indices)) != len(indices):
                raise ArtifactError(f"duplicate master indices in frozen subset seed {seed}")
            if np.any(indices < 0) or np.any(indices >= master_size):
                raise ArtifactError(f"out-of-range master index in frozen subset seed {seed}")
            if self.id_hashes.get(seed) != ordered_id_hash(ids):
                raise ArtifactError(f"frozen subset hash mismatch for seed {seed}")


def verify_split_manifest(
    splits_dir: Path,
    *,
    expected_sha256: str,
    split_names: Sequence[str],
) -> dict[str, Any]:
    manifest_path = splits_dir / "manifest.json"
    if sha256_path(manifest_path) != expected_sha256:
        raise ArtifactError("split manifest hash differs from prompt registry")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid split manifest: {manifest_path}") from exc
    records = manifest.get("splits") if isinstance(manifest, dict) else None
    if not isinstance(records, dict):
        raise ArtifactError("split manifest has no splits mapping")
    loaded_ids: dict[str, tuple[str, ...]] = {}
    for name in split_names:
        record = records.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("example_ids"), list):
            raise ArtifactError(f"split manifest has no ordered IDs for {name}")
        declared_ids = tuple(str(value) for value in record["example_ids"])
        actual_ids = tuple(example.id for example in load_jsonl_split(splits_dir / f"{name}.jsonl"))
        if declared_ids != actual_ids:
            raise ArtifactError(f"split file differs from manifest ordered IDs: {name}")
        if int(record.get("size", -1)) != len(actual_ids):
            raise ArtifactError(f"split size differs from manifest: {name}")
        if str(record.get("id_hash")) != ordered_id_hash(actual_ids):
            raise ArtifactError(f"split ID hash differs from manifest: {name}")
        _require_unique(actual_ids, source=name)
        loaded_ids[name] = actual_ids
    if "probe_train" in loaded_ids and "probe_val" in loaded_ids:
        overlap = set(loaded_ids["probe_train"]) & set(loaded_ids["probe_val"])
        if overlap:
            raise ArtifactError("probe train and validation IDs overlap")
    return manifest


def assemble_probe_dataset(
    activation_store: Path,
    split_file: Path,
    labels: Sequence[str],
    *,
    position: ProbePosition,
    expected_metadata: Mapping[str, object] | None = None,
) -> ProbeDataset:
    batch = load_activation_store(activation_store)
    examples = load_jsonl_split(split_file)
    split_ids = tuple(example.id for example in examples)
    _require_unique(batch.example_ids, source=str(activation_store))
    _require_unique(split_ids, source=str(split_file))
    if batch.example_ids != split_ids:
        raise ArtifactError("activation store and split have different ordered example IDs")
    expected_id_hash = ordered_id_hash(split_ids)
    if batch.metadata.get("example_id_hash") != expected_id_hash:
        raise ArtifactError("activation example_id_hash does not match ordered split IDs")
    if batch.metadata.get("split") != split_file.name:
        raise ArtifactError("activation split metadata does not match split file")
    if batch.metadata.get("split_sha256") != sha256_path(split_file):
        raise ArtifactError("activation split SHA-256 does not match split file")
    if expected_metadata:
        for key, expected in expected_metadata.items():
            if batch.metadata.get(key) != expected:
                raise ArtifactError(f"activation metadata differs for {key}")
    allowed = set(labels)
    unknown = sorted({label for example in examples for label in example.labels} - allowed)
    if unknown:
        raise ArtifactError(f"split contains unknown labels: {unknown}")
    activations = getattr(batch, position)
    targets = multilabel_matrix([example.labels for example in examples], labels)
    result = ProbeDataset(
        example_ids=split_ids,
        activations=activations,
        targets=targets,
        text_lengths=np.asarray([len(example.text) for example in examples], dtype=np.int32),
        label_counts=targets.sum(axis=1).astype(np.int8),
        metadata={
            **batch.metadata,
            "position": position,
            "labels": list(labels),
            "split_path": str(split_file.resolve()),
        },
    )
    result.validate()
    return result


def build_frozen_probe_subsets(
    master: ProbeDataset,
    subset_files: Mapping[int, Path],
    *,
    expected_size: int,
) -> FrozenProbeSubsets:
    master.validate()
    if expected_size <= 0 or expected_size > len(master.example_ids):
        raise ArtifactError("frozen probe subset size must fit the master pool")
    index_by_id = {example_id: index for index, example_id in enumerate(master.example_ids)}
    seeds: list[int] = []
    indices: list[np.ndarray] = []
    ids_by_seed: list[tuple[str, ...]] = []
    id_hashes: dict[int, str] = {}
    for seed, path in sorted(subset_files.items()):
        examples = load_jsonl_split(path)
        example_ids = tuple(example.id for example in examples)
        _require_unique(example_ids, source=str(path))
        if len(example_ids) != expected_size:
            raise ArtifactError(
                f"frozen subset seed {seed} has {len(example_ids)} rows, expected {expected_size}"
            )
        outside = sorted(set(example_ids) - set(index_by_id))
        if outside:
            raise ArtifactError(
                f"frozen subset seed {seed} contains IDs outside master: {outside[:3]}"
            )
        seeds.append(seed)
        indices.append(np.asarray([index_by_id[example_id] for example_id in example_ids]))
        ids_by_seed.append(example_ids)
        id_hashes[seed] = ordered_id_hash(example_ids)
    result = FrozenProbeSubsets(
        seeds=tuple(seeds),
        indices=tuple(indices),
        example_ids=tuple(ids_by_seed),
        id_hashes=id_hashes,
        master_id_hash=ordered_id_hash(master.example_ids),
    )
    result.validate(master_size=len(master.example_ids), expected_size=expected_size)
    return result


def save_frozen_probe_subsets(path: Path, subsets: FrozenProbeSubsets) -> None:
    payload = {
        "master_id_hash": subsets.master_id_hash,
        "subsets": [
            {
                "seed": seed,
                "example_ids": list(example_ids),
                "id_hash": subsets.id_hashes[seed],
            }
            for seed, example_ids in zip(subsets.seeds, subsets.example_ids, strict=True)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "FrozenProbeSubsets",
    "ProbeDataset",
    "ProbePosition",
    "assemble_probe_dataset",
    "build_frozen_probe_subsets",
    "ordered_id_hash",
    "save_frozen_probe_subsets",
    "verify_split_manifest",
]
