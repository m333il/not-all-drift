from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from .artifacts import sha256_path
from .errors import ArtifactError
from .metrics import evaluate_multilabel_arrays
from .probe_data import ordered_id_hash
from .probes import train_layerwise_probes


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid probing artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"probing artifact must be a mapping: {path}")
    return payload


def _atomic_directory(output: Path) -> Path:
    if output.exists():
        raise ArtifactError(f"probing output already exists: {output}")
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise ArtifactError(f"temporary probing output already exists: {temporary}")
    temporary.mkdir(parents=True)
    return temporary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def fit_frozen_probes(
    assembled: Path,
    output: Path,
    *,
    c_values: tuple[float, ...],
    threshold: float,
) -> Path:
    arrays_path = assembled / "arrays.npz"
    source = np.load(arrays_path)
    provenance = _read_json(assembled / "provenance.json")
    required = {
        "x_train",
        "y_train",
        "x_val",
        "y_val",
        "subset_seeds",
        "subset_indices",
        "subset_hashes",
        "example_ids_train",
        "example_ids_val",
    }
    if missing := required - set(source.files):
        raise ArtifactError(f"assembled probe arrays are missing: {sorted(missing)}")
    x_train = source["x_train"]
    y_train = source["y_train"]
    x_val = source["x_val"]
    y_val = source["y_val"]
    seeds = [int(value) for value in source["subset_seeds"]]
    subset_indices = source["subset_indices"]
    subset_hashes = [str(value) for value in source["subset_hashes"]]
    train_ids = tuple(str(value) for value in source["example_ids_train"])
    val_ids = tuple(str(value) for value in source["example_ids_val"])
    if len(seeds) != len(subset_indices) or len(seeds) != len(subset_hashes):
        raise ArtifactError("assembled frozen subset arrays are misaligned")
    if tuple(seeds) != tuple(int(value) for value in provenance["subset_hashes"]):
        raise ArtifactError("assembled subset seeds differ from provenance")
    if list(c_values) != provenance.get("c_values") or threshold != provenance.get("threshold"):
        raise ArtifactError("probe hyperparameters differ from assembled provenance")
    if ordered_id_hash(train_ids) != provenance.get("train_id_hash"):
        raise ArtifactError("assembled training IDs differ from provenance")
    if ordered_id_hash(val_ids) != provenance.get("val_id_hash"):
        raise ArtifactError("assembled validation IDs differ from provenance")
    subsets_payload = _read_json(assembled / "subsets.json")
    subset_records = subsets_payload.get("subsets")
    if not isinstance(subset_records, list) or len(subset_records) != len(seeds):
        raise ArtifactError("assembled subsets.json is incomplete")
    subset_by_seed = {int(record["seed"]): record for record in subset_records}
    for seed, indices, subset_hash in zip(seeds, subset_indices, subset_hashes, strict=True):
        if not np.issubdtype(indices.dtype, np.integer):
            raise ArtifactError(f"frozen subset indices are not integers for seed {seed}")
        if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
            raise ArtifactError(f"frozen subset indices are invalid for seed {seed}")
        if np.any(indices < 0) or np.any(indices >= len(train_ids)):
            raise ArtifactError(f"frozen subset indices are out of bounds for seed {seed}")
        selected_ids = tuple(train_ids[int(index)] for index in indices)
        record = subset_by_seed.get(seed)
        if not isinstance(record, dict):
            raise ArtifactError(f"missing frozen subset record for seed {seed}")
        if tuple(str(value) for value in record.get("example_ids", ())) != selected_ids:
            raise ArtifactError(f"frozen subset IDs differ for seed {seed}")
        expected_hash = ordered_id_hash(selected_ids)
        if (
            subset_hash != expected_hash
            or str(record.get("id_hash")) != expected_hash
            or provenance["subset_hashes"].get(str(seed)) != expected_hash
        ):
            raise ArtifactError(f"frozen subset hash differs for seed {seed}")
    temporary = _atomic_directory(output)
    rows: list[dict[str, Any]] = []
    try:
        for seed, indices, subset_hash in zip(
            seeds, subset_indices, subset_hashes, strict=True
        ):
            results = train_layerwise_probes(
                x_train[indices],
                y_train[indices],
                x_val,
                y_val,
                c_values=c_values,
                threshold=threshold,
            )
            np.savez_compressed(
                temporary / f"probe_seed{seed}.npz",
                weights=np.stack([result.weights for result in results]),
                intercepts=np.stack([result.intercepts for result in results]),
                thresholds=np.stack([result.thresholds for result in results]),
                validation_f1=np.asarray([result.validation_f1 for result in results]),
                validation_aurocs=np.asarray(
                    [
                        [np.nan if value is None else value for value in result.validation_aurocs]
                        for result in results
                    ]
                ),
                c=np.asarray([result.c for result in results]),
                subset_seed=np.asarray(seed),
                subset_hash=np.asarray(subset_hash),
            )
            for layer, result in enumerate(results):
                logits = x_val[:, layer] @ result.weights.T + result.intercepts
                probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
                predictions = probabilities >= result.thresholds
                metrics = evaluate_multilabel_arrays(y_val, predictions, probabilities)
                rows.append(
                    {
                        "subset_seed": seed,
                        "subset_hash": subset_hash,
                        "subset_size": len(indices),
                        "layer": layer,
                        "selected_c": result.c,
                        "selection_f1_samples_empty_aware": result.validation_f1,
                        "condition_control_status": "omitted_stage_a_single_condition",
                        **metrics,
                    }
                )
        _write_jsonl(temporary / "metrics.jsonl", rows)
        fit_manifest = {
            "schema_version": 1,
            "assembled": str(assembled.resolve()),
            "assembled_sha256": sha256_path(assembled),
            "prompt_id": provenance["prompt_id"],
            "prompt_kind": provenance["prompt_kind"],
            "prompt_registry_sha256": provenance["prompt_registry_sha256"],
            "model_id": provenance["model_id"],
            "model_revision": provenance["model_revision"],
            "tokenizer_revision": provenance["tokenizer_revision"],
            "template_hash": provenance["template_hash"],
            "adapter_hash": provenance["adapter_hash"],
            "non_thinking": provenance["non_thinking"],
            "split_manifest_sha256": provenance["split_manifest_sha256"],
            "val_split_sha256": provenance["val_split_sha256"],
            "val_id_hash": provenance["val_id_hash"],
            "val_rows": provenance["val_rows"],
            "validation_example_ids": list(val_ids),
            "position": provenance["position"],
            "labels": provenance["labels"],
            "layers": provenance["layers"],
            "hidden_size": provenance["hidden_size"],
            "subset_seeds": seeds,
            "subset_hashes": dict(zip((str(seed) for seed in seeds), subset_hashes, strict=True)),
            "c_values": list(c_values),
            "threshold": threshold,
            "preprocessing": "none",
            "sklearn_version": sklearn.__version__,
        }
        (temporary / "fit_manifest.json").write_text(
            json.dumps(fit_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _assert_transfer_compatible(source: dict[str, Any], target: dict[str, Any]) -> None:
    for key in (
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "template_hash",
        "adapter_hash",
        "non_thinking",
        "position",
        "labels",
        "layers",
        "hidden_size",
        "prompt_registry_sha256",
        "split_manifest_sha256",
        "val_split_sha256",
        "val_id_hash",
        "val_rows",
    ):
        if source.get(key) != target.get(key):
            raise ArtifactError(f"cross-condition probing metadata differs for {key}")


def evaluate_frozen_probes(fitted: Path, assembled: Path, output: Path) -> Path:
    manifest = _read_json(fitted / "fit_manifest.json")
    target_provenance = _read_json(assembled / "provenance.json")
    _assert_transfer_compatible(manifest, target_provenance)
    arrays = np.load(assembled / "arrays.npz")
    x_val = arrays["x_val"]
    y_val = arrays["y_val"]
    target_val_ids = tuple(str(value) for value in arrays["example_ids_val"])
    if tuple(manifest["validation_example_ids"]) != target_val_ids:
        raise ArtifactError("cross-condition validation example IDs differ")
    if ordered_id_hash(target_val_ids) != target_provenance["val_id_hash"]:
        raise ArtifactError("target validation IDs differ from provenance")
    if len(target_val_ids) != x_val.shape[0] or x_val.shape[0] != y_val.shape[0]:
        raise ArtifactError("target validation arrays have inconsistent row counts")
    rows: list[dict[str, Any]] = []
    for seed in manifest["subset_seeds"]:
        model = np.load(fitted / f"probe_seed{int(seed)}.npz")
        if str(model["subset_hash"]) != str(manifest["subset_hashes"][str(seed)]):
            raise ArtifactError(f"probe subset hash mismatch for seed {seed}")
        for layer in range(x_val.shape[1]):
            logits = x_val[:, layer] @ model["weights"][layer].T + model["intercepts"][layer]
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
            predictions = probabilities >= model["thresholds"][layer]
            rows.append(
                {
                    "train_prompt_id": manifest["prompt_id"],
                    "eval_prompt_id": target_provenance["prompt_id"],
                    "subset_seed": int(seed),
                    "subset_hash": str(model["subset_hash"]),
                    "layer": layer,
                    **evaluate_multilabel_arrays(y_val, predictions, probabilities),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ArtifactError(f"probe evaluation output already exists: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    _write_jsonl(temporary, rows)
    temporary.replace(output)
    return output


def select_common_layer(fitted: Path, output: Path) -> Path:
    manifest = _read_json(fitted / "fit_manifest.json")
    if manifest.get("prompt_id") != "C_seed" or manifest.get("prompt_kind") != "C_seed":
        raise ArtifactError("common layer selection requires fitted C_seed probes")
    layer_scores: dict[int, list[float]] = defaultdict(list)
    observed: set[tuple[int, int]] = set()
    for line in (fitted / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (int(row["layer"]), int(row["subset_seed"]))
        if key in observed:
            raise ArtifactError("seed prompt layer metrics contain duplicate seed-layer rows")
        observed.add(key)
        layer_scores[key[0]].append(float(row["selection_f1_samples_empty_aware"]))
    expected_seeds = len(manifest["subset_seeds"])
    expected_grid = {
        (layer, int(seed))
        for layer in range(int(manifest["layers"]))
        for seed in manifest["subset_seeds"]
    }
    if observed != expected_grid or any(
        len(values) != expected_seeds for values in layer_scores.values()
    ):
        raise ArtifactError("seed prompt layer metrics are incomplete")
    means = {layer: float(np.mean(values)) for layer, values in layer_scores.items()}
    selected = min(means, key=lambda layer: (-means[layer], layer))
    ledger = {
        "schema_version": 1,
        "selection_condition": manifest["prompt_id"],
        "selection_metric": "f1_samples_empty_aware",
        "selected_layer": selected,
        "tie_break": "lower_layer_index",
        "mean_validation_scores": {str(layer): means[layer] for layer in sorted(means)},
        "fit_manifest_sha256": sha256_path(fitted / "fit_manifest.json"),
        "fit_artifact_sha256": sha256_path(fitted),
        "metrics_sha256": sha256_path(fitted / "metrics.jsonl"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ArtifactError(f"selection ledger already exists: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return output


__all__ = ["evaluate_frozen_probes", "fit_frozen_probes", "select_common_layer"]
