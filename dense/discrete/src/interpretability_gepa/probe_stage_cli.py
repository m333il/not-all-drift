from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from .artifacts import sha256_path
from .config import load_config
from .datasets import labels_for_dataset
from .errors import ArtifactError
from .preregistration import load_preregistration
from .probe_data import (
    ProbePosition,
    assemble_probe_dataset,
    build_frozen_probe_subsets,
    save_frozen_probe_subsets,
    verify_split_manifest,
)
from .probe_workflow import evaluate_frozen_probes, fit_frozen_probes, select_common_layer
from .prompt_registry import (
    build_prompt_registry,
    load_prompt_registry,
    save_prompt_registry,
)
from .prompts import SEED_INSTRUCTIONS


def _integer_list(value: str, *, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"{name} must be a comma-separated integer list") from exc
    if not result or len(result) != len(set(result)):
        raise typer.BadParameter(f"{name} must contain unique integers")
    return result


def build_registry_command(
    run_dir: Annotated[
        list[Path], typer.Option("--run-dir", exists=True, file_okay=False, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option("--output")],
    seeds: str = "42,43,44",
    train_sizes: str = "100,200,500",
    legacy_assertions: Annotated[
        Path | None, typer.Option("--legacy-assertions", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Build an immutable registry for the completed GEPA prompt grid."""
    if not run_dir:
        raise typer.BadParameter("at least one --run-dir is required")
    assertion_payload: dict[str, dict[str, Any]] = {}
    if legacy_assertions is not None:
        decoded = json.loads(legacy_assertions.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict) or not isinstance(decoded.get("runs"), dict):
            raise typer.BadParameter("legacy assertions must contain a runs mapping")
        assertion_payload = decoded["runs"]
    first_setup_path = run_dir[0] / "setup.json"
    if first_setup_path.exists():
        first_setup = json.loads(first_setup_path.read_text(encoding="utf-8"))
    else:
        first_assertion = assertion_payload.get(str(run_dir[0].resolve()))
        if not isinstance(first_assertion, dict) or not isinstance(
            first_assertion.get("setup"), dict
        ):
            raise typer.BadParameter("first run requires setup or a legacy assertion")
        first_setup = first_assertion["setup"]
    dataset_id = str(first_setup["dataset"])
    try:
        seed_instruction = SEED_INSTRUCTIONS[dataset_id]
    except KeyError as exc:
        raise typer.BadParameter(f"no frozen seed instruction for {dataset_id}") from exc
    expected_cells = tuple(
        (seed, train_size)
        for seed in _integer_list(seeds, name="seeds")
        for train_size in _integer_list(train_sizes, name="train-sizes")
    )
    registry = build_prompt_registry(
        run_dir,
        seed_instruction=seed_instruction,
        expected_cells=expected_cells,
        setup_assertions=assertion_payload,
    )
    save_prompt_registry(output, registry)
    typer.echo(output)


def _assert_pair_compatible(
    train_metadata: dict[str, object], val_metadata: dict[str, object]
) -> None:
    for key in (
        "model",
        "model_revision",
        "tokenizer_revision",
        "template_hash",
        "adapter_hash",
        "non_thinking",
        "condition",
        "prompt_id",
        "instruction_sha256",
        "prompt_registry_sha256",
    ):
        if train_metadata.get(key) != val_metadata.get(key):
            raise ArtifactError(f"probe activation metadata differs for {key}")


def assemble_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    preregistration: Annotated[
        Path, typer.Option("--preregistration", exists=True, dir_okay=False)
    ],
    prompt_registry: Annotated[
        Path, typer.Option("--prompt-registry", exists=True, dir_okay=False)
    ],
    prompt_id: Annotated[str, typer.Option("--prompt-id")],
    train_store: Annotated[Path, typer.Option("--train-store", exists=True, file_okay=False)],
    val_store: Annotated[Path, typer.Option("--val-store", exists=True, file_okay=False)],
    splits_dir: Annotated[Path, typer.Option("--splits-dir", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    position: ProbePosition = "last_prompt",
    robustness: Annotated[bool, typer.Option("--robustness")] = False,
) -> None:
    """Assemble one ID-checked activation condition for frozen probing."""
    cfg = load_config(config)
    prereg = load_preregistration(preregistration)
    registry = load_prompt_registry(prompt_registry)
    prompt = registry.prompt(prompt_id)
    model_key = registry.model_key or registry.model_id
    if cfg.dataset.id != registry.dataset_id:
        raise typer.BadParameter("config dataset differs from prompt registry")
    model = cfg.model(model_key)
    if model.id != registry.model_id or model.revision != registry.model_revision:
        raise typer.BadParameter("config model differs from prompt registry")
    labels = labels_for_dataset(cfg.dataset.id)
    if labels != registry.labels:
        raise typer.BadParameter("config labels differ from prompt registry")
    probe_contract = prereg.payload.get("probe")
    if not isinstance(probe_contract, dict):
        raise typer.BadParameter("preregistration has no probe contract")
    registered_position = str(probe_contract.get("position"))
    if robustness and position != "mean_text":
        raise typer.BadParameter("robustness analysis requires mean_text position")
    if not robustness and position != registered_position:
        raise typer.BadParameter("probe position differs from preregistration")
    if tuple(float(value) for value in probe_contract.get("c_values", ())) != cfg.probes.c_values:
        raise typer.BadParameter("config probe C grid differs from preregistration")
    if tuple(int(value) for value in probe_contract.get("seeds", ())) != cfg.probes.seeds:
        raise typer.BadParameter("config probe seeds differ from preregistration")
    prereg.assert_selection_split("probe_val")
    expected_metadata = {
        "model": registry.model_id,
        "model_revision": registry.model_revision,
        "tokenizer_revision": registry.tokenizer_revision,
        "adapter_hash": None,
        "non_thinking": registry.non_thinking,
        "condition": prompt.kind,
        "prompt_id": prompt.prompt_id,
        "instruction_sha256": prompt.instruction_sha256,
        "prompt_registry_sha256": registry.registry_sha256,
    }
    train_split = splits_dir / "probe_train.jsonl"
    val_split = splits_dir / "probe_val.jsonl"
    split_names = (
        "probe_train",
        "probe_val",
        *(f"probe_train_seed{seed}" for seed in cfg.probes.seeds),
    )
    split_manifest = verify_split_manifest(
        splits_dir,
        expected_sha256=registry.split_manifest_sha256,
        split_names=split_names,
    )
    split_contract = split_manifest.get("contract")
    if not isinstance(split_contract, dict):
        raise ArtifactError("split manifest has no contract")
    if int(split_contract.get("probe_train_size", -1)) != cfg.dataset.probe_train_size:
        raise ArtifactError("config probe train size differs from split manifest")
    if int(split_contract.get("probe_val_size", -1)) != cfg.dataset.probe_val_size:
        raise ArtifactError("config probe validation size differs from split manifest")
    if split_manifest.get("dataset_revision") != cfg.dataset.revision:
        raise ArtifactError("config dataset revision differs from split manifest")
    if tuple(split_manifest.get("labels", ())) != labels:
        raise ArtifactError("config labels differ from split manifest")
    train = assemble_probe_dataset(
        train_store,
        train_split,
        labels,
        position=position,
        expected_metadata=expected_metadata,
    )
    val = assemble_probe_dataset(
        val_store,
        val_split,
        labels,
        position=position,
        expected_metadata=expected_metadata,
    )
    _assert_pair_compatible(train.metadata, val.metadata)
    if train.activations.shape[1:] != val.activations.shape[1:]:
        raise ArtifactError("probe train and validation activation shapes differ")
    subset_files = {
        seed: splits_dir / f"probe_train_seed{seed}.jsonl" for seed in cfg.probes.seeds
    }
    missing = [str(path) for path in subset_files.values() if not path.exists()]
    if missing:
        raise typer.BadParameter(f"missing frozen probe subset files: {missing}")
    train_sizes = tuple(int(value) for value in probe_contract["train_sizes"])
    if (
        not train_sizes
        or any(value <= 0 for value in train_sizes)
        or tuple(sorted(set(train_sizes))) != train_sizes
        or train_sizes[-1] > len(train.example_ids)
    ):
        raise typer.BadParameter("invalid preregistered probe train sizes")
    subset_size = int(split_contract["probe_train_size"]) * float(
        split_contract["probe_fraction"]
    )
    if not subset_size.is_integer():
        raise typer.BadParameter("frozen split contract has non-integral probe subset size")
    expected_subset_size = int(subset_size)
    if train_sizes[-1] != expected_subset_size:
        raise typer.BadParameter("probe train-size ceiling differs from frozen split contract")
    subsets = build_frozen_probe_subsets(
        train,
        subset_files,
        expected_size=train_sizes[-1],
    )
    if output.exists():
        raise typer.BadParameter(f"output already exists: {output}")
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise typer.BadParameter(f"temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        np.savez(
            temporary / "arrays.npz",
            x_train=train.activations,
            y_train=train.targets,
            x_val=val.activations,
            y_val=val.targets,
            length_train=train.text_lengths,
            length_val=val.text_lengths,
            label_count_train=train.label_counts,
            label_count_val=val.label_counts,
            example_ids_train=np.asarray(train.example_ids),
            example_ids_val=np.asarray(val.example_ids),
            subset_seeds=np.asarray(subsets.seeds),
            subset_indices=np.stack(subsets.indices),
            subset_hashes=np.asarray([subsets.id_hashes[seed] for seed in subsets.seeds]),
        )
        save_frozen_probe_subsets(temporary / "subsets.json", subsets)
        provenance = {
            "schema_version": 1,
            "prompt_id": prompt.prompt_id,
            "prompt_kind": prompt.kind,
            "instruction_sha256": prompt.instruction_sha256,
            "prompt_registry": str(prompt_registry.resolve()),
            "prompt_registry_sha256": registry.registry_sha256,
            "model_id": registry.model_id,
            "model_revision": registry.model_revision,
            "config": str(config.resolve()),
            "config_sha256": cfg.content_hash(),
            "preregistration": str(preregistration.resolve()),
            "preregistration_sha256": sha256_path(preregistration),
            "split_manifest_sha256": registry.split_manifest_sha256,
            "train_store": str(train_store.resolve()),
            "train_store_sha256": sha256_path(train_store),
            "val_store": str(val_store.resolve()),
            "val_store_sha256": sha256_path(val_store),
            "train_id_hash": subsets.master_id_hash,
            "val_id_hash": str(val.metadata["example_id_hash"]),
            "train_split_sha256": train.metadata.get("split_sha256"),
            "val_split_sha256": val.metadata.get("split_sha256"),
            "tokenizer_revision": train.metadata.get("tokenizer_revision"),
            "template_hash": train.metadata.get("template_hash"),
            "adapter_hash": train.metadata.get("adapter_hash"),
            "non_thinking": train.metadata.get("non_thinking"),
            "subset_hashes": {str(seed): subsets.id_hashes[seed] for seed in subsets.seeds},
            "position": position,
            "labels": list(labels),
            "train_rows": len(train.example_ids),
            "val_rows": len(val.example_ids),
            "layers": train.activations.shape[1],
            "hidden_size": train.activations.shape[2],
            "train_sizes": list(train_sizes),
            "c_values": list(cfg.probes.c_values),
            "threshold": cfg.probes.threshold,
            "analysis_role": "robustness" if robustness else "primary",
        }
        (temporary / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    typer.echo(output)


def fit_frozen_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    assembled: Annotated[Path, typer.Option("--assembled", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Fit all layers on each externally frozen probe subset."""
    cfg = load_config(config)
    provenance = json.loads((assembled / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("config_sha256") != cfg.content_hash():
        raise typer.BadParameter("fit config differs from assembled probe config")
    fit_frozen_probes(
        assembled,
        output,
        c_values=cfg.probes.c_values,
        threshold=cfg.probes.threshold,
    )
    typer.echo(output)


def evaluate_frozen_command(
    fitted: Annotated[Path, typer.Option("--fitted", exists=True, file_okay=False)],
    assembled: Annotated[Path, typer.Option("--assembled", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Evaluate frozen-subset probes on an aligned target condition."""
    evaluate_frozen_probes(fitted, assembled, output)
    typer.echo(output)


def select_layer_command(
    fitted_seed: Annotated[
        Path, typer.Option("--fitted-seed", exists=True, file_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Select one common layer from C_seed validation performance only."""
    select_common_layer(fitted_seed, output)
    typer.echo(output)


def register_probe_stage_commands(app: typer.Typer) -> None:
    app.command("build-registry")(build_registry_command)
    app.command("assemble")(assemble_command)
    app.command("fit-frozen")(fit_frozen_command)
    app.command("evaluate-frozen")(evaluate_frozen_command)
    app.command("select-layer")(select_layer_command)


__all__ = ["register_probe_stage_commands"]
