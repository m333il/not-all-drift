from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from .artifacts import RunDirectory, sha256_path
from .config import ExperimentConfig, load_config
from .datasets import load_huggingface_rows, prepare_from_rows
from .errors import ArtifactError, InterpError, ReflectorUnavailable
from .logit_lens_cli import app as logit_lens_app
from .phase0_cli import app as phase0_app
from .pipeline_cli import app as pipeline_app
from .probe_stage_cli import register_probe_stage_commands
from .tuned_lens_cli import app as tuned_lens_app

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
data_app = typer.Typer(no_args_is_help=True)
gepa_app = typer.Typer(no_args_is_help=True)
activations_app = typer.Typer(no_args_is_help=True)
probes_app = typer.Typer(no_args_is_help=True)
register_probe_stage_commands(probes_app)
causal_app = typer.Typer(no_args_is_help=True)
geometry_app = typer.Typer(no_args_is_help=True)
report_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(gepa_app, name="gepa")
app.add_typer(activations_app, name="activations")
app.add_typer(probes_app, name="probes")
app.add_typer(causal_app, name="causal")
app.add_typer(geometry_app, name="geometry")
app.add_typer(report_app, name="report")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(phase0_app, name="phase0")
app.add_typer(logit_lens_app, name="logit-lens")
app.add_typer(tuned_lens_app, name="tuned-lens")

ConfigPath = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]
Overrides = Annotated[list[str] | None, typer.Option("--set")]


def _load(path: Path, overrides: list[str] | None) -> ExperimentConfig:
    return load_config(path, overrides)


def _artifact_slug(value: str, *, max_length: int = 32) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    if not slug:
        return "unknown"
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    prefix = slug[: max_length - len(digest) - 1].rstrip("-")
    return f"{prefix}-{digest}"


@app.command()
def preflight(config: ConfigPath, set_: Overrides = None) -> None:
    """Validate configuration and local prerequisites without network access."""
    cfg = _load(config, set_)
    failures: list[str] = []
    if sys.version_info[:2] != (3, 12):
        failures.append(f"expected Python 3.12, found {sys.version.split()[0]}")
    if "REPLACE_WITH_" in cfg.gepa.local_reflector.model:
        failures.append("local reflector model ID is still a placeholder")
    if not cfg.gepa.local_reflector.revision or cfg.gepa.local_reflector.revision == "main":
        failures.append("local reflector revision must be an immutable commit")
    if cfg.gepa.api_reflector and "REPLACE_WITH_" in cfg.gepa.api_reflector.model:
        failures.append("API reflector model ID is still a placeholder")
    if cfg.dataset.revision == "main":
        failures.append("dataset revision must be an immutable revision, not main")
    for model in cfg.models:
        if model.revision == "main" or model.tokenizer_revision == "main":
            failures.append(f"model/tokenizer revision is not pinned: {model.id}")
    for provider in (cfg.task_provider, cfg.gepa.local_reflector, cfg.gepa.api_reflector):
        if provider and provider.api_key_env and not os.environ.get(provider.api_key_env):
            failures.append(f"missing environment variable {provider.api_key_env}")
    usage = shutil.disk_usage(
        cfg.output.root.parent if cfg.output.root.parent.exists() else Path.cwd()
    )
    typer.echo(
        json.dumps(
            {
                "config_hash": cfg.content_hash(),
                "free_gib": usage.free // 2**30,
                "failures": failures,
            },
            indent=2,
        )
    )
    if failures:
        raise typer.Exit(2)


@data_app.command("prepare")
def data_prepare(config: ConfigPath, set_: Overrides = None) -> None:
    """Download pinned source data and write leakage-checked split manifests."""
    cfg = _load(config, set_)
    if cfg.dataset.id == "synthetic":
        raise typer.BadParameter("synthetic data has no Hugging Face source")
    train, test, labels = load_huggingface_rows(
        cfg.dataset.id,
        cfg.dataset.revision,
        binarization_threshold=cfg.dataset.binarization_threshold,
    )
    d = cfg.dataset
    prepared = prepare_from_rows(
        dataset=d.id,
        labels=labels,
        train_rows=train,
        test_rows=test,
        seeds=cfg.gepa.seeds,
        n_train=d.train_size,
        n_optimizer_val=d.optimizer_val_size,
        n_probe_train=d.probe_train_size,
        n_probe_val=d.probe_val_size,
        n_intervention_val=d.intervention_val_size,
        n_mechanistic_eval=d.mechanistic_eval_size,
        n_natural_eval=d.natural_eval_size,
        revision=d.revision,
        group_field=d.group_field,
    )
    with RunDirectory(cfg, "data.prepare") as run:
        prepared.write(run.path / "splits")
        typer.echo(run.path)


@app.command("evaluate")
def evaluate_command(
    config: ConfigPath,
    split_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    instruction_file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    condition: str = "seed",
    model_key: str | None = None,
    set_: Overrides = None,
) -> None:
    """Run deterministic endpoint evaluation and retain every prediction."""
    import pandas as pd

    from .datasets import labels_for_dataset, load_jsonl_split
    from .evaluation import evaluate_examples
    from .metrics import evaluate_multilabel
    from .prompts import SEED_INSTRUCTIONS
    from .providers import UsageTrackingProvider, build_provider

    cfg = _load(config, set_)
    examples = load_jsonl_split(split_file)
    labels = labels_for_dataset(cfg.dataset.id)
    instruction = (
        instruction_file.read_text(encoding="utf-8")
        if instruction_file
        else SEED_INSTRUCTIONS[cfg.dataset.id]
    )
    provider = UsageTrackingProvider(build_provider(cfg.provider_for_model(model_key)))
    predictions = evaluate_examples(
        provider,
        examples,
        condition=condition,
        instruction=instruction,
        labels=labels,
        max_tokens=cfg.generation.max_new_tokens,
    )
    gold = [x.labels for x in examples]
    metrics = evaluate_multilabel(
        gold,
        [x.labels for x in predictions],
        labels,
        [x.parse_ok for x in predictions],
    )
    # Set-parser score of the same responses.
    lenient = evaluate_multilabel(gold, [x.lenient_labels for x in predictions], labels)
    metrics["f1_samples_lenient"] = lenient["f1_samples"]
    metrics["order_violation_rate"] = (
        sum(x.order_violation for x in predictions) / len(predictions) if predictions else 0.0
    )
    with RunDirectory(cfg, "evaluate") as run:
        pd.DataFrame([x.to_dict() for x in predictions]).to_parquet(
            run.path / "predictions.parquet", index=False
        )
        run.write_json("metrics.json", metrics)
        run.write_json("budget.json", provider.summary())
        typer.echo(run.path)


@gepa_app.command("optimize")
def gepa_optimize(
    config: ConfigPath,
    splits_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    seed: int = 42,
    reflector: str = "local",
    model_key: str | None = None,
    set_: Overrides = None,
) -> None:
    """Run GEPA against optimizer train/val only and freeze the selected prompt."""
    from .datasets import (
        labels_for_dataset,
        load_jsonl_split,
        optimizer_train_split_path,
        optimizer_val_split_path,
    )
    from .gepa_runner import optimize
    from .prompts import CIVIL_COMMENTS_PROMPT_CONTRACT_ID, SEED_INSTRUCTIONS, render_messages
    from .providers import UsageTrackingProvider, build_provider

    cfg = _load(config, set_)
    train_path = optimizer_train_split_path(
        splits_dir, cfg.dataset.id, seed=seed, train_size=cfg.dataset.train_size
    )
    validation_path = optimizer_val_split_path(
        splits_dir, seed=seed, val_size=cfg.dataset.optimizer_val_size
    )
    train = load_jsonl_split(train_path)
    validation = load_jsonl_split(validation_path)
    if len(train) != cfg.dataset.train_size:
        raise typer.BadParameter(
            f"train split has {len(train)} rows, expected {cfg.dataset.train_size}: {train_path}"
        )
    if len(validation) != cfg.dataset.optimizer_val_size:
        raise typer.BadParameter(
            f"validation split has {len(validation)} rows, expected "
            f"{cfg.dataset.optimizer_val_size}: {validation_path}"
        )
    reflection_cfg = cfg.gepa.local_reflector if reflector == "local" else cfg.gepa.api_reflector
    if reflection_cfg is None:
        raise typer.BadParameter("API reflector is not configured")
    labels = labels_for_dataset(cfg.dataset.id)
    task_model = cfg.model(model_key)
    task_provider = UsageTrackingProvider(build_provider(cfg.provider_for_model(model_key)))
    reflection_provider = UsageTrackingProvider(
        build_provider(reflection_cfg), failure_limit=cfg.gepa.reflection_failure_limit
    )
    seed_instruction = SEED_INSTRUCTIONS[cfg.dataset.id]
    seed_prompt = render_messages(seed_instruction, labels, "{text}").messages[0]["content"]
    task_name = task_model.key or task_model.id
    reflector_name = reflection_cfg.model.rsplit("/", 1)[-1]
    run_command = (
        f"gepa.optimize-{_artifact_slug(cfg.dataset.id)}-task-{_artifact_slug(task_name)}-"
        f"reflect-{_artifact_slug(reflector_name)}-train-{cfg.dataset.train_size}-"
        f"val-{cfg.dataset.optimizer_val_size}-seed-{seed}"
    )
    with RunDirectory(cfg, run_command) as run:
        gepa_dir = run.path / "gepa"
        gepa_dir.mkdir()
        (run.path / "seed_instructions.txt").write_text(seed_instruction, encoding="utf-8")
        (run.path / "seed_prompt.txt").write_text(seed_prompt, encoding="utf-8")
        seed_prompt_hash = hashlib.sha256(seed_prompt.encode()).hexdigest()
        (run.path / "seed_prompt.sha256").write_text(seed_prompt_hash + "\n", encoding="utf-8")
        run.write_json(
            "prompt_contract.json",
            {
                "id": (
                    CIVIL_COMMENTS_PROMPT_CONTRACT_ID
                    if cfg.dataset.id == "civil_comments"
                    else "legacy_json_v1"
                ),
                "labels": list(labels),
                "seed_instruction_sha256": hashlib.sha256(seed_instruction.encode()).hexdigest(),
                "seed_prompt_sha256": seed_prompt_hash,
            },
        )
        inputs = {
            "seed": seed,
            "train": {
                "path": str(train_path.resolve()),
                "rows": len(train),
                "sha256": sha256_path(train_path),
            },
            "validation": {
                "path": str(validation_path.resolve()),
                "rows": len(validation),
                "sha256": sha256_path(validation_path),
            },
        }
        manifest_path = splits_dir / "manifest.json"
        if manifest_path.exists():
            inputs["manifest"] = {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_path(manifest_path),
            }
        run.write_json("inputs.json", inputs)
        run.write_json(
            "setup.json",
            {
                "dataset": cfg.dataset.id,
                "seed": seed,
                "train_size": len(train),
                "validation_size": len(validation),
                "task": {
                    "key": task_model.key,
                    "model": task_model.id,
                    "revision": task_model.revision,
                    "endpoint": task_model.endpoint,
                },
                "reflector": {
                    "selector": reflector,
                    "kind": reflection_cfg.kind,
                    "model": reflection_cfg.model,
                    "revision": reflection_cfg.revision,
                    "endpoint": reflection_cfg.base_url,
                },
            },
        )
        result = optimize(
            train=train,
            validation=validation,
            seed_instruction=seed_instruction,
            task_provider=task_provider,
            reflection_provider=reflection_provider,
            labels=labels,
            seed=seed,
            max_metric_calls=cfg.gepa.max_metric_calls,
            reflection_minibatch_size=cfg.gepa.reflection_minibatch_size,
            reflection_max_tokens=cfg.gepa.reflection_max_tokens,
            reflection_instruction_budget_tokens=cfg.gepa.reflection_instruction_budget_tokens,
            run_dir=gepa_dir,
        )
        if not reflection_provider.calls:
            # No reflection succeeded: the result would just be the seed prompt.
            raise ReflectorUnavailable(
                f"no reflection succeeded in {reflection_provider.failed_calls} attempts"
            )
        (run.path / "optimized_instructions.txt").write_text(
            result.selected_instruction, encoding="utf-8"
        )
        rendered = render_messages(result.selected_instruction, labels, "{text}")
        optimized_prompt = rendered.messages[0]["content"]
        (run.path / "optimized_prompt.txt").write_text(optimized_prompt, encoding="utf-8")
        run.write_json("candidates.json", list(result.candidates))
        prompt_hash = hashlib.sha256(result.selected_instruction.encode()).hexdigest()
        (run.path / "prompt.sha256").write_text(prompt_hash + "\n", encoding="utf-8")
        optimized_prompt_hash = hashlib.sha256(optimized_prompt.encode()).hexdigest()
        (run.path / "optimized_prompt.sha256").write_text(
            optimized_prompt_hash + "\n", encoding="utf-8"
        )
        run.write_json("gepa/result.json", result.result_snapshot())
        (gepa_dir / "candidate_tree.html").write_text(
            result.candidate_tree_html(), encoding="utf-8"
        )
        run.write_json(
            "budget.json",
            {"task": task_provider.summary(), "reflection": reflection_provider.summary()},
        )
        if reflection_provider.truncated_calls:
            typer.echo(
                f"WARNING: {reflection_provider.truncated_calls} of "
                f"{reflection_provider.calls} reflections hit "
                f"gepa.reflection_max_tokens={cfg.gepa.reflection_max_tokens}",
                err=True,
            )
        typer.echo(run.path)


@activations_app.command("extract")
def activations_extract(
    config: ConfigPath,
    split_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    instruction_file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    prompt_registry: Annotated[
        Path | None, typer.Option("--prompt-registry", exists=True, dir_okay=False)
    ] = None,
    prompt_id: str | None = None,
    adapter: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    condition: str | None = None,
    model_key: str | None = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 8,
    pilot_verify: Annotated[bool, typer.Option("--pilot-verify")] = False,
    set_: Overrides = None,
) -> None:
    """Extract compact last-prompt and mean-text states with HF."""
    from .activations import extract_hf, save_activation_store
    from .datasets import labels_for_dataset, load_jsonl_split
    from .prompt_registry import load_prompt_registry
    from .prompts import SEED_INSTRUCTIONS, apply_chat_template, render_condition_messages

    cfg = _load(config, set_)
    model_cfg = cfg.model(model_key)
    examples = load_jsonl_split(split_file)
    source_rows = len(examples)
    if limit is not None:
        examples = examples[:limit]
    if pilot_verify and limit is None:
        raise typer.BadParameter("--pilot-verify requires an explicit --limit")
    labels = labels_for_dataset(cfg.dataset.id)
    registry_hash: str | None = None
    resolved_prompt_id: str
    resolved_condition: str
    if (prompt_registry is None) != (prompt_id is None):
        raise typer.BadParameter("--prompt-registry and --prompt-id must be supplied together")
    if prompt_registry is not None and prompt_id is not None:
        if instruction_file is not None:
            raise typer.BadParameter("--instruction-file cannot be combined with prompt registry")
        if adapter is not None:
            raise typer.BadParameter("--adapter cannot be combined with discrete prompt registry")
        registry = load_prompt_registry(prompt_registry)
        record = registry.prompt(prompt_id)
        if cfg.dataset.id != registry.dataset_id or labels != registry.labels:
            raise typer.BadParameter("prompt registry dataset or labels differ from config")
        if model_cfg.id != registry.model_id or model_cfg.revision != registry.model_revision:
            raise typer.BadParameter("prompt registry model differs from selected model")
        if (
            model_cfg.tokenizer_revision or model_cfg.revision
        ) != registry.tokenizer_revision or model_cfg.non_thinking != registry.non_thinking:
            raise typer.BadParameter(
                "prompt registry tokenizer contract differs from selected model"
            )
        if condition is not None and condition != record.kind:
            raise typer.BadParameter("condition differs from registered prompt kind")
        resolved_condition = record.kind
        resolved_prompt_id = record.prompt_id
        instruction = record.instruction
        instruction_hash = record.instruction_sha256
        registry_hash = registry.registry_sha256
    else:
        resolved_condition = condition or "C_seed"
        if resolved_condition == "C_adapt" and instruction_file is None:
            raise typer.BadParameter("C_adapt requires --instruction-file or prompt registry")
        if resolved_condition == "C_seed" and instruction_file is not None:
            raise typer.BadParameter("C_seed cannot use an adapted --instruction-file")
        instruction = (
            instruction_file.read_text(encoding="utf-8")
            if instruction_file
            else SEED_INSTRUCTIONS[cfg.dataset.id]
        )
        instruction_hash = hashlib.sha256(instruction.encode()).hexdigest()
        resolved_prompt_id = resolved_condition
    from .modeling import load_hf_model

    model, tokenizer = load_hf_model(model_cfg)
    if adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise typer.BadParameter("install the models extra") from exc
        model = PeftModel.from_pretrained(model, adapter)
    prompts = [
        apply_chat_template(
            tokenizer,
            render_condition_messages(resolved_condition, instruction, labels, x.text),
            non_thinking=model_cfg.non_thinking,
        )
        for x in examples
    ]
    batch = extract_hf(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        texts=[x.text for x in examples],
        example_ids=[x.id for x in examples],
        batch_size=batch_size,
        verify_logits=pilot_verify,
        metadata={
            "model": model_cfg.id,
            "model_revision": model_cfg.revision,
            "tokenizer_revision": model_cfg.tokenizer_revision or model_cfg.revision,
            "split": split_file.name,
            "split_sha256": sha256_path(split_file),
            "condition": resolved_condition,
            "prompt_id": resolved_prompt_id,
            "instruction_sha256": instruction_hash,
            "prompt_registry_sha256": registry_hash,
            "example_id_hash": hashlib.sha256(
                "\n".join(x.id for x in examples).encode()
            ).hexdigest(),
            "prompt_hash": hashlib.sha256("\0".join(prompts).encode()).hexdigest(),
            "template_hash": hashlib.sha256(
                str(getattr(tokenizer, "chat_template", "")).encode()
            ).hexdigest(),
            "adapter_hash": None if adapter is None else sha256_path(adapter),
            "non_thinking": model_cfg.non_thinking,
            "source_rows": source_rows,
            "selected_rows": len(examples),
            "batch_size": batch_size,
            "pilot_verification": pilot_verify,
        },
    )
    import numpy as np

    expected_layers = None if model_cfg.layers is None else model_cfg.layers + 1
    if expected_layers is not None and batch.last_prompt.shape[1] != expected_layers:
        raise ArtifactError(
            f"captured {batch.last_prompt.shape[1]} residual layers, expected {expected_layers}"
        )
    if not np.isfinite(batch.last_prompt).all() or not np.isfinite(batch.mean_text).all():
        raise ArtifactError("activation extraction produced NaN or infinite values")
    with RunDirectory(cfg, "activations.extract") as run:
        store = run.path / "activations"
        save_activation_store(store, batch, compress=cfg.output.activation_compressor == "zstd")
        if pilot_verify:
            from .activations import load_activation_store

            restored = load_activation_store(store)
            if (
                restored.example_ids != batch.example_ids
                or not np.array_equal(restored.last_prompt, batch.last_prompt)
                or not np.array_equal(restored.mean_text, batch.mean_text)
            ):
                raise ArtifactError("activation pilot Zarr round trip differs")
            run.write_json(
                "pilot_verification.json",
                {
                    "passed": True,
                    "final_logits_verified": True,
                    "finite": True,
                    "round_trip_verified": True,
                    "last_prompt_rule": "right_padded_attention_mask_sum_minus_one",
                    "mean_text_rule": "offset_overlap_with_input_text_only",
                    "rows": len(batch.example_ids),
                    "shape": list(batch.last_prompt.shape),
                },
            )
        typer.echo(run.path)


@probes_app.command("train")
def probes_train(
    arrays: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Path,
) -> None:
    """Fit all-layer probes from an NPZ contract."""
    import numpy as np

    from .probes import train_layerwise_probes

    data = np.load(arrays)
    results = train_layerwise_probes(data["x_train"], data["y_train"], data["x_val"], data["y_val"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        weights=np.stack([x.weights for x in results]),
        intercepts=np.stack([x.intercepts for x in results]),
        thresholds=np.stack([x.thresholds for x in results]),
        validation_f1=np.asarray([x.validation_f1 for x in results]),
        validation_aurocs=np.asarray(
            [
                [np.nan if value is None else value for value in result.validation_aurocs]
                for result in results
            ]
        ),
        c=np.asarray([x.c for x in results]),
    )


@probes_app.command("evaluate")
def probes_evaluate(
    probes: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    activations: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    targets: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Path,
) -> None:
    """Evaluate fitted probes on any aligned condition for cross-condition transfer."""
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    from .metrics import empty_aware_sample_f1_rows

    fitted = np.load(probes)
    x, y = np.load(activations), np.load(targets)
    required = {"weights", "intercepts", "thresholds"}
    if not required <= set(fitted.files):
        raise typer.BadParameter(f"probe file is missing: {sorted(required - set(fitted.files))}")
    if x.ndim != 3 or y.ndim != 2 or len(x) != len(y):
        raise typer.BadParameter("arrays must align as [examples,layers,hidden] and labels")
    rows = []
    for layer in range(x.shape[1]):
        logits = x[:, layer] @ fitted["weights"][layer].T + fitted["intercepts"][layer]
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        predictions = probabilities >= fitted["thresholds"][layer]
        aurocs = [
            None
            if len(np.unique(y[:, label])) < 2
            else float(roc_auc_score(y[:, label], probabilities[:, label]))
            for label in range(y.shape[1])
        ]
        rows.append(
            {
                "layer": layer,
                # Empty-aware samples-F1, as used for probe selection.
                "f1_samples": float(empty_aware_sample_f1_rows(y, predictions).mean()),
                "f1_samples_legacy": float(
                    f1_score(y, predictions, average="samples", zero_division=0)
                ),
                "f1_micro": float(f1_score(y, predictions, average="micro", zero_division=0)),
                "f1_macro": float(f1_score(y, predictions, average="macro", zero_division=0)),
                "per_label_auroc": aurocs,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    typer.echo(output)


@probes_app.command("audit")
def probes_audit(
    config: ConfigPath,
    preregistration: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    arrays: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Path,
    selection_split: str = "probe_val",
    set_: Overrides = None,
) -> None:
    """Run the preregistered multi-seed selectivity and capacity-control battery."""
    import numpy as np

    from .preregistration import load_preregistration
    from .probes import run_probe_protocol

    cfg = _load(config, set_)
    prereg = load_preregistration(preregistration)
    prereg.assert_selection_split(selection_split)
    data = np.load(arrays)
    controls = {
        name: (data[f"{name}_train"], data[f"{name}_val"]) for name in ("length", "label_count")
    }
    if ("condition_train" in data.files) != ("condition_val" in data.files):
        raise typer.BadParameter(
            "probe arrays must contain both condition_train and condition_val or neither"
        )
    if "condition_train" in data.files and "condition_val" in data.files:
        controls["condition"] = (data["condition_train"], data["condition_val"])
    required_subset_arrays = {"subset_seeds", "subset_indices", "subset_hashes"}
    if missing := required_subset_arrays - set(data.files):
        raise typer.BadParameter(f"probe arrays are missing frozen subsets: {sorted(missing)}")
    subset_seeds = [int(value) for value in data["subset_seeds"]]
    if tuple(subset_seeds) != tuple(cfg.probes.seeds):
        raise typer.BadParameter("frozen subset seeds differ from probe config")
    subset_indices = {
        seed: np.asarray(indices)
        for seed, indices in zip(subset_seeds, data["subset_indices"], strict=True)
    }
    subset_hashes = {
        seed: str(value) for seed, value in zip(subset_seeds, data["subset_hashes"], strict=True)
    }
    probe_cfg = prereg.payload["probe"]
    rows = run_probe_protocol(
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        controls=controls,
        subset_indices=subset_indices,
        subset_hashes=subset_hashes,
        train_sizes=tuple(int(value) for value in probe_cfg["train_sizes"]),
        c_values=cfg.probes.c_values,
        threshold=cfg.probes.threshold,
        permutations=int(probe_cfg["permutations"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    typer.echo(output)


@causal_app.command("steer")
def causal_steer(seed_acts: Path, target_acts: Path, output: Path, layer: int) -> None:
    """Compile paired activations into one selected-layer steering direction."""
    import numpy as np

    from .activations import induced_shift_batches, load_activation_store

    _, direction = induced_shift_batches(
        load_activation_store(seed_acts), load_activation_store(target_acts)
    )
    np.save(output, direction[layer])


@causal_app.command("patch")
def causal_patch(seed_acts: Path, target_acts: Path, output: Path, layer: int) -> None:
    """Create aligned selected-layer values for per-input prompt-only replacement."""
    import numpy as np

    from .activations import induced_shift_batches, load_activation_store

    seed = load_activation_store(seed_acts)
    target = load_activation_store(target_acts)
    induced_shift_batches(seed, target)
    np.save(output, target.last_prompt[:, layer])


@causal_app.command("erase")
def causal_erase(activations: Path, concepts: Path, output: Path) -> None:
    """Learn a label/count subspace and write erased activations plus its basis."""
    import numpy as np

    from .causal import apply_leace, fit_leace

    x, z = np.load(activations), np.load(concepts)
    eraser = fit_leace(x, z)
    np.savez_compressed(
        output, erased=apply_leace(x, eraser), matrix=eraser.matrix, bias=eraser.bias
    )


@causal_app.command("generate")
def causal_generate(
    config: ConfigPath,
    split_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    value_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    layer: int,
    mode: str = "add",
    positions: str = "last",
    alpha: float = 1.0,
    model_key: str | None = None,
    set_: Overrides = None,
) -> None:
    """Run greedy HF generation with a frozen residual intervention cell."""
    import numpy as np
    import pandas as pd

    try:
        import torch
    except ImportError as exc:
        raise typer.BadParameter("install the models extra") from exc
    from .causal import EditSpec, generate_with_edit
    from .datasets import labels_for_dataset, load_jsonl_split
    from .prompts import SEED_INSTRUCTIONS, apply_chat_template, parse_labels, render_messages

    if mode not in {"add", "replace", "erase"}:
        raise typer.BadParameter("mode must be add, replace, or erase")
    if positions not in {"last", "prompt_last", "all"}:
        raise typer.BadParameter("positions must be last, prompt_last, or all")
    cfg = _load(config, set_)
    from .modeling import load_hf_model

    model_cfg = cfg.model(model_key)
    model, tokenizer = load_hf_model(model_cfg)
    examples = load_jsonl_split(split_file)
    labels = labels_for_dataset(cfg.dataset.id)
    prompts = [
        apply_chat_template(
            tokenizer,
            render_messages(SEED_INSTRUCTIONS[cfg.dataset.id], labels, example.text),
            non_thinking=model_cfg.non_thinking,
        )
        for example in examples
    ]
    loaded = np.load(value_file)
    value: object
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if mode != "erase" or not {"matrix", "bias"} <= set(loaded.files):
            raise typer.BadParameter("NPZ intervention must contain LEACE matrix and bias")
        value = (torch.as_tensor(loaded["matrix"]), torch.as_tensor(loaded["bias"]))
    else:
        value = torch.as_tensor(loaded)
    generated = generate_with_edit(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        spec=EditSpec(
            layer,
            cast(Literal["add", "replace", "erase"], mode),
            cast(Literal["last", "prompt_last", "all"], positions),
            alpha,
        ),
        value=value,
        max_new_tokens=cfg.generation.max_new_tokens,
    )
    rows = []
    for example, raw in zip(examples, generated, strict=True):
        try:
            parsed, error = parse_labels(raw, labels), None
        except Exception as exc:
            parsed, error = (), str(exc)
        rows.append(
            {
                "example_id": example.id,
                "raw_response": raw,
                "labels": list(parsed),
                "parse_error": error,
            }
        )
    with RunDirectory(cfg, "causal.generate") as run:
        pd.DataFrame(rows).to_parquet(run.path / "predictions.parquet", index=False)
        typer.echo(run.path)


@geometry_app.command("analyze")
def geometry_analyze(first: Path, second: Path, output: Path) -> None:
    """Compute causal-branch similarity summaries from two [n,layer,d] arrays."""
    from .activations import induced_shift_batches, load_activation_store
    from .geometry import cosine, linear_cka, procrustes_distance

    first_batch, second_batch = load_activation_store(first), load_activation_store(second)
    induced_shift_batches(first_batch, second_batch)
    x, y = first_batch.last_prompt, second_batch.last_prompt
    rows = [
        {
            "layer": layer,
            "mean_cosine": cosine(x[:, layer].mean(0), y[:, layer].mean(0)),
            "linear_cka": linear_cka(x[:, layer], y[:, layer]),
            "procrustes": procrustes_distance(x[:, layer], y[:, layer]),
        }
        for layer in range(x.shape[1])
    ]
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


@report_app.command("build")
def report_build(rows_json: Path, output_dir: Path) -> None:
    """Build CSV/Parquet/JSON result companions from JSON rows."""
    from .reporting import build_report

    rows = json.loads(rows_json.read_text(encoding="utf-8"))
    build_report(rows, output_dir)


@app.command("synthetic-smoke")
def synthetic_smoke(config: ConfigPath, set_: Overrides = None) -> None:
    """Run an offline data→metrics→report smoke pipeline."""
    from .metrics import evaluate_multilabel
    from .reporting import build_report

    cfg = _load(config, set_)
    with RunDirectory(cfg, "synthetic-smoke") as run:
        metrics = evaluate_multilabel([("a",), (), ("b",)], [("a",), ("a",), ()], ("a", "b"))
        rows = [
            {"condition": "synthetic", "metric": key, "value": value}
            for key, value in metrics.items()
            if isinstance(value, float)
        ]
        build_report(rows, run.path / "report")
        run.write_json("smoke.json", {"ok": True, "metrics": metrics})
        typer.echo(run.path)


def main() -> None:
    try:
        app()
    except InterpError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc


if __name__ == "__main__":
    main()
