#!/usr/bin/env python3
"""Audit Gemma Scope reconstruction fidelity on fixed Civil Comments samples."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_optimization.civil_comments import (
    canonical_target,
    canonicalize_labels,
    compute_binary_metrics,
    compute_multilabel_metrics,
    parse_binary_prediction,
    parse_prediction,
)
from prompt_optimization.gemma_scope import (
    GemmaScopeJumpReLU,
    OneShotReconstructionHook,
    ReconstructionAccumulator,
    ReconstructionScope,
    build_reconstruction_mask,
    hidden_from_decoder_output,
    resid_post_hidden_state_index,
    resolve_gemma_decoder_layer,
    resolve_gemma_final_norm,
)
from prompt_optimization.prefix_cache import install_gemma_prefix_cache_compatibility
from prompt_optimization.frozen_text_prompt import (
    encode_prompts_strict,
    load_prompt_template,
    prompt_sha256,
    render_prompt as render_frozen_prompt,
)

from scripts.probing.extract_civil_comments_activations import load_fixed_datasets, load_model
from scripts.adapters.train_civil_comments_peft import render_prompt

CONDITIONS = ("manual", "adapted", "frozen_prompt")
SCOPES = ("anchor", "all_valid")
DEFAULT_SAE_REVISION = "fd571b47c1c64851e9b1989792367b9babb4af63"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument(
        "--condition-name",
        help="Display name recorded in outputs; defaults to --condition.",
    )
    parser.add_argument(
        "--frozen-prompt-file",
        type=Path,
        help="Prompt template containing one {text}; required for frozen_prompt.",
    )
    parser.add_argument("--sae-path", type=Path, required=True)
    parser.add_argument("--sae-revision", default=DEFAULT_SAE_REVISION)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--scope", choices=SCOPES, default="anchor")
    parser.add_argument("--split", choices=("probe_train", "probe_val", "test"), default="test")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument(
        "--max-input-length",
        type=int,
        help="Override reference-run max_length (needed for long frozen prompts).",
    )
    parser.add_argument(
        "--sample-ids-file",
        type=Path,
        help=(
            "JSON sample manifest. It may be a list of IDs, an object with `ids`, "
            "or an object with `subsets[split].ids`. IDs are selected in listed order."
        ),
    )
    parser.add_argument(
        "--original-predictions-file",
        type=Path,
        help="Reuse a validated original-pass prediction file from the same condition.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sae-chunk-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def encode_prompts(
    tokenizer: Any,
    batch: dict[str, list[Any]],
    *,
    labels: tuple[str, ...],
    setup: str,
    binary_output_format: str = "safe-toxic",
    max_length: int,
    device: torch.device,
    frozen_prompt_template: str | None = None,
) -> dict[str, torch.Tensor]:
    if frozen_prompt_template is not None:
        prompts = [
            render_frozen_prompt(tokenizer, frozen_prompt_template, text)
            for text in batch["text"]
        ]
        encoded, _ = encode_prompts_strict(
            tokenizer,
            prompts,
            max_input_length=max_length,
            padding_side=tokenizer.padding_side,
            device=device,
        )
        return encoded
    prompts = [
        render_prompt(
            tokenizer,
            text,
            labels,
            setup,
            binary_output_format,
        )
        for text in batch["text"]
    ]
    return tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)


def state_type_masks(
    tokenizer: Any,
    encoded: dict[str, torch.Tensor],
    *,
    prepended_virtual_tokens: int,
) -> dict[str, torch.Tensor]:
    """Partition valid states into virtual, special-text, and non-special text."""
    attention_mask = encoded["attention_mask"].to(dtype=torch.bool)
    input_ids = encoded["input_ids"]
    special_text = torch.zeros_like(attention_mask)
    for token_id in tokenizer.all_special_ids:
        special_text |= input_ids == int(token_id)
    special_text &= attention_mask
    non_special_text = attention_mask & ~special_text
    if prepended_virtual_tokens:
        virtual = torch.ones(
            (len(attention_mask), prepended_virtual_tokens),
            dtype=torch.bool,
            device=attention_mask.device,
        )
        false_prefix = torch.zeros_like(virtual)
        return {
            "virtual_tokens": torch.cat((virtual, torch.zeros_like(attention_mask)), dim=1),
            "special_text_tokens": torch.cat((false_prefix, special_text), dim=1),
            "non_special_text_tokens": torch.cat((false_prefix, non_special_text), dim=1),
        }
    return {
        "special_text_tokens": special_text,
        "non_special_text_tokens": non_special_text,
    }


def verify_hook_alignment(
    model: Any,
    encoded: dict[str, torch.Tensor],
    *,
    layer: int,
) -> dict[str, Any]:
    decoder_layer = resolve_gemma_decoder_layer(model, layer)
    captured: list[torch.Tensor] = []
    final_norm_inputs: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        captured.append(hidden_from_decoder_output(output).detach())

    handle = decoder_layer.register_forward_hook(capture)
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder_layers = base_model.model.layers
    final_norm_handle = None
    if layer == len(decoder_layers) - 1:
        final_norm = resolve_gemma_final_norm(model)

        def capture_final_norm_input(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
        ) -> None:
            final_norm_inputs.append(inputs[0].detach())

        final_norm_handle = final_norm.register_forward_pre_hook(capture_final_norm_input)
    try:
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
    finally:
        handle.remove()
        if final_norm_handle is not None:
            final_norm_handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one layer hook call, observed {len(captured)}")
    if final_norm_handle is None:
        hidden_state_index: int | None = resid_post_hidden_state_index(layer)
        expected = outputs.hidden_states[hidden_state_index]
        comparison_source = "hidden_states[layer + 1]"
    else:
        if len(final_norm_inputs) != 1:
            raise RuntimeError(
                f"Expected one final norm input, observed {len(final_norm_inputs)}"
            )
        hidden_state_index = None
        expected = final_norm_inputs[0]
        comparison_source = "final_norm_input"
    if captured[0].shape != expected.shape:
        raise RuntimeError(
            f"Hook shape {tuple(captured[0].shape)} differs from hidden state "
            f"{tuple(expected.shape)}"
        )
    difference = (captured[0].float() - expected.float()).abs()
    max_absolute_difference = float(difference.max().item())
    mean_absolute_difference = float(difference.mean().item())
    if max_absolute_difference > 1e-6:
        raise RuntimeError(
            f"Decoder hook does not match {comparison_source}: "
            f"max_abs={max_absolute_difference:.6g}"
        )
    return {
        "status": "matched",
        "layer": layer,
        "hidden_state_index": hidden_state_index,
        "comparison_source": comparison_source,
        "shape": list(expected.shape),
        "max_absolute_difference": max_absolute_difference,
        "mean_absolute_difference": mean_absolute_difference,
    }


def serialize_prediction(prediction: Any | None, *, setup: str) -> Any | None:
    if prediction is None or setup == "binary":
        return prediction
    return list(prediction)


def predict_pass(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    *,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    labels: tuple[str, ...],
    setup: str,
    binary_output_format: str = "safe-toxic",
    decoder_layer: torch.nn.Module | None = None,
    sae: GemmaScopeJumpReLU | None = None,
    scope: ReconstructionScope = "anchor",
    prepended_virtual_tokens: int = 0,
    sae_chunk_size: int = 128,
    frozen_prompt_template: str | None = None,
) -> tuple[list[Any], list[Any | None], list[str], dict[str, Any] | None]:
    if (decoder_layer is None) != (sae is None):
        raise ValueError("decoder_layer and sae must either both be set or both be absent")
    targets: list[Any] = []
    predictions: list[Any | None] = []
    raw_outputs: list[str] = []
    top_error_states: list[dict[str, Any]] = []
    accumulator = (
        ReconstructionAccumulator(retain_state_metrics=True) if sae is not None else None
    )
    group_accumulators: dict[str, ReconstructionAccumulator] = {}
    model.eval()
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]
            encoded = encode_prompts(
                tokenizer,
                batch,
                labels=labels,
                setup=setup,
                binary_output_format=binary_output_format,
                max_length=max_length,
                device=model.device,
                frozen_prompt_template=frozen_prompt_template,
            )
            handle = None
            controller = None
            if sae is not None and decoder_layer is not None:
                selection_mask = build_reconstruction_mask(
                    encoded["attention_mask"],
                    prepended_virtual_tokens=prepended_virtual_tokens,
                    scope=scope,
                )
                metric_group_masks = {
                    name: mask & selection_mask
                    for name, mask in state_type_masks(
                        tokenizer,
                        encoded,
                        prepended_virtual_tokens=prepended_virtual_tokens,
                    ).items()
                }
                for name in metric_group_masks:
                    group_accumulators.setdefault(
                        name,
                        ReconstructionAccumulator(retain_state_metrics=True),
                    )
                controller = OneShotReconstructionHook(
                    sae,
                    selection_mask,
                    chunk_size=sae_chunk_size,
                    accumulator=accumulator,
                    diagnostic_top_k=8,
                    metric_group_masks=metric_group_masks,
                    metric_group_accumulators=group_accumulators,
                )
                handle = decoder_layer.register_forward_hook(controller)
            try:
                with torch.inference_mode():
                    output_ids = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
            finally:
                if handle is not None:
                    handle.remove()
            if controller is not None and not controller.applied:
                raise RuntimeError("SAE reconstruction hook was not applied")
            if controller is not None:
                for diagnostic in controller.top_error_states:
                    batch_index = int(diagnostic["batch_index"])
                    sequence_position = int(diagnostic["sequence_position"])
                    text_position = sequence_position - prepended_virtual_tokens
                    token_id = (
                        int(encoded["input_ids"][batch_index, text_position].item())
                        if text_position >= 0
                        else None
                    )
                    top_error_states.append(
                        {
                            **diagnostic,
                            "sample_id": str(batch["id"][batch_index]),
                            "text_position": text_position if text_position >= 0 else None,
                            "virtual_token": text_position < 0,
                            "token_id": token_id,
                            "token": (
                                tokenizer.convert_ids_to_tokens(token_id)
                                if token_id is not None
                                else None
                            ),
                            "decoded_token": (
                                tokenizer.decode([token_id]) if token_id is not None else None
                            ),
                        }
                    )

            continuation = output_ids[:, encoded["input_ids"].shape[1] :]
            generated = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            raw_outputs.extend(generated)
            if setup == "binary":
                predictions.extend(
                    parse_binary_prediction(text, output_format=binary_output_format)
                    for text in generated
                )
                targets.extend(str(row) for row in batch["binary_label"])
            else:
                predictions.extend(
                    parse_prediction(text, allowed_labels=labels) for text in generated
                )
                targets.extend(
                    canonicalize_labels(row, allowed_labels=labels) for row in batch["labels"]
                )
    finally:
        tokenizer.padding_side = old_padding_side

    reconstruction_metrics = None
    if accumulator is not None:
        reconstruction_metrics = {
            **accumulator.compute().as_dict(),
            "per_state_quantiles": accumulator.distribution_summary(),
            "top_error_states": sorted(
                top_error_states,
                key=lambda row: float(row["mse"]),
                reverse=True,
            )[:20],
        }
        reconstruction_metrics["state_groups"] = {
            name: {
                **group_accumulator.compute().as_dict(),
                "per_state_quantiles": group_accumulator.distribution_summary(),
            }
            for name, group_accumulator in group_accumulators.items()
            if group_accumulator.states
        }
    return targets, predictions, raw_outputs, reconstruction_metrics


def score_pass(
    targets: list[Any],
    predictions: list[Any | None],
    *,
    labels: tuple[str, ...],
    setup: str,
) -> dict[str, float]:
    if setup == "binary":
        return compute_binary_metrics(targets, predictions)
    return compute_multilabel_metrics(targets, predictions, labels=labels)


def prediction_rows(
    dataset: Dataset,
    targets: list[Any],
    predictions: list[Any | None],
    raw_outputs: list[str],
    *,
    setup: str,
    labels: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "id": example_id,
            "target": target if setup == "binary" else list(target),
            "target_text": (
                target
                if setup == "binary"
                else canonical_target(target, allowed_labels=labels)
            ),
            "prediction": serialize_prediction(prediction, setup=setup),
            "raw_output": raw_output,
            "valid_format": prediction is not None,
            "exact_match": prediction == target,
        }
        for example_id, target, prediction, raw_output in zip(
            dataset["id"], targets, predictions, raw_outputs, strict=True
        )
    ]


def paired_summary(
    targets: list[Any],
    original: list[Any | None],
    reconstructed: list[Any | None],
) -> dict[str, float | int]:
    if not (len(targets) == len(original) == len(reconstructed)):
        raise ValueError("Paired prediction arrays must have equal lengths")
    original_correct = [prediction == target for prediction, target in zip(original, targets)]
    reconstructed_correct = [
        prediction == target for prediction, target in zip(reconstructed, targets)
    ]
    return {
        "samples": len(targets),
        "prediction_agreement": sum(
            left == right for left, right in zip(original, reconstructed)
        )
        / max(len(targets), 1),
        "wrong_to_correct": sum(
            not before and after
            for before, after in zip(original_correct, reconstructed_correct)
        ),
        "correct_to_wrong": sum(
            before and not after
            for before, after in zip(original_correct, reconstructed_correct)
        ),
    }


def sample_ids_from_manifest(path: Path, split: str) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        ids = payload
    elif isinstance(payload, dict) and isinstance(payload.get("ids"), list):
        ids = payload["ids"]
    elif isinstance(payload, dict) and isinstance(payload.get("subsets"), dict):
        split_payload = payload["subsets"].get(split)
        if not isinstance(split_payload, dict) or not isinstance(
            split_payload.get("ids"), list
        ):
            raise ValueError(f"Sample manifest has no subsets[{split!r}].ids")
        ids = split_payload["ids"]
    else:
        raise ValueError(f"Unsupported sample manifest schema: {path}")
    normalized = [str(value) for value in ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Sample manifest contains duplicate IDs: {path}")
    if not normalized:
        raise ValueError(f"Sample manifest contains no IDs: {path}")
    return normalized


def select_dataset_by_ids(dataset: Dataset, ids: list[str]) -> Dataset:
    positions: dict[str, int] = {}
    for index, example_id in enumerate(dataset["id"]):
        key = str(example_id)
        if key in positions:
            raise ValueError(f"Dataset contains duplicate ID: {key}")
        positions[key] = index
    missing = [example_id for example_id in ids if example_id not in positions]
    if missing:
        raise ValueError(f"Sample IDs are absent from split: {missing[:5]}")
    return dataset.select([positions[example_id] for example_id in ids])


def load_original_predictions(
    path: Path,
    dataset: Dataset,
    *,
    setup: str,
) -> tuple[list[Any], list[Any | None], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected a prediction list in {path}")
    expected_ids = [str(value) for value in dataset["id"]]
    observed_ids = [str(row.get("id")) for row in payload if isinstance(row, dict)]
    if observed_ids != expected_ids or len(payload) != len(expected_ids):
        raise ValueError("Reused original predictions do not match selected dataset IDs")
    targets: list[Any] = []
    predictions: list[Any | None] = []
    raw_outputs: list[str] = []
    for row in payload:
        target = row.get("target")
        prediction = row.get("prediction")
        if setup != "binary":
            if not isinstance(target, list):
                raise TypeError("Expected list-valued multilabel target")
            target = tuple(str(value) for value in target)
            if prediction is not None:
                if not isinstance(prediction, list):
                    raise TypeError("Expected list-valued multilabel prediction")
                prediction = tuple(str(value) for value in prediction)
        targets.append(target)
        predictions.append(prediction)
        raw_outputs.append(str(row.get("raw_output", "")))
    return targets, predictions, raw_outputs


def main() -> None:
    args = parse_args()
    if (args.condition == "frozen_prompt") != (args.frozen_prompt_file is not None):
        raise ValueError(
            "--frozen-prompt-file is required exactly when --condition=frozen_prompt"
        )
    if args.layer < 0:
        raise ValueError("--layer must be non-negative")
    for name in ("max_samples", "batch_size", "sae_chunk_size", "cpu_threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_input_length is not None and args.max_input_length <= 0:
        raise ValueError("--max-input-length must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("SAE fidelity evaluation requires CUDA")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(args.cpu_threads)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")

    config = read_json(args.reference_run / "config.json")
    binary_output_format = str(config.get("binary_output_format", "safe-toxic"))
    max_input_length = int(args.max_input_length or config["max_length"])
    datasets, split_contract, labels, setup = load_fixed_datasets(config)
    requested_ids = (
        sample_ids_from_manifest(args.sample_ids_file, args.split)
        if args.sample_ids_file is not None
        else None
    )
    if requested_ids is not None:
        if args.max_samples != 16:
            raise ValueError("Do not combine --sample-ids-file with --max-samples")
        dataset = select_dataset_by_ids(datasets[args.split], requested_ids)
    else:
        dataset = datasets[args.split].select(
            range(min(args.max_samples, len(datasets[args.split])))
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        revision=config.get("model_revision"),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    frozen_prompt_template = None
    frozen_prompt_metadata = None
    if args.condition == "frozen_prompt":
        frozen_prompt_template = load_prompt_template(args.frozen_prompt_file)
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            revision=config.get("model_revision"),
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            local_files_only=True,
        )
        model.config.use_cache = False
        hidden_offset, adapter_path, peft_type, seed_equivalence = 0, None, None, None
        frozen_prompt_metadata = {
            "path": str(args.frozen_prompt_file.resolve()),
            "sha256": prompt_sha256(frozen_prompt_template),
            "characters": len(frozen_prompt_template),
        }
    else:
        model, hidden_offset, adapter_path, peft_type, seed_equivalence = load_model(
            config,
            condition=args.condition,
            reference_run=args.reference_run,
            seed_adapter_run=None,
            equivalent_seed_adapter_runs=(),
        )
    prefix_compatibility = None
    if peft_type == "PREFIX_TUNING":
        prefix_compatibility = install_gemma_prefix_cache_compatibility(
            model,
            num_virtual_tokens=int(config["num_virtual_tokens"]),
        )
    decoder_layer = resolve_gemma_decoder_layer(model, args.layer)
    sae = GemmaScopeJumpReLU.from_npz(
        args.sae_path,
        device=model.device,
        dtype=torch.float32,
    )
    sae.eval()

    alignment_batch = dataset[: min(args.batch_size, len(dataset))]
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        alignment_encoded = encode_prompts(
            tokenizer,
            alignment_batch,
            labels=labels,
            setup=setup,
            binary_output_format=binary_output_format,
            max_length=max_input_length,
            device=model.device,
            frozen_prompt_template=frozen_prompt_template,
        )
        hook_alignment = verify_hook_alignment(model, alignment_encoded, layer=args.layer)
    finally:
        tokenizer.padding_side = old_padding_side

    torch.cuda.reset_peak_memory_stats()
    started_at = time.time()
    if args.original_predictions_file is None:
        original_targets, original_predictions, original_raw, _ = predict_pass(
            model,
            tokenizer,
            dataset,
            batch_size=args.batch_size,
            max_length=max_input_length,
            max_new_tokens=int(config["max_new_tokens"]),
            labels=labels,
            setup=setup,
            binary_output_format=binary_output_format,
            frozen_prompt_template=frozen_prompt_template,
        )
    else:
        original_targets, original_predictions, original_raw = load_original_predictions(
            args.original_predictions_file,
            dataset,
            setup=setup,
        )
    reconstructed_targets, reconstructed_predictions, reconstructed_raw, reconstruction = (
        predict_pass(
            model,
            tokenizer,
            dataset,
            batch_size=args.batch_size,
            max_length=max_input_length,
            max_new_tokens=int(config["max_new_tokens"]),
            labels=labels,
            setup=setup,
            binary_output_format=binary_output_format,
            decoder_layer=decoder_layer,
            sae=sae,
            scope=args.scope,
            prepended_virtual_tokens=hidden_offset,
            sae_chunk_size=args.sae_chunk_size,
            frozen_prompt_template=frozen_prompt_template,
        )
    )
    if original_targets != reconstructed_targets:
        raise RuntimeError("Original and reconstructed passes produced different targets")

    original_metrics = score_pass(
        original_targets,
        original_predictions,
        labels=labels,
        setup=setup,
    )
    reconstructed_metrics = score_pass(
        reconstructed_targets,
        reconstructed_predictions,
        labels=labels,
        setup=setup,
    )
    original_rows = prediction_rows(
        dataset,
        original_targets,
        original_predictions,
        original_raw,
        setup=setup,
        labels=labels,
    )
    reconstructed_rows = prediction_rows(
        dataset,
        reconstructed_targets,
        reconstructed_predictions,
        reconstructed_raw,
        setup=setup,
        labels=labels,
    )
    write_json(args.output_dir / "original_predictions.json", original_rows)
    write_json(args.output_dir / "reconstructed_predictions.json", reconstructed_rows)

    payload = {
        "status": "done",
        "condition": args.condition_name or args.condition,
        "condition_loader": args.condition,
        "scope": args.scope,
        "layer": args.layer,
        "split": args.split,
        "samples": len(dataset),
        "max_input_length": max_input_length,
        "labels": list(labels),
        "setup": setup,
        "reference_run": str(args.reference_run.resolve()),
        "adapter_path": adapter_path,
        "peft_type": peft_type,
        "prepended_virtual_tokens": hidden_offset,
        "seed_adapter_equivalence": seed_equivalence,
        "frozen_prompt": frozen_prompt_metadata,
        "sample_selection": {
            "manifest": (
                str(args.sample_ids_file.resolve())
                if args.sample_ids_file is not None
                else None
            ),
            "ids": [str(value) for value in dataset["id"]],
        },
        "original_predictions_source": (
            str(args.original_predictions_file.resolve())
            if args.original_predictions_file is not None
            else "generated_in_run"
        ),
        "prefix_cache_compatibility": prefix_compatibility,
        "sae": {
            "path": str(args.sae_path.resolve()),
            "source_revision": args.sae_revision,
            "size_bytes": args.sae_path.stat().st_size,
            "d_in": sae.d_in,
            "d_sae": sae.d_sae,
        },
        "hook_alignment": hook_alignment,
        "reconstruction": reconstruction,
        "original_metrics": original_metrics,
        "reconstructed_metrics": reconstructed_metrics,
        "metric_delta": {
            key: reconstructed_metrics[key] - original_metrics[key]
            for key in original_metrics.keys() & reconstructed_metrics.keys()
        },
        "paired": paired_summary(
            original_targets,
            original_predictions,
            reconstructed_predictions,
        ),
        "split_contract": split_contract[args.split],
        "elapsed_seconds": time.time() - started_at,
        "resource_usage": {
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "environment": {
            "git_revision": git_revision(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "visible_devices": visible,
        },
    }
    write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
