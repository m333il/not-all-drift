#!/usr/bin/env python3
"""Evaluate SAE fidelity on Prompt virtual tokens or shared fixed prompt tokens."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_optimization.civil_comments import (
    load_manifest,
    prepended_virtual_tokens,
    read_jsonl,
    sha256_directory,
    sha256_file,
    split_file_name,
    validate_rows,
)
from prompt_optimization.frozen_text_prompt import load_prompt_template, render_prompt
from prompt_optimization.gemma_scope import GemmaScopeJumpReLU, ReconstructionAccumulator
from prompt_optimization.instruction_sae import (
    OneShotInstructionSAEHook,
    build_instruction_window_mask,
)
from prompt_optimization.dense_carrier_screen import (
    CROSS_METHOD_CARRIERS,
    DEFAULT_CARRIERS,
    INTERNAL_CARRIERS,
    carrier_positions,
)
from prompt_optimization.instruction_spans import (
    normalized_content_start,
    task_carrier_character_spans,
    token_positions_by_group,
    token_positions_for_spans,
    virtual_token_positions,
)
from prompt_optimization.prefix_cache import install_gemma_prefix_cache_compatibility
from prompt_optimization.sae_intervention import distribution_metrics

from scripts.sae.evaluate_civil_comments_sae_collective_direct import canonical_sae_entry  # noqa: E402
from scripts.sae.evaluate_civil_comments_sae_fidelity import select_dataset_by_ids  # noqa: E402


Condition = Literal["manual", "prompt", "prefix"]
SpanKind = Literal["prompt_virtual", "fixed"]
INTERVENTIONS = ("sae_reconstruction", "zero_feature", "shuffled_reconstruction")
DEFAULT_MODEL_REVISION = "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition", choices=("manual", "prompt", "prefix"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--adapter-checkpoint", default="best_adapter")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--sample-ids-file", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--span-kind", choices=("prompt_virtual", "fixed"), required=True)
    parser.add_argument(
        "--fixed-groups",
        nargs="+",
        default=("labels", "text_marker", "output_rules", "answer", "all_fixed"),
    )
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--sae-manifest", type=Path, required=True)
    parser.add_argument("--sae-snapshot", type=Path, required=True)
    parser.add_argument("--sae-layers", type=int, nargs="+", default=(6, 13, 20, 24))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--distribution-batch-size", type=int, default=8)
    parser.add_argument("--sae-chunk-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-input-length", type=int, default=8_144)
    parser.add_argument("--shuffle-seed", type=int, default=9101)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def sample_ids(path: Path, split: str) -> list[str]:
    payload = read_json(path)
    if isinstance(payload, list):
        return [str(value) for value in payload]
    if "subsets" in payload:
        return [str(value) for value in payload["subsets"][split]["ids"]]
    return [str(value) for value in payload.get("ids", payload[split])]


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest(args.split_root)
    labels = tuple(map(str, manifest["labels"]))
    setup = str(manifest.get("setup", manifest.get("contract", {}).get("setup")))
    if setup != "multilabel":
        raise ValueError("This experiment requires the v2 multilabel setup")
    source = args.split_root / split_file_name(args.split)
    rows = read_jsonl(source)
    validate_rows(rows, allowed_labels=labels, setup=setup)
    from datasets import Dataset

    rows = select_dataset_by_ids(Dataset.from_list(rows), sample_ids(args.sample_ids_file, args.split)).to_list()
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    return rows, {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "ids": [str(row["id"]) for row in rows],
        "labels": list(labels),
    }


def peft_type_name(model: Any) -> str:
    value = model.peft_config["default"].peft_type
    return str(getattr(value, "value", value))


def load_model_and_template(
    args: argparse.Namespace,
) -> tuple[Any, Any, str, int, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise TypeError("Semantic span extraction requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    provenance: dict[str, Any] = {}
    if args.condition == "manual":
        if args.prompt_file is None or args.run_dir is not None:
            raise ValueError("Manual requires --prompt-file and forbids --run-dir")
        model = base
        template = load_prompt_template(args.prompt_file)
        hidden_offset = 0
        provenance["prompt_file"] = str(args.prompt_file.resolve())
        provenance["prompt_sha256"] = sha256_file(args.prompt_file)
    else:
        if args.run_dir is None or args.prompt_file is not None:
            raise ValueError("Continuous conditions require --run-dir and forbid --prompt-file")
        config = read_json(args.run_dir / "config.json")
        expected = f"{args.condition}_tuning"
        if config["method"] != expected:
            raise ValueError(f"Run method {config['method']!r} does not match {expected!r}")
        if int(config["train_samples"]) != 20_000 or int(config["num_virtual_tokens"]) != 20:
            raise ValueError("Expected continuous N=20000,m=20")
        if int(config["training_seed"]) != args.seed:
            raise ValueError("Run seed does not match --seed")
        adapter = args.run_dir / args.adapter_checkpoint
        model = PeftModel.from_pretrained(base, adapter)
        peft_type = peft_type_name(model)
        num_virtual_tokens = int(model.peft_config["default"].num_virtual_tokens)
        hidden_offset = prepended_virtual_tokens(peft_type, num_virtual_tokens)
        if args.condition == "prefix":
            install_gemma_prefix_cache_compatibility(
                model,
                num_virtual_tokens=num_virtual_tokens,
            )
        template = load_prompt_template(args.run_dir / "prompt.txt")
        provenance.update(
            {
                "run_dir": str(args.run_dir.resolve()),
                "adapter": str(adapter.resolve()),
                "adapter_sha256": sha256_directory(adapter),
                "peft_type": peft_type,
                "num_virtual_tokens": num_virtual_tokens,
            }
        )
    model.eval()
    model.config.use_cache = False
    return model, tokenizer, template, hidden_offset, provenance


def encode_batch(
    tokenizer: Any,
    template: str,
    rows: list[dict[str, Any]],
    *,
    span_kind: SpanKind,
    fixed_group: str | None,
    hidden_offset: int,
    max_input_length: int,
    carrier_task: str = "civil_multilabel",
) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
    rendered = [render_prompt(tokenizer, template, str(row["text"])) for row in rows]
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_offsets_mapping=span_kind == "fixed",
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = old_padding_side
    if int(encoded["attention_mask"].sum(1).max()) > max_input_length:
        raise ValueError("A prompt exceeds --max-input-length; refusing truncation")
    if span_kind == "prompt_virtual":
        if hidden_offset <= 0:
            raise ValueError("prompt_virtual requires Prompt Tuning virtual residual positions")
        position_rows = [virtual_token_positions(hidden_offset) for _ in rows]
    else:
        if fixed_group is None:
            raise ValueError("fixed span requires a fixed group")
        position_rows = []
        for index, (row, rendered_prompt) in enumerate(zip(rows, rendered, strict=True)):
            content = template.format(text=str(row["text"]))
            content_start = normalized_content_start(rendered_prompt, content)
            groups = task_carrier_character_spans(
                content,
                str(row["text"]),
                task=carrier_task,
            )
            if fixed_group in {
                *DEFAULT_CARRIERS,
                *CROSS_METHOD_CARRIERS,
                *INTERNAL_CARRIERS,
                "answer_literal_last",
                "output_rules",
            }:
                token_groups = token_positions_by_group(
                    encoded["offset_mapping"][index],
                    encoded["attention_mask"][index],
                    content_start=content_start,
                    groups=groups,
                    hidden_offset=hidden_offset,
                )
                token_groups["valid_real"] = [
                    token_index + hidden_offset
                    for token_index, valid in enumerate(
                        encoded["attention_mask"][index].tolist()
                    )
                    if bool(valid)
                ]
                position_rows.append(carrier_positions(token_groups, fixed_group))
                continue
            single_token_groups = {
                "answer_last": ("answer", "last"),
                "labels_first": ("labels", "first"),
            }
            span_group, selector = single_token_groups.get(
                fixed_group, (fixed_group, None)
            )
            if span_group not in groups:
                raise ValueError(f"Unknown fixed group: {fixed_group}")
            positions = token_positions_for_spans(
                encoded["offset_mapping"][index],
                encoded["attention_mask"][index],
                content_start=content_start,
                spans=groups[span_group],
                hidden_offset=hidden_offset,
            )
            if selector is not None:
                if not positions:
                    raise ValueError(f"The {span_group} group produced no tokenizer positions")
                positions = positions[-1:] if selector == "last" else positions[:1]
            position_rows.append(positions)
        encoded.pop("offset_mapping")
    return dict(encoded), position_rows


def forward_pass(
    model: Any,
    tokenizer: Any,
    template: str,
    rows: list[dict[str, Any]],
    *,
    span_kind: SpanKind,
    fixed_group: str | None,
    hidden_offset: int,
    max_input_length: int,
    batch_size: int,
    decoder_layer: torch.nn.Module | None = None,
    sae: GemmaScopeJumpReLU | None = None,
    mode: str | None = None,
    sae_chunk_size: int = 128,
    shuffle_seed: int = 0,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    use_intervention = decoder_layer is not None
    if use_intervention != (sae is not None and mode is not None):
        raise ValueError("decoder_layer, sae, and mode must be supplied together")
    logits: list[torch.Tensor] = []
    selected_counts: list[int] = []
    accumulator = ReconstructionAccumulator(retain_state_metrics=True) if use_intervention else None
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded_cpu, positions = encode_batch(
            tokenizer,
            template,
            batch,
            span_kind=span_kind,
            fixed_group=fixed_group,
            hidden_offset=hidden_offset,
            max_input_length=max_input_length,
        )
        encoded = {key: value.to(model.device) for key, value in encoded_cpu.items()}
        handle = None
        controller = None
        if use_intervention:
            sequence_length = encoded["input_ids"].shape[1] + hidden_offset
            mask = build_instruction_window_mask(
                positions,
                sequence_length=sequence_length,
                window="all",
                device=model.device,
            )
            controller = OneShotInstructionSAEHook(
                sae,  # type: ignore[arg-type]
                mask,
                mode=mode,  # type: ignore[arg-type]
                chunk_size=sae_chunk_size,
                shuffle_seed=shuffle_seed + start,
                accumulator=accumulator,
            )
            handle = decoder_layer.register_forward_hook(controller)
        try:
            with torch.inference_mode():
                output = model(**encoded, return_dict=True, use_cache=False)
        finally:
            if handle is not None:
                handle.remove()
        if controller is not None:
            if not controller.applied or controller.result is None:
                raise RuntimeError("SAE span intervention hook was not applied")
            selected_counts.append(controller.result.selected_states)
        logits.append(output.logits[:, -1].detach().cpu().to(torch.bfloat16))
        del output, encoded
    diagnostics = None
    if accumulator is not None:
        metrics = accumulator.compute()
        diagnostics = {
            **metrics.as_dict(),
            "raw_energy_recovered": (
                1.0 - metrics.normalized_mse if metrics.normalized_mse is not None else None
            ),
            "per_state_quantiles": accumulator.distribution_summary(),
            "selected_states_by_batch": selected_counts,
        }
    return torch.cat(logits), diagnostics


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Span SAE evaluation requires CUDA")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(args.cpu_threads)
    started = time.time()
    rows, split_contract = load_rows(args)
    model, tokenizer, template, hidden_offset, provenance = load_model_and_template(args)
    if args.span_kind == "prompt_virtual" and args.condition != "prompt":
        raise ValueError("prompt_virtual is defined only for Prompt Tuning")
    groups = ("prompt_virtual",) if args.span_kind == "prompt_virtual" else tuple(args.fixed_groups)
    sae_manifest = read_json(args.sae_manifest)
    baseline_logits, _ = forward_pass(
        model,
        tokenizer,
        template,
        rows,
        span_kind=args.span_kind,
        fixed_group=None if args.span_kind == "prompt_virtual" else groups[0],
        hidden_offset=hidden_offset,
        max_input_length=args.max_input_length,
        batch_size=args.batch_size,
    )
    aggregate: list[dict[str, Any]] = []
    reconstruction: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    for layer in args.sae_layers:
        entry = canonical_sae_entry(sae_manifest, layer)
        sae_path = args.sae_snapshot / entry["path"]
        if sha256_file(sae_path) != entry["sha256"]:
            raise ValueError(f"SAE hash mismatch for layer {layer}")
        sae = GemmaScopeJumpReLU.from_npz(sae_path, device=model.device, dtype=torch.float32)
        for group in groups:
            condition_metrics: dict[str, dict[str, list[float | int]]] = {}
            for mode in INTERVENTIONS:
                print(json.dumps({"layer": layer, "group": group, "mode": mode}), flush=True)
                candidate, diagnostics = forward_pass(
                    model,
                    tokenizer,
                    template,
                    rows,
                    span_kind=args.span_kind,
                    fixed_group=None if args.span_kind == "prompt_virtual" else group,
                    hidden_offset=hidden_offset,
                    max_input_length=args.max_input_length,
                    batch_size=args.batch_size,
                    decoder_layer=model.get_base_model().model.layers[layer]
                    if hasattr(model, "get_base_model")
                    else model.model.layers[layer],
                    sae=sae,
                    mode=mode,
                    sae_chunk_size=args.sae_chunk_size,
                    shuffle_seed=args.shuffle_seed,
                )
                distances = distribution_metrics(
                    baseline_logits,
                    candidate,
                    batch_size=args.distribution_batch_size,
                    device=model.device,
                )
                condition_metrics[mode] = distances
                aggregate.append(
                    {
                        "condition": args.condition,
                        "seed": args.seed,
                        "span_kind": args.span_kind,
                        "group": group,
                        "sae_layer": layer,
                        "intervention": mode,
                        "mean_next_token_kl": float(np.mean(distances["kl"])),
                        "median_next_token_kl": float(np.median(distances["kl"])),
                        "mean_next_token_js": float(np.mean(distances["js"])),
                        "top1_agreement": float(np.mean(distances["top1_agreement"])),
                    }
                )
                for row, kl, js, agreement in zip(
                    rows,
                    distances["kl"],
                    distances["js"],
                    distances["top1_agreement"],
                    strict=True,
                ):
                    per_sample.append(
                        {
                            "sample_id": str(row["id"]),
                            "condition": args.condition,
                            "seed": args.seed,
                            "span_kind": args.span_kind,
                            "group": group,
                            "sae_layer": layer,
                            "intervention": mode,
                            "next_token_kl": float(kl),
                            "next_token_js": float(js),
                            "top1_agreement": int(agreement),
                        }
                    )
                if diagnostics is None:
                    raise RuntimeError("Missing SAE reconstruction diagnostics")
                reconstruction.append(
                    {
                        "condition": args.condition,
                        "seed": args.seed,
                        "span_kind": args.span_kind,
                        "group": group,
                        "sae_layer": layer,
                        "intervention": mode,
                        **diagnostics,
                    }
                )
                del candidate
            zero = np.asarray(condition_metrics["zero_feature"]["kl"], dtype=float)
            recon = np.asarray(condition_metrics["sae_reconstruction"]["kl"], dtype=float)
            valid = zero > 1e-8
            recovery = np.full_like(zero, np.nan)
            recovery[valid] = 1.0 - recon[valid] / zero[valid]
            aggregate.append(
                {
                    "condition": args.condition,
                    "seed": args.seed,
                    "span_kind": args.span_kind,
                    "group": group,
                    "sae_layer": layer,
                    "intervention": "sae_reconstruction_recovery_vs_zero",
                    "mean_kl_recovery": float(np.nanmean(recovery)),
                    "median_kl_recovery": float(np.nanmedian(recovery)),
                    "valid_recovery_samples": int(valid.sum()),
                }
            )
        del sae
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)
    write_csv(args.output_dir / "per_sample_metrics.csv", per_sample)
    write_json(args.output_dir / "reconstruction_metrics.json", reconstruction)
    write_json(
        args.output_dir / "summary.json",
        {
            "status": "done",
            "scope": "instruction_span_sae_prefill_fidelity",
            "condition": args.condition,
            "seed": args.seed,
            "span_kind": args.span_kind,
            "groups": list(groups),
            "layers": list(args.sae_layers),
            "samples": len(rows),
            "hook_site": "resid_post decoder block output",
            "kl_direction": "KL(p_original || p_intervened), full vocabulary, next token",
            "recovery_definition": "1 - KL(original||SAE reconstruction) / KL(original||zero feature)",
            "provenance": provenance,
            "split_contract": split_contract,
            "sae": {
                "manifest": str(args.sae_manifest.resolve()),
                "manifest_sha256": sha256_file(args.sae_manifest),
                "snapshot": str(args.sae_snapshot.resolve()),
            },
            "code": {
                "git_revision": git_revision(),
                "entrypoint": str(Path(__file__).resolve()),
                "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
                "instruction_spans_sha256": sha256_file(
                    Path(__file__).resolve().parents[1]
                    / "src/prompt_optimization/instruction_spans.py"
                ),
                "instruction_sae_sha256": sha256_file(
                    Path(__file__).resolve().parents[1]
                    / "src/prompt_optimization/instruction_sae.py"
                ),
            },
            "elapsed_seconds": time.time() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "peft": importlib.metadata.version("peft"),
                "cuda": torch.version.cuda,
                "visible_device": visible,
            },
        },
    )
    print(json.dumps({"status": "done", "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
