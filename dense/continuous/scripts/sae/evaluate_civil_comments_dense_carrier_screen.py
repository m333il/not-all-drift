#!/usr/bin/env python3
"""Screen shared prompt carriers with Manual-centered dense residual patches.

This is deliberately a fast first-token/full-vocabulary screen.  It identifies
carriers worth a later autoregressive output-set and task-metric evaluation
without persisting large activation banks.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prompt_optimization.civil_comments import (
    load_manifest,
    read_jsonl,
    sha256_file,
    split_file_name,
    validate_rows,
)
from prompt_optimization.dense_carrier_screen import (
    DEFAULT_CARRIERS,
    aggregate_recovery,
    indices_within_superset,
    shuffled_rows,
)
from prompt_optimization.instruction_sae import (
    OneShotPositionCaptureHook,
    OneShotPositionDeltaHook,
    build_instruction_window_mask,
)
from prompt_optimization.sae_intervention import distribution_metrics, kl_from_logits

from scripts.sae.evaluate_civil_comments_sae_fidelity import sample_ids_from_manifest  # noqa: E402
from scripts.sae.evaluate_civil_comments_span_sae_prefill import (  # noqa: E402
    DEFAULT_MODEL_REVISION,
    encode_batch,
    load_model_and_template,
)


@dataclass(frozen=True)
class ConditionCapture:
    logits: torch.Tensor
    states: dict[int, torch.Tensor]
    carrier_indices: dict[str, torch.Tensor]
    carrier_offsets: dict[str, list[int]]
    carrier_token_ids: dict[str, list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-condition", choices=("prompt", "prefix"), required=True
    )
    parser.add_argument(
        "--task",
        choices=("civil_multilabel", "civil_binary_yesno", "amazon_rating"),
        default="civil_multilabel",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--manual-prompt-file", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", default="best_adapter")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split", default="intervention_val")
    parser.add_argument("--sample-ids-file", type=Path)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--layers", type=int, nargs="+", default=(6, 13, 20, 24))
    parser.add_argument("--carriers", nargs="+", default=DEFAULT_CARRIERS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--distribution-batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=8_144)
    parser.add_argument("--shuffle-seed", type=int, default=31_081)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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


def decoder_layers(model: Any) -> Any:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.layers


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = args.split_root / "manifest.json"
    if args.task in {"civil_multilabel", "civil_binary_yesno"}:
        manifest = load_manifest(args.split_root)
        labels = tuple(map(str, manifest["labels"]))
        setup = str(manifest.get("setup", manifest.get("contract", {}).get("setup")))
        expected_setup = "multilabel" if args.task == "civil_multilabel" else "binary"
        if setup != expected_setup:
            raise ValueError(
                f"Task {args.task} requires setup={expected_setup}, observed {setup}"
            )
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        labels = tuple(map(str, manifest.get("classes", ("1", "2", "3", "4", "5"))))
        setup = "multiclass_rating"
    source = args.split_root / split_file_name(args.split)
    rows = read_jsonl(source)
    if args.task in {"civil_multilabel", "civil_binary_yesno"}:
        validate_rows(rows, allowed_labels=labels, setup=setup)
    else:
        ids: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError(f"Amazon row {index} has an invalid id")
            if row["id"] in ids:
                raise ValueError(f"Amazon split contains duplicate id {row['id']}")
            ids.add(row["id"])
            if not isinstance(row.get("text"), str) or not row["text"]:
                raise ValueError(f"Amazon row {index} has invalid text")
    if args.sample_ids_file is not None:
        requested = sample_ids_from_manifest(args.sample_ids_file, args.split)
        by_id = {str(row["id"]): row for row in rows}
        missing = [sample_id for sample_id in requested if sample_id not in by_id]
        if missing:
            raise ValueError(f"Sample IDs are absent from {args.split}: {missing[:5]}")
        rows = [by_id[sample_id] for sample_id in requested]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No validation rows were selected")
    return rows, {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "split": args.split,
        "sample_ids": [str(row["id"]) for row in rows],
        "labels": list(labels),
        "setup": setup,
        "task": args.task,
    }


def condition_namespace(
    args: argparse.Namespace,
    *,
    condition: str,
) -> argparse.Namespace:
    namespace = argparse.Namespace(**vars(args))
    namespace.condition = condition
    if condition == "manual":
        namespace.run_dir = None
        namespace.prompt_file = args.manual_prompt_file
    else:
        namespace.run_dir = args.source_run_dir
        namespace.prompt_file = None
    return namespace


def encoded_carriers(
    tokenizer: Any,
    template: str,
    rows: list[dict[str, Any]],
    *,
    carriers: tuple[str, ...],
    hidden_offset: int,
    max_input_length: int,
    carrier_task: str,
) -> tuple[dict[str, torch.Tensor], dict[str, list[list[int]]]]:
    # Capture one superset that also contains the chat-template suffix.  This
    # keeps ``generation_anchor`` distinct from the literal ``Answer:`` token
    # while preserving the original content-only carrier selections.
    capture_group = "common_to_generation"
    encoded, superset_rows = encode_batch(
        tokenizer,
        template,
        rows,
        span_kind="fixed",
        fixed_group=capture_group,
        hidden_offset=hidden_offset,
        max_input_length=max_input_length,
        carrier_task=carrier_task,
    )
    positions = {capture_group: superset_rows}
    for carrier in carriers:
        if carrier == capture_group:
            continue
        candidate, carrier_rows = encode_batch(
            tokenizer,
            template,
            rows,
            span_kind="fixed",
            fixed_group=carrier,
            hidden_offset=hidden_offset,
            max_input_length=max_input_length,
            carrier_task=carrier_task,
        )
        if not torch.equal(candidate["input_ids"], encoded["input_ids"]):
            raise ValueError(f"Carrier {carrier} changed encoded input IDs")
        if not torch.equal(candidate["attention_mask"], encoded["attention_mask"]):
            raise ValueError(f"Carrier {carrier} changed the attention mask")
        positions[carrier] = carrier_rows
    return encoded, positions


def capture_condition(
    model: Any,
    tokenizer: Any,
    template: str,
    rows: list[dict[str, Any]],
    *,
    layers: tuple[int, ...],
    carriers: tuple[str, ...],
    hidden_offset: int,
    max_input_length: int,
    batch_size: int,
    carrier_task: str,
) -> ConditionCapture:
    state_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    carrier_indices: dict[str, list[int]] = {carrier: [] for carrier in carriers}
    carrier_offsets: dict[str, list[int]] = {carrier: [0] for carrier in carriers}
    carrier_token_ids: dict[str, list[int]] = {carrier: [] for carrier in carriers}
    logits: list[torch.Tensor] = []
    superset_offset = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded_cpu, positions = encoded_carriers(
            tokenizer,
            template,
            batch,
            carriers=carriers,
            hidden_offset=hidden_offset,
            max_input_length=max_input_length,
            carrier_task=carrier_task,
        )
        superset_rows = positions["common_to_generation"]
        for carrier in carriers:
            local = indices_within_superset(superset_rows, positions[carrier])
            carrier_indices[carrier].extend(superset_offset + index for index in local)
            for selected_positions in positions[carrier]:
                carrier_offsets[carrier].append(
                    carrier_offsets[carrier][-1] + len(selected_positions)
                )
            for batch_index, selected_positions in enumerate(positions[carrier]):
                for position in selected_positions:
                    input_position = position - hidden_offset
                    if input_position < 0:
                        raise ValueError("A real-token carrier overlapped virtual positions")
                    carrier_token_ids[carrier].append(
                        int(encoded_cpu["input_ids"][batch_index, input_position])
                    )
        encoded = {key: value.to(model.device) for key, value in encoded_cpu.items()}
        selection = build_instruction_window_mask(
            superset_rows,
            sequence_length=encoded["input_ids"].shape[1] + hidden_offset,
            window="all",
            device=model.device,
        )
        controllers = {
            layer: OneShotPositionCaptureHook(selection) for layer in layers
        }
        handles = [
            decoder_layers(model)[layer].register_forward_hook(controllers[layer])
            for layer in layers
        ]
        try:
            with torch.inference_mode():
                output = model(
                    **encoded,
                    return_dict=True,
                    use_cache=False,
                    logits_to_keep=1,
                )
        finally:
            for handle in handles:
                handle.remove()
        logits.append(output.logits[:, -1].detach().cpu().to(torch.bfloat16))
        for layer, controller in controllers.items():
            if not controller.applied or controller.states is None:
                raise RuntimeError(f"Capture hook did not run for layer {layer}")
            state_chunks[layer].append(controller.states)
        superset_offset += sum(map(len, superset_rows))
        print(
            json.dumps({"phase": "capture", "done": min(start + len(batch), len(rows)), "total": len(rows)}),
            flush=True,
        )
        del output, encoded
    return ConditionCapture(
        logits=torch.cat(logits),
        states={layer: torch.cat(chunks) for layer, chunks in state_chunks.items()},
        carrier_indices={
            carrier: torch.tensor(indices, dtype=torch.long)
            for carrier, indices in carrier_indices.items()
        },
        carrier_offsets=carrier_offsets,
        carrier_token_ids=carrier_token_ids,
    )


def validate_alignment(
    source: ConditionCapture,
    manual: ConditionCapture,
    *,
    carriers: tuple[str, ...],
) -> None:
    for carrier in carriers:
        if source.carrier_offsets[carrier] != manual.carrier_offsets[carrier]:
            raise ValueError(f"Carrier {carrier} has unequal source/Manual token counts")
        if source.carrier_token_ids[carrier] != manual.carrier_token_ids[carrier]:
            raise ValueError(f"Carrier {carrier} has unequal source/Manual token IDs")


def patched_logits(
    model: Any,
    tokenizer: Any,
    template: str,
    rows: list[dict[str, Any]],
    *,
    layer: int,
    carrier: str,
    delta: torch.Tensor,
    carrier_offsets: list[int],
    expected_token_ids: list[int],
    hidden_offset: int,
    max_input_length: int,
    batch_size: int,
    carrier_task: str,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded_cpu, position_rows = encode_batch(
            tokenizer,
            template,
            batch,
            span_kind="fixed",
            fixed_group=carrier,
            hidden_offset=hidden_offset,
            max_input_length=max_input_length,
            carrier_task=carrier_task,
        )
        observed: list[int] = []
        for batch_index, positions in enumerate(position_rows):
            observed.extend(
                int(encoded_cpu["input_ids"][batch_index, position - hidden_offset])
                for position in positions
            )
        left = carrier_offsets[start]
        right = carrier_offsets[start + len(batch)]
        if observed != expected_token_ids[left:right]:
            raise ValueError(f"Live Manual tokens disagree for carrier {carrier}")
        encoded = {key: value.to(model.device) for key, value in encoded_cpu.items()}
        mask = build_instruction_window_mask(
            position_rows,
            sequence_length=encoded["input_ids"].shape[1] + hidden_offset,
            window="all",
            device=model.device,
        )
        controller = OneShotPositionDeltaHook(mask, delta[left:right])
        handle = decoder_layers(model)[layer].register_forward_hook(controller)
        try:
            with torch.inference_mode():
                output = model(
                    **encoded,
                    return_dict=True,
                    use_cache=False,
                    logits_to_keep=1,
                )
        finally:
            handle.remove()
        if not controller.applied:
            raise RuntimeError("Dense carrier hook was not applied")
        chunks.append(output.logits[:, -1].detach().cpu().to(torch.bfloat16))
        del output, encoded
    return torch.cat(chunks)


def stable_shuffle_seed(base: int, *, layer: int, carrier: str) -> int:
    digest = hashlib.sha256(f"{base}:{layer}:{carrier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Dense carrier screening requires CUDA")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    for name in ("batch_size", "distribution_batch_size", "max_input_length", "cpu_threads"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.source_run_dir is None:
        raise ValueError("Prompt/Prefix require --source-run-dir")
    carriers = tuple(dict.fromkeys(map(str, args.carriers)))
    unknown = sorted(set(carriers) - set(DEFAULT_CARRIERS))
    if unknown:
        raise ValueError(f"Unknown carriers: {unknown}")
    layers = tuple(dict.fromkeys(map(int, args.layers)))
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(args.cpu_threads)
    started = time.time()
    rows, split_contract = load_rows(args)

    source_args = condition_namespace(args, condition=args.source_condition)
    source_model, source_tokenizer, source_template, source_offset, source_provenance = (
        load_model_and_template(source_args)
    )
    if min(layers) < 0 or max(layers) >= len(decoder_layers(source_model)):
        raise ValueError("Requested layer is outside the decoder")
    source = capture_condition(
        source_model,
        source_tokenizer,
        source_template,
        rows,
        layers=layers,
        carriers=carriers,
        hidden_offset=source_offset,
        max_input_length=args.max_input_length,
        batch_size=args.batch_size,
        carrier_task=args.task,
    )
    del source_model, source_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    manual_args = condition_namespace(args, condition="manual")
    manual_model, manual_tokenizer, manual_template, manual_offset, manual_provenance = (
        load_model_and_template(manual_args)
    )
    manual = capture_condition(
        manual_model,
        manual_tokenizer,
        manual_template,
        rows,
        layers=layers,
        carriers=carriers,
        hidden_offset=manual_offset,
        max_input_length=args.max_input_length,
        batch_size=args.batch_size,
        carrier_task=args.task,
    )
    validate_alignment(source, manual, carriers=carriers)

    baseline_distances = distribution_metrics(
        source.logits,
        manual.logits,
        batch_size=args.distribution_batch_size,
        device=manual_model.device,
    )
    baseline_kl = torch.tensor(baseline_distances["kl"], dtype=torch.float32)
    aggregate: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    for layer in layers:
        for carrier in carriers:
            source_states = source.states[layer][source.carrier_indices[carrier]]
            manual_states = manual.states[layer][manual.carrier_indices[carrier]]
            if source_states.shape != manual_states.shape:
                raise ValueError(f"State shape mismatch for layer={layer} carrier={carrier}")
            dense = source_states.float() - manual_states.float()
            components = {
                "dense": dense,
                "shuffled_dense": shuffled_rows(
                    dense,
                    seed=stable_shuffle_seed(args.shuffle_seed, layer=layer, carrier=carrier),
                ),
            }
            for component, delta in components.items():
                print(
                    json.dumps(
                        {
                            "phase": "patch",
                            "source": args.source_condition,
                            "layer": layer,
                            "carrier": carrier,
                            "component": component,
                        }
                    ),
                    flush=True,
                )
                candidate = patched_logits(
                    manual_model,
                    manual_tokenizer,
                    manual_template,
                    rows,
                    layer=layer,
                    carrier=carrier,
                    delta=delta,
                    carrier_offsets=manual.carrier_offsets[carrier],
                    expected_token_ids=manual.carrier_token_ids[carrier],
                    hidden_offset=manual_offset,
                    max_input_length=args.max_input_length,
                    batch_size=args.batch_size,
                    carrier_task=args.task,
                )
                source_distances = distribution_metrics(
                    source.logits,
                    candidate,
                    batch_size=args.distribution_batch_size,
                    device=manual_model.device,
                )
                manual_distances = distribution_metrics(
                    manual.logits,
                    candidate,
                    batch_size=args.distribution_batch_size,
                    device=manual_model.device,
                )
                source_kl = torch.tensor(source_distances["kl"], dtype=torch.float32)
                recovery = aggregate_recovery(baseline_kl, source_kl)
                aggregate.append(
                    {
                        "source_condition": args.source_condition,
                        "seed": args.seed,
                        "carrier": carrier,
                        "layer": layer,
                        "component": component,
                        "samples": len(rows),
                        "selected_states": len(delta),
                        "mean_kl_source_to_manual": float(baseline_kl.mean()),
                        "mean_kl_source_to_patched": float(source_kl.mean()),
                        "mean_kl_manual_to_patched": float(np.mean(manual_distances["kl"])),
                        "top1_agreement_with_source": float(
                            np.mean(source_distances["top1_agreement"])
                        ),
                        "top1_agreement_with_manual": float(
                            np.mean(manual_distances["top1_agreement"])
                        ),
                        **recovery,
                    }
                )
                for index, row in enumerate(rows):
                    denominator = float(baseline_kl[index])
                    numerator = float(source_kl[index])
                    per_sample.append(
                        {
                            "sample_id": str(row["id"]),
                            "source_condition": args.source_condition,
                            "seed": args.seed,
                            "carrier": carrier,
                            "layer": layer,
                            "component": component,
                            "kl_source_to_manual": denominator,
                            "kl_source_to_patched": numerator,
                            "recovery": (
                                1.0 - numerator / denominator
                                if denominator > 1e-8
                                else None
                            ),
                            "top1_agreement_with_source": int(
                                source_distances["top1_agreement"][index]
                            ),
                            "top1_agreement_with_manual": int(
                                manual_distances["top1_agreement"][index]
                            ),
                        }
                    )
                # Keep compact, inspectable checkpoints so the first useful
                # carrier result is visible before the complete grid finishes.
                write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)
                write_csv(args.output_dir / "per_sample_metrics.csv", per_sample)
                del candidate, source_kl, delta
            del dense, source_states, manual_states
            gc.collect()
            torch.cuda.empty_cache()

    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)
    write_csv(args.output_dir / "per_sample_metrics.csv", per_sample)
    write_json(
        args.output_dir / "summary.json",
        {
            "status": "done",
            "scope": "manual_centered_dense_carrier_first_token_screen",
            "task": args.task,
            "source_condition": args.source_condition,
            "seed": args.seed,
            "layers": list(layers),
            "carriers": list(carriers),
            "components": ["dense", "shuffled_dense"],
            "samples": len(rows),
            "estimand": "Manual carrier + (source carrier state - Manual carrier state)",
            "recovery": (
                "1 - sum KL(p_source || p_patched) / "
                "sum KL(p_source || p_manual)"
            ),
            "kl_scope": "full vocabulary at the first generated token",
            "selection_policy": (
                "validation screen only; autoregressive output-set/F1 diagnostics are deferred "
                "until a carrier passes the preregistered dense gate"
            ),
            "shuffle_control": (
                "deterministic global derangement of aligned token-state deltas; "
                "the multiset of delta vectors is preserved"
            ),
            "hook_site": "resid_post decoder block output during prefill",
            "split_contract": split_contract,
            "source_provenance": source_provenance,
            "manual_provenance": manual_provenance,
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "git_revision": git_revision(),
            "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "helper_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/prompt_optimization/dense_carrier_screen.py"
            ),
            "span_helper_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/prompt_optimization/instruction_spans.py"
            ),
            "elapsed_seconds": time.time() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "visible_device": visible,
        },
    )
    print(json.dumps({"status": "done", "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
