#!/usr/bin/env python3
"""SAE-decompose directed cross-method replacement at dense-gated carriers."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prompt_optimization.civil_comments import sha256_file
from prompt_optimization.dense_carrier_screen import (
    CROSS_METHOD_CARRIERS,
    aggregate_recovery,
    replacement_sae_components,
)
from prompt_optimization.gemma_scope import GemmaScopeJumpReLU

from scripts.sae.evaluate_civil_comments_dense_carrier_screen import (  # noqa: E402
    git_revision,
    load_rows,
    patched_logits,
    write_csv,
    write_json,
)
from prompt_optimization.sae_intervention import distribution_metrics
from scripts.sae.evaluate_civil_comments_sae_collective_direct import canonical_sae_entry  # noqa: E402
from scripts.sae.evaluate_civil_comments_span_sae_prefill import (  # noqa: E402
    DEFAULT_MODEL_REVISION,
    load_model_and_template,
)
from scripts.sae.evaluate_cross_method_dense_carrier_screen import (  # noqa: E402
    capture_all_conditions,
    condition_namespace,
    method_names,
    transition_shuffle_seed,
)


COMPONENTS = ("dense", "sae", "residual", "shuffled_dense", "shuffled_sae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("civil_multilabel", "civil_binary_yesno", "amazon_rating"),
        default="civil_multilabel",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prompt-run-dir", type=Path, required=True)
    parser.add_argument("--prefix-run-dir", type=Path, required=True)
    parser.add_argument(
        "--gepa-prompt-file",
        type=Path,
        help=(
            "Optional frozen GEPA prompt template containing one {text} placeholder. "
            "When supplied, all directed Prompt/Prefix/GEPA transitions are evaluated."
        ),
    )
    parser.add_argument("--adapter-checkpoint", default="best_adapter")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split", default="intervention_val")
    parser.add_argument("--sample-ids-file", type=Path)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--sae-manifest", type=Path, required=True)
    parser.add_argument("--sae-snapshot", type=Path, required=True)
    parser.add_argument(
        "--cells",
        nargs="+",
        default=("generation_anchor:20", "generation_anchor:24", "all_common_real:13"),
        help="Exact carrier:block cells selected by the preceding dense gate.",
    )
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=COMPONENTS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--distribution-batch-size", type=int, default=8)
    parser.add_argument("--sae-chunk-size", type=int, default=128)
    parser.add_argument("--max-input-length", type=int, default=8_144)
    parser.add_argument("--shuffle-seed", type=int, default=123_017)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_cell(value: str) -> tuple[str, int]:
    carrier, separator, raw_layer = value.rpartition(":")
    if not separator or carrier not in CROSS_METHOD_CARRIERS:
        raise ValueError(f"Invalid --cells entry: {value!r}")
    try:
        layer = int(raw_layer)
    except ValueError as error:
        raise ValueError(f"Invalid decoder block in --cells entry: {value!r}") from error
    if layer < 0:
        raise ValueError("Decoder blocks must be non-negative")
    return carrier, layer


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def reconstruct_states(
    sae: GemmaScopeJumpReLU,
    states: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float | int | None]]:
    inputs = states.to(device=sae.W_enc.device, dtype=torch.float32)
    reconstruction, metrics = sae.reconstruct_chunked(inputs, chunk_size=chunk_size)
    output = reconstruction.detach().to(device="cpu", dtype=torch.float32)
    del inputs, reconstruction
    torch.cuda.empty_cache()
    return output, metrics.as_dict()


def delta_geometry(dense: torch.Tensor, component: torch.Tensor) -> dict[str, float]:
    dense_float = dense.float().reshape(-1)
    component_float = component.float().reshape(-1)
    dense_energy = float(torch.dot(dense_float, dense_float))
    component_energy = float(torch.dot(component_float, component_float))
    dot = float(torch.dot(dense_float, component_float))
    denominator = math.sqrt(max(dense_energy * component_energy, 0.0))
    return {
        "dense_delta_energy": dense_energy,
        "component_delta_energy": component_energy,
        "component_energy_fraction": (
            component_energy / dense_energy if dense_energy > 0 else math.nan
        ),
        "component_dense_cosine": dot / denominator if denominator > 0 else math.nan,
    }


def per_sample_geometry(
    dense: torch.Tensor,
    component: torch.Tensor,
    offsets: list[int],
) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    for left, right in zip(offsets[:-1], offsets[1:], strict=True):
        output.append(delta_geometry(dense[left:right], component[left:right]))
    return output


def prefixed_recovery(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Cross-method SAE carrier analysis requires CUDA")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    for name in (
        "batch_size",
        "distribution_batch_size",
        "sae_chunk_size",
        "max_input_length",
        "cpu_threads",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    cells = tuple(dict.fromkeys(parse_cell(value) for value in args.cells))
    carriers = tuple(dict.fromkeys(carrier for carrier, _layer in cells))
    layers = tuple(dict.fromkeys(layer for _carrier, layer in cells))
    requested = tuple(dict.fromkeys(map(str, args.components)))
    if "dense" not in requested:
        raise ValueError("The dense component is required for R_component|dense")
    components = ("dense", *(name for name in requested if name != "dense"))
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    rows, split_contract = load_rows(args)
    methods = method_names(args)
    captures, provenance = capture_all_conditions(
        args,
        rows,
        layers=layers,
        carriers=carriers,
    )

    manifest = read_json(args.sae_manifest)
    saes: dict[int, GemmaScopeJumpReLU] = {}
    sae_provenance: dict[str, dict[str, Any]] = {}
    for layer in layers:
        entry = canonical_sae_entry(manifest, layer)
        path = args.sae_snapshot / str(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"SAE hash mismatch for layer {layer}: {path}")
        saes[layer] = GemmaScopeJumpReLU.from_npz(
            path,
            device="cuda:0",
            dtype=torch.float32,
        )
        sae_provenance[str(layer)] = {
            "path": str(path.resolve()),
            "sha256": entry["sha256"],
            "width": int(saes[layer].d_sae),
            "roles": list(entry["roles"]),
        }

    reconstructions: dict[tuple[str, str, int], torch.Tensor] = {}
    reconstruction_rows: list[dict[str, Any]] = []
    for condition in methods:
        capture = captures[condition]
        for carrier, layer in cells:
            selected = capture.states[layer][capture.carrier_indices[carrier]]
            reconstruction, metrics = reconstruct_states(
                saes[layer],
                selected,
                chunk_size=args.sae_chunk_size,
            )
            reconstructions[(condition, carrier, layer)] = reconstruction
            reconstruction_rows.append(
                {
                    "condition": condition,
                    "carrier": carrier,
                    "layer": layer,
                    **metrics,
                }
            )
            print(
                json.dumps(
                    {
                        "phase": "reconstruct",
                        "condition": condition,
                        "carrier": carrier,
                        "layer": layer,
                        "states": len(selected),
                    }
                ),
                flush=True,
            )
    write_csv(args.output_dir / "reconstruction_metrics.csv", reconstruction_rows)
    del saes
    gc.collect()
    torch.cuda.empty_cache()

    aggregate: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    for host in methods:
        namespace = condition_namespace(args, host)
        model, tokenizer, template, hidden_offset, _details = load_model_and_template(namespace)
        host_capture = captures[host]
        baseline: dict[str, torch.Tensor] = {}
        for goal in methods:
            if goal == host:
                continue
            distances = distribution_metrics(
                captures[goal].logits,
                host_capture.logits,
                batch_size=args.distribution_batch_size,
                device=model.device,
            )
            baseline[goal] = torch.tensor(distances["kl"], dtype=torch.float32)

        for carrier, layer in cells:
            offsets = host_capture.carrier_offsets[carrier]
            for goal in methods:
                if goal == host:
                    continue
                goal_capture = captures[goal]
                host_states = host_capture.states[layer][host_capture.carrier_indices[carrier]]
                goal_states = goal_capture.states[layer][goal_capture.carrier_indices[carrier]]
                available = replacement_sae_components(
                    host_states,
                    goal_states,
                    reconstructions[(host, carrier, layer)],
                    reconstructions[(goal, carrier, layer)],
                    shuffle_seed=transition_shuffle_seed(
                        args.shuffle_seed,
                        host=host,
                        goal=goal,
                        layer=layer,
                        carrier=carrier,
                    ),
                )
                geometry = {
                    component: delta_geometry(available["dense"], available[component])
                    for component in components
                }
                sample_geometry = {
                    component: per_sample_geometry(
                        available["dense"], available[component], offsets
                    )
                    for component in components
                }
                dense_candidate: torch.Tensor | None = None
                dense_to_host: torch.Tensor | None = None
                for component in components:
                    print(
                        json.dumps(
                            {
                                "phase": "replace",
                                "transition": f"{host}->{goal}",
                                "carrier": carrier,
                                "layer": layer,
                                "component": component,
                            }
                        ),
                        flush=True,
                    )
                    candidate = patched_logits(
                        model,
                        tokenizer,
                        template,
                        rows,
                        layer=layer,
                        carrier=carrier,
                        delta=available[component],
                        carrier_offsets=offsets,
                        expected_token_ids=host_capture.carrier_token_ids[carrier],
                        hidden_offset=hidden_offset,
                        max_input_length=args.max_input_length,
                        batch_size=args.batch_size,
                        carrier_task=args.task,
                    )
                    if component == "dense":
                        dense_candidate = candidate
                        dense_host_distances = distribution_metrics(
                            dense_candidate,
                            host_capture.logits,
                            batch_size=args.distribution_batch_size,
                            device=model.device,
                        )
                        dense_to_host = torch.tensor(
                            dense_host_distances["kl"], dtype=torch.float32
                        )
                    if dense_candidate is None or dense_to_host is None:
                        raise RuntimeError("Dense endpoint must be evaluated first")
                    goal_distances = distribution_metrics(
                        goal_capture.logits,
                        candidate,
                        batch_size=args.distribution_batch_size,
                        device=model.device,
                    )
                    host_distances = distribution_metrics(
                        host_capture.logits,
                        candidate,
                        batch_size=args.distribution_batch_size,
                        device=model.device,
                    )
                    dense_distances = distribution_metrics(
                        dense_candidate,
                        candidate,
                        batch_size=args.distribution_batch_size,
                        device=model.device,
                    )
                    goal_kl = torch.tensor(goal_distances["kl"], dtype=torch.float32)
                    dense_kl = torch.tensor(dense_distances["kl"], dtype=torch.float32)
                    total_recovery = aggregate_recovery(baseline[goal], goal_kl)
                    dense_recovery = aggregate_recovery(dense_to_host, dense_kl)
                    aggregate.append(
                        {
                            "host_condition": host,
                            "goal_condition": goal,
                            "transition": f"{host}->{goal}",
                            "seed": args.seed,
                            "carrier": carrier,
                            "layer": layer,
                            "component": component,
                            "samples": len(rows),
                            "selected_states": len(available[component]),
                            "mean_kl_goal_to_host": float(baseline[goal].mean()),
                            "mean_kl_goal_to_component": float(goal_kl.mean()),
                            "mean_kl_dense_to_host": float(dense_to_host.mean()),
                            "mean_kl_dense_to_component": float(dense_kl.mean()),
                            "mean_kl_host_to_component": float(np.mean(host_distances["kl"])),
                            "top1_agreement_with_goal": float(
                                np.mean(goal_distances["top1_agreement"])
                            ),
                            "top1_agreement_with_dense": float(
                                np.mean(dense_distances["top1_agreement"])
                            ),
                            **geometry[component],
                            **prefixed_recovery("total", total_recovery),
                            **prefixed_recovery("given_dense", dense_recovery),
                        }
                    )
                    for sample_index, row in enumerate(rows):
                        goal_den = float(baseline[goal][sample_index])
                        goal_num = float(goal_kl[sample_index])
                        dense_den = float(dense_to_host[sample_index])
                        dense_num = float(dense_kl[sample_index])
                        per_sample.append(
                            {
                                "sample_id": str(row["id"]),
                                "host_condition": host,
                                "goal_condition": goal,
                                "transition": f"{host}->{goal}",
                                "seed": args.seed,
                                "carrier": carrier,
                                "layer": layer,
                                "component": component,
                                "kl_goal_to_host": goal_den,
                                "kl_goal_to_component": goal_num,
                                "kl_dense_to_host": dense_den,
                                "kl_dense_to_component": dense_num,
                                "total_recovery": (
                                    1.0 - goal_num / goal_den if goal_den > 1e-8 else None
                                ),
                                "given_dense_recovery": (
                                    1.0 - dense_num / dense_den if dense_den > 1e-8 else None
                                ),
                                "goal_top1_agreement": int(
                                    goal_distances["top1_agreement"][sample_index]
                                ),
                                "dense_top1_agreement": int(
                                    dense_distances["top1_agreement"][sample_index]
                                ),
                                **sample_geometry[component][sample_index],
                            }
                        )
                    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)
                    write_csv(args.output_dir / "per_sample_metrics.csv", per_sample)
                    if component != "dense":
                        del candidate
                    del goal_kl, dense_kl
                if dense_candidate is not None:
                    del dense_candidate
                del available, geometry, sample_geometry
                gc.collect()
                torch.cuda.empty_cache()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    write_json(
        args.output_dir / "summary.json",
        {
            "status": "done",
            "scope": "directed_cross_method_sae_carrier_gate",
            "task": args.task,
            "seed": args.seed,
            "methods": list(methods),
            "directions": [
                f"{host}->{goal}" for host in methods for goal in methods if host != goal
            ],
            "cells": [f"{carrier}:{layer}" for carrier, layer in cells],
            "components": list(components),
            "samples": len(rows),
            "batch_size": args.batch_size,
            "distribution_batch_size": args.distribution_batch_size,
            "sae_chunk_size": args.sae_chunk_size,
            "cpu_threads": args.cpu_threads,
            "dense_delta": "h_goal - h_host",
            "sae_delta": "reconstruct(h_goal) - reconstruct(h_host)",
            "residual_delta": "dense_delta - sae_delta",
            "total_recovery": (
                "1 - sum KL(p_goal || p_component) / sum KL(p_goal || p_host)"
            ),
            "given_dense_recovery": (
                "1 - sum KL(p_dense || p_component) / sum KL(p_dense || p_host)"
            ),
            "kl_scope": "full vocabulary at the first generated token",
            "selection_policy": (
                "cells fixed from the preceding 200-example seed-42 dense gate; "
                "block 25 and failed task_first/all_fixed carriers excluded"
            ),
            "shuffle_control": (
                "deterministic global derangement of token-state deltas; SAE and dense "
                "use separate derangements"
            ),
            "split_contract": split_contract,
            "condition_provenance": provenance,
            "sae_manifest": str(args.sae_manifest.resolve()),
            "sae_manifest_sha256": sha256_file(args.sae_manifest),
            "sae_snapshot": str(args.sae_snapshot.resolve()),
            "sae_entries": sae_provenance,
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "git_revision": git_revision(),
            "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "carrier_helper_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/prompt_optimization/dense_carrier_screen.py"
            ),
            "elapsed_seconds": time.time() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "visible_device": visible,
        },
    )
    print(json.dumps({"status": "done", "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
