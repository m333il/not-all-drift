#!/usr/bin/env python3
"""Causally test sparse SAE feature subsets in directed method replacement."""

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
from prompt_optimization.dense_carrier_screen import aggregate_recovery, replacement_delta, shuffled_rows
from prompt_optimization.feature_subset_replacement import masked_feature_delta, norm_match_rows
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
from scripts.sae.evaluate_cross_method_sae_carrier_gate import (  # noqa: E402
    delta_geometry,
    per_sample_geometry,
    prefixed_recovery,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("civil_multilabel", "civil_binary_yesno", "amazon_rating"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prompt-run-dir", type=Path, required=True)
    parser.add_argument("--prefix-run-dir", type=Path, required=True)
    parser.add_argument(
        "--gepa-prompt-file",
        type=Path,
        help=(
            "Optional frozen GEPA prompt template containing one {text} placeholder. "
            "The feature-set manifest must contain every resulting pair."
        ),
    )
    parser.add_argument("--adapter-checkpoint", default="best_adapter")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split", default="intervention_val")
    parser.add_argument("--sample-ids-file", type=Path)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--feature-sets", type=Path, required=True)
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--sae-manifest", type=Path, required=True)
    parser.add_argument("--sae-snapshot", type=Path, required=True)
    parser.add_argument("--carrier", default="generation_anchor")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--distribution-batch-size", type=int, default=8)
    parser.add_argument("--sae-chunk-size", type=int, default=128)
    parser.add_argument("--max-input-length", type=int, default=8_144)
    parser.add_argument("--shuffle-seed", type=int, default=913_042)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def encode_and_reconstruct_cpu(
    sae: GemmaScopeJumpReLU,
    states: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_chunks: list[torch.Tensor] = []
    reconstruction_chunks: list[torch.Tensor] = []
    for start in range(0, len(states), chunk_size):
        features = sae.encode(states[start : start + chunk_size].to(sae.W_enc.device))
        reconstruction = sae.decode(features)
        feature_chunks.append(features.detach().cpu().float())
        reconstruction_chunks.append(reconstruction.detach().cpu().float())
    return torch.cat(feature_chunks), torch.cat(reconstruction_chunks)


def pair_key(left: str, right: str) -> str:
    return "__".join(sorted((left, right)))


def component_vectors(
    *,
    host: str,
    goal: str,
    host_states: torch.Tensor,
    goal_states: torch.Tensor,
    host_features: torch.Tensor,
    goal_features: torch.Tensor,
    host_reconstruction: torch.Tensor,
    goal_reconstruction: torch.Tensor,
    decoder: torch.Tensor,
    feature_sets: dict[str, list[int]],
    shuffle_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int | None]]:
    feature_delta = goal_features.float() - host_features.float()
    dense = replacement_delta(host_states, goal_states)
    full_sae = goal_reconstruction.float() - host_reconstruction.float()
    values: dict[str, torch.Tensor] = {"dense": dense, "full_sae": full_sae}
    counts: dict[str, int | None] = {"dense": None, "full_sae": decoder.shape[0]}
    for k in (8, 16, 32, 64, 128):
        name = f"direct_top{k}"
        indices = feature_sets[name]
        values[name] = masked_feature_delta(feature_delta, decoder, indices)
        counts[name] = len(indices)
    for k in (32, 64, 128):
        name = f"aligned_shared_top{k}"
        indices = feature_sets[name]
        values[name] = masked_feature_delta(feature_delta, decoder, indices)
        counts[name] = len(indices)
    for role, method in (("host", host), ("goal", goal)):
        source_name = f"{method}_unique_top64"
        name = f"{role}_unique_top64"
        indices = feature_sets[source_name]
        values[name] = masked_feature_delta(feature_delta, decoder, indices)
        counts[name] = len(indices)

    reference = values["direct_top64"]
    random_indices = feature_sets["random_top64"]
    random = masked_feature_delta(feature_delta, decoder, random_indices)
    values["random_norm_matched_top64"] = norm_match_rows(random, reference)
    counts["random_norm_matched_top64"] = len(random_indices)
    values["sign_flipped_direct_top64"] = -reference
    counts["sign_flipped_direct_top64"] = counts["direct_top64"]
    values["shuffled_direct_top64"] = shuffled_rows(reference, seed=shuffle_seed)
    counts["shuffled_direct_top64"] = counts["direct_top64"]
    return values, counts


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Feature-subset replacement requires CUDA")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    for name in (
        "max_samples",
        "batch_size",
        "distribution_batch_size",
        "sae_chunk_size",
        "max_input_length",
        "cpu_threads",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.layer < 0 or args.output_dir.exists():
        raise ValueError("Layer must be non-negative and output must not exist")
    selection = read_json(args.feature_sets)
    if selection.get("status") != "done":
        raise ValueError("Feature-set manifest is incomplete")
    expected = {
        "task": args.task,
        "optimizer_seed": args.seed,
        "carrier": args.carrier,
        "layer": args.layer,
    }
    mismatches = {
        key: (selection.get(key), value)
        for key, value in expected.items()
        if selection.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Feature-set contract mismatch: {mismatches}")
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.reset_peak_memory_stats()
    args.output_dir.mkdir(parents=True)
    started = time.time()

    rows, split_contract = load_rows(args)
    methods = method_names(args)
    calibration_ids = set(map(str, selection["calibration_ids"]))
    evaluation_rows = [row for row in rows if str(row["id"]) not in calibration_ids]
    if len(evaluation_rows) != len(rows) - len(calibration_ids.intersection(
        str(row["id"]) for row in rows
    )):
        raise RuntimeError("Calibration/evaluation ID accounting failed")
    if not evaluation_rows or calibration_ids.intersection(
        str(row["id"]) for row in evaluation_rows
    ):
        raise ValueError("Causal evaluation must be non-empty and disjoint from calibration")
    rows = evaluation_rows
    split_contract["sample_ids_before_calibration_exclusion"] = split_contract["sample_ids"]
    split_contract["sample_ids"] = [str(row["id"]) for row in rows]
    split_contract["calibration_ids_excluded"] = sorted(calibration_ids)

    captures, provenance = capture_all_conditions(
        args,
        rows,
        layers=(args.layer,),
        carriers=(args.carrier,),
    )
    manifest = read_json(args.sae_manifest)
    entry = canonical_sae_entry(manifest, args.layer)
    sae_path = args.sae_snapshot / str(entry["path"])
    if sha256_file(sae_path) != str(entry["sha256"]):
        raise ValueError(f"SAE hash mismatch: {sae_path}")
    sae = GemmaScopeJumpReLU.from_npz(sae_path, device="cuda:0", dtype=torch.float32)
    selected_states = {
        method: captures[method].states[args.layer][
            captures[method].carrier_indices[args.carrier]
        ].float()
        for method in methods
    }
    encoded = {
        method: encode_and_reconstruct_cpu(
            sae, state, chunk_size=args.sae_chunk_size
        )
        for method, state in selected_states.items()
    }
    feature_activations = {method: values[0] for method, values in encoded.items()}
    reconstructions = {method: values[1] for method, values in encoded.items()}
    decoder = sae.W_dec.detach().cpu().float()
    del sae
    gc.collect()
    torch.cuda.empty_cache()

    aggregate: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    for host in methods:
        namespace = condition_namespace(args, host)
        model, tokenizer, template, hidden_offset, _ = load_model_and_template(namespace)
        host_capture = captures[host]
        offsets = host_capture.carrier_offsets[args.carrier]
        for goal in methods:
            if goal == host:
                continue
            goal_capture = captures[goal]
            baseline_metrics = distribution_metrics(
                goal_capture.logits,
                host_capture.logits,
                batch_size=args.distribution_batch_size,
                device=model.device,
            )
            baseline = torch.tensor(baseline_metrics["kl"], dtype=torch.float32)
            key = pair_key(host, goal)
            sets = selection["feature_sets"][key]
            available, counts = component_vectors(
                host=host,
                goal=goal,
                host_states=selected_states[host],
                goal_states=selected_states[goal],
                host_features=feature_activations[host],
                goal_features=feature_activations[goal],
                host_reconstruction=reconstructions[host],
                goal_reconstruction=reconstructions[goal],
                decoder=decoder,
                feature_sets=sets,
                shuffle_seed=transition_shuffle_seed(
                    args.shuffle_seed,
                    host=host,
                    goal=goal,
                    layer=args.layer,
                    carrier=args.carrier,
                ),
            )
            component_order = tuple(available)
            geometry = {
                component: delta_geometry(available["dense"], delta)
                for component, delta in available.items()
            }
            sample_geometry = {
                component: per_sample_geometry(available["dense"], delta, offsets)
                for component, delta in available.items()
            }
            endpoints: dict[str, torch.Tensor] = {}
            for component in ("dense", "full_sae"):
                endpoints[component] = patched_logits(
                    model,
                    tokenizer,
                    template,
                    rows,
                    layer=args.layer,
                    carrier=args.carrier,
                    delta=available[component],
                    carrier_offsets=offsets,
                    expected_token_ids=host_capture.carrier_token_ids[args.carrier],
                    hidden_offset=hidden_offset,
                    max_input_length=args.max_input_length,
                    batch_size=args.batch_size,
                    carrier_task=args.task,
                )
            dense_to_host = torch.tensor(
                distribution_metrics(
                    endpoints["dense"], host_capture.logits,
                    batch_size=args.distribution_batch_size, device=model.device,
                )["kl"],
                dtype=torch.float32,
            )
            sae_to_host = torch.tensor(
                distribution_metrics(
                    endpoints["full_sae"], host_capture.logits,
                    batch_size=args.distribution_batch_size, device=model.device,
                )["kl"],
                dtype=torch.float32,
            )
            full_sae_goal_kl = torch.tensor(
                distribution_metrics(
                    goal_capture.logits, endpoints["full_sae"],
                    batch_size=args.distribution_batch_size, device=model.device,
                )["kl"],
                dtype=torch.float32,
            )
            full_sae_total = aggregate_recovery(baseline, full_sae_goal_kl)["recovery"]

            for component in component_order:
                print(
                    json.dumps(
                        {
                            "phase": "feature_subset_replace",
                            "transition": f"{host}->{goal}",
                            "component": component,
                        }
                    ),
                    flush=True,
                )
                candidate = endpoints.get(component)
                if candidate is None:
                    candidate = patched_logits(
                        model,
                        tokenizer,
                        template,
                        rows,
                        layer=args.layer,
                        carrier=args.carrier,
                        delta=available[component],
                        carrier_offsets=offsets,
                        expected_token_ids=host_capture.carrier_token_ids[args.carrier],
                        hidden_offset=hidden_offset,
                        max_input_length=args.max_input_length,
                        batch_size=args.batch_size,
                        carrier_task=args.task,
                    )
                goal_metrics = distribution_metrics(
                    goal_capture.logits, candidate,
                    batch_size=args.distribution_batch_size, device=model.device,
                )
                dense_metrics = distribution_metrics(
                    endpoints["dense"], candidate,
                    batch_size=args.distribution_batch_size, device=model.device,
                )
                sae_metrics = distribution_metrics(
                    endpoints["full_sae"], candidate,
                    batch_size=args.distribution_batch_size, device=model.device,
                )
                goal_kl = torch.tensor(goal_metrics["kl"], dtype=torch.float32)
                dense_kl = torch.tensor(dense_metrics["kl"], dtype=torch.float32)
                sae_kl = torch.tensor(sae_metrics["kl"], dtype=torch.float32)
                total = aggregate_recovery(baseline, goal_kl)
                given_dense = aggregate_recovery(dense_to_host, dense_kl)
                given_sae = aggregate_recovery(sae_to_host, sae_kl)
                energy_fraction = (
                    geometry[component]["component_delta_energy"]
                    / max(geometry["full_sae"]["component_delta_energy"], 1e-12)
                )
                aggregate.append(
                    {
                        "task": args.task,
                        "seed": args.seed,
                        "host_condition": host,
                        "goal_condition": goal,
                        "transition": f"{host}->{goal}",
                        "carrier": args.carrier,
                        "layer": args.layer,
                        "component": component,
                        "requested_feature_count": counts[component],
                        "evaluation_samples": len(rows),
                        "mean_kl_goal_to_host": float(baseline.mean()),
                        "mean_kl_goal_to_component": float(goal_kl.mean()),
                        "mean_kl_dense_to_component": float(dense_kl.mean()),
                        "mean_kl_full_sae_to_component": float(sae_kl.mean()),
                        "top1_agreement_with_goal": float(np.mean(goal_metrics["top1_agreement"])),
                        "top1_agreement_with_dense": float(np.mean(dense_metrics["top1_agreement"])),
                        "top1_agreement_with_full_sae": float(np.mean(sae_metrics["top1_agreement"])),
                        "component_energy_fraction_of_full_sae": energy_fraction,
                        "recovery_fraction_of_full_sae_total": (
                            float(total["recovery"]) / float(full_sae_total)
                            if math.isfinite(float(full_sae_total)) and abs(float(full_sae_total)) > 1e-8
                            else math.nan
                        ),
                        **geometry[component],
                        **prefixed_recovery("total", total),
                        **prefixed_recovery("given_dense", given_dense),
                        **prefixed_recovery("given_full_sae", given_sae),
                    }
                )
                for sample_index, row in enumerate(rows):
                    per_sample.append(
                        {
                            "sample_id": str(row["id"]),
                            "task": args.task,
                            "seed": args.seed,
                            "transition": f"{host}->{goal}",
                            "component": component,
                            "kl_goal_to_host": float(baseline[sample_index]),
                            "kl_goal_to_component": float(goal_kl[sample_index]),
                            "kl_dense_to_component": float(dense_kl[sample_index]),
                            "kl_full_sae_to_component": float(sae_kl[sample_index]),
                            **sample_geometry[component][sample_index],
                        }
                    )
                write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)
                write_csv(args.output_dir / "per_sample_metrics.csv", per_sample)
                if component not in endpoints:
                    del candidate
                del goal_kl, dense_kl, sae_kl
            del endpoints, available, geometry, sample_geometry
            gc.collect()
            torch.cuda.empty_cache()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    write_json(
        args.output_dir / "summary.json",
        {
            "status": "done",
            "scope": "directed_cross_method_sae_feature_subset_gate",
            "task": args.task,
            "seed": args.seed,
            "methods": list(methods),
            "carrier": args.carrier,
            "layer": args.layer,
            "calibration_samples": len(calibration_ids),
            "evaluation_samples": len(rows),
            "calibration_evaluation_disjoint": True,
            "components": list(component_order),
            "dense_delta": "h_goal - h_host",
            "full_sae_delta": "(z_goal-z_host) @ W_dec",
            "masked_delta": "P_S(z_goal-z_host) @ W_dec; decoder bias omitted",
            "total_recovery": "1 - sum KL(goal||component) / sum KL(goal||host)",
            "given_full_sae_recovery": (
                "1 - sum KL(full_SAE_endpoint||component) / "
                "sum KL(full_SAE_endpoint||host)"
            ),
            "selection_manifest": str(args.feature_sets.resolve()),
            "selection_manifest_sha256": sha256_file(args.feature_sets),
            "split_contract": split_contract,
            "condition_provenance": provenance,
            "sae_path": str(sae_path.resolve()),
            "sae_sha256": str(entry["sha256"]),
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "git_revision": git_revision(),
            "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "elapsed_seconds": time.time() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "visible_device": visible,
            "cpu_threads": args.cpu_threads,
        },
    )
    print(json.dumps({"status": "done", "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
