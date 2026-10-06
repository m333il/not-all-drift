#!/usr/bin/env python3
"""Screen carriers for directed dense replacement between prompt methods.

For a directed transition ``host A -> goal B`` this evaluator runs the host
condition and patches aligned resid-post carrier states with ``h_B - h_A``.
The primary metric is the ratio-of-sums recovery of the full-vocabulary
first-generated-token KL gap from A to B.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
    replacement_delta,
    shuffled_rows,
)

from scripts.sae.evaluate_civil_comments_dense_carrier_screen import (  # noqa: E402
    ConditionCapture,
    capture_condition,
    decoder_layers,
    git_revision,
    load_rows,
    patched_logits,
    validate_alignment,
    write_csv,
    write_json,
)
from prompt_optimization.sae_intervention import distribution_metrics
from scripts.sae.evaluate_civil_comments_span_sae_prefill import (  # noqa: E402
    DEFAULT_MODEL_REVISION,
    load_model_and_template,
)


CONTINUOUS_METHODS = ("prompt", "prefix")
COMPONENTS = ("dense", "shuffled_dense")


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
    parser.add_argument("--layers", type=int, nargs="+", default=(20, 13, 24, 25))
    parser.add_argument(
        "--carriers",
        nargs="+",
        default=(
            "generation_anchor",
            "task_first",
            "all_fixed",
            "all_common_real",
            "chat_suffix",
        ),
    )
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=COMPONENTS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--distribution-batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=8_144)
    parser.add_argument("--shuffle-seed", type=int, default=91_031)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def condition_namespace(
    args: argparse.Namespace,
    condition: str,
) -> argparse.Namespace:
    namespace = argparse.Namespace(**vars(args))
    if condition == "prompt":
        namespace.condition = "prompt"
        namespace.run_dir = args.prompt_run_dir
        namespace.prompt_file = None
    elif condition == "prefix":
        namespace.condition = "prefix"
        namespace.run_dir = args.prefix_run_dir
        namespace.prompt_file = None
    elif condition == "gepa":
        if args.gepa_prompt_file is None:
            raise ValueError("GEPA requires --gepa-prompt-file")
        namespace.condition = "manual"
        namespace.run_dir = None
        namespace.prompt_file = args.gepa_prompt_file
    else:  # pragma: no cover - guarded by internal callers
        raise ValueError(f"Unknown condition: {condition}")
    return namespace


def method_names(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the methods in the requested comparison, in stable display order."""
    if args.gepa_prompt_file is not None:
        return (*CONTINUOUS_METHODS, "gepa")
    return CONTINUOUS_METHODS


def transition_shuffle_seed(
    base: int,
    *,
    host: str,
    goal: str,
    layer: int,
    carrier: str,
) -> int:
    payload = f"{base}:{host}:{goal}:{layer}:{carrier}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def capture_all_conditions(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    layers: tuple[int, ...],
    carriers: tuple[str, ...],
) -> tuple[dict[str, ConditionCapture], dict[str, dict[str, Any]]]:
    methods = method_names(args)
    captures: dict[str, ConditionCapture] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for condition in methods:
        namespace = condition_namespace(args, condition)
        model, tokenizer, template, hidden_offset, details = load_model_and_template(namespace)
        if min(layers) < 0 or max(layers) >= len(decoder_layers(model)):
            raise ValueError("Requested layer is outside the decoder")
        captures[condition] = capture_condition(
            model,
            tokenizer,
            template,
            rows,
            layers=layers,
            carriers=carriers,
            hidden_offset=hidden_offset,
            max_input_length=args.max_input_length,
            batch_size=args.batch_size,
            carrier_task=args.task,
        )
        provenance[condition] = details
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1 :]:
            validate_alignment(captures[left], captures[right], carriers=carriers)
    return captures, provenance


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Cross-method dense carrier screening requires CUDA")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    for name in (
        "batch_size",
        "distribution_batch_size",
        "max_input_length",
        "cpu_threads",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    layers = tuple(dict.fromkeys(map(int, args.layers)))
    carriers = tuple(dict.fromkeys(map(str, args.carriers)))
    unknown = sorted(set(carriers) - set(CROSS_METHOD_CARRIERS))
    if unknown:
        raise ValueError(f"Unknown cross-method carriers: {unknown}")
    components = tuple(dict.fromkeys(map(str, args.components)))
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(args.cpu_threads)
    started = time.time()
    rows, split_contract = load_rows(args)
    methods = method_names(args)

    captures, provenance = capture_all_conditions(
        args,
        rows,
        layers=layers,
        carriers=carriers,
    )
    aggregate: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []

    # Carrier-first ordering makes the expected strong generation-anchor result
    # available before the longer instruction-carrier grid finishes.
    for host in methods:
        namespace = condition_namespace(args, host)
        model, tokenizer, template, hidden_offset, _ = load_model_and_template(namespace)
        host_capture = captures[host]
        baseline: dict[str, dict[str, Any]] = {}
        for goal in methods:
            if goal == host:
                continue
            distances = distribution_metrics(
                captures[goal].logits,
                host_capture.logits,
                batch_size=args.distribution_batch_size,
                device=model.device,
            )
            baseline[goal] = {
                "distances": distances,
                "kl": torch.tensor(distances["kl"], dtype=torch.float32),
            }

        for carrier in carriers:
            for layer in layers:
                for goal in methods:
                    if goal == host:
                        continue
                    goal_capture = captures[goal]
                    host_states = host_capture.states[layer][
                        host_capture.carrier_indices[carrier]
                    ]
                    goal_states = goal_capture.states[layer][
                        goal_capture.carrier_indices[carrier]
                    ]
                    dense = replacement_delta(host_states, goal_states)
                    available = {
                        "dense": dense,
                        "shuffled_dense": shuffled_rows(
                            dense,
                            seed=transition_shuffle_seed(
                                args.shuffle_seed,
                                host=host,
                                goal=goal,
                                layer=layer,
                                carrier=carrier,
                            ),
                        ),
                    }
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
                            carrier_offsets=host_capture.carrier_offsets[carrier],
                            expected_token_ids=host_capture.carrier_token_ids[carrier],
                            hidden_offset=hidden_offset,
                            max_input_length=args.max_input_length,
                            batch_size=args.batch_size,
                            carrier_task=args.task,
                        )
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
                        goal_kl = torch.tensor(goal_distances["kl"], dtype=torch.float32)
                        denominator = baseline[goal]["kl"]
                        recovery = aggregate_recovery(denominator, goal_kl)
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
                                "mean_kl_goal_to_host": float(denominator.mean()),
                                "mean_kl_goal_to_intervened": float(goal_kl.mean()),
                                "mean_kl_host_to_intervened": float(
                                    np.mean(host_distances["kl"])
                                ),
                                "top1_agreement_with_goal": float(
                                    np.mean(goal_distances["top1_agreement"])
                                ),
                                "top1_agreement_with_host": float(
                                    np.mean(host_distances["top1_agreement"])
                                ),
                                **recovery,
                            }
                        )
                        for sample_index, row in enumerate(rows):
                            den = float(denominator[sample_index])
                            num = float(goal_kl[sample_index])
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
                                    "kl_goal_to_host": den,
                                    "kl_goal_to_intervened": num,
                                    "kl_host_to_intervened": float(
                                        host_distances["kl"][sample_index]
                                    ),
                                    "recovery": 1.0 - num / den if den > 1e-8 else None,
                                    "top1_agreement_with_goal": int(
                                        goal_distances["top1_agreement"][sample_index]
                                    ),
                                    "top1_agreement_with_host": int(
                                        host_distances["top1_agreement"][sample_index]
                                    ),
                                }
                            )
                        write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)
                        write_csv(args.output_dir / "per_sample_metrics.csv", per_sample)
                        del candidate, goal_kl
                    del dense, available, host_states, goal_states
                    gc.collect()
                    torch.cuda.empty_cache()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    write_json(
        args.output_dir / "summary.json",
        {
            "status": "done",
            "scope": "directed_cross_method_dense_carrier_first_token_screen",
            "task": args.task,
            "seed": args.seed,
            "methods": list(methods),
            "directions": [
                f"{host}->{goal}"
                for host in methods
                for goal in methods
                if host != goal
            ],
            "layers": list(layers),
            "carriers": list(carriers),
            "components": list(components),
            "samples": len(rows),
            "replacement": "run host A and add h_goal_B - h_host_A at aligned carrier states",
            "recovery": (
                "1 - sum KL(p_goal_B || p_intervened_A_to_B) / "
                "sum KL(p_goal_B || p_host_A)"
            ),
            "kl_scope": "full vocabulary at the first generated token",
            "carrier_semantics": {
                "generation_anchor": "actual last non-padding rendered chat token",
                "task_first": "first token of the immutable task suffix",
                "all_fixed": "all immutable task tokens excluding sample text",
                "all_common_real": "immutable task suffix including sample text",
                "chat_suffix": "rendered chat-template tokens after literal Answer:",
            },
            "shuffle_control": (
                "deterministic global derangement of the aligned goal-minus-host "
                "token-state deltas"
            ),
            "selection_policy": (
                "validation dense screen; SAE decomposition and autoregressive task metrics "
                "are deferred until a directed carrier passes the recovery gate"
            ),
            "split_contract": split_contract,
            "condition_provenance": provenance,
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "git_revision": git_revision(),
            "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "carrier_helper_sha256": sha256_file(
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
