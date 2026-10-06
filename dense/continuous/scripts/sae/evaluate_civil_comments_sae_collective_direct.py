#!/usr/bin/env python3
"""Evaluate collective SAE shift interventions at the final residual block."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_optimization.civil_comments import sha256_file
from prompt_optimization.gemma_scope import (
    GemmaScopeJumpReLU,
    resolve_gemma_final_norm,
)
from prompt_optimization.sae_intervention import (
    build_collective_interventions,
    decompose_collective_shift,
    js_from_logits,
    kl_from_logits,
    normalized_recovery,
    random_norm_matched,
    shuffled_norm_matched,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--manual-activation-dir", type=Path, required=True)
    parser.add_argument("--method-activation-dir", type=Path, required=True)
    parser.add_argument("--sae-manifest", type=Path, required=True)
    parser.add_argument("--sae-snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sae-chunk-size", type=int, default=128)
    parser.add_argument("--control-seeds", type=int, nargs="+", default=(9101, 9102, 9103, 9104, 9105))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty metric table")
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
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


def load_activation_directory(
    directory: Path,
    *,
    expected_condition: str,
    layer: int,
    max_samples: int | None,
) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    summary = read_json(directory / "summary.json")
    metadata = read_json(directory / "metadata.json")
    if summary.get("status") != "done" or summary.get("condition") != expected_condition:
        raise ValueError(f"Invalid {expected_condition} activation directory: {directory}")
    layers = [int(value) for value in summary["layers"]]
    if layer not in layers:
        raise ValueError(f"Layer {layer} missing from {directory}")
    rows = metadata.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"Activation metadata rows missing from {directory}")
    states = load_file(str(directory / "states.safetensors"), device="cpu")["states"]
    if list(states.shape) != summary.get("shape") or len(states) != len(rows):
        raise ValueError(f"Activation shape/metadata mismatch in {directory}")
    limit = len(states) if max_samples is None else min(max_samples, len(states))
    return states[:limit, layers.index(layer)].contiguous(), summary, rows[:limit]


def canonical_sae_entry(manifest: dict[str, Any], layer: int) -> dict[str, Any]:
    matches = [
        entry
        for entry in manifest["entries"]
        if int(entry["layer"]) == layer and "canonical_depth" in entry["roles"]
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one canonical SAE for layer {layer}, found {len(matches)}")
    return matches[0]


def resolve_lm_head(model: Any) -> torch.nn.Module:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    lm_head = getattr(base_model, "lm_head", None)
    if not isinstance(lm_head, torch.nn.Module):
        raise TypeError("Expected a causal LM exposing lm_head")
    return lm_head


def project_states(
    states: torch.Tensor,
    *,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
) -> torch.Tensor:
    parameter = next(lm_head.parameters())
    normalized = final_norm(states.to(dtype=parameter.dtype))
    return lm_head(normalized)


def first_token_ids(tokenizer: Any, labels: list[str]) -> dict[str, int]:
    output: dict[str, int] = {}
    for label in (*labels, "NONE"):
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Tokenizer produced no token for {label!r}")
        output[label] = int(token_ids[0])
    return output


def safe_float(value: torch.Tensor) -> float | None:
    number = float(value.item())
    return number if math.isfinite(number) else None


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["operation"]), str(row["condition"]), row.get("control_seed"))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    metrics = (
        "kl_to_target",
        "js_to_target",
        "normalized_kl_recovery",
        "logit_shift_cosine",
        "logit_shift_norm_ratio",
        "top1_target_agreement",
        "top1_baseline_agreement",
    )
    for (operation, condition, control_seed), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "operation": operation,
            "condition": condition,
            "control_seed": control_seed,
            "samples": len(group),
        }
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in group if row.get(metric) is not None],
                dtype=np.float64,
            )
            item[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    for name in ("batch_size", "sae_chunk_size", "cpu_threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if len(set(args.control_seeds)) != len(args.control_seeds):
        raise ValueError("--control-seeds must be unique")
    if not torch.cuda.is_available():
        raise RuntimeError("Direct collective intervention evaluation requires CUDA")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(args.cpu_threads)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        args.output_dir / "summary.json",
        args.output_dir / "per_sample_metrics.csv",
        args.output_dir / "aggregate_metrics.csv",
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"Refusing to overwrite artifacts in {args.output_dir}")

    manual_cpu, manual_summary, manual_rows = load_activation_directory(
        args.manual_activation_dir,
        expected_condition="manual",
        layer=args.layer,
        max_samples=args.max_samples,
    )
    method_condition = str(read_json(args.method_activation_dir / "summary.json")["condition"])
    if method_condition not in {"prompt", "prefix"}:
        raise ValueError(f"Unsupported method condition: {method_condition}")
    adapted_cpu, method_summary, method_rows = load_activation_directory(
        args.method_activation_dir,
        expected_condition=method_condition,
        layer=args.layer,
        max_samples=args.max_samples,
    )
    manual_ids = [str(row["id"]) for row in manual_rows]
    if [str(row["id"]) for row in method_rows] != manual_ids:
        raise ValueError("Manual and adapted sample order differs")
    if [row["labels"] for row in method_rows] != [row["labels"] for row in manual_rows]:
        raise ValueError("Manual and adapted gold labels differ")

    config = read_json(args.reference_run / "config.json")
    if Path(method_summary["reference_run"]).resolve() != args.reference_run.resolve():
        raise ValueError("Method activation reference run differs from --reference-run")
    sae_manifest = read_json(args.sae_manifest)
    sae_entry = canonical_sae_entry(sae_manifest, args.layer)
    sae_path = args.sae_snapshot / sae_entry["path"]
    if sha256_file(sae_path) != sae_entry["sha256"]:
        raise ValueError("SAE file hash differs from manifest")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        revision=config.get("model_revision"),
        local_files_only=True,
    )
    token_ids = first_token_ids(tokenizer, list(config["labels"]))
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        revision=config.get("model_revision"),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )
    model.eval()
    final_norm = resolve_gemma_final_norm(model)
    lm_head = resolve_lm_head(model)
    for module in (final_norm, lm_head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    sae = GemmaScopeJumpReLU.from_npz(
        sae_path,
        device=model.device,
        dtype=torch.float32,
    )

    manual = manual_cpu.to(device=model.device, dtype=torch.float32)
    adapted = adapted_cpu.to(device=model.device, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    started_at = time.time()
    decomposition = decompose_collective_shift(
        manual,
        adapted,
        sae,
        chunk_size=args.sae_chunk_size,
    )
    core_states = build_collective_interventions(
        manual,
        adapted,
        decomposition.sparse,
        shuffle_seed=args.control_seeds[0],
        random_seed=args.control_seeds[0],
    )
    state_conditions: dict[str, torch.Tensor] = {
        name: states
        for name, states in core_states.items()
        if not name.startswith(("shuffled_", "random_"))
    }
    for control_seed in args.control_seeds:
        shuffled = shuffled_norm_matched(decomposition.sparse, seed=control_seed)
        random = random_norm_matched(decomposition.sparse, seed=control_seed)
        state_conditions[f"shuffled_add_s{control_seed}"] = manual + shuffled
        state_conditions[f"shuffled_remove_s{control_seed}"] = adapted - shuffled
        state_conditions[f"random_add_s{control_seed}"] = manual + random
        state_conditions[f"random_remove_s{control_seed}"] = adapted - random

    max_dense_state_error = max(
        float((state_conditions["dense_add"] - adapted).abs().max().item()),
        float((state_conditions["dense_remove"] - manual).abs().max().item()),
    )
    if max_dense_state_error > 1e-6:
        raise RuntimeError(f"Dense intervention identity failed: {max_dense_state_error}")

    rows: list[dict[str, Any]] = []
    max_dense_logit_error = 0.0
    for start in range(0, len(manual), args.batch_size):
        stop = min(start + args.batch_size, len(manual))
        with torch.inference_mode():
            manual_logits = project_states(
                manual[start:stop], final_norm=final_norm, lm_head=lm_head
            )
            adapted_logits = project_states(
                adapted[start:stop], final_norm=final_norm, lm_head=lm_head
            )
            add_baseline_kl = kl_from_logits(adapted_logits, manual_logits)
            remove_baseline_kl = kl_from_logits(manual_logits, adapted_logits)
            add_dense_shift = adapted_logits.float() - manual_logits.float()
            remove_dense_shift = -add_dense_shift
            manual_top1 = manual_logits.argmax(dim=-1)
            adapted_top1 = adapted_logits.argmax(dim=-1)

            for condition, condition_states in state_conditions.items():
                if condition in {"manual", "adapted"}:
                    continue
                operation = "add" if condition.endswith("add") or "_add_" in condition else "remove"
                candidate_logits = project_states(
                    condition_states[start:stop],
                    final_norm=final_norm,
                    lm_head=lm_head,
                )
                target_logits = adapted_logits if operation == "add" else manual_logits
                baseline_logits = manual_logits if operation == "add" else adapted_logits
                baseline_kl = add_baseline_kl if operation == "add" else remove_baseline_kl
                dense_logit_shift = add_dense_shift if operation == "add" else remove_dense_shift
                candidate_shift = candidate_logits.float() - baseline_logits.float()
                kl = kl_from_logits(target_logits, candidate_logits)
                js = js_from_logits(target_logits, candidate_logits)
                recovery = normalized_recovery(baseline_kl, kl)
                cosine = functional.cosine_similarity(
                    dense_logit_shift,
                    candidate_shift,
                    dim=-1,
                    eps=1e-12,
                )
                dense_norm = dense_logit_shift.norm(dim=-1).clamp_min(1e-12)
                norm_ratio = candidate_shift.norm(dim=-1) / dense_norm
                candidate_top1 = candidate_logits.argmax(dim=-1)
                target_top1 = adapted_top1 if operation == "add" else manual_top1
                baseline_top1 = manual_top1 if operation == "add" else adapted_top1
                if condition in {"dense_add", "dense_remove"}:
                    max_dense_logit_error = max(
                        max_dense_logit_error,
                        float((candidate_logits.float() - target_logits.float()).abs().max().item()),
                    )
                control_seed = None
                if "_s" in condition:
                    control_seed = int(condition.rsplit("_s", 1)[1])
                short_condition = condition.split("_s", 1)[0]
                for offset in range(stop - start):
                    item: dict[str, Any] = {
                        "sample_id": manual_ids[start + offset],
                        "labels": "|".join(map(str, manual_rows[start + offset]["labels"])) or "NONE",
                        "method": method_condition,
                        "train_samples": int(config["train_samples"]),
                        "num_virtual_tokens": int(config["num_virtual_tokens"]),
                        "training_seed": int(config["training_seed"]),
                        "layer": args.layer,
                        "operation": operation,
                        "condition": short_condition,
                        "control_seed": control_seed,
                        "kl_to_target": safe_float(kl[offset]),
                        "js_to_target": safe_float(js[offset]),
                        "baseline_kl": safe_float(baseline_kl[offset]),
                        "normalized_kl_recovery": safe_float(recovery[offset]),
                        "logit_shift_cosine": safe_float(cosine[offset]),
                        "logit_shift_norm_ratio": safe_float(norm_ratio[offset]),
                        "top1_token_id": int(candidate_top1[offset].item()),
                        "top1_target_agreement": int(candidate_top1[offset] == target_top1[offset]),
                        "top1_baseline_agreement": int(candidate_top1[offset] == baseline_top1[offset]),
                    }
                    for label, token_id in token_ids.items():
                        item[f"logit_{label}"] = float(candidate_logits[offset, token_id].float().item())
                    rows.append(item)

    if max_dense_logit_error > 1e-5:
        raise RuntimeError(f"Dense final-layer logits differ from target: {max_dense_logit_error}")
    aggregate_rows = aggregate(rows)
    write_csv(args.output_dir / "per_sample_metrics.csv", rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate_rows)
    summary = {
        "status": "done",
        "scope": "collective_sae_direct_readout",
        "method": method_condition,
        "config": {
            "reference_run": str(args.reference_run.resolve()),
            "train_samples": int(config["train_samples"]),
            "num_virtual_tokens": int(config["num_virtual_tokens"]),
            "training_seed": int(config["training_seed"]),
        },
        "samples": len(manual),
        "layer": args.layer,
        "anchor": "last_common_textual_prompt_token",
        "conditions": sorted({str(row["condition"]) for row in rows}),
        "control_seeds": args.control_seeds,
        "first_token_ids": token_ids,
        "numerical_checks": {
            "max_dense_state_error": max_dense_state_error,
            "max_dense_logit_error": max_dense_logit_error,
        },
        "artifacts": {
            "per_sample_metrics": str((args.output_dir / "per_sample_metrics.csv").resolve()),
            "aggregate_metrics": str((args.output_dir / "aggregate_metrics.csv").resolve()),
        },
        "inputs": {
            "manual_activation_summary": str((args.manual_activation_dir / "summary.json").resolve()),
            "method_activation_summary": str((args.method_activation_dir / "summary.json").resolve()),
            "sae_manifest": str(args.sae_manifest.resolve()),
            "sae_path": str(sae_path.resolve()),
            "sae_sha256": sae_entry["sha256"],
            "manual_states_sha256": manual_summary["states_file"]["sha256"],
            "method_states_sha256": method_summary["states_file"]["sha256"],
        },
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
            "cuda": torch.version.cuda,
            "visible_devices": visible,
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"status": "done", "samples": len(manual), "rows": len(rows)}))


if __name__ == "__main__":
    main()
