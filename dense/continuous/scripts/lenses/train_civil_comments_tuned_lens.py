#!/usr/bin/env python3
"""Train a shared Tuned Lens on held-out manual Civil Comments activations."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from prompt_optimization.tuned_lens import (
    evaluate_lens_bank,
    evaluate_lens_layer,
    fit_tuned_lens_layer,
    load_translator,
    save_translator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--manual-activation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_states(directory: Path, split: str) -> torch.Tensor:
    path = directory / f"manual_{split}.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    states = load_file(path, device="cpu").get("states")
    if states is None or states.ndim != 3:
        raise ValueError(f"Invalid states tensor in {path}")
    return states


def validate_activation_contract(directory: Path) -> dict[str, Any]:
    summary_path = directory / "manual_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = read_json(summary_path)
    if summary.get("status") != "done" or summary.get("condition") != "manual":
        raise ValueError("Tuned Lens must use a completed manual activation condition")
    return summary


def resolve_decoder(model: Any) -> tuple[torch.nn.Module, torch.nn.Module]:
    backbone = getattr(model, "model", None)
    final_norm = getattr(backbone, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if final_norm is None or lm_head is None:
        raise TypeError("Expected a causal LM exposing model.norm and lm_head")
    return final_norm, lm_head


def normalized_layers(requested: list[int] | None, depth_points: int) -> list[int]:
    candidates = list(range(depth_points - 1)) if requested is None else requested
    layers = sorted(set(candidates))
    if not layers:
        raise ValueError("At least one non-final depth point is required")
    if layers[0] < 0 or layers[-1] >= depth_points - 1:
        raise ValueError(
            f"layers must be between 0 and {depth_points - 2}; final depth point is the teacher"
        )
    return layers


def run_contract(
    args: argparse.Namespace,
    *,
    model_config: dict[str, Any],
    layers: list[int],
    shapes: dict[str, list[int]],
) -> dict[str, Any]:
    return {
        "version": "civil_comments_tuned_lens_v1",
        "reference_run": str(args.reference_run.resolve()),
        "manual_activation_dir": str(args.manual_activation_dir.resolve()),
        "model_name": model_config["model_name"],
        "model_revision": model_config.get("model_revision"),
        "layers": layers,
        "shapes": shapes,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "evaluation_interval": args.evaluation_interval,
        "patience": args.patience,
        "seed": args.seed,
    }


def prepare_output(directory: Path, contract: dict[str, Any], *, resume: bool) -> None:
    contract_path = directory / "run_config.json"
    if directory.exists() and any(directory.iterdir()):
        if not resume:
            raise FileExistsError(f"Refusing to overwrite non-empty {directory}; pass --resume")
        if not contract_path.is_file() or read_json(contract_path) != contract:
            raise ValueError("Existing Tuned Lens run_config.json does not match requested run")
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(contract_path, contract)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Tuned Lens training requires a CUDA GPU")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise RuntimeError("Expose exactly one physical GPU with CUDA_VISIBLE_DEVICES")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)

    activation_summary = validate_activation_contract(args.manual_activation_dir)
    model_config = read_json(args.reference_run / "config.json")
    train_states = load_states(args.manual_activation_dir, "probe_train")
    validation_states = load_states(args.manual_activation_dir, "probe_val")
    test_states = load_states(args.manual_activation_dir, "test")
    shapes = {
        "probe_train": list(train_states.shape),
        "probe_val": list(validation_states.shape),
        "test": list(test_states.shape),
    }
    if train_states.shape[1:] != validation_states.shape[1:] or train_states.shape[
        1:
    ] != test_states.shape[1:]:
        raise ValueError("Manual activation split shapes do not share depth and hidden size")
    layers = normalized_layers(args.layers, train_states.shape[1])
    contract = run_contract(args, model_config=model_config, layers=layers, shapes=shapes)
    prepare_output(args.output_dir, contract, resume=args.resume)

    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name"],
        revision=model_config.get("model_revision"),
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    final_norm, lm_head = resolve_decoder(model)
    for module in (final_norm, lm_head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    # Only the final norm and unembedding are needed once activations have been
    # extracted. Keeping the full 2B model alive would waste shared GPU memory.
    del model
    torch.cuda.empty_cache()

    started_at = time.time()
    layer_results: list[dict[str, Any]] = []
    translators = [None] * (train_states.shape[1] - 1)
    for layer in layers:
        translator_path = args.output_dir / f"layer_{layer:02d}.safetensors"
        metrics_path = args.output_dir / f"layer_{layer:02d}.json"
        if translator_path.is_file() and metrics_path.is_file():
            translator = load_translator(
                translator_path,
                hidden_size=train_states.shape[2],
                device=device,
            )
            layer_result = read_json(metrics_path)
        elif translator_path.exists() or metrics_path.exists():
            raise RuntimeError(f"Incomplete checkpoint pair for depth point {layer}")
        else:
            layer_started_at = time.time()
            fit = fit_tuned_lens_layer(
                train_states[:, layer],
                train_states[:, -1],
                validation_states[:, layer],
                validation_states[:, -1],
                final_norm=final_norm,
                lm_head=lm_head,
                device=device,
                steps=args.steps,
                batch_size=args.batch_size,
                evaluation_batch_size=args.evaluation_batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                temperature=args.temperature,
                evaluation_interval=args.evaluation_interval,
                patience=args.patience,
                seed=args.seed + layer,
            )
            translator = fit.translator
            direct_test = evaluate_lens_layer(
                test_states[:, layer],
                test_states[:, -1],
                final_norm=final_norm,
                lm_head=lm_head,
                device=device,
                translator=None,
                batch_size=args.evaluation_batch_size,
                temperature=args.temperature,
            )
            tuned_test = evaluate_lens_layer(
                test_states[:, layer],
                test_states[:, -1],
                final_norm=final_norm,
                lm_head=lm_head,
                device=device,
                translator=translator,
                batch_size=args.evaluation_batch_size,
                temperature=args.temperature,
            )
            layer_result = {
                "status": "done",
                "depth_point": layer,
                "best_step": fit.best_step,
                "best_validation_kl": fit.best_validation_kl,
                "validation": {
                    "direct": fit.direct_validation.as_dict(),
                    "tuned": fit.tuned_validation.as_dict(),
                },
                "test": {
                    "direct": direct_test.as_dict(),
                    "tuned": tuned_test.as_dict(),
                },
                "history": list(fit.history),
                "elapsed_seconds": time.time() - layer_started_at,
            }
            save_translator(
                translator_path,
                translator,
                metadata={
                    "version": contract["version"],
                    "depth_point": str(layer),
                    "model_revision": str(contract["model_revision"]),
                },
            )
            write_json(metrics_path, layer_result)
        translators[layer] = translator
        layer_results.append(layer_result)
        write_json(
            args.output_dir / "summary.partial.json",
            {
                "status": "running",
                "completed_layers": sorted(
                    item["depth_point"] for item in layer_results if item.get("status") == "done"
                ),
                "requested_layers": layers,
            },
        )

    test_trajectory = evaluate_lens_bank(
        test_states,
        translators=translators,
        final_norm=final_norm,
        lm_head=lm_head,
        device=device,
        batch_size=args.evaluation_batch_size,
        temperature=args.temperature,
    )
    manual_direct_test_trajectory = evaluate_lens_bank(
        test_states,
        translators=[None] * (test_states.shape[1] - 1),
        final_norm=final_norm,
        lm_head=lm_head,
        device=device,
        batch_size=args.evaluation_batch_size,
        temperature=args.temperature,
    )
    summary = {
        "status": "done",
        "contract": contract,
        "activation_summary": activation_summary,
        "layer_results": layer_results,
        "manual_test_trajectory": test_trajectory,
        "manual_direct_test_trajectory": manual_direct_test_trajectory,
        "elapsed_seconds": time.time() - started_at,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "visible_devices": visible,
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    partial = args.output_dir / "summary.partial.json"
    if partial.exists():
        partial.unlink()


if __name__ == "__main__":
    main()
