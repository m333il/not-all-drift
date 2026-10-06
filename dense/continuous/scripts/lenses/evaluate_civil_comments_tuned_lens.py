#!/usr/bin/env python3
"""Evaluate one shared Tuned Lens on manual, seed, or adapted activations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from prompt_optimization.tuned_lens import evaluate_lens_bank, load_translator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens-dir", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=("manual", "seed", "adapted"),
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=("probe_train", "probe_val", "test"),
        default="test",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_and_validate_states(
    activation_dir: Path,
    *,
    condition: str,
    split: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    summary_path = activation_dir / f"{condition}_summary.json"
    if not summary_path.is_file():
        summary_path = activation_dir / "summary.json"
    tensor_path = activation_dir / f"{condition}_{split}.safetensors"
    if not summary_path.is_file() or not tensor_path.is_file():
        raise FileNotFoundError(f"Missing activation summary or tensor in {activation_dir}")
    summary = read_json(summary_path)
    if summary.get("status") != "done" or summary.get("condition") != condition:
        raise ValueError(f"Activation condition {condition!r} is not complete")
    if split not in summary.get("splits", {}):
        raise ValueError(f"Split {split!r} is absent from the activation summary")
    states = load_file(tensor_path, device="cpu").get("states")
    if states is None or states.ndim != 3:
        raise ValueError(f"Invalid states tensor in {tensor_path}")
    if list(states.shape) != summary["splits"][split]["shape"]:
        raise ValueError("Activation tensor shape differs from its recorded contract")
    return states, summary


def resolve_decoder(model: Any) -> tuple[torch.nn.Module, torch.nn.Module]:
    backbone = getattr(model, "model", None)
    final_norm = getattr(backbone, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if final_norm is None or lm_head is None:
        raise TypeError("Expected a causal LM exposing model.norm and lm_head")
    return final_norm, lm_head


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.cpu_threads <= 0:
        raise ValueError("batch size and CPU thread count must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Tuned Lens evaluation requires a CUDA GPU")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise RuntimeError("Expose exactly one physical GPU with CUDA_VISIBLE_DEVICES")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(args.cpu_threads)

    run_config_path = args.lens_dir / "run_config.json"
    lens_summary_path = args.lens_dir / "summary.json"
    if not run_config_path.is_file() or not lens_summary_path.is_file():
        raise FileNotFoundError("Tuned Lens directory is incomplete")
    run_config = read_json(run_config_path)
    lens_summary = read_json(lens_summary_path)
    if lens_summary.get("status") != "done":
        raise ValueError("Tuned Lens training has not completed")

    states, activation_summary = load_and_validate_states(
        args.activation_dir,
        condition=args.condition,
        split=args.split,
    )
    expected_shape = run_config["shapes"].get(args.split)
    if expected_shape is not None and list(states.shape[1:]) != expected_shape[1:]:
        raise ValueError("Activation depth or hidden size does not match the trained lens")

    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        run_config["model_name"],
        revision=run_config.get("model_revision"),
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    final_norm, lm_head = resolve_decoder(model)
    for module in (final_norm, lm_head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    del model
    torch.cuda.empty_cache()

    translators = [None] * (states.shape[1] - 1)
    for layer in run_config["layers"]:
        translators[layer] = load_translator(
            args.lens_dir / f"layer_{layer:02d}.safetensors",
            hidden_size=states.shape[2],
            device=device,
        )
    trajectory = evaluate_lens_bank(
        states,
        translators=translators,
        final_norm=final_norm,
        lm_head=lm_head,
        device=device,
        batch_size=args.batch_size,
        temperature=float(run_config["temperature"]),
    )
    direct_trajectory = evaluate_lens_bank(
        states,
        translators=[None] * (states.shape[1] - 1),
        final_norm=final_norm,
        lm_head=lm_head,
        device=device,
        batch_size=args.batch_size,
        temperature=float(run_config["temperature"]),
    )
    write_json(
        args.output,
        {
            "status": "done",
            "lens_dir": str(args.lens_dir.resolve()),
            "activation_dir": str(args.activation_dir.resolve()),
            "condition": args.condition,
            "split": args.split,
            "samples": len(states),
            "trajectory": trajectory,
            "direct_trajectory": direct_trajectory,
            "contracts": {
                "tuned_lens": run_config,
                "activation_reference_run": activation_summary["reference_run"],
            },
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "visible_devices": visible,
            },
        },
    )


if __name__ == "__main__":
    main()
