#!/usr/bin/env python3
"""Evaluate the vanilla Logit Lens on a Civil Comments activation bank."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_optimization.logit_lens import direct_token_probabilities
from prompt_optimization.tuned_lens import evaluate_lens_bank


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_activation_bank(
    directory: Path, condition: str, split: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    summary_path = directory / f"{condition}_summary.json"
    if not summary_path.is_file():
        summary_path = directory / "summary.json"
    tensor_path = directory / f"{condition}_{split}.safetensors"
    if not summary_path.is_file() or not tensor_path.is_file():
        raise FileNotFoundError("Activation summary or tensor is missing")
    summary = read_json(summary_path)
    if summary.get("status") != "done" or summary.get("condition") != condition:
        raise ValueError("Activation condition is incomplete or mismatched")
    states = load_file(tensor_path, device="cpu").get("states")
    if states is None or states.ndim != 3:
        raise ValueError("Activation bank must contain states[samples, depth, hidden]")
    expected = summary.get("splits", {}).get(split, {}).get("shape")
    if expected is not None and list(states.shape) != expected:
        raise ValueError("Activation tensor differs from its recorded shape")
    return states, summary


def resolve_decoder(model: Any) -> tuple[torch.nn.Module, torch.nn.Module]:
    backbone = getattr(model, "model", None)
    final_norm = getattr(backbone, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if final_norm is None or lm_head is None:
        raise TypeError("Expected a causal LM exposing model.norm and lm_head")
    return final_norm, lm_head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument(
        "--condition", choices=("manual", "seed", "adapted"), required=True
    )
    parser.add_argument(
        "--split", choices=("probe_train", "probe_val", "test"), default="test"
    )
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--model-revision")
    parser.add_argument("--label-names", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.batch_size <= 0 or args.temperature <= 0 or args.cpu_threads <= 0:
        raise ValueError("Batch size, temperature, and CPU threads must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or not torch.cuda.is_available():
        raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(args.cpu_threads)

    states, activation_summary = load_activation_bank(
        args.activation_dir, args.condition, args.split
    )
    label_names = list(args.label_names or activation_summary.get("labels", ()))
    if not label_names:
        raise ValueError("Provide --label-names or an activation summary containing labels")
    if activation_summary.get("setup", "multilabel") == "multilabel" and "NONE" not in label_names:
        label_names.append("NONE")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    label_token_ids: list[int] = []
    for label in label_names:
        encoded = tokenizer(label, add_special_tokens=False)["input_ids"]
        if not encoded:
            raise ValueError(f"Label has no tokenizer IDs: {label!r}")
        label_token_ids.append(int(encoded[0]))
    if len(set(label_token_ids)) != len(label_token_ids):
        raise ValueError("Two label names share the same first token; use explicit unique labels")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    final_norm, lm_head = resolve_decoder(model)
    for module in (final_norm, lm_head):
        module.eval().requires_grad_(False)
    device = torch.device("cuda:0")
    trajectory_rows = evaluate_lens_bank(
        states,
        translators=[None] * (states.shape[1] - 1),
        final_norm=final_norm,
        lm_head=lm_head,
        device=device,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )
    trajectory = [
        {"depth_point": row["depth_point"], **row["direct"]}
        for row in trajectory_rows
    ]
    label_probabilities = direct_token_probabilities(
        states,
        token_ids=label_token_ids,
        final_norm=final_norm,
        lm_head=lm_head,
        device=device,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )

    args.output_dir.mkdir(parents=True)
    save_file(
        {"label_probabilities": label_probabilities.contiguous()},
        str(args.output_dir / "label_probabilities.safetensors"),
        metadata={
            "labels": json.dumps(label_names),
            "token_ids": json.dumps(label_token_ids),
        },
    )
    mean_probabilities = label_probabilities.mean(0)
    write_json(
        args.output_dir / "summary.json",
        {
            "status": "done",
            "condition": args.condition,
            "split": args.split,
            "samples": len(states),
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "temperature": args.temperature,
            "labels": label_names,
            "label_first_token_ids": label_token_ids,
            "trajectory": trajectory,
            "mean_label_first_token_probability": {
                label: mean_probabilities[:, index].tolist()
                for index, label in enumerate(label_names)
            },
            "terminal_state_already_normalized": True,
            "activation_contract": {
                "condition": activation_summary["condition"],
                "shape": list(states.shape),
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
