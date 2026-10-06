#!/usr/bin/env python3
"""Fit a source-to-final residual predictor with normalized DENSE+KL loss."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from prompt_optimization.conditions import final_readout
from prompt_optimization.early_exit import ResidualLinear, forward_kl


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def batches(data: dict[str, torch.Tensor], size: int, target_key: str):
    for start in range(0, len(data["source_states"]), size):
        stop = min(start + size, len(data["source_states"]))
        yield data["source_states"][start:stop], data[target_key][start:stop]


@torch.no_grad()
def score(predictor, data, model, batch_size: int, target_key: str):
    dense = kl = cosine = 0.0
    count = 0
    for source_cpu, target_cpu in batches(data, batch_size, target_key):
        source = source_cpu.cuda().float()
        target = target_cpu.cuda().float()
        predicted = predictor.state_prediction(source)
        dense += float((predicted - target).square().sum(-1).sum())
        kl += float(forward_kl(final_readout(model, target), final_readout(model, predicted)).sum())
        cosine += float(
            torch.nn.functional.cosine_similarity(predicted - source, target - source, dim=-1).sum()
        )
        count += len(source)
    return {"dense": dense / count, "kl": kl / count, "residual_cosine": cosine / count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--target", choices=("adapted", "baseline"), default="adapted")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    contract = read_json(args.cache_dir / "contract.json")
    if args.condition not in contract["conditions"]:
        raise ValueError(f"Unknown cached condition: {args.condition}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    condition = contract["conditions"][args.condition]
    data = {
        split: load_file(
            str(args.cache_dir / condition["splits"][split]["path"]), device="cpu"
        )
        for split in ("train", "val", "test")
    }
    target_key = "target_final" if args.target == "adapted" else "baseline_final"
    model = AutoModelForCausalLM.from_pretrained(
        contract["model_name"],
        revision=contract.get("model_revision"),
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval().requires_grad_(False)
    train = data["train"]
    mean_residual = (train[target_key].float() - train["source_states"].float()).mean(0).cuda()
    predictor = ResidualLinear(mean_residual).float().cuda()
    batch_size = int(config["train_batch_positions"])
    baseline = score(predictor, train, model, batch_size, target_key)
    scales = {name: max(baseline[name], 1e-8) for name in ("dense", "kl")}
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    best_objective, best_state, best_step, stale = math.inf, None, -1, 0
    history = []
    steps = int(config["training_steps"])
    for step in range(steps + 1):
        if step:
            predictor.train()
            optimizer.zero_grad(set_to_none=True)
            chunk_count = math.ceil(len(train["source_states"]) / batch_size)
            for source_cpu, target_cpu in batches(train, batch_size, target_key):
                source = source_cpu.cuda().float()
                target = target_cpu.cuda().float()
                predicted = predictor.state_prediction(source)
                dense = (predicted - target).square().sum(-1).mean()
                kl = forward_kl(
                    final_readout(model, target), final_readout(model, predicted)
                ).mean()
                loss = (dense / scales["dense"] + kl / scales["kl"]) / chunk_count
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite DENSE+KL loss")
                loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
        if step % int(config["eval_every"]) and step != steps:
            continue
        predictor.eval()
        validation = score(predictor, data["val"], model, batch_size, target_key)
        objective = validation["dense"] / scales["dense"] + validation["kl"] / scales["kl"]
        history.append({"step": step, "val_objective": objective, **validation})
        if objective < best_objective:
            best_objective, best_step = objective, step
            best_state = copy.deepcopy(predictor.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(config["patience_evals"]):
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced")
    predictor.load_state_dict(best_state)
    predictor.eval()
    args.output_dir.mkdir(parents=True)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in predictor.state_dict().items()},
        str(args.output_dir / "predictor.safetensors"),
    )
    summary = {
        "status": "done",
        "condition": args.condition,
        "target": args.target,
        "source_block": contract["source_block"],
        "target_block": contract["target_block"],
        "objective": "normalized_dense_plus_kl",
        "scales": scales,
        "best_step": best_step,
        "history": history,
        "validation": score(predictor, data["val"], model, batch_size, target_key),
        "test": score(predictor, data["test"], model, batch_size, target_key),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
