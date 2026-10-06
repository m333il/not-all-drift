#!/usr/bin/env python3
"""Train a Linear or GELU-MLP predictor on adapted-native trajectories."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from prompt_optimization.conditions import final_readout
from prompt_optimization.gemma_scope import GemmaScopeJumpReLU
from prompt_optimization.residual_predictor import (
    BiasOnlyResidualPredictor,
    LinearResidualPredictor,
    LowRankResidualPredictor,
    MLPResidualPredictor,
)
from prompt_optimization.trajectory_objectives import (
    OBJECTIVE_COMPONENTS,
    TrainingMode,
    apply_packed_predictor,
    mean_shift_statistics,
    normalized_objective,
    trajectory_objective_parts,
    validate_packed_trajectories,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(cache_dir: Path, contract: dict[str, Any], split: str) -> dict[str, torch.Tensor]:
    data = load_file(str(cache_dir / contract["splits"][split]["path"]), device="cpu")
    required = {"baseline", "adapted", "token_ids", "row_index", "step_index", "offsets"}
    if set(data) != required:
        raise ValueError(f"Unexpected tensors in {split} cache: {sorted(data)}")
    validate_packed_trajectories(data["baseline"], data["offsets"])
    if data["adapted"].shape != data["baseline"].shape:
        raise ValueError(f"Baseline/adapted states do not align in {split}")
    return data


def row_chunks(
    data: dict[str, torch.Tensor], rows_per_chunk: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    offsets = data["offsets"].long()
    row_count = len(offsets) - 1
    for start in range(0, row_count, rows_per_chunk):
        stop = min(start + rows_per_chunk, row_count)
        left, right = int(offsets[start]), int(offsets[stop])
        yield (
            data["baseline"][left:right].cuda(non_blocking=True).float(),
            data["adapted"][left:right].cuda(non_blocking=True).float(),
            (offsets[start : stop + 1] - left).cuda(non_blocking=True),
            right - left,
        )


class MeanShift(torch.nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value.detach().clone())

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value.expand_as(states)


def build_predictor(
    architecture: str,
    mean_shift: torch.Tensor,
    input_mean: torch.Tensor,
    input_scale: torch.Tensor,
    width: int,
    rank: int | None,
) -> torch.nn.Module:
    if architecture == "bias_only":
        return BiasOnlyResidualPredictor(mean_shift)
    if architecture == "linear":
        return LinearResidualPredictor(mean_shift)
    if architecture == "mlp":
        return MLPResidualPredictor(
            mean_shift.numel(), width, input_mean, input_scale, mean_shift
        )
    if architecture == "low_rank":
        if rank is None:
            raise ValueError("low_rank architecture requires --rank")
        return LowRankResidualPredictor(mean_shift, rank)
    raise ValueError(f"Unknown architecture: {architecture}")


@torch.no_grad()
def score(
    predictor: torch.nn.Module,
    data: dict[str, torch.Tensor],
    mode: TrainingMode,
    model: Any,
    sae: GemmaScopeJumpReLU | None,
    rows_per_chunk: int,
) -> dict[str, float]:
    totals = {name: 0.0 for name in ("dense", "enc", "dec", "kl")}
    cosine_sum = 0.0
    relative_norm_error_sum = 0.0
    positions_total = len(data["baseline"])
    for baseline, adapted, offsets, positions in row_chunks(data, rows_per_chunk):
        predicted = apply_packed_predictor(predictor, baseline, offsets, mode)
        parts = trajectory_objective_parts(
            predicted,
            adapted,
            readout=lambda states: final_readout(model, states),
            sae=sae,
        )
        for name, value in parts.items():
            totals[name] += float(value) * positions
        true_delta = adapted - baseline
        predicted_delta = predicted - baseline
        cosine_sum += float(
            torch.nn.functional.cosine_similarity(
                predicted_delta, true_delta, dim=-1
            ).sum()
        )
        relative_norm_error_sum += float(
            (
                (predicted_delta.norm(dim=-1) - true_delta.norm(dim=-1)).abs()
                / true_delta.norm(dim=-1).clamp_min(1e-8)
            ).sum()
        )
    metrics = {
        name: value / positions_total
        for name, value in totals.items()
        if sae is not None or name not in {"enc", "dec"}
    }
    metrics.update(
        {
            "delta_cosine": cosine_sum / positions_total,
            "relative_delta_norm_error": relative_norm_error_sum / positions_total,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("bias_only", "linear", "low_rank", "mlp"), required=True
    )
    parser.add_argument(
        "--mode",
        choices=("prefill_once", "fixed_recurrent", "repredict_recurrent"),
        required=True,
    )
    parser.add_argument(
        "--objective", choices=tuple(OBJECTIVE_COMPONENTS), required=True
    )
    parser.add_argument("--sae-npz", type=Path)
    parser.add_argument("--mlp-width", type=int, default=4608)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if (args.objective == "dense_sae_kl") != (args.sae_npz is not None):
        raise ValueError("dense_sae_kl requires --sae-npz; other objectives do not use it")
    if (args.architecture == "low_rank") != (args.rank is not None):
        raise ValueError("--rank is required exactly for --architecture low_rank")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")

    config = read_json(args.config)
    contract = read_json(args.cache_dir / "contract.json")
    if contract.get("status") != "done":
        raise ValueError("Teacher cache is incomplete")
    data = {
        split: load_split(args.cache_dir, contract, split)
        for split in ("train", "val", "test")
    }
    model = AutoModelForCausalLM.from_pretrained(
        contract["model_name"],
        revision=contract.get("model_revision"),
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval().requires_grad_(False)
    sae = (
        GemmaScopeJumpReLU.from_npz(args.sae_npz, device="cuda")
        .eval()
        .requires_grad_(False)
        if args.sae_npz is not None
        else None
    )

    train = data["train"]
    mean_shift, input_mean, input_scale = mean_shift_statistics(
        train["baseline"], train["adapted"], train["offsets"], args.mode
    )
    mean_shift = mean_shift.cuda()
    mean_predictor = MeanShift(mean_shift)
    rows_per_chunk = int(config["trajectory_chunk_rows"])
    initial = score(
        mean_predictor, train, args.mode, model, sae, rows_per_chunk
    )
    scales = {
        name: max(initial[name], 1e-12)
        for name in OBJECTIVE_COMPONENTS[args.objective]
    }

    torch.manual_seed(int(config.get("seed", 42)))
    predictor = build_predictor(
        args.architecture,
        mean_shift,
        input_mean.cuda(),
        input_scale.cuda(),
        args.mlp_width,
        args.rank,
    ).float().cuda()
    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    best_objective = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_step = -1
    stale = 0
    history: list[dict[str, float | int]] = []
    training_steps = int(config["training_steps"])
    total_positions = len(train["baseline"])

    for step in range(training_steps + 1):
        if step:
            predictor.train()
            optimizer.zero_grad(set_to_none=True)
            for baseline, adapted, offsets, positions in row_chunks(
                train, rows_per_chunk
            ):
                predicted = apply_packed_predictor(
                    predictor, baseline, offsets, args.mode
                )
                parts = trajectory_objective_parts(
                    predicted,
                    adapted,
                    readout=lambda states: final_readout(model, states),
                    sae=sae,
                )
                loss = normalized_objective(
                    parts, scales, args.objective
                ) * (positions / total_positions)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite trajectory objective")
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                predictor.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()

        if step % int(config["eval_every"]) and step != training_steps:
            continue
        predictor.eval()
        validation = score(
            predictor, data["val"], args.mode, model, sae, rows_per_chunk
        )
        objective = sum(
            validation[name] / scales[name]
            for name in OBJECTIVE_COMPONENTS[args.objective]
        )
        row: dict[str, float | int] = {
            "step": step,
            "val_objective": objective,
            **validation,
        }
        history.append(row)
        if objective < best_objective:
            best_objective = objective
            best_step = step
            best_state = copy.deepcopy(predictor.state_dict())
            stale = 0
        else:
            stale += 1
        if (
            step >= int(config.get("minimum_training_steps", 0))
            and stale >= int(config["patience_evals"])
        ):
            break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced")
    predictor.load_state_dict(best_state)
    predictor.eval().requires_grad_(False)
    args.output_dir.mkdir(parents=True)
    checkpoint = args.output_dir / "predictor.safetensors"
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in predictor.state_dict().items()
        },
        str(checkpoint),
    )
    summary = {
        "status": "done",
        "condition_name": contract["condition_name"],
        "block": contract["block"],
        "architecture": args.architecture,
        "mlp_width": args.mlp_width if args.architecture == "mlp" else None,
        "rank": args.rank if args.architecture == "low_rank" else None,
        "mode": args.mode,
        "objective": args.objective,
        "objective_components": list(OBJECTIVE_COMPONENTS[args.objective]),
        "scales": scales,
        "best_step": best_step,
        "best_validation_objective": best_objective,
        "history": history,
        "validation": score(
            predictor, data["val"], args.mode, model, sae, rows_per_chunk
        ),
        "test_teacher_forced": score(
            predictor, data["test"], args.mode, model, sae, rows_per_chunk
        ),
        "parameter_count": sum(parameter.numel() for parameter in predictor.parameters()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
