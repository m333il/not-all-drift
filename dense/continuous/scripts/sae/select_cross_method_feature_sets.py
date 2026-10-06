#!/usr/bin/env python3
"""Select calibration-only SAE feature sets for causal method replacement.

For each method pair, ``direct_topK`` ranks features by
``abs(mean(z_right - z_left)) * ||W_dec[j]||_2``.  Method-vs-manual rankings
also produce sign-aligned shared and method-unique controls.  The calibration
IDs stored in the output manifest must be excluded from causal evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prompt_optimization.civil_comments import sha256_file
from prompt_optimization.feature_subset_replacement import build_pair_feature_sets
from prompt_optimization.gemma_scope import GemmaScopeJumpReLU


def parse_state(value: str) -> tuple[str, Path, str]:
    name, separator, location = value.partition("=")
    path, key_separator, key = location.partition("::")
    if not separator or not key_separator or not name or not path or not key:
        raise argparse.ArgumentTypeError("State must be METHOD=/path/states.safetensors::KEY")
    return name, Path(path), key


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_state(path: Path, key: str) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise KeyError(f"Missing state {key!r} in {path}")
        values = handle.get_tensor(key)
    if values.ndim != 2 or not len(values) or not torch.isfinite(values).all():
        raise ValueError(f"Expected finite non-empty [samples, hidden] state: {path}::{key}")
    return values.float()


@torch.inference_mode()
def encode(sae: GemmaScopeJumpReLU, states: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [
            sae.encode(states[start : start + chunk_size].to(sae.W_enc.device))
            .float()
            .cpu()
            for start in range(0, len(states), chunk_size)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", type=parse_state, required=True)
    parser.add_argument("--calibration-ids", type=Path, required=True)
    parser.add_argument("--sae-npz", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--carrier", default="generation_anchor")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--top-k", nargs="+", type=int, default=(8, 16, 32, 64, 128))
    parser.add_argument("--shared-k", nargs="+", type=int, default=(32, 64, 128))
    parser.add_argument("--sign-epsilon", type=float, default=1e-8)
    parser.add_argument("--random-seed", type=int, default=831_042)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    specs = {name: (path, key) for name, path, key in args.state}
    if len(specs) != len(args.state) or "manual" not in specs or len(specs) < 3:
        raise ValueError("Provide unique manual and at least two adapted states")
    top_ks = tuple(sorted(set(args.top_k)))
    shared_ks = tuple(sorted(set(args.shared_k)))
    if not top_ks or min(top_ks) <= 0 or not set(shared_ks).issubset(top_ks):
        raise ValueError("shared-k must be a positive subset of top-k")
    calibration_ids = list(map(str, read_json(args.calibration_ids)))
    if not calibration_ids or len(calibration_ids) != len(set(calibration_ids)):
        raise ValueError("Calibration IDs must be a non-empty unique JSON list")
    states = {name: load_state(*spec) for name, spec in specs.items()}
    shapes = {tuple(values.shape) for values in states.values()}
    if len(shapes) != 1 or len(next(iter(states.values()))) != len(calibration_ids):
        raise ValueError("State shapes and calibration ID count must agree")
    sae = GemmaScopeJumpReLU.from_npz(args.sae_npz, device="cuda:0", dtype=torch.float32)
    features = {name: encode(sae, values, args.chunk_size) for name, values in states.items()}
    manual = features.pop("manual")
    method_shifts = {name: (values - manual).mean(0) for name, values in features.items()}
    decoder_norm = sae.W_dec.float().norm(dim=1).cpu()
    feature_sets = build_pair_feature_sets(
        method_shifts,
        decoder_norm,
        top_ks=top_ks,
        sign_epsilon=args.sign_epsilon,
        random_seed=args.random_seed,
        random_k=64,
    )
    args.output_dir.mkdir(parents=True)
    tensors: dict[str, torch.Tensor] = {"decoder_norm": decoder_norm}
    for method, shift in method_shifts.items():
        tensors[f"{method}_mean_delta"] = shift.float()
        tensors[f"{method}_importance"] = shift.abs().float() * decoder_norm
    save_file(tensors, str(args.output_dir / "feature_signatures.safetensors"))
    manifest = {
        "status": "done",
        "scope": "cross_method_calibration_feature_selection",
        "selection_split_role": "calibration only; exclude these IDs from causal evaluation",
        "task": args.task,
        "optimizer_seed": args.seed,
        "carrier": args.carrier,
        "layer": args.layer,
        "methods": sorted(method_shifts),
        "top_k": list(top_ks),
        "shared_k": list(shared_ks),
        "sign_epsilon": args.sign_epsilon,
        "importance": "abs(mean SAE activation shift versus Manual) * decoder-row L2 norm",
        "direct_importance": "abs(mean pairwise SAE activation difference) * decoder-row L2 norm",
        "calibration_ids": calibration_ids,
        "calibration_ids_sha256": sha256_file(args.calibration_ids),
        "feature_sets": feature_sets,
        "states": {
            name: {"path": str(path.resolve()), "key": key, "sha256": sha256_file(path)}
            for name, (path, key) in specs.items()
        },
        "sae_path": str(args.sae_npz.resolve()),
        "sae_sha256": sha256_file(args.sae_npz),
        "sae_width": sae.d_sae,
    }
    (args.output_dir / "feature_sets.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
