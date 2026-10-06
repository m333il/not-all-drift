#!/usr/bin/env python3
"""Collect expert usage frequencies for one arm on a calibration split.

Use this instead of the published ``expert_counts.npz`` when the pruning
decision should be made on the same sample and settings as the evaluation. The
split must not overlap the test set: choosing which experts to delete by
looking at the test data is selection on the outcome.

    uv run scripts/calibrate_frequencies.py \\
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --arm prompt_tuning:checkpoints/epoch_002 \\
        --data data/civil_multilabel_calib.jsonl --n-examples 256 \\
        --out counts/prompt_tuning_calib.npz
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from mrd_pruning.arms import ArmSpec, load_arm, provenance  # noqa: E402
from mrd_pruning.calibrate import calibrate  # noqa: E402
from mrd_pruning.task import SEED_SYSTEM_PROMPT, render_chat, render_user_prompt  # noqa: E402

logger = logging.getLogger("calibrate")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--arm", required=True, help="NAME or NAME:ADAPTER_PATH")
    p.add_argument("--gepa-prompt", type=Path, default=None)
    p.add_argument("--system-policy", default="as_trained",
                   choices=["as_trained", "seed_all", "none_all"])
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--n-examples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--allow-unpinned-checkpoint", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args(argv)


def build_arm(token: str, args: argparse.Namespace) -> ArmSpec:
    name, _, adapter = token.partition(":")
    if name == "base":
        return ArmSpec(name="base", kind="base", system_prompt_text=SEED_SYSTEM_PROMPT)
    if name == "gepa":
        if args.gepa_prompt is None:
            raise SystemExit("--arm gepa needs --gepa-prompt")
        return ArmSpec(name="gepa", kind="gepa",
                       system_prompt_text=args.gepa_prompt.read_text().strip())
    if name in ("prompt_tuning", "prefix_tuning"):
        if not adapter:
            raise SystemExit(f"--arm {name} needs an adapter path")
        return ArmSpec(name=name, kind=name, adapter_path=Path(adapter),
                       system_prompt_text=None,
                       allow_unpinned_checkpoint=args.allow_unpinned_checkpoint)
    raise SystemExit(f"unknown arm {name!r}")


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = parse_args(argv)
    arm = build_arm(args.arm, args)
    arm.validate()

    comments = []
    with args.data.open() as handle:
        for line in handle:
            if len(comments) >= args.n_examples:
                break
            comments.append(json.loads(line)["comment"])
    if len(comments) < args.n_examples:
        raise SystemExit(f"{args.data} holds {len(comments)} rows, {args.n_examples} requested")

    model, tokenizer = load_arm(arm, model_id=args.model, revision=args.revision, dtype=args.dtype)
    system_prompt = arm.system_prompt(args.system_policy)
    rendered = [
        render_chat(tokenizer, render_user_prompt(c), system_prompt, tokenize=False)
        for c in comments
    ]

    batches = []
    for start in range(0, len(rendered), args.batch_size):
        encoded = tokenizer(rendered[start:start + args.batch_size], return_tensors="pt",
                            padding=True, add_special_tokens=False)
        batches.append({k: v.to(model.device) for k, v in encoded.items()})

    counts = calibrate(model, batches, top_k=args.top_k,
                       source=f"calibrate:{arm.name}:{args.data.name}",
                       n_examples=len(comments), stage="prompt")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "layer_ids": list(counts.layer_ids),
        "num_experts": counts.n_experts,
        "top_k": args.top_k,
        "n_examples": counts.n_examples,
        "stage": counts.stage,
        "arm_provenance": provenance(arm),
        "system_policy": args.system_policy,
        "entries": {f"{arm.name}|prompt": {"total_assignments": float(counts.counts.sum())}},
    }
    np.savez(args.out, _meta=json.dumps(meta, ensure_ascii=False),
             **{f"{arm.name}|prompt": counts.counts})
    logger.info("wrote %s (%d layers x %d experts)", args.out, counts.n_layers, counts.n_experts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
