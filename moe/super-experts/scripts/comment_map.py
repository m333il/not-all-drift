#!/usr/bin/env python3
"""Expert routing of the text arms on the 500 calibration comments, rendered as trained.

``measure_routing_map.py`` feeds a GEPA arm its ``optimized_prompt.txt``, the
instruction already wrapped in the output contract with a literal ``{text}``. This
script measures the text arms with the rendering every score in this repository
uses: the v25 user-only contract and the instruction component alone. The GEPA maps
of the paper come from here. The base arm is the control: its comment-token counts
must match the base map of ``measure_routing_map.py``, which differs only in the
harness.

Counts are top-k selections per layer and expert, split into the tokens of the
classified comment, the other prompt tokens, and the generated tokens (greedy, every
decode step), as ``int64`` arrays of shape ``[layers, experts]``. The comment tokens
precede the answer, so their routing does not depend on the answer at all.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from se_gepa.arms import (  # noqa: E402
    SEED_KEY, build_base, check_contract, check_instructions, device_of, load_contract, render,
    resolve_instructions,
)
from train_router import comment_span  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--only", required=True, help="comma-separated text arms")
    parser.add_argument("--comments", type=Path, required=True)
    parser.add_argument("--contract-sample", type=Path, required=True)
    parser.add_argument("--contract-rows", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--chat-pins")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def routers(model):
    found = [module for name, module in model.named_modules()
             if name.endswith(("mlp.gate", "mlp.router")) and isinstance(getattr(module, "weight", None),
                                                                           torch.nn.Parameter)]
    if not found:
        raise RuntimeError("no router modules found")
    return found


def main():
    from transformers import AutoTokenizer

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    contract = load_contract(json.loads(args.chat_pins) if args.chat_pins else None)
    _labels, seeds, _render, _apply = contract
    wanted = args.only.split(",")
    arms = [arm for arm in json.loads(args.arms.read_text()) if arm["name"] in wanted]
    if len(arms) != len(wanted) or any(arm["kind"] != "text" for arm in arms):
        raise SystemExit("--only must name text arms present in --arms")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    resolve_instructions(arms, tokenizer, seeds)
    check_instructions(tokenizer, arms, contract)
    rows_for_contract = [json.loads(line) for line in args.contract_rows.read_text().splitlines() if line.strip()]
    print(f"CONTRACT_TOKENS_VERIFIED={check_contract(tokenizer, contract, rows_for_contract, json.loads(args.contract_sample.read_text()))}",
          flush=True)
    comments = [json.loads(line) for line in args.comments.read_text().splitlines() if line.strip()][: args.limit]
    model = build_base(args.model)
    gates = routers(model)
    n_layers, n_experts = len(gates), gates[0].weight.shape[0]

    state = {}

    def make_hook(layer):
        def hook(_module, _inputs, output):
            indices = output[2].reshape(-1, output[2].shape[-1])
            calls = state["calls"]
            calls[layer] += 1
            counts = state["counts"]
            if calls[layer] == 1:
                lo, hi = state["span"]
                prompt = indices[-state["length"]:]
                flat_comment = prompt[lo:hi].flatten()
                other = torch.cat([prompt[:lo], prompt[hi:]]).flatten()
                counts["comment"][layer].index_add_(0, flat_comment, torch.ones_like(flat_comment))
                counts["prompt_other"][layer].index_add_(0, other, torch.ones_like(other))
            else:
                flat = indices.flatten()
                counts["generated"][layer].index_add_(0, flat, torch.ones_like(flat))
        return hook

    hooks = [gate.register_forward_hook(make_hook(layer)) for layer, gate in enumerate(gates)]
    summary = {}
    try:
        for arm in arms:
            instruction = arm["instruction"]
            device = device_of(model)
            state["counts"] = {stage: torch.zeros(n_layers, n_experts, dtype=torch.long, device=device)
                               for stage in ("comment", "prompt_other", "generated")}
            lengths, generated, skipped = [], [], 0
            started = time.time()
            for row in comments:
                ids = render(tokenizer, instruction, row["comment"], contract)
                span = comment_span(tokenizer, instruction, row["comment"], contract, ids)
                if span is None:
                    skipped += 1
                    continue
                state.update(span=span, length=len(ids), calls=[0] * n_layers)
                tensor = torch.tensor([ids], device=device)
                with torch.no_grad():
                    output = model.generate(
                        input_ids=tensor, attention_mask=torch.ones_like(tensor), do_sample=False,
                        max_new_tokens=args.max_new_tokens, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id)
                lengths.append(len(ids))
                generated.append(output.shape[1] - len(ids))
            arrays = {stage: tensor.cpu().numpy() for stage, tensor in state["counts"].items()}
            np.savez_compressed(args.out / f"{arm['name']}.npz", **arrays)
            summary[arm["name"]] = {
                "rows": len(lengths), "skipped": skipped, "prompt_tokens_mean": float(np.mean(lengths)),
                "generated_tokens_mean": float(np.mean(generated)),
                "comment_tokens_mean": float(arrays["comment"][0].sum() / gates[0].top_k / max(1, len(lengths)))
                if hasattr(gates[0], "top_k") else None,
                "seconds": round(time.time() - started, 1)}
            print("COMMENT_MAP_SUMMARY " + json.dumps({arm["name"]: summary[arm["name"]]}), flush=True)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            blob = base64.b64encode(gzip.compress(buffer.getvalue())).decode()
            for start in range(0, len(blob), 1000):
                print(f"COMMENT_MAP_B64 {arm['name']} {start // 1000} {blob[start:start + 1000]}", flush=True)
            print(f"COMMENT_MAP_B64_END {arm['name']} {(len(blob) + 999) // 1000}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
