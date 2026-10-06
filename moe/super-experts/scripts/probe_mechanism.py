#!/usr/bin/env python3
"""Where the massive activation and the attention sink sit, arm by arm.

The Super-Expert profile shows some arms silence the experts. It cannot say
whether the mechanism those experts feed is gone or is being supplied another
way, and under prefix tuning that is exactly the open question: the adapter
injects learned keys and values at every layer, so the sink can live on the
prefix while every expert stays quiet.

Two quantities, neither of which involves an expert:

* the largest absolute value in the residual stream at each position after each
  decoder layer - a massive activation shows up here no matter what produced it;
* where early-layer attention actually goes, split between the injected prefix
  keys and the real positions.

Expensive per example: ``output_attentions`` materialises weights for every layer
at once, which is quadratic in sequence length and cost a CUDA OOM at the GEPA
arm's 2,073-token instruction. The attention half therefore runs on sequences
truncated to ``--attention-tokens``; the sink sits in the first few positions, so
truncating the tail preserves exactly what is being asked. The residual half is
cheap and runs on the full sequence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from se_gepa.arms import (
    SEED_KEY, build_base, device_of, group_arms, load_contract, render,
    resolve_instructions, virtual_offset, wrap,
)
from contextlib import nullcontext

from se_gepa.ablation import ExpertAblation
from se_gepa.residual import ResidualNormProbe, attention_sink_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--attention-layers", default="0,1,2,3")
    parser.add_argument("--attention-tokens", type=int, default=256,
                        help="Truncate sequences for the attention forward only; memory for all "
                             "layers' weights is quadratic in this")
    parser.add_argument("--attention-examples", type=int, default=4)
    parser.add_argument("--chat-pins", help="JSON rendering pins for GPT-OSS")
    parser.add_argument("--only", help="Comma-separated arm names; default is all of them")
    parser.add_argument("--ablate", default="", help="layer:expert pairs zeroed under the ablated condition")
    parser.add_argument("--conditions", default="intact", help="intact and/or ablated")
    parser.add_argument("--report-layers", type=int, default=6,
                        help="How many early layers to report position detail for")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def injected_prefix(arm) -> int:
    """Keys the adapter prepends that are not positions in the stream.

    Prefix tuning writes ``past_key_values`` and adds no positions, so its virtual
    tokens are *only* keys. Prompt tuning adds positions instead, so it injects no
    extra keys - its virtual tokens are already counted as real ones here.
    """
    if arm["kind"] != "peft":
        return 0
    receipt = json.loads((Path(arm["dir"]) / "receipt.json").read_text())
    return receipt["num_virtual_tokens"] if receipt["peft_type"] == "PREFIX_TUNING" else 0


def probe_arm(model, arm, sequences, args):
    offset = virtual_offset(arm)
    prefix = injected_prefix(arm)
    layers = [int(part) for part in args.attention_layers.split(",")]
    rows, attention = [], []
    with ResidualNormProbe(model) as probe:
        for index, ids in enumerate(sequences):
            tensor = torch.tensor([ids], device=device_of(model))
            probe.begin_example(index)
            with torch.no_grad():
                model(input_ids=tensor, use_cache=False)
    for record in probe.records:
        peak = max(range(len(record.per_position)), key=lambda p: record.per_position[p])
        rows.append({"example": record.example, "layer": record.layer,
                     "max": record.per_position[peak], "argmax_position": peak,
                     "at_position_0": record.per_position[0],
                     "at_offset": record.per_position[offset] if offset < len(record.per_position) else None})
    for index, ids in enumerate(sequences[: min(args.attention_examples, len(sequences))]):
        clipped = ids[: args.attention_tokens]
        tensor = torch.tensor([clipped], device=device_of(model))
        for row in attention_sink_summary(model, tensor, layers, prefix, offset):
            attention.append({"example": index, "tokens": len(clipped), **row})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"arm": arm["name"], "virtual_offset": offset, "injected_prefix_keys": prefix,
            "attention_tokens": args.attention_tokens,
            "residual": rows, "attention": attention}


def print_means(result):
    """Attention split and residual peak per layer, averaged over every example, one line each."""
    tag = f"arm={result['arm']} condition={result['condition']}"
    by_layer = {}
    for row in result["attention"]:
        by_layer.setdefault(row["layer"], []).append(row)
    for layer, rows in sorted(by_layer.items()):
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in
                ("from3_virtual", "from3_key0", "from3_key1", "from3_key2", "from3_rest", "from3_learned_sink")}
        print(f"ATTN {tag} layer={layer} n={len(rows)} " + " ".join(f"{k[6:]}={v:.4f}" for k, v in mean.items()),
              flush=True)
    peaks = {}
    for row in result["residual"]:
        peaks.setdefault(row["layer"], []).append(row)
    for layer, rows in sorted(peaks.items())[:8]:
        print(f"PEAK {tag} layer={layer} max={sum(r['max'] for r in rows) / len(rows):.4g} "
              f"at_pos0={sum(r['at_position_0'] for r in rows) / len(rows):.4g} "
              f"argmax_positions={sorted({r['argmax_position'] for r in rows})[:5]}", flush=True)


def summarise(result, report_layers):
    print(f"\n--- {result['arm']}  offset={result['virtual_offset']}"
          f"  injected_prefix_keys={result['injected_prefix_keys']}", flush=True)
    by_layer = {}
    for row in result["residual"]:
        by_layer.setdefault(row["layer"], []).append(row)
    print("    residual stream, max |channel| per layer (means over examples):", flush=True)
    for layer in sorted(by_layer)[:report_layers]:
        group = by_layer[layer]
        peak = sum(row["max"] for row in group) / len(group)
        positions = {row["argmax_position"] for row in group}
        print(f"      layer {layer:>2}: max={peak:.4g} argmax_positions={sorted(positions)[:5]}"
              f" at_pos0={sum(row['at_position_0'] for row in group) / len(group):.4g}", flush=True)
    overall = max(result["residual"], key=lambda row: row["max"])
    print(f"    largest anywhere: {overall['max']:.4g} at layer {overall['layer']}"
          f" position {overall['argmax_position']}", flush=True)
    for row in result["attention"]:
        if row["example"]:
            continue
        print(f"    attention layer {row['layer']}: learned_sink_mass={row['learned_sink_mass_mean']:.4g}"
              f" prefix_mass={row['prefix_mass_mean']:.4g}"
              f" key0={row['real_key_0_mass_mean']:.4g} key1={row['real_key_1_mass_mean']:.4g}"
              f" argmax_key={row['argmax_real_key_mode']} share={row['argmax_real_key_share']:.2f}"
              f" | from3 virtual={row['from3_virtual']:.3f} k0={row['from3_key0']:.3f}"
              f" k1={row['from3_key1']:.3f} k2={row['from3_key2']:.3f} rest={row['from3_rest']:.3f}"
              f" sink={row['from3_learned_sink']:.3f}",
              flush=True)


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arms = json.loads(args.arms.read_text())
    if args.only:
        wanted = set(args.only.split(","))
        arms = [arm for arm in arms if arm["name"] in wanted]
        if len(arms) != len(wanted):
            raise SystemExit(f"Unknown arm names: {sorted(wanted - {arm['name'] for arm in arms})}")
    contract = load_contract(json.loads(args.chat_pins) if args.chat_pins else None)
    _labels, seeds, _render, _apply = contract
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    resolve_instructions(arms, tokenizer, seeds)
    model = build_base(args.model, args.device)

    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()][: args.limit]
    sequences = {instruction: [render(tokenizer, instruction, row["text"], contract) for row in rows]
                 for instruction in {arm.get("instruction", seeds[SEED_KEY]) for arm in arms}}

    results = []
    conditions = args.conditions.split(",")
    ablate = {tuple(int(x) for x in pair.split(":")) for pair in args.ablate.split(",") if pair}
    if "ablated" in conditions and not ablate:
        raise SystemExit("--conditions ablated needs --ablate")

    def run(target, arm, seqs):
        for condition in conditions:
            with ExpertAblation(target, ablate) if condition == "ablated" else nullcontext():
                result = probe_arm(target, arm, seqs, args)
            result["condition"] = condition
            results.append(result)
            summarise(result, args.report_layers)
            print_means(result)

    text_arms, groups = group_arms(arms)
    for arm in text_arms:
        run(model, arm, sequences[arm["instruction"]])
    for peft_type, group in groups.items():
        wrapped = wrap(model, group)
        print(f"PEFT_GROUP={peft_type} arms={[arm['name'] for arm in group]}", flush=True)
        for arm in group:
            wrapped.set_adapter(arm["name"])
            run(wrapped, arm, sequences[seeds[SEED_KEY]])
        del wrapped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (args.out / "mechanism.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"PROBED_ARMS={len(results)}", flush=True)
    print(f"ARTIFACT_DIR={args.out}", flush=True)


if __name__ == "__main__":
    main()
