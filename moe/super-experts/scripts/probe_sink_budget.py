#!/usr/bin/env python3
"""Where GPT-OSS puts its attention, with and without its Super Experts.

Zeroing GPT-OSS's two Super Experts, (17,5) and (6,5), leaves its Civil score and
WikiText perplexity almost where they were, while removing Qwen's three collapses
generation. One reading is that every GPT-OSS head already has a place to put
attention that reads nothing -- a learned sink logit in the softmax denominator --
so the model has no use for a positional sink built on a massive activation. This
probe measures what that reading predicts, on the frozen base and the same inputs,
under three conditions: intact, ``down_proj`` zeroed (the paper's intervention)
and the router mask (the experts leave the top-k).

Per layer it records

* the residual stream's max |channel| at stream position 0, and the largest value
  anywhere else, so it is visible whether the massive activation survives;
* where the attention budget goes, averaged over heads and over queries from
  ``--first-query`` on: the learned sink (the returned weights' deficit from one,
  because GPT-OSS drops the sink column before returning them), key 0, key 1,
  key 2 and every later key.

Queries before ``--first-query`` are excluded: position 0 can attend only to itself
and the sink, so its budget says nothing about where the rest of the sequence parks
attention. In sliding-window layers a query more than the window past position 0
cannot see key 0 at all; the layer type is recorded so the two kinds are read apart.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from se_gepa.ablation import ExpertAblation, RouterMask
from se_gepa.residual import ResidualNormProbe

CONDITIONS = ("intact", "zeroed", "router_masked")
GROUPS = ("learned_sink", "key_0", "key_1", "key_2", "rest")
PER_HEAD = ("learned_sink", "key_0")


def parse_experts(text: str) -> set[tuple[int, int]]:
    pairs = set()
    for part in text.split(","):
        layer, expert = part.split(":")
        pairs.add((int(layer), int(expert)))
    if not pairs:
        raise ValueError("No experts named")
    return pairs


def attention_budget(weights: torch.Tensor, first_query: int) -> tuple[dict, dict]:
    """Split one layer's attention budget; ``weights`` is [heads, queries, keys].

    Returns the per-group mean over heads and queries, and the per-head means of the
    learned sink and key 0. The groups sum to one for every query.
    """
    if weights.shape[-1] < 4:
        raise ValueError("Need at least four keys to separate keys 0-2 from the rest")
    weights = weights[:, first_query:, :].float()
    if weights.shape[1] == 0:
        raise ValueError("No queries at or after first_query")
    parts = {
        "learned_sink": (1.0 - weights.sum(dim=-1)).clamp(min=0.0),
        "key_0": weights[..., 0],
        "key_1": weights[..., 1],
        "key_2": weights[..., 2],
        "rest": weights[..., 3:].sum(dim=-1),
    }
    means = {name: float(value.mean()) for name, value in parts.items()}
    heads = {name: parts[name].mean(dim=-1).tolist() for name in PER_HEAD}
    return means, heads


def residual_summary(per_position: list[float]) -> dict:
    rest = per_position[1:]
    peak = max(range(len(rest)), key=rest.__getitem__) + 1 if rest else None
    return {"position_0": per_position[0],
            "position_1": per_position[1] if len(per_position) > 1 else None,
            "position_2": per_position[2] if len(per_position) > 2 else None,
            "max_elsewhere": per_position[peak] if peak is not None else None,
            "argmax_elsewhere": peak}


def intervention(model, condition: str, experts: set[tuple[int, int]]):
    if condition == "intact":
        return nullcontext()
    if condition == "zeroed":
        return ExpertAblation(model, experts)
    if condition == "router_masked":
        return RouterMask(model, experts)
    raise ValueError(f"Unknown condition {condition}")


def logits_digest(model, ids: list[int]) -> tuple[str, torch.Tensor]:
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids], device=model.device), use_cache=False).logits
    logits = logits.detach().float().cpu()
    return hashlib.sha256(logits.numpy().tobytes()).hexdigest(), logits


@torch.no_grad()
def probe(model, sequences: list[list[int]], experts: set[tuple[int, int]], first_query: int) -> dict:
    """Measure every condition on every sequence; returns per-example rows and layer means."""
    layer_types = list(getattr(model.config, "layer_types", None) or [])
    reference_sha, reference = logits_digest(model, sequences[0])
    rows = []
    for condition in CONDITIONS:
        with intervention(model, condition, experts):
            for index, ids in enumerate(sequences):
                with ResidualNormProbe(model) as residual:
                    residual.begin_example(index)
                    outputs = model(input_ids=torch.tensor([ids], device=model.device),
                                    output_attentions=True, use_cache=False)
                norms = {record.layer: record.per_position for record in residual.records}
                if len(outputs.attentions) != len(norms):
                    raise RuntimeError("Attention and residual layer counts differ")
                for layer, weights in enumerate(outputs.attentions):
                    if weights is None:
                        raise RuntimeError("No attention weights; load with attn_implementation='eager'")
                    budget, heads = attention_budget(weights[0], first_query)
                    rows.append({"condition": condition, "example": index, "layer": layer,
                                 "layer_type": layer_types[layer] if layer_types else None,
                                 **{f"mass_{name}": value for name, value in budget.items()},
                                 **{f"head_{name}": value for name, value in heads.items()},
                                 **{f"residual_{name}": value
                                    for name, value in residual_summary(norms[layer]).items()}})
                del outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    after_sha, after = logits_digest(model, sequences[0])
    restoration = {"intact_logits_sha256_before": reference_sha,
                   "intact_logits_sha256_after": after_sha,
                   "max_abs_difference": float((after - reference).abs().max())}
    if restoration["max_abs_difference"] > 1e-2:
        raise RuntimeError("The model did not return to its intact state after the interventions")
    return {"rows": rows, "restoration": restoration, "layer_types": layer_types,
            "summary": summarise(rows, len(layer_types) or 1 + max(row["layer"] for row in rows))}


def summarise(rows: list[dict], layers: int) -> dict:
    """Mean over examples per condition and layer; min/max for the sink and key 0."""
    summary = {}
    for condition in CONDITIONS:
        per_layer = []
        for layer in range(layers):
            group = [row for row in rows if row["condition"] == condition and row["layer"] == layer]
            if not group:
                continue
            entry = {"layer": layer, "layer_type": group[0]["layer_type"], "examples": len(group)}
            for name in GROUPS:
                values = [row[f"mass_{name}"] for row in group]
                entry[name] = sum(values) / len(values)
                if name in PER_HEAD:
                    entry[f"{name}_min"] = min(values)
                    entry[f"{name}_max"] = max(values)
            for name in ("position_0", "max_elsewhere"):
                values = [row[f"residual_{name}"] for row in group if row[f"residual_{name}"] is not None]
                entry[f"residual_{name}"] = sum(values) / len(values) if values else None
            entry["residual_argmax_elsewhere"] = sorted({row["residual_argmax_elsewhere"] for row in group})[:8]
            per_layer.append(entry)
        summary[condition] = per_layer
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--experts", required=True, help="layer:expert pairs, comma-separated")
    parser.add_argument("--attention-tokens", type=int, default=256,
                        help="Truncate each sequence; attention weights for every layer are "
                             "materialised at once and cost memory quadratic in this")
    parser.add_argument("--first-query", type=int, default=3)
    parser.add_argument("--chat-pins", help="JSON rendering pins for GPT-OSS")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    from se_gepa.arms import SEED_KEY, build_base, load_contract, render

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    experts = parse_experts(args.experts)
    contract = load_contract(json.loads(args.chat_pins) if args.chat_pins else None)
    _labels, seeds, _render, _apply = contract
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()][: args.limit]
    if len(rows) != args.limit:
        raise ValueError(f"Asked for {args.limit} rows, found {len(rows)}")
    rendered = [render(tokenizer, seeds[SEED_KEY], row["text"], contract) for row in rows]
    sequences = [ids[: args.attention_tokens] for ids in rendered]
    inputs = [{"id": row.get("id"), "tokens": len(ids), "truncated": len(ids) > args.attention_tokens,
               "first_ids": ids[:8], "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
              for row, ids in zip(rows, rendered)]
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")

    model = build_base(args.model, args.device)
    result = probe(model, sequences, experts, args.first_query)
    config = model.config
    meta = {"model": args.model, "experts": sorted(experts), "conditions": list(CONDITIONS),
            "first_query": args.first_query, "attention_tokens": args.attention_tokens,
            "examples": len(sequences), "rows_file": args.rows.name,
            "sliding_window": getattr(config, "sliding_window", None),
            "layer_types": result["layer_types"], "restoration": result["restoration"],
            "attention_implementation": config._attn_implementation,
            "dtype": str(next(model.parameters()).dtype)}
    (args.out / "rows.json").write_text(json.dumps(result["rows"]) + "\n")
    (args.out / "summary.json").write_text(json.dumps({**meta, "summary": result["summary"]}, indent=2) + "\n")

    for condition, per_layer in result["summary"].items():
        print(f"--- {condition}", flush=True)
        for entry in per_layer:
            kind = (entry["layer_type"] or "?")[:4]
            print(f"  L{entry['layer']:>2} {kind} sink={entry['learned_sink']:.3f} k0={entry['key_0']:.3f}"
                  f" k1={entry['key_1']:.3f} k2={entry['key_2']:.3f} rest={entry['rest']:.3f}"
                  f" r0={entry['residual_position_0']:.4g} rmax={entry['residual_max_elsewhere']:.4g}",
                  flush=True)
    compact = {condition: [[entry["layer"], round(entry["learned_sink"], 4), round(entry["key_0"], 4),
                            round(entry["key_1"], 4), round(entry["key_2"], 4), round(entry["rest"], 4),
                            round(entry["residual_position_0"], 2), round(entry["residual_max_elsewhere"], 2)]
                           for entry in per_layer]
               for condition, per_layer in result["summary"].items()}
    print("SINK_BUDGET_SUMMARY=" + json.dumps(compact, separators=(",", ":")), flush=True)
    print("SINK_BUDGET_RESTORATION=" + json.dumps(result["restoration"]), flush=True)
    print(f"ARTIFACT_DIR={args.out}", flush=True)


if __name__ == "__main__":
    main()
