#!/usr/bin/env python3
"""WikiText-2 perplexity per arm, with and without the Super Experts pruned.

The task score answers "does this arm still need the Super Experts for the job it
was trained on". It cannot answer "did the arm lose general capability", because
every arm is trained to emit one label list and would score zero on anything that
asks for a different shape of answer -- for reasons that have nothing to do with
experts.

Perplexity does not ask the model to answer. It asks for probabilities over text
the model never has to produce, so an adapter trained on one output format is not
punished for that format. It is also the one column of the paper's Table 3 that
needs no generation harness, which makes it directly comparable to their numbers.

Protocol follows the upstream repository exactly: WikiText-2 *test*, joined with
blank lines, tokenized whole, cut into non-overlapping windows of ``--seqlen``
with the tail dropped, negative log-likelihood averaged inside each window and
perplexity taken as ``exp`` of the mean over windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from se_gepa.ablation import ExpertAblation, RouterMask
from se_gepa.arms import build_base, device_of, group_arms, wrap

WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--seqlen", type=int, default=2048, help="Upstream uses 2048")
    parser.add_argument("--limit", type=int, default=0, help="Windows to score; 0 means all")
    parser.add_argument("--only", help="Comma-separated arm names")
    parser.add_argument("--ablate", required=True,
                        help="Super experts to prune, as layer:expert pairs (see configs/super_experts.json)")
    parser.add_argument("--conditions", default="intact,ablated")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def windows(tokenizer, seqlen, limit):
    from datasets import load_dataset

    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
                        revision=WIKITEXT_REVISION)
    encoded = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    count = encoded.numel() // seqlen
    if limit:
        count = min(count, limit)
    return encoded[:, : count * seqlen].view(count, seqlen)


@torch.no_grad()
def perplexity(model, rows, device):
    """Mean per-window NLL, exponentiated.

    ``labels[0]`` is masked on purpose. Without an adapter the shift drops the
    first token's target anyway, but PEFT prepends its own -100 block for prompt
    tuning, so the last virtual position would predict the window's first real
    token and that arm alone would score one extra target. Masking it scores the
    same positions under every arm.
    """
    losses = []
    for index in range(rows.shape[0]):
        ids = rows[index : index + 1].to(device)
        labels = ids.clone()
        labels[:, 0] = -100
        losses.append(model(input_ids=ids, labels=labels).loss.float())
        if (index + 1) % 20 == 0:
            print(f"  {index + 1}/{rows.shape[0]} running ppl "
                  f"{torch.exp(torch.stack(losses).mean()).item():.4f}", flush=True)
    return torch.exp(torch.stack(losses).mean()).item(), len(losses)


class _Nothing:
    def __init__(self, *_args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None


CONDITIONS = {"intact": _Nothing, "ablated": ExpertAblation, "router_masked": RouterMask}
"""``ablated`` zeroes down_proj, which is what the paper does and which also
attenuates the block output by the expert's share of the gate weights.
``router_masked`` keeps the expert out of the top-k, which is what deleting it
from a shipped model does: the freed slot goes to the next expert and the weights
renormalise, so the mixture keeps its magnitude. Perplexity asks for no output
format, so it is the one place the two can be compared without a task in the way."""


def score(model, arm, rows, args, condition, ablate):
    if arm["kind"] == "peft":
        model.set_adapter(arm["name"])
    context = CONDITIONS[condition](model, ablate)
    started = time.time()
    with context:
        value, scored = perplexity(model, rows, device_of(model))
    row = {"arm": arm["name"], "condition": condition, "perplexity": value,
           "windows": scored, "seqlen": args.seqlen,
           "ablated_experts": sorted(ablate) if condition != "intact" else [],
           "elapsed_seconds": round(time.time() - started, 1)}
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ablate = {tuple(int(part) for part in pair.split(":")) for pair in args.ablate.split(",")}
    conditions = args.conditions.split(",")
    unknown = [name for name in conditions if name not in CONDITIONS]
    if unknown:
        raise SystemExit(f"Unknown conditions {unknown}; choose from {sorted(CONDITIONS)}")

    arms = json.loads(args.arms.read_text())
    if args.only:
        wanted = set(args.only.split(","))
        arms = [arm for arm in arms if arm["name"] in wanted]
        if len(arms) != len(wanted):
            raise SystemExit(f"Unknown arm names: {sorted(wanted - {arm['name'] for arm in arms})}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = build_base(args.model, args.device)
    rows = windows(tokenizer, args.seqlen, args.limit)
    print(f"WINDOWS={rows.shape[0]} SEQLEN={args.seqlen}", flush=True)

    results = []

    def record(row):
        results.append(row)
        (args.out / "perplexity.json").write_text(json.dumps(results, indent=2) + "\n")

    text_arms, groups = group_arms(arms)
    for arm in text_arms:
        for condition in conditions:
            record(score(model, arm, rows, args, condition, ablate))
    for peft_type, group in groups.items():
        wrapped = wrap(model, group)
        print(f"PEFT_GROUP={peft_type} arms={[arm['name'] for arm in group]}", flush=True)
        for arm in group:
            for condition in conditions:
                record(score(wrapped, arm, rows, args, condition, ablate))
        del wrapped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"PPL_CELLS={len(results)}", flush=True)
    print(f"ARTIFACT_DIR={args.out}", flush=True)


if __name__ == "__main__":
    main()
