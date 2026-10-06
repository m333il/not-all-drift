#!/usr/bin/env python3
"""Held-out task quality per arm, with and without the Super Experts pruned.

This is the causal half of the question the profile raised. The profile says the
Super Experts fall silent under some arms; it cannot say whether the model still
needs them. The paper's own evidence for necessity is pruning, so this reproduces
that intervention arm by arm: zero the three experts' down projections and score
free generation on held-out test rows.

Read as a 2x2 per arm. If the base arm collapses when pruned and an adapted arm
does not, the adapted arm no longer depends on the mechanism - which is a
different and stronger statement than "its experts went quiet".

Generation follows the contract these cells were trained and evaluated under:
native chat template, single user turn, greedy, free generation to EOS, eager
attention, ``grouped_mm`` experts. One sequence at a time, because batching a
prompt-tuning arm puts left padding between the virtual tokens and the text and
the position ids stop matching what the adapter was trained on.

``--max-new-tokens`` defaults well below the v25 contract's 1024 on purpose. A
valid answer is a comma-separated label list -- v25 measured 7 completion tokens --
so 64 leaves an order of magnitude of headroom for every answer that parses. What
the cap changes is the cost of an answer that does *not* parse, and pruning Super
Experts is reported to make models repeat themselves: a degenerate arm never emits
EOS and runs to the ceiling on every example, which is 150x the decode steps at
1024. Truncation is therefore reported per cell rather than hidden, and a
truncated answer scores zero whether the ceiling is 64 or 1024.

Per-cell summaries are written as each cell finishes, and rows are flushed as they
are produced, so a job that runs out of wall clock still leaves the cells it did
complete.
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
from se_gepa.arms import (
    SEED_KEY, build_base, check_contract, check_instructions, device_of, group_arms, load_contract,
    render, resolve_instructions, wrap,
)

MAX_NEW_TOKENS = 64
CONTRACT_MAX_NEW_TOKENS = 1024
"""What v25 generated under; kept for the record, not as this measurement's default."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True, help="Held-out test rows")
    parser.add_argument("--contract-sample", type=Path)
    parser.add_argument("--contract-rows", type=Path)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--only", help="Comma-separated arm names to score; default is all of them")
    pruned = parser.add_mutually_exclusive_group(required=True)
    pruned.add_argument("--ablate",
                        help="Super Experts to prune in the ablated condition, as layer:expert pairs "
                             "(see configs/super_experts.json)")
    pruned.add_argument("--ablate-file", type=Path,
                        help="JSON list of [layer, expert] pairs to prune instead (frequency masks)")
    parser.add_argument("--offset", type=int, default=0, help="skip this many rows first (to split a run)")
    parser.add_argument("--loop-window", type=int, default=0,
                        help="stop an answer whose last N tokens are one exact repeating block; 0 = off")
    parser.add_argument("--loop-period", type=int, default=32, help="longest block the loop stop looks for")
    parser.add_argument("--conditions", default="intact,ablated")
    parser.add_argument("--chat-pins", help="JSON of chat-template values this leg must pin, "
                                            "e.g. reasoning_effort and date for GPT-OSS")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


CHANNEL_MARKER = "<|channel|>"


def split_channels(tokenizer, ids):
    """Return ``(answer, analysis)`` for a completion, harmony-aware.

    A reasoning model writes its trace and its answer into separate channels of
    one token stream. Decoding with ``skip_special_tokens`` drops the channel
    headers but keeps the trace's text, so the parser would see reasoning prose
    in front of the labels and score every reasoning answer zero. Ported from the
    project's own ``decode_completion``; the marker test means a model without
    channels takes the plain path unchanged.
    """
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    if CHANNEL_MARKER not in raw:
        return tokenizer.decode(ids, skip_special_tokens=True).strip(), ""
    answer, analysis = "", ""
    for part in ("<|start|>assistant" + raw).split("<|start|>"):
        if part.startswith("assistant<|channel|>final<|message|>"):
            answer = part.split("<|message|>", 1)[1].split("<|return|>", 1)[0]
        elif part.startswith("assistant<|channel|>analysis<|message|>"):
            analysis = part.split("<|message|>", 1)[1].split("<|end|>", 1)[0]
    return answer.strip(), analysis.strip()


def score_one(tokenizer, generated, prompt_length, labels, contract):
    from interpretability_gepa.errors import ParseError
    from interpretability_gepa.metrics import empty_aware_sample_f1_rows, multilabel_matrix
    from interpretability_gepa.prompts import parse_labels

    names = contract[0]
    ids = generated[0, prompt_length:].tolist()
    text, analysis = split_channels(tokenizer, ids)
    try:
        prediction = parse_labels(text, names, enforce_order=False)
        valid = True
    except ParseError:
        prediction, valid = (), False
    score = float(empty_aware_sample_f1_rows(
        multilabel_matrix([labels], names), multilabel_matrix([prediction], names))[0]) if valid else 0.0
    return {"text": text, "analysis": analysis, "prediction": list(prediction),
            "score": score, "valid": valid,
            "finished": bool(ids and ids[-1] == tokenizer.eos_token_id),
            "completion_tokens": len(ids)}


class LoopStop:
    """Stop generation once the last ``window`` tokens are one block repeated exactly.

    A pruned model that has lost its super experts repeats a phrase or whitespace
    until the ceiling; those rows are unreadable whether they stop at the ceiling
    or here, so stopping them changes their cost, not their score. The rule is
    strict (an exact period of at most ``period`` tokens over the whole window), so
    ordinary text and reasoning never trigger it.
    """

    def __init__(self, window, period, prompt_length):
        self.window, self.period, self.prompt_length = window, period, prompt_length
        self.fired = False

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0, self.prompt_length:]
        stop = False
        if generated.numel() >= self.window:
            tail = generated[-self.window:]
            stop = any(torch.equal(tail[p:], tail[:-p]) for p in range(1, self.period + 1))
        self.fired = self.fired or stop
        return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)


@torch.no_grad()
def score_arm(model, tokenizer, arm, sequences, rows, contract, args, condition, ablate, stream):
    if arm["kind"] == "peft":
        model.set_adapter(arm["name"])
    context = CONDITIONS[condition](model, ablate)
    started = time.time()
    scores, valid, finished, truncated, empty_scores, loops = [], [], [], [], [], []
    with context:
        for index, ids in enumerate(sequences):
            tensor = torch.tensor([ids], device=device_of(model))
            extra = {}
            loop = None
            if args.loop_window:
                from transformers import StoppingCriteriaList
                loop = LoopStop(args.loop_window, args.loop_period, len(ids))
                extra["stopping_criteria"] = StoppingCriteriaList([loop])
            output = model.generate(
                input_ids=tensor, attention_mask=torch.ones_like(tensor), do_sample=False,
                max_new_tokens=args.max_new_tokens, use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, **extra,
            )
            row = score_one(tokenizer, output, len(ids), rows[index]["labels"], contract)
            row.update(arm=arm["name"], condition=condition, key=rows[index]["id"],
                       loop_stopped=bool(loop and loop.fired))
            loops.append(row["loop_stopped"])
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            scores.append(row["score"])
            empty_scores.append(row["score"] if row["valid"] else float(not rows[index]["labels"]))
            valid.append(row["valid"])
            finished.append(row["finished"])
            truncated.append(row["completion_tokens"] >= args.max_new_tokens)
            if (index + 1) % 200 == 0:
                print(f"{arm['name']}/{condition}: {index + 1}/{len(sequences)} "
                      f"running score {sum(scores) / len(scores):.4f}", flush=True)
    summary = {
        "arm": arm["name"], "condition": condition, "n": len(scores),
        "score": sum(scores) / len(scores) if scores else None,
        "score_unreadable_as_empty": sum(empty_scores) / len(empty_scores) if empty_scores else None,
        "valid": sum(valid) / len(valid) if valid else None,
        "finished": sum(finished) / len(finished) if finished else None,
        "truncated": sum(truncated) / len(truncated) if truncated else None,
        "max_new_tokens": args.max_new_tokens,
        "loop_stopped": sum(loops) / len(loops) if loops else None,
        "elapsed_seconds": round(time.time() - started, 1),
        "ablated_experts": (sorted(ablate) if len(ablate) <= 16 else f"{len(ablate)} pairs")
                           if condition != "intact" else [],
    }
    print(json.dumps(summary), flush=True)
    return summary


class _Nothing:
    def __init__(self, *_args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None


CONDITIONS = {"intact": _Nothing, "ablated": ExpertAblation, "router_masked": RouterMask}
"""``ablated`` zeroes down_proj, which is the paper's intervention; ``router_masked``
keeps the expert out of the top-k, which is what deleting it from a shipped model
would do. They are different questions and can disagree."""


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ablate = ({tuple(pair) for pair in json.loads(args.ablate_file.read_text())} if args.ablate_file
              else {tuple(int(part) for part in pair.split(":")) for pair in args.ablate.split(",")})
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
    contract = load_contract(json.loads(args.chat_pins) if args.chat_pins else None)
    _labels, seeds, _render, _apply = contract
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    resolve_instructions(arms, tokenizer, seeds)
    check_instructions(tokenizer, arms, contract)
    model = build_base(args.model, args.device)

    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    rows = rows[args.offset: args.offset + args.limit]
    if args.contract_sample:
        source = args.contract_rows or args.rows
        checked = check_contract(
            tokenizer, contract,
            [json.loads(line) for line in source.read_text().splitlines() if line.strip()],
            json.loads(args.contract_sample.read_text()))
        print(f"CONTRACT_TOKENS_VERIFIED={checked}", flush=True)

    sequences = {}
    for instruction in {arm.get("instruction", seeds[SEED_KEY]) for arm in arms}:
        sequences[instruction] = [render(tokenizer, instruction, row["text"], contract) for row in rows]

    summaries = []

    def record(summary):
        summaries.append(summary)
        (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    with (args.out / "rows.jsonl").open("w") as stream:
        text_arms, groups = group_arms(arms)
        for arm in text_arms:
            for condition in conditions:
                record(score_arm(model, tokenizer, arm, sequences[arm["instruction"]],
                                 rows, contract, args, condition, ablate, stream))
        for peft_type, group in groups.items():
            wrapped = wrap(model, group)
            print(f"PEFT_GROUP={peft_type} arms={[arm['name'] for arm in group]}", flush=True)
            for arm in group:
                for condition in conditions:
                    record(score_arm(wrapped, tokenizer, arm, sequences[seeds[SEED_KEY]],
                                     rows, contract, args, condition, ablate, stream))
            del wrapped
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"SCORED_CELLS={len(summaries)}", flush=True)
    print(f"ARTIFACT_DIR={args.out}", flush=True)


if __name__ == "__main__":
    main()
