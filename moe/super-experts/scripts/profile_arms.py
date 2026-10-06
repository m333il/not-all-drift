#!/usr/bin/env python3
"""Experiment 1: do the adaptation arms move the Super Experts?

One frozen backbone, one rendering contract, four kinds of arm, two corpora.
Everything that differs between arms is either the instruction text or the
installed PEFT adapter; the user text is identical across arms within a corpus.

Three quantities are kept apart on purpose, because an arm can move one without
moving the others:

* **membership** -- which experts the published criterion calls Super Experts
  under this arm at all;
* **magnitude** -- the maximum ``|down_proj output|`` at the *fixed* Super
  Experts measured on the unadapted checkpoint, which is what makes arms
  comparable even when membership changes;
* **placement** -- which position produced it, reported against two references
  that only coincide for the arms that add no positions: stream position 0, and
  the position of the chat template's opening token.

Prompt tuning is the reason placement needs two references. It prepends
``num_virtual_tokens`` continuous positions, so stream position 0 is a learned
vector while ``<|im_start|>`` has moved back. Prefix tuning injects into
``past_key_values`` and adds no positions, so for it the two references are the
same position and its expert inputs still differ from the base arm's.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from se_gepa.arms import (
    SEED_KEY, build_base, check_contract, check_instructions, device_of, group_arms, load_contract,
    render, resolve_instructions, virtual_offset, wrap,
)
from se_gepa.criterion import identify, output_max_map
from se_gepa.profiler import FusedExpertProfiler, profile_to_json
from se_gepa.router import RouterScoreProbe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local directory of the staged base model")
    parser.add_argument("--arms", type=Path, required=True, help="JSON list of arm specifications")
    parser.add_argument("--civil-rows", type=Path, required=True)
    parser.add_argument("--contract-sample", type=Path,
                        help="Archived baseline rows (key + input_ids) the rendering must reproduce")
    parser.add_argument("--contract-rows", type=Path,
                        help="Split the archived rows came from; the arms were scored on validation, "
                             "while profiling runs on held-out test rows, so the two differ")
    parser.add_argument("--chat-pins", help="JSON rendering pins for models whose template uses date or reasoning effort")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--wiki-segments", type=int, default=32)
    parser.add_argument("--wiki-seqlen", type=int, default=512)
    parser.add_argument("--track", required=True,
                        help="Fixed Super Experts to follow across arms, as layer:expert pairs")
    parser.add_argument("--corpora", default="civil,wikitext2")
    parser.add_argument("--device", default="cuda",
                        help="Device map. Only cuda is a real measurement; cpu exists so the entry "
                             "point can be smoke-tested on a tiny model.")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def wikitext_segments(tokenizer, count, seqlen):
    import datasets

    rows = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
                                 revision="b08601e04326c79dfdd32d625aee71d232d685c3")
    encoded = tokenizer("\n\n".join(rows["text"]), return_tensors="pt").input_ids[0]
    return [tokenizer.decode(encoded[index * seqlen: (index + 1) * seqlen])
            for index in range(count)]


def summarise(profiler, probe, tracked, offset, example_count):
    """Per-example placement and routing for the tracked Super Experts."""
    rows = []
    for index in range(example_count):
        trace = profiler.traces[index]
        for layer, expert in sorted(tracked):
            values = trace.values.get((layer, expert), {})
            if not values:
                continue
            observation = next((row for row in probe.observations
                                if (row.layer, row.expert) == (layer, expert) and row.example == index), None)
            best = max(values, key=values.get)
            elsewhere = [value for position, value in values.items() if position not in (0, offset)]
            rows.append({
                "example": index, "layer": layer, "expert": expert,
                "argmax_position": best,
                "argmax_at_stream_start": best == 0,
                "argmax_at_chat_start": best == offset,
                "activation_at_max": values[best],
                "activation_at_stream_start": values.get(0),
                "activation_at_chat_start": values.get(offset),
                "activation_max_elsewhere": max(elsewhere) if elsewhere else None,
                "router_probability_stream_start": observation.probability[0] if observation else None,
                "router_probability_chat_start": (
                    observation.probability[offset] if observation and offset < len(observation.probability) else None),
                "router_selected_stream_start": observation.selected[0] if observation else None,
                "positions_routed": len(values),
            })
    return rows


def profile_arm(model, arm, sequences, tracked, out_dir, label):
    offset = virtual_offset(arm)
    started = time.time()
    if arm["kind"] == "peft":
        model.set_adapter(arm["name"])
    with FusedExpertProfiler(model, track=tracked) as profiler, RouterScoreProbe(model, tracked) as probe:
        for index, ids in enumerate(sequences):
            tensor = torch.tensor([ids], device=device_of(model))
            profiler.begin_example(index, ids, virtual_tokens=offset)
            probe.begin_example(index)
            with torch.no_grad():
                model(input_ids=tensor)
            if (index + 1) % 25 == 0:
                print(f"{label}: {index + 1}/{len(sequences)}", flush=True)
    records = profiler.records
    backbone = model.get_base_model() if hasattr(model, "get_base_model") else model
    super_experts = identify(output_max_map(records), total_layers=backbone.config.num_hidden_layers)
    placement = summarise(profiler, probe, tracked, offset, len(sequences))
    (out_dir / f"{label}.profile.json").write_text(json.dumps(profile_to_json(profiler), indent=2) + "\n")
    with (out_dir / f"{label}.placement.jsonl").open("w") as stream:
        for row in placement:
            stream.write(json.dumps(row) + "\n")
    summary = {
        "arm": arm["name"], "kind": arm["kind"], "corpus": label.split("@")[-1],
        "virtual_offset": offset, "examples": len(sequences),
        "elapsed_seconds": round(time.time() - started, 1),
        "super_experts": [asdict(row) for row in super_experts],
        "tracked": {f"{layer}:{expert}": (asdict(records[(layer, expert)]) if (layer, expert) in records else None)
                    for layer, expert in sorted(tracked)},
        "placement_rates": {
            f"{layer}:{expert}": {
                "argmax_at_stream_start": _rate(placement, layer, expert, "argmax_at_stream_start"),
                "argmax_at_chat_start": _rate(placement, layer, expert, "argmax_at_chat_start"),
                "examples": sum(1 for row in placement if (row["layer"], row["expert"]) == (layer, expert)),
            } for layer, expert in sorted(tracked)},
    }
    print(json.dumps({k: summary[k] for k in ("arm", "corpus", "virtual_offset", "placement_rates")}), flush=True)
    print("SE_TRACKED=" + json.dumps({"arm": arm["name"], "corpus": summary["corpus"], "tracked": {
        key: None if row is None else {k: row[k] for k in ("output_max", "position", "example") if k in row}
        for key, row in summary["tracked"].items()},
        "criterion": [[row["layer"], row["expert"], round(row["output_max"], 2)] for row in summary["super_experts"]]}),
        flush=True)
    return summary


def _rate(rows, layer, expert, field):
    values = [row[field] for row in rows if (row["layer"], row["expert"]) == (layer, expert)]
    return sum(values) / len(values) if values else None


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tracked = {tuple(int(part) for part in pair.split(":")) for pair in args.track.split(",")}
    from transformers import AutoTokenizer

    arms = json.loads(args.arms.read_text())
    contract = load_contract(json.loads(args.chat_pins) if args.chat_pins else None)
    _labels, seeds, _render, _apply = contract
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    resolve_instructions(arms, tokenizer, seeds)
    check_instructions(tokenizer, arms, contract)
    model = build_base(args.model, args.device)

    rows = [json.loads(line) for line in args.civil_rows.read_text().splitlines() if line.strip()]
    if args.contract_sample:
        sample = json.loads(args.contract_sample.read_text())
        source = args.contract_rows or args.civil_rows
        contract_rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        checked = check_contract(tokenizer, contract, contract_rows, sample)
        print(f"CONTRACT_TOKENS_VERIFIED={checked}", flush=True)

    corpora = {}
    if "civil" in args.corpora:
        corpora["civil"] = [row["text"] for row in rows[: args.limit]]
    if "wikitext2" in args.corpora:
        corpora["wikitext2"] = wikitext_segments(tokenizer, args.wiki_segments, args.wiki_seqlen)

    sequences = {corpus: {} for corpus in corpora}
    for corpus, texts in corpora.items():
        for instruction in {arm.get("instruction", seeds[SEED_KEY]) for arm in arms}:
            sequences[corpus][instruction] = [render(tokenizer, instruction, text, contract)
                                              for text in texts]

    text_arms, groups = group_arms(arms)
    summaries = []
    for arm in text_arms:
        for corpus in corpora:
            summaries.append(profile_arm(model, arm, sequences[corpus][arm["instruction"]],
                                         tracked, args.out, f"{arm['name']}@{corpus}"))
    for peft_type, group in groups.items():
        wrapped = wrap(model, group)
        print(f"PEFT_GROUP={peft_type} arms={[arm['name'] for arm in group]}", flush=True)
        for arm in group:
            for corpus in corpora:
                summaries.append(profile_arm(wrapped, arm, sequences[corpus][seeds[SEED_KEY]],
                                             tracked, args.out, f"{arm['name']}@{corpus}"))
        del wrapped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"ARMS_PROFILED={len(summaries)}", flush=True)
    print(f"ARTIFACT_DIR={args.out}", flush=True)


if __name__ == "__main__":
    main()
