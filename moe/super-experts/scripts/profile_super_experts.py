#!/usr/bin/env python3
"""Paper-faithful Super-Expert profile of one checkpoint on one corpus.

This is the replication gate: run it on ``Qwen/Qwen3-30B-A3B`` over WikiText-2
and the published criterion must return exactly the three Super Experts reported
in arXiv:2507.23279 Table 2 (layer 1 expert 68, layer 2 expert 92, layer 3
expert 82). Until that matches, a profile of any adapted checkpoint says nothing
-- a disagreement would be indistinguishable from a bug in the port.

The corpus follows upstream: the WikiText-2 *test* split joined with blank lines,
tokenized once, then cut into ``--seqlen`` segments, no chat template, one
segment per forward pass.
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

from se_gepa.criterion import identify, output_max_map
from se_gepa.profiler import FusedExpertProfiler, SharedExpertRecorder, profile_to_json
from se_gepa.router import RouterScoreProbe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2"])
    parser.add_argument("--nsamples", type=int, default=32,
                        help="Number of seqlen-token segments. The statistic is a maximum, "
                             "so this is part of the measurement, not a speed knob.")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-fraction", type=float, default=0.75)
    parser.add_argument("--backend", default="grouped_mm", choices=["grouped_mm", "eager"],
                        help="Experts implementation. grouped_mm is the production backend and the "
                             "one profiling should normally use; eager is the reference loop.")
    parser.add_argument("--device", default="cuda",
                        help="Device map for the model. Only cuda is a real measurement; cpu exists "
                             "so the whole entry point can be smoke-tested on a tiny model.")
    parser.add_argument("--token-samples", type=int, default=4,
                        help="Segments re-run with per-token tracking and gate scores, to test the "
                             "paper's claim that Super Experts fire on the sink token. 0 skips it.")
    return parser.parse_args()


def token_level_pass(model, segments, super_experts, count):
    """Second pass over a few segments: where does each Super Expert actually fire?

    Reports, per segment, the position of that expert's own maximum, its
    activation at the first position against the largest anywhere else, and the
    gate's softmax probability for it at the first position against the mean
    over the remaining positions. The paper argues the sink-token claim from
    router scores; this keeps the activation and the routing side separate
    instead of reading one off the other.
    """
    targets = {(row.layer, row.expert) for row in super_experts}
    if not targets or count <= 0:
        return None
    with FusedExpertProfiler(model, track=targets) as profiler, RouterScoreProbe(model, targets) as probe:
        for index in range(min(count, segments.shape[0])):
            ids = segments[index : index + 1].to(model.device)
            profiler.begin_example(index, ids[0].tolist())
            probe.begin_example(index)
            with torch.no_grad():
                model(input_ids=ids)
    rows = []
    for layer, expert in sorted(targets):
        traces = [trace.values.get((layer, expert), {}) for trace in profiler.traces]
        observations = [row for row in probe.observations if (row.layer, row.expert) == (layer, expert)]
        entry = {"layer": layer, "expert": expert, "segments": []}
        for trace, observation in zip(traces, observations):
            if not trace:
                continue
            best = max(trace, key=trace.get)
            elsewhere = [value for position, value in trace.items() if position != 0]
            other = [value for position, value in enumerate(observation.probability) if position != 0]
            entry["segments"].append({
                "argmax_position": best,
                "activation_at_max": trace[best],
                "activation_at_first_position": trace.get(0),
                "activation_max_elsewhere": max(elsewhere) if elsewhere else None,
                "router_probability_first": observation.probability[0],
                "router_rank_first": observation.rank[0],
                "router_selected_first": observation.selected[0],
                "router_probability_elsewhere_mean": sum(other) / len(other) if other else None,
                "router_selected_rate_elsewhere": (
                    sum(observation.selected[1:]) / len(observation.selected[1:])
                    if len(observation.selected) > 1 else None),
                "positions_routed": len(trace),
            })
        fired_first = [row["argmax_position"] == 0 for row in entry["segments"]]
        entry["argmax_at_first_position_rate"] = (
            sum(fired_first) / len(fired_first) if fired_first else None)
        rows.append(entry)
    return rows


WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1", "b08601e04326c79dfdd32d625aee71d232d685c3")


def corpus(tokenizer, name, nsamples, seqlen):
    import datasets

    if name != "wikitext2":
        raise NotImplementedError(name)
    repo, configuration, revision = WIKITEXT
    rows = datasets.load_dataset(repo, configuration, split="test", revision=revision)
    encoded = tokenizer("\n\n".join(rows["text"]), return_tensors="pt").input_ids
    available = encoded.shape[1] // seqlen
    if nsamples > available:
        raise ValueError(f"{name} yields {available} segments of {seqlen} tokens, {nsamples} requested")
    return encoded[0, : nsamples * seqlen].view(nsamples, seqlen)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        device_map=args.device, experts_implementation=args.backend,
    ).eval()
    segments = corpus(tokenizer, args.dataset, args.nsamples, args.seqlen)

    started = time.time()
    with FusedExpertProfiler(model) as profiler, SharedExpertRecorder(model) as shared:
        for index in range(segments.shape[0]):
            ids = segments[index : index + 1].to(model.device)
            profiler.begin_example(index, ids[0].tolist())
            with torch.no_grad():
                model(input_ids=ids)
            print(f"segment {index + 1}/{segments.shape[0]}", flush=True)

    super_experts = identify(
        output_max_map(profiler.records),
        total_layers=model.config.num_hidden_layers,
        include_fraction=args.include_fraction,
    )
    manifest = {
        "model": args.model,
        "revision": args.revision,
        "resolved_dtype": "bfloat16",
        "experts_implementation": model.config._experts_implementation,
        "dataset": args.dataset,
        "dataset_revision": WIKITEXT[2],
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_experts": (getattr(model.config, "num_experts", None)
                        or model.config.num_local_experts),
        "include_fraction": args.include_fraction,
        "elapsed_seconds": round(time.time() - started, 1),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }
    token_level = token_level_pass(model, segments, super_experts, args.token_samples)
    if token_level is not None:
        (args.out / "token_level.json").write_text(json.dumps(token_level, indent=2) + "\n")
        for row in token_level:
            print(f"SE layer {row['layer']} expert {row['expert']}: maximum at the first position in "
                  f"{row['argmax_at_first_position_rate']:.0%} of the {len(row['segments'])} sampled segments",
                  flush=True)
    (args.out / "profile.json").write_text(json.dumps(profile_to_json(profiler), indent=2) + "\n")
    (args.out / "super_experts.json").write_text(
        json.dumps([asdict(row) for row in super_experts], indent=2) + "\n")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if shared.records:
        (args.out / "shared_experts.json").write_text(json.dumps(
            [{"layer": layer, **record} for layer, record in sorted(shared.records.items())], indent=2) + "\n")
        routed_max = max(row.output_max for row in super_experts) if super_experts else None
        for layer, record in sorted(shared.records.items(), key=lambda item: -item[1]["max"])[:5]:
            print(f"SHARED layer {layer} output_max {record['max']:.4g} at position {record['position']}"
                  f" channel {record['channel']} (largest routed Super Expert: {routed_max})", flush=True)
    for row in super_experts:
        record = profiler.records[(row.layer, row.expert)]
        token = tokenizer.decode([record.token_id]) if record.token_id >= 0 else "<virtual>"
        print(f"SE rank {row.rank}: layer {row.layer} expert {row.expert} "
              f"output_max {row.output_max:.4g} at position {record.position} "
              f"token {record.token_id} ({token!r})", flush=True)
    print(f"SUPER_EXPERTS={len(super_experts)}", flush=True)
    print(f"ARTIFACT_DIR={args.out}", flush=True)


if __name__ == "__main__":
    main()
