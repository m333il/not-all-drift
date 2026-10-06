#!/usr/bin/env python3
"""Calibrate and test constant replacement of a prefix's direct value contribution."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from se_gepa.arms import SEED_KEY, build_base, check_contract, device_of, load_contract, render, wrap
from se_gepa.prefix_intervention import PrefixValueIntervention
from probe_attention_contributions import arm_metadata, sequence_hash
from score_arms import score_one


def select_rows(rows, calibration_limit, eval_offset, limit):
    if calibration_limit < 1 or limit < 1 or eval_offset < calibration_limit:
        raise ValueError("Calibration and evaluation must be positive, disjoint row ranges")
    if eval_offset + limit > len(rows):
        raise ValueError("Not enough rows for the requested split")
    calibration, evaluation = rows[:calibration_limit], rows[eval_offset:eval_offset + limit]
    for field in ("id", "text"):
        if {r[field] for r in calibration} & {r[field] for r in evaluation}:
            raise ValueError(f"Calibration/evaluation overlap in {field}")
    if len({r['id'] for r in evaluation}) != len(evaluation):
        raise ValueError("Duplicate evaluation IDs")
    return calibration, evaluation


def paired_summary(rows, reference, seed):
    delta = torch.tensor([r["score"] - reference[r["key"]]["row"]["score"] for r in rows], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(rows), (2000, len(rows)), generator=generator)
    interval = torch.quantile(delta[indices].mean(dim=1), torch.tensor([.025, .975], dtype=torch.float64))
    return {"n": len(rows), "score": sum(r["score"] for r in rows) / len(rows),
            "valid": sum(r["valid"] for r in rows) / len(rows),
            "finished": sum(r["finished"] for r in rows) / len(rows),
            "truncated": sum(r["truncated"] for r in rows) / len(rows),
            "mean_completion_tokens": sum(r["completion_tokens"] for r in rows) / len(rows),
            "paired_score_delta": float(delta.mean()), "paired_bootstrap_ci95": interval.tolist(),
            "improved": int((delta > 0).sum()), "worsened": int((delta < 0).sum()),
            "first_token_kl_from_intact": sum(r["first_token_kl_from_intact"] for r in rows) / len(rows)}


@torch.no_grad()
def calibrate(model, sequences, layers, scope):
    samples = {layer: [] for layer in layers}
    from contextlib import ExitStack
    for index, ids in enumerate(sequences):
        tensor = torch.tensor([ids], device=device_of(model))
        native = model(input_ids=tensor, use_cache=False, logits_to_keep=1).logits if index == 0 else None
        with ExitStack() as stack:
            hooks = {layer: stack.enter_context(PrefixValueIntervention(model, layer, scope=scope)) for layer in layers}
            observed = model(input_ids=tensor, use_cache=False, logits_to_keep=1).logits
            if native is not None and not torch.equal(native, observed):
                raise RuntimeError("Calibration hooks changed native logits")
            for layer, hook in hooks.items():
                samples[layer].append(hook.last_contribution.float().mean(dim=0).cpu())
        del observed, native
    vectors, stats = {}, {}
    for layer, values in samples.items():
        matrix = torch.stack(values)
        mean = matrix.mean(dim=0)
        energy = matrix.square().sum(dim=1).mean()
        vectors[layer] = mean
        stats[layer] = {"examples": len(values), "vector": mean.tolist(), "norm": float(mean.norm()),
                        "example_mean_rms": float(energy.sqrt()),
                        "constant_fraction_of_example_mean_energy": float(mean.square().sum() / energy) if energy > 0 else None}
    return vectors, stats


@torch.no_grad()
def evaluate(model, tokenizer, arm, sequences, rows, contract, vectors, args, stream):
    reference, summaries = {}, []
    conditions = [("intact", None)] + [(name, layer) for layer in args.layers
                                      for name in ("zero", "restore", "constant", "random")]
    random_vectors = {}
    for layer, vector in vectors.items():
        random = torch.randn(vector.shape, generator=torch.Generator().manual_seed(args.seed + layer))
        random_vectors[layer] = random / random.norm() * vector.norm()
    for condition, layer in conditions:
        started = time.time()
        results = []
        for source, ids in zip(rows, sequences):
            tensor = torch.tensor([ids], device=device_of(model))
            context = (nullcontext(None) if condition == "intact" else PrefixValueIntervention(
                model, layer, mode="constant" if condition == "random" else condition,
                vector=random_vectors[layer] if condition == "random" else vectors[layer], scope=args.scope))
            with context as hook:
                generated = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                    max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=True)
            row = score_one(tokenizer, generated.sequences, len(ids), source["labels"], contract)
            row["truncated"] = not row["finished"] and row["completion_tokens"] >= args.max_new_tokens
            row["parsed_score"] = row["score"]
            if row["truncated"]:
                row["score"] = 0.0
            first = generated.scores[0][0].float().cpu()
            digest = hashlib.sha256()
            for logits in generated.scores:
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite generation scores")
                digest.update(logits.float().cpu().numpy().tobytes())
            ids_out = generated.sequences[0, len(ids):].tolist()
            if hook is not None and hook.calls != len(ids_out):
                raise RuntimeError("Intervention did not run on every generation step")
            if condition == "intact":
                reference[source["id"]] = {"row": dict(row), "first": first, "tokens": ids_out,
                                             "scores_sha256": digest.hexdigest()}
            ref = reference[source["id"]]
            if condition == "restore" and (ids_out != ref["tokens"] or digest.hexdigest() != ref["scores_sha256"]):
                raise RuntimeError(f"Exact-donor restoration changed generation: {source['id']}")
            log_p, log_q = ref["first"].log_softmax(dim=-1), first.log_softmax(dim=-1)
            kl = float((log_p.exp() * (log_p - log_q)).sum())
            row.update(arm=arm["name"], condition=condition, layer=layer, key=source["id"],
                       sequence_sha256=sequence_hash(ids), first_token_kl_from_intact=kl,
                       generated_ids=ids_out, generation_scores_sha256=digest.hexdigest(),
                       intervention_calls=hook.calls if hook is not None else 0,
                       decode_calls=hook.decode_calls if hook is not None else 0)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            results.append(row)
            del generated, first
        summary = {"condition": condition, "layer": layer, **paired_summary(results, reference, args.seed),
                   "elapsed_seconds": time.time() - started}
        summaries.append(summary)
        (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    return summaries, {str(layer): vector.tolist() for layer, vector in random_vectors.items()}


def main():
    import peft
    import transformers
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--arms-spec", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--contract-sample", type=Path, required=True)
    parser.add_argument("--contract-rows", type=Path, required=True)
    parser.add_argument("--calibration-limit", type=int, default=32)
    parser.add_argument("--eval-offset", type=int, default=32)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--layers", default="3,5")
    parser.add_argument("--scope", choices=["last", "all"], default="last")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.layers = [int(x) for x in args.layers.split(",")]
    if not args.layers or len(set(args.layers)) != len(args.layers) or args.max_new_tokens < 1:
        raise ValueError("Require distinct layers and positive generation cap")
    if transformers.__version__ != "5.16.1" or peft.__version__ != "0.20.0":
        raise RuntimeError("Runtime must match transformers 5.16.1 and peft 0.20.0")
    arms = json.loads(args.arms_spec.read_text())
    if len(arms) != 1 or arms[0]["kind"] != "peft":
        raise ValueError("Run exactly one prefix adapter per job")
    arm = arms[0]
    metadata = arm_metadata(arm)
    receipt = metadata["receipt"]
    if receipt["peft_type"] != "PREFIX_TUNING" or receipt["base_revision"] != args.model_revision:
        raise ValueError("Expected a prefix adapter on the pinned base revision")
    source_rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    calibration, evaluation = select_rows(source_rows, args.calibration_limit, args.eval_offset, args.limit)
    contract = load_contract()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    checked = check_contract(tokenizer, contract,
        [json.loads(line) for line in args.contract_rows.read_text().splitlines() if line.strip()],
        json.loads(args.contract_sample.read_text()))
    sequences = {name: [render(tokenizer, contract[1][SEED_KEY], r["text"], contract) for r in rows]
                 for name, rows in [("calibration", calibration), ("evaluation", evaluation)]}
    if {sequence_hash(s) for s in sequences["calibration"]} & {sequence_hash(s) for s in sequences["evaluation"]}:
        raise ValueError("Identical rendered input in calibration and evaluation")
    args.out.mkdir(parents=True, exist_ok=False)
    model = wrap(build_base(args.model_dir, args.device), arms)
    if any(layer < 0 or layer >= model.config.num_hidden_layers for layer in args.layers):
        raise ValueError("Layer is outside the model")
    vectors, calibration_stats = calibrate(model, sequences["calibration"], args.layers, args.scope)
    (args.out / "calibration.json").write_text(json.dumps(calibration_stats, indent=2) + "\n")
    with (args.out / "rows.jsonl").open("w") as stream:
        summaries, random_vectors = evaluate(model, tokenizer, arm, sequences["evaluation"], evaluation,
                                              contract, vectors, args, stream)
    (args.out / "random-vectors.json").write_text(json.dumps(random_vectors) + "\n")
    expected = args.limit * (1 + 4 * len(args.layers))
    actual = len((args.out / "rows.jsonl").read_text().splitlines())
    if actual != expected:
        raise RuntimeError(f"Expected {expected} score rows, found {actual}")
    manifest = {"status": "PASS", "probe": "prefix_value_causal_v1", "arm": metadata,
                "model_revision": args.model_revision, "layers": args.layers, "scope": args.scope,
                "scope_semantics": "last query of prefill and every decode step" if args.scope == "last" else "all real prefill and decode queries",
                "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                "calibration_limit": args.calibration_limit, "eval_offset": args.eval_offset,
                "evaluation_limit": args.limit, "contract_examples_verified": checked,
                "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
                "contract_sample_sha256": hashlib.sha256(args.contract_sample.read_bytes()).hexdigest(),
                "sequence_hashes": {name: [{"key": r["id"], "sha256": sequence_hash(ids), "tokens": len(ids)}
                                          for r, ids in zip(rows, sequences[name])]
                                    for name, rows in [("calibration", calibration), ("evaluation", evaluation)]},
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "transformers": transformers.__version__, "peft": peft.__version__},
                "metric_rows": actual, "cells": len(summaries), "calibration_noop_exact": True,
                "exact_donor_generation_parity": True,
                "evaluation_role": "disjoint diagnostic validation subset; not independent final test",
                "files_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(args.out.iterdir()) if f.is_file()}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"PREFIX_CAUSAL_ROWS={actual}", flush=True)
    print(f"ARTIFACT_DIR={args.out}", flush=True)


if __name__ == "__main__":
    main()
