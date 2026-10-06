#!/usr/bin/env python3
"""Measure Civil quality under the early-layer WikiText interventions."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from se_gepa.arms import SEED_KEY, build_base, check_contract, device_of, load_contract, render, wrap
from se_gepa.prefix_intervention import PrefixKeyMask, PrefixValueIntervention, localization_designs
from probe_attention_contributions import arm_metadata, sequence_hash
from probe_prefix_causal import paired_summary
from score_arms import score_one


def check_mask(layer, records, intervention=None):
    def hook(_module, _args, output):
        weights = output[1]
        selected = intervention.selected_queries if intervention is not None else slice(None)
        if not torch.isfinite(weights).all() or torch.count_nonzero(weights[..., selected, :records[layer]["prefix_tokens"]]):
            raise RuntimeError("Masked prefix has nonzero or non-finite attention")
        error = float((weights.float().sum(-1) - 1).abs().max())
        if error > .02:
            raise RuntimeError("Masked attention is not normalized")
        records[layer]["calls"] += 1
        records[layer]["normalization_max_error"] = max(records[layer]["normalization_max_error"], error)
    return hook


@torch.no_grad()
def generate(model, tokenizer, ids, labels, contract, condition, layers, max_new_tokens, mask_scope="all"):
    tensor = torch.tensor([ids], device=device_of(model))
    observed = {}
    with ExitStack() as stack:
        hooks = {}
        for layer in layers if condition != "intact" else []:
            if condition == "mask_keys":
                hook = stack.enter_context(PrefixKeyMask(model, layer, scope=mask_scope))
                observed[layer] = {"prefix_tokens": hook.prefix_tokens, "calls": 0, "normalization_max_error": 0.0}
                handle = hook.attention.register_forward_hook(check_mask(layer, observed, hook))
                stack.callback(handle.remove)
            else:
                hook = stack.enter_context(PrefixValueIntervention(
                    model, layer, mode="restore" if condition == "restore" else "zero", scope="all"))
            hooks[layer] = hook
        generated = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=True)
    row = score_one(tokenizer, generated.sequences, len(ids), labels, contract)
    row["parsed_score"] = row["score"]
    row["truncated"] = not row["finished"] and row["completion_tokens"] >= max_new_tokens
    if row["truncated"]:
        row["score"] = 0.0
    if any(h.calls != row["completion_tokens"] for h in hooks.values()):
        raise RuntimeError("Intervention did not execute on every generation step")
    if any(r["calls"] != row["completion_tokens"] for r in observed.values()):
        raise RuntimeError("Missing masked-attention checks")
    for layer, record in observed.items():
        total_queries = len(ids) + row["completion_tokens"] - 1
        expected = min(3, total_queries) if mask_scope == "early3" else max(0, total_queries - 3) if mask_scope == "after3" else total_queries
        if hooks[layer].masked_queries != expected:
            raise RuntimeError("Mask did not cover the expected real query positions")
        record.update(scope=mask_scope, masked_queries=hooks[layer].masked_queries)
    digest = hashlib.sha256()
    for logits in generated.scores:
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite generation scores")
        digest.update(logits.float().cpu().numpy().tobytes())
    row.update(generated_ids=generated.sequences[0, len(ids):].tolist(), generation_scores_sha256=digest.hexdigest(),
               intervention_calls={str(k): h.calls for k, h in hooks.items()},
               mask_checks={str(k): v for k, v in observed.items()})
    return row, generated.scores[0][0].float().cpu()


def evaluate(model, tokenizer, arm, sequences, rows, contract, layers, max_new_tokens, out, designs=None):
    conditions = ["intact", "zero_values", "mask_keys"] if arm["kind"] == "peft" else ["intact"]
    designs = designs or [{"name": c, "mode": c, "layers": layers, "scope": "all"} for c in conditions]
    reference, summaries, restorations = {}, [], []
    with (out / "rows.jsonl").open("w") as stream:
        for design in designs:
            condition = design["name"]
            started, results = time.time(), []
            for index, (source, ids) in enumerate(zip(rows, sequences)):
                row, first = generate(model, tokenizer, ids, source["labels"], contract, design["mode"],
                                      design["layers"], max_new_tokens, design["scope"])
                if condition == "intact":
                    reference[source["id"]] = {"row": dict(row), "first": first}
                ref = reference[source["id"]]
                log_p, log_q = ref["first"].log_softmax(-1), first.log_softmax(-1)
                row.update(arm=arm["name"], condition=condition, key=source["id"],
                           sequence_sha256=sequence_hash(ids),
                           first_token_kl_from_intact=float((log_p.exp() * (log_p - log_q)).sum()))
                stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
                results.append(row)
                if (index + 1) % 25 == 0:
                    print(f"CIVIL_PROGRESS condition={condition} examples={index + 1}", flush=True)
                if condition == "intact" and index < 5 and arm["kind"] == "peft":
                    restored, _ = generate(model, tokenizer, ids, source["labels"], contract, "restore", layers, max_new_tokens)
                    if any(restored[field] != row[field] for field in ("generated_ids", "generation_scores_sha256", "score")):
                        raise RuntimeError("Exact donor restoration changed generation")
                    restorations.append({"key": source["id"], "generation_exact": True,
                                         "generation_scores_sha256": row["generation_scores_sha256"],
                                         "intervention_calls": restored["intervention_calls"]})
            summary = {"condition": condition, **paired_summary(results, reference, 42), "elapsed_seconds": time.time() - started}
            summaries.append(summary)
            (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
    (out / "restore-checks.json").write_text(json.dumps(restorations, indent=2) + "\n")
    return summaries


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
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--layers", default="0,1,2,3,4,5")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--localization-panel", choices=["layers", "positions"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1 or args.max_new_tokens < 1:
        raise ValueError("Require positive row and generation limits")
    if transformers.__version__ != "5.16.1" or peft.__version__ != "0.20.0":
        raise RuntimeError("Runtime differs from pinned attention implementation")
    arms = json.loads(args.arms_spec.read_text())
    if len(arms) != 1:
        raise ValueError("Run one arm per GPU")
    arm = arms[0]
    metadata = arm_metadata(arm)
    if arm["kind"] == "peft" and (metadata["receipt"]["peft_type"] != "PREFIX_TUNING" or
            metadata["receipt"]["base_revision"] != args.model_revision):
        raise ValueError("Expected prefix adapter on the pinned base")
    source = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    rows = source[:args.limit]
    if len(rows) != args.limit or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Insufficient rows or duplicate IDs")
    contract = load_contract()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    checked = check_contract(tokenizer, contract, source, json.loads(args.contract_sample.read_text()))
    sequences = [render(tokenizer, contract[1][SEED_KEY], row["text"], contract) for row in rows]
    model = build_base(args.model_dir, "cuda")
    layers = [int(x) for x in args.layers.split(",")]
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0 or max(layers) >= model.config.num_hidden_layers:
        raise ValueError("Invalid intervention layers")
    if arm["kind"] == "peft":
        model = wrap(model, arms)
    args.out.mkdir(parents=True, exist_ok=False)
    designs = localization_designs(args.localization_panel) if args.localization_panel else None
    if designs and (arm["kind"] != "peft" or layers != list(range(6))):
        raise ValueError("Localization requires a prefix and observation layers L0-L5")
    summaries = evaluate(model, tokenizer, arm, sequences, rows, contract, layers, args.max_new_tokens, args.out, designs)
    count = len((args.out / "rows.jsonl").read_text().splitlines())
    if count != len(summaries) * len(rows):
        raise RuntimeError("Missing metric rows")
    manifest = {"status": "PASS", "probe": "prefix_civil_keys_values_v1", "arm": metadata,
                "model_revision": args.model_revision, "layers": layers, "scope": "all prefill queries and every cached decode step",
                "evaluation_role": "existing validation set used in checkpoint selection; not independent confirmation",
                "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(), "examples": len(rows),
                "sequence_hashes": [{"key": r["id"], "sha256": sequence_hash(ids), "tokens": len(ids)} for r, ids in zip(rows, sequences)],
                "max_new_tokens": args.max_new_tokens, "seed": 42, "metric_rows": count,
                "conditions": [s["condition"] for s in summaries], "contract_examples_verified": checked,
                "contract_sample_sha256": hashlib.sha256(args.contract_sample.read_bytes()).hexdigest(),
                "restore_examples_verified": min(5, len(rows)) if arm["kind"] == "peft" else 0,
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
                "files_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in args.out.iterdir() if f.is_file()}}
    if designs:
        manifest.update(probe="prefix_civil_localization_v1", localization_panel=args.localization_panel,
                        intervention_designs=designs, scope="per-condition real query positions; see intervention_designs")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PREFIX_CIVIL_ROWS={count}", flush=True)


if __name__ == "__main__":
    main()
