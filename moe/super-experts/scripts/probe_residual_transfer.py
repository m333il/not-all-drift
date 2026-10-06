#!/usr/bin/env python3
"""Transfer prefix-trained early residual states into the unadapted backbone."""
import argparse
import base64
import zlib
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from se_gepa.arms import SEED_KEY, build_base, check_contract, device_of, load_contract, render, wrap
from se_gepa.residual_shift import EarlyResidualShift
from probe_attention_contributions import arm_metadata, sequence_hash
from probe_prefix_causal import paired_summary
from probe_prefix_civil import generate
from probe_prefix_wikitext import logits_hash
from eval_ppl_arms import windows, WIKITEXT_REVISION

CONDITIONS = ("base", "teacher", "donor_state", "donor_delta", "constant", "random", "shuffled")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def packed_tensor(tensor):
    raw = tensor.contiguous().numpy().astype("<f4").tobytes()
    return {"dtype": "float32-little-endian", "shape": list(tensor.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "zlib_base64": base64.b64encode(zlib.compress(raw)).decode()}


def select_civil(calibration_source, evaluation_source, calibration_limit, offset, limit):
    calibration = calibration_source[:calibration_limit]
    evaluation = evaluation_source[offset:offset + limit]
    if calibration_limit < 1 or offset < 0 or limit < 1 or len(calibration) != calibration_limit or len(evaluation) != limit:
        raise ValueError("Unavailable calibration/evaluation rows")
    combined = calibration + evaluation
    if len({r["id"] for r in combined}) != len(combined) or len({r["text"] for r in combined}) != len(combined):
        raise ValueError("Duplicate or overlapping calibration/evaluation rows")
    return calibration, evaluation


@torch.no_grad()
def capture(model, ids, layer, check=False):
    tensor = torch.tensor([ids], device=device_of(model))
    options = dict(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False, logits_to_keep=1)
    native = model(**options).logits if check else None
    with EarlyResidualShift(model, layer) as hook:
        observed = model(**options).logits
    if check and not torch.equal(native, observed):
        raise RuntimeError("Residual observation changed logits")
    states = torch.cat(hook.before)
    if hook.calls != 1 or states.shape[0] != 3 or not torch.isfinite(states).all():
        raise RuntimeError("Missing or non-finite early residual states")
    return states


def vector_statistics(deltas, mean):
    energy = deltas.square().sum(-1).mean(0)
    error = (deltas - mean).square().sum(-1).mean(0)
    return {"examples": len(deltas), "mean_norm_per_position": mean.norm(dim=-1).tolist(),
            "rms_norm_per_position": energy.sqrt().tolist(),
            "relative_mse_per_position": (error / energy.clamp_min(1e-30)).tolist(),
            "cosine_to_constant_per_example": torch.nn.functional.cosine_similarity(deltas, mean[None], dim=-1).tolist()}


def controls(mean):
    random = torch.randn(mean.shape, generator=torch.Generator().manual_seed(42))
    return random * (mean.norm(dim=-1, keepdim=True) / random.norm(dim=-1, keepdim=True))


@torch.no_grad()
def measure(model, ids, task, tokenizer, labels, contract, max_new_tokens):
    if task == "civil":
        return generate(model, tokenizer, ids, labels, contract, "intact", [], max_new_tokens)
    tensor = torch.tensor([ids], device=device_of(model))
    target = tensor.clone()
    target[:, 0] = -100
    output = model(input_ids=tensor, attention_mask=torch.ones_like(tensor), labels=target, use_cache=False)
    if not torch.isfinite(output.loss) or not torch.isfinite(output.logits).all():
        raise RuntimeError("Non-finite WikiText output")
    return {"nll": float(output.loss), "targets": len(ids) - 1,
            "logits_sha256": logits_hash(output.logits)}, None


def equal_outputs(a, b, task):
    if task == "wiki":
        return a["logits_sha256"] == b["logits_sha256"] and a["nll"] == b["nll"]
    return all(a[k] == b[k] for k in ("generated_ids", "generation_scores_sha256", "score"))


@torch.no_grad()
def evaluate(teacher, sequences, sources, calibration_sequences, task, tokenizer, contract, out, layer=3, max_new_tokens=64):
    base = teacher.get_base_model()
    calibration_delta = torch.stack([capture(teacher, ids, layer, i == 0) - capture(base, ids, layer, i == 0)
                                     for i, ids in enumerate(calibration_sequences)])
    mean = calibration_delta.mean(0)
    random = controls(mean)
    write_json(out / "vectors.json", {"constant": mean.tolist(), "random": random.tolist(),
        "calibration_deltas": packed_tensor(calibration_delta), "statistics": vector_statistics(calibration_delta, mean),
        "distinct_first3_token_sequences": len({tuple(ids[:3]) for ids in calibration_sequences})})
    donor_states, base_states = [], []
    for i, ids in enumerate(sequences):
        base_states.append(capture(base, ids, layer, i == 0))
        donor_states.append(capture(teacher, ids, layer, i == 0))
    donor_states, base_states = torch.stack(donor_states), torch.stack(base_states)
    deltas = donor_states - base_states
    write_json(out / "donors.json", {"base_states": packed_tensor(base_states), "teacher_states": packed_tensor(donor_states),
        "delta_statistics": vector_statistics(deltas, mean),
        "distinct_first3_token_sequences": len({tuple(ids[:3]) for ids in sequences}),
        "shuffled_indices": [(i + 1) % len(sequences) for i in range(len(sequences))]})
    references, summaries, gates = {}, [], []
    with (out / "rows.jsonl").open("w") as stream:
        for condition in CONDITIONS:
            rows = []
            for i, (ids, source) in enumerate(zip(sequences, sources)):
                vector = {"donor_state": donor_states[i], "donor_delta": deltas[i], "constant": mean,
                          "random": random, "shuffled": deltas[(i + 1) % len(sequences)]}.get(condition)
                mode = "replace" if condition == "donor_state" else "add"
                context = EarlyResidualShift(base, layer, mode, vector) if vector is not None else nullcontext()
                with context as hook:
                    row, first = measure(teacher if condition == "teacher" else base, ids, task, tokenizer,
                                         source.get("labels"), contract, max_new_tokens)
                row.update(key=source["id"], condition=condition, sequence_sha256=sequence_hash(ids), prompt_tokens=len(ids))
                if hook is not None:
                    expected_calls = row["completion_tokens"] if task == "civil" else 1
                    if hook.patched != 3 or hook.calls != expected_calls:
                        raise RuntimeError("Wrong residual intervention coverage")
                    actual = torch.cat(hook.after)
                    row.update(patched_positions=hook.patched, intervention_calls=hook.calls,
                               patched_state_max_abs_per_position=actual.abs().amax(-1).tolist(),
                               donor_state_max_error=float((actual - donor_states[i]).abs().max()))
                    if condition == "donor_state" and not torch.equal(actual, donor_states[i]):
                        raise RuntimeError("Donor state replacement is not exact")
                if condition == "base":
                    references[source["id"]] = {"row": dict(row), "first": first}
                    if i < (5 if task == "civil" else 1):
                        with EarlyResidualShift(base, layer, "add", torch.zeros_like(mean)) as identity:
                            restored, _ = measure(base, ids, task, tokenizer, source.get("labels"), contract, max_new_tokens)
                        if not equal_outputs(row, restored, task) or identity.patched != 3:
                            raise RuntimeError("Zero-shift identity failed")
                        gates.append({"key": source["id"], "zero_shift_exact": True})
                ref = references[source["id"]]
                if task == "civil":
                    log_p, log_q = ref["first"].log_softmax(-1), first.log_softmax(-1)
                    row["first_token_kl_from_intact"] = float((log_p.exp() * (log_p - log_q)).sum())
                else:
                    row["delta_nll"] = row["nll"] - ref["row"]["nll"]
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
            if condition == "teacher":
                base_again, _ = measure(base, sequences[0], task, tokenizer, sources[0].get("labels"), contract, max_new_tokens)
                if not equal_outputs(references[sources[0]["id"]]["row"], base_again, task):
                    raise RuntimeError("Teacher evaluation contaminated base recipient")
                gates.append({"key": sources[0]["id"], "base_after_teacher_exact": True})
            if task == "civil":
                summary = paired_summary(rows, references, 42)
            else:
                delta = torch.tensor([r["delta_nll"] for r in rows], dtype=torch.float64)
                draws = torch.randint(len(rows), (10000, len(rows)), generator=torch.Generator().manual_seed(42))
                interval = torch.quantile(delta[draws].mean(-1), torch.tensor([.025, .975], dtype=torch.float64))
                nll = sum(r["nll"] for r in rows) / len(rows)
                summary = {"n": len(rows), "nll": nll, "ppl": math.exp(nll), "paired_delta_nll": float(delta.mean()),
                           "paired_bootstrap_ci95": interval.tolist(), "targets": sum(r["targets"] for r in rows)}
            summaries.append({"condition": condition, **summary})
            write_json(out / "summary.json", summaries)
            print(json.dumps(summaries[-1]), flush=True)
    write_json(out / "gates.json", gates)
    return summaries


def main():
    import peft
    import transformers
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--arms-spec", type=Path, required=True)
    parser.add_argument("--task", choices=["civil", "wiki"], required=True)
    parser.add_argument("--contract-sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if transformers.__version__ != "5.16.1" or peft.__version__ != "0.20.0":
        raise RuntimeError("Unexpected measurement runtime")
    arms = json.loads(args.arms_spec.read_text())
    if len(arms) != 1 or arms[0]["kind"] != "peft":
        raise ValueError("Exactly one prefix teacher per job")
    metadata = arm_metadata(arms[0])
    if metadata["receipt"]["peft_type"] != "PREFIX_TUNING" or metadata["receipt"]["base_revision"] != args.model_revision:
        raise ValueError("Wrong teacher architecture or base revision")
    root = Path(__file__).resolve().parents[1]
    paths = [root / "data" / name for name in ("civil_v2_val_seed42_n200.jsonl", "civil_v2_test_head2000.jsonl")]
    calibration_source, evaluation_source = [[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
    calibration_rows, civil_rows = select_civil(calibration_source, evaluation_source, 32, 1800, 200)
    tokenizer, contract = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True), load_contract()
    checked = check_contract(tokenizer, contract, calibration_source, json.loads(args.contract_sample.read_text()))
    calibration_sequences = [render(tokenizer, contract[1][SEED_KEY], r["text"], contract) for r in calibration_rows]
    if args.task == "civil":
        sources = civil_rows
        sequences = [render(tokenizer, contract[1][SEED_KEY], r["text"], contract) for r in sources]
    else:
        tokens = windows(tokenizer, 2048, 32)
        if tokens.shape != (32, 2048):
            raise RuntimeError("Missing WikiText evaluation windows")
        sequences = tokens[16:32].tolist()
        sources = [{"id": f"wiki-test-window-{i}"} for i in range(16, 32)]
    if any(len(ids) < 3 for ids in calibration_sequences + sequences):
        raise ValueError("Early-three protocol requires at least three tokens")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "inputs.json", {"calibration": [{**r, "input_ids": ids} for r, ids in zip(calibration_rows, calibration_sequences)],
                                         "evaluation": [{**r, "input_ids": ids} for r, ids in zip(sources, sequences)]})
    base = build_base(args.model_dir, "cuda")
    tensor = torch.tensor([calibration_sequences[0]], device=device_of(base))
    with torch.no_grad():
        before_wrap = base(input_ids=tensor, use_cache=False, logits_to_keep=1).logits
        teacher = wrap(base, arms)
        after_wrap = teacher.get_base_model()(input_ids=tensor, use_cache=False, logits_to_keep=1).logits
    if not torch.equal(before_wrap, after_wrap):
        raise RuntimeError("Wrapping changed the base recipient")
    del before_wrap, after_wrap, tensor
    summaries = evaluate(teacher, sequences, sources, calibration_sequences, args.task, tokenizer, contract, args.out)
    count = len((args.out / "rows.jsonl").read_text().splitlines())
    if count != len(CONDITIONS) * len(sources):
        raise RuntimeError("Missing evaluation rows")
    write_json(args.out / "manifest.json", {"status": "PASS", "probe": "early_residual_transfer_v1", "task": args.task,
        "arm": metadata, "model_revision": args.model_revision, "layer": 3, "real_positions": [0, 1, 2],
        "recipient": "base without prefix", "donor": "intact trained prefix on the same input",
        "delta": "teacher post-L3 residual minus base post-L3 residual", "strength": 1.0,
        "constant": "mean of 32 Civil validation donor deltas, separately per position; also used unchanged on WikiText",
        "conditions": list(CONDITIONS), "metric_rows": count, "max_new_tokens": 64, "seed": 42,
        "calibration_selection": "first32 Civil validation; already used for checkpoint selection",
        "evaluation_selection": "Civil test head2000[1800:2000] or WikiText test windows[16:32] length2048",
        "evaluation_history": "disjoint from calibration and prior localization; Civil test used in historical baseline evaluation; not a newly collected test set",
        "wikitext_revision": WIKITEXT_REVISION, "contract_examples_verified": checked,
        "contract_sample_sha256": hashlib.sha256(args.contract_sample.read_bytes()).hexdigest(),
        "source_rows_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "base_wrap_logits_exact": True, "observation_logits_exact": True,
        "cache_scope": "patch changes downstream-layer K/V; current and earlier layer caches remain base",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}})
    print(f"RESIDUAL_TRANSFER_ROWS={count}", flush=True)


if __name__ == "__main__":
    main()
