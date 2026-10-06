#!/usr/bin/env python3
"""Stream Qwen3 contributions for virtual, real 0/1/2, and remaining keys."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from se_gepa.arms import (
    SEED_KEY, build_base, check_contract, check_instructions, device_of, group_arms, load_contract,
    render, resolve_instructions, wrap,
)
from se_gepa.attention_contributions import AttentionContributionProbe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--arms-spec", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--contract-sample", type=Path, required=True)
    parser.add_argument("--contract-rows", type=Path, required=True)
    parser.add_argument("--layers", default="0,1,2,3,4,5")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-real-tokens", type=int, default=256)
    parser.add_argument("--only", help="Comma-separated arm names; the restored spec may already be filtered")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-mean-vectors", action="store_true",
                        help="Include each group's full query-mean output vector in metrics.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sequence_hash(ids: list[int]) -> str:
    payload = json.dumps(ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def geometry(arm):
    if arm["kind"] == "text":
        return 0, 0, None
    receipt_path = Path(arm["dir"]) / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    count = int(receipt["num_virtual_tokens"])
    if receipt["peft_type"] == "PROMPT_TUNING":
        return count, 0, receipt
    if receipt["peft_type"] == "PREFIX_TUNING":
        return 0, count, receipt
    raise ValueError(f"{arm['name']}: unsupported PEFT type {receipt['peft_type']!r}")


def arm_metadata(arm):
    row = {key: value for key, value in arm.items() if key != "dir"}
    if arm["kind"] == "peft":
        directory = Path(arm["dir"])
        receipt_path = directory / "receipt.json"
        config_path = directory / "adapter" / "adapter_config.json"
        row.update(
            receipt=json.loads(receipt_path.read_text()),
            receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            adapter_config=json.loads(config_path.read_text()),
            adapter_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        )
    return row


def tensor_errors(actual, replay):
    delta = (actual.float() - replay.float()).flatten()
    reference = actual.float().flatten()
    return {
        "exact": bool(torch.equal(actual, replay)),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(delta) /
                             torch.linalg.vector_norm(reference).clamp_min(1e-30)),
    }


def probe_arm(model, arm, sequences, layers, stream, audit_vector_dims, reconstruction_rtol):
    virtual_queries, prefix_keys, _receipt = geometry(arm)
    count = 0
    reconstruction_maxima = {
        "before_o_proj": {"max_abs": 0.0, "relative_l2": 0.0},
        "after_o_proj": {"max_abs": 0.0, "relative_l2": 0.0},
    }

    def emit(row):
        nonlocal count
        for stage, errors in row["reconstruction"].items():
            for metric, value in errors.items():
                reconstruction_maxima[stage][metric] = max(reconstruction_maxima[stage][metric], value)
        stream.write(json.dumps({"arm": arm["name"], **row}, separators=(",", ":"),
                                allow_nan=False) + "\n")
        stream.flush()
        count += 1

    first = torch.tensor([sequences[0]["ids"]], device=device_of(model))
    with torch.no_grad():
        reference_logits = model(input_ids=first, use_cache=False).logits
    replay = None
    with AttentionContributionProbe(
        model, layers, virtual_query_tokens=virtual_queries, prefix_key_tokens=prefix_keys,
        emit=emit, audit_vector_dims=audit_vector_dims, reconstruction_rtol=reconstruction_rtol,
    ) as probe:
        for index, item in enumerate(sequences):
            ids = item["ids"]
            probe.begin_example(index, len(ids), item["sha256"])
            tensor = torch.tensor([ids], device=device_of(model))
            with torch.no_grad():
                outputs = model(input_ids=tensor, use_cache=False)
            if not torch.isfinite(outputs.logits).all():
                raise RuntimeError(f"{arm['name']} example {index} produced non-finite logits")
            if index == 0:
                replay = tensor_errors(reference_logits, outputs.logits)
                if not replay["exact"]:
                    raise RuntimeError(f"{arm['name']}: attention hooks changed the model logits: {replay}")
            del outputs
    del reference_logits
    return count, replay, reconstruction_maxima


def main() -> None:
    import peft
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    args = parse_args()
    if transformers.__version__ != "5.16.1" or peft.__version__ != "0.20.0":
        raise RuntimeError(
            "This probe is pinned to transformers==5.16.1 and peft==0.20.0; got "
            f"transformers=={transformers.__version__}, peft=={peft.__version__}"
        )
    if args.limit <= 0 or args.max_real_tokens <= 0:
        raise SystemExit("--limit and --max-real-tokens must be positive")
    layers = [int(value) for value in args.layers.split(",") if value]
    if not layers:
        raise SystemExit("--layers must select at least one layer")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory {args.output_dir}")

    arms_payload = json.loads(args.arms_spec.read_text())
    arms = arms_payload["arms"] if isinstance(arms_payload, dict) else arms_payload
    if args.only:
        wanted = set(args.only.split(","))
        arms = [arm for arm in arms if arm["name"] in wanted]
        missing = wanted - {arm["name"] for arm in arms}
        if missing:
            raise SystemExit(f"Unknown arm names: {sorted(missing)}")
    if not arms:
        raise SystemExit("No arms selected")
    for arm in arms:
        if arm["kind"] == "peft":
            receipt = json.loads((Path(arm["dir"]) / "receipt.json").read_text())
            if receipt.get("base_revision") != args.model_revision:
                raise RuntimeError(
                    f"{arm['name']} was trained on {receipt.get('base_revision')}, "
                    f"not --model-revision {args.model_revision}"
                )

    model_config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    if model_config.model_type not in {"qwen3", "qwen3_moe"}:
        raise TypeError(f"Only Qwen3 is supported, got {model_config.model_type!r}")

    contract = load_contract()
    _labels, seeds, _render, _apply = contract
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    resolve_instructions(arms, tokenizer, seeds)
    check_instructions(tokenizer, arms, contract)
    contract_rows = [json.loads(line) for line in args.contract_rows.read_text().splitlines() if line.strip()]
    checked = check_contract(tokenizer, contract, contract_rows, json.loads(args.contract_sample.read_text()))
    print(f"CONTRACT_TOKENS_VERIFIED={checked}", flush=True)

    source_rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()][:args.limit]
    if not source_rows:
        raise RuntimeError("--rows contains no examples")
    instructions = {arm.get("instruction", seeds[SEED_KEY]) for arm in arms}
    sequences = {}
    for instruction in instructions:
        rendered = [render(tokenizer, instruction, row["text"], contract) for row in source_rows]
        sequences[instruction] = []
        for row, ids in zip(source_rows, rendered):
            clipped = ids[:args.max_real_tokens]
            if not clipped:
                raise RuntimeError(f"Rendered example {row.get('id')!r} has no tokens")
            sequences[instruction].append({
                "key": row.get("id"), "ids": clipped, "sha256": sequence_hash(clipped),
                "full_real_tokens": len(ids), "measured_real_tokens": len(clipped),
            })

    args.output_dir.mkdir(parents=True)
    model = build_base(args.model_dir, args.device)
    if model.config.model_type not in {"qwen3", "qwen3_moe"}:
        raise TypeError(f"Only Qwen3 is supported, got {model.config.model_type!r}")

    reconstruction_rtol = 0.02 if next(model.parameters()).dtype == torch.bfloat16 else 1e-5
    audit_vector_dims = model.config.hidden_size if args.save_mean_vectors else 0
    metrics_path = args.output_dir / "metrics.jsonl"
    metric_rows = 0
    no_op_replay = {}
    reconstruction_maxima = {}
    text_arms, groups = group_arms(arms)
    with metrics_path.open("w") as stream:
        for arm in text_arms:
            count, replay, maxima = probe_arm(
                model, arm, sequences[arm["instruction"]], layers, stream,
                audit_vector_dims, reconstruction_rtol,
            )
            metric_rows += count
            no_op_replay[arm["name"]] = replay
            reconstruction_maxima[arm["name"]] = maxima
            print(f"CONTRIBUTIONS_ARM={arm['name']}", flush=True)
        for peft_type, group in groups.items():
            wrapped = wrap(model, group)
            print(f"PEFT_GROUP={peft_type} arms={[arm['name'] for arm in group]}", flush=True)
            for arm in group:
                wrapped.set_adapter(arm["name"])
                count, replay, maxima = probe_arm(
                    wrapped, arm, sequences[seeds[SEED_KEY]], layers, stream,
                    audit_vector_dims, reconstruction_rtol,
                )
                metric_rows += count
                no_op_replay[arm["name"]] = replay
                reconstruction_maxima[arm["name"]] = maxima
                print(f"CONTRIBUTIONS_ARM={arm['name']}", flush=True)
            del wrapped
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    expected_metric_rows = len(arms) * len(source_rows) * len(layers)
    if metric_rows != expected_metric_rows:
        raise RuntimeError(f"Wrote {metric_rows} metric rows, expected {expected_metric_rows}")

    sequence_manifest = {}
    for arm in arms:
        instruction = arm.get("instruction", seeds[SEED_KEY])
        sequence_manifest[arm["name"]] = [
            {key: value for key, value in item.items() if key != "ids"}
            for item in sequences[instruction]
        ]
    manifest = {
        "status": "PASS",
        "probe": "qwen3_attention_contributions_v1",
        "prefill_only": True,
        "model_dir": str(Path(args.model_dir).resolve()),
        "model_revision": args.model_revision,
        "model_type": model.config.model_type,
        "dtype": str(next(model.parameters()).dtype),
        "attention_backend": model.config._attn_implementation,
        "experts_backend": getattr(model.config, "_experts_implementation", None),
        "layers": layers,
        "limit": args.limit,
        "max_real_tokens": args.max_real_tokens,
        "contract_examples_verified": checked,
        "rows_path": str(args.rows),
        "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "contract_sample_sha256": hashlib.sha256(args.contract_sample.read_bytes()).hexdigest(),
        "contract_rows_sha256": hashlib.sha256(args.contract_rows.read_bytes()).hexdigest(),
        "sequence_hashes": sequence_manifest,
        "arms": [arm_metadata(arm) for arm in arms],
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
        },
        "metrics_rows": metric_rows,
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "mean_vectors_saved": args.save_mean_vectors,
        "audit_vector_dims": audit_vector_dims,
        "reconstruction_relative_l2_tolerance": reconstruction_rtol,
        "reconstruction_maxima": reconstruction_maxima,
        "no_op_replay": no_op_replay,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(f"CONTRIBUTION_ROWS={metric_rows}", flush=True)
    print(f"ARTIFACT_DIR={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
