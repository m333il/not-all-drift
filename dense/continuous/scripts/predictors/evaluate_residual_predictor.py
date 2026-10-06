#!/usr/bin/env python3
"""Evaluate a trained residual predictor during baseline free generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from prompt_optimization.autoregressive_intervention import AutoregressiveDeltaHook
from prompt_optimization.civil_comments import (
    canonicalize_labels,
    compute_multilabel_metrics,
    parse_prediction,
    read_jsonl,
)
from prompt_optimization.conditions import base_model, decoder_layers, load_condition, render_prompt
from prompt_optimization.residual_predictor import (
    BiasOnlyResidualPredictor,
    LinearResidualPredictor,
    LowRankResidualPredictor,
    MLPResidualPredictor,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictor(summary, state):
    if summary["architecture"] == "bias_only":
        predictor = BiasOnlyResidualPredictor(torch.zeros(state["bias"].numel()))
    elif summary["architecture"] == "linear":
        predictor = LinearResidualPredictor(torch.zeros(state["bias"].numel()))
    elif summary["architecture"] == "low_rank":
        predictor = LowRankResidualPredictor(
            torch.zeros(state["bias"].numel()), int(summary["rank"])
        )
    elif summary["architecture"] == "mlp":
        predictor = MLPResidualPredictor(
            state["input_mean"].numel(),
            int(summary["mlp_width"]),
            state["input_mean"],
            state["input_scale"],
            torch.zeros(state["input_mean"].numel()),
        )
    else:
        raise ValueError(f"Unknown architecture: {summary['architecture']}")
    predictor.load_state_dict({key: value.float() for key, value in state.items()})
    return predictor.cuda().float().eval().requires_grad_(False)


def select_rows(path: Path, ids: list[str]):
    rows = read_jsonl(path)
    by_id = {str(row["id"]): row for row in rows}
    return [by_id[value] for value in ids]


def invalid_zero_sample_f1(
    gold: tuple[str, ...], prediction: tuple[str, ...] | None
) -> float:
    """Set F1 with malformed generations scored as zero, not as NONE."""
    if prediction is None:
        return 0.0
    gold_set, prediction_set = set(gold), set(prediction)
    if not gold_set and not prediction_set:
        return 1.0
    denominator = len(gold_set) + len(prediction_set)
    return 2 * len(gold_set & prediction_set) / denominator if denominator else 0.0


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--baseline-prompt-template", type=Path, required=True)
    parser.add_argument("--baseline-adapter-path", type=Path)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    contract = read_json(args.cache_dir / "contract.json")
    summary = read_json(args.fit_dir / "summary.json")
    if summary["condition_name"] != contract["condition_name"]:
        raise ValueError("Predictor and teacher cache describe different conditions")
    mode = summary["mode"]
    ids = [str(value) for value in contract["splits"]["test"]["ids"]]
    if args.limit is not None:
        ids = ids[: args.limit]
    rows = select_rows(args.test_file, ids)
    labels = tuple(args.labels)
    loaded = load_condition(
        model_name=contract["model_name"],
        model_revision=contract.get("model_revision"),
        prompt_template=args.baseline_prompt_template,
        adapter_path=args.baseline_adapter_path,
        local_files_only=args.local_files_only,
    )
    state = load_file(str(args.fit_dir / "predictor.safetensors"), device="cpu")
    predictor = load_predictor(summary, state)
    block = int(contract["block"])
    layers = decoder_layers(loaded.model)
    if block != len(layers) - 1:
        raise ValueError("Autoregressive final-readout evaluation requires the last decoder block")
    loaded.tokenizer.padding_side = "left"
    records, diagnostics = [], []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [
            render_prompt(loaded.tokenizer, loaded.prompt_template, str(row["text"]))
            for row in batch
        ]
        encoded = loaded.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
            return_tensors="pt",
        ).to(loaded.model.device)
        positions = torch.full(
            (len(batch),),
            encoded.input_ids.shape[1] - 1 + loaded.hidden_state_offset,
            dtype=torch.long,
            device=loaded.model.device,
        )
        hook = AutoregressiveDeltaHook(positions, predictor=predictor, mode=mode)
        handle = decoder_layers(loaded.model)[block].register_forward_hook(hook)
        try:
            generated = loaded.model.generate(
                **encoded,
                do_sample=False,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=loaded.tokenizer.pad_token_id,
                eos_token_id=loaded.tokenizer.eos_token_id,
            )
        finally:
            handle.remove()
        diagnostics.append(hook.diagnostics())
        continuation = generated[:, encoded.input_ids.shape[1] :]
        texts = loaded.tokenizer.batch_decode(continuation, skip_special_tokens=True)
        for row, text, token_ids in zip(batch, texts, continuation, strict=True):
            prediction = parse_prediction(text, allowed_labels=labels)
            records.append(
                {
                    "id": str(row["id"]),
                    "gold": row["labels"],
                    "generated_text": text,
                    "prediction": list(prediction) if prediction is not None else None,
                    "truncated": bool(
                        loaded.tokenizer.eos_token_id is not None
                        and not (token_ids == loaded.tokenizer.eos_token_id).any()
                    ),
                }
            )
    targets = [canonicalize_labels(row["labels"], allowed_labels=labels) for row in rows]
    predictions = [
        tuple(record["prediction"]) if record["prediction"] is not None else None
        for record in records
    ]
    metrics = compute_multilabel_metrics(targets, predictions, labels=labels)
    metrics["invalid_zero_samples_f1"] = sum(
        invalid_zero_sample_f1(target, prediction)
        for target, prediction in zip(targets, predictions, strict=True)
    ) / len(targets)
    metrics["set_accuracy"] = sum(
        prediction is not None and set(prediction) == set(target)
        for target, prediction in zip(targets, predictions, strict=True)
    ) / len(targets)
    metrics["truncated_rate"] = sum(record["truncated"] for record in records) / len(records)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "contract.json").write_text(
        json.dumps(
            {
                "status": "done",
                "condition_name": contract["condition_name"],
                "architecture": summary["architecture"],
                "objective": summary["objective"],
                "mode": mode,
                "block": block,
                "metrics": metrics,
                "hook_diagnostics": diagnostics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
