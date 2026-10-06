#!/usr/bin/env python3
"""Evaluate a constant train-mean residual shift during autoregressive decoding.

Each ``BLOCK=CACHE_DIR`` cell supplies aligned baseline/adapted training states.
The intervention vector is their train-split mean difference.  It is added at
the selected decoder block at prompt prefill and every subsequent decode step.
Layer and scale selection are performed on the supplied validation rows; this
script intentionally does not report the selected cell as an untouched test.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
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
    sha256_file,
)
from prompt_optimization.conditions import decoder_layers, load_condition, render_prompt
from prompt_optimization.residual_predictor import BiasOnlyResidualPredictor


@dataclass(frozen=True)
class Cell:
    block: int
    cache_dir: Path


def parse_cell(value: str) -> Cell:
    raw_block, separator, raw_path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Cell must be BLOCK=CACHE_DIR")
    try:
        block = int(raw_block)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid block in {value!r}") from error
    return Cell(block, Path(raw_path))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_rows(path: Path, ids_path: Path | None, limit: int | None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if ids_path is not None:
        ids = list(map(str, read_json(ids_path)))
        by_id = {str(row["id"]): row for row in rows}
        missing = [value for value in ids if value not in by_id]
        if missing:
            raise ValueError(f"Validation IDs are absent from input: {missing[:3]}")
        rows = [by_id[value] for value in ids]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No validation rows selected")
    return rows


def mean_shift(cell: Cell) -> tuple[torch.Tensor, dict[str, Any], Path, int]:
    contract = read_json(cell.cache_dir / "contract.json")
    if int(contract.get("block", cell.block)) != cell.block:
        raise ValueError(f"Block/cache mismatch for {cell.cache_dir}")
    entry = contract["splits"]["train"]
    path = cell.cache_dir / entry["path"]
    data = load_file(str(path), device="cpu")
    baseline_key = "baseline" if "baseline" in data else "manual"
    adapted_key = "adapted" if "adapted" in data else "target"
    if baseline_key not in data or adapted_key not in data:
        raise ValueError(f"Aligned states are missing from {path}")
    baseline, adapted = data[baseline_key].float(), data[adapted_key].float()
    if baseline.shape != adapted.shape or baseline.ndim != 2 or not len(baseline):
        raise ValueError(f"Malformed train cache: {path}")
    return (adapted - baseline).mean(0), contract, path, len(baseline)


def invalid_zero_sample_f1(
    gold: tuple[str, ...], prediction: tuple[str, ...] | None
) -> float:
    if prediction is None:
        return 0.0
    gold_set, prediction_set = set(gold), set(prediction)
    if not gold_set and not prediction_set:
        return 1.0
    denominator = len(gold_set) + len(prediction_set)
    return 2.0 * len(gold_set & prediction_set) / denominator if denominator else 0.0


@torch.inference_mode()
def evaluate_cell(
    *,
    loaded: Any,
    rows: list[dict[str, Any]],
    labels: tuple[str, ...],
    block: int,
    shift: torch.Tensor,
    scale: float,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    predictor = BiasOnlyResidualPredictor(shift * scale).to(
        device=loaded.model.device, dtype=torch.float32
    )
    predictor.eval().requires_grad_(False)
    records: list[dict[str, Any]] = []
    loaded.tokenizer.padding_side = "left"
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            render_prompt(loaded.tokenizer, loaded.prompt_template, str(row["text"]))
            for row in batch
        ]
        encoded = loaded.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        ).to(loaded.model.device)
        positions = (
            torch.arange(encoded.attention_mask.shape[1], device=loaded.model.device)
            .expand_as(encoded.attention_mask)
            .masked_fill(~encoded.attention_mask.bool(), -1)
            .max(1)
            .values
        )
        hook = AutoregressiveDeltaHook(
            positions, predictor=predictor, mode="repredict_recurrent"
        )
        handle = decoder_layers(loaded.model)[block].register_forward_hook(hook)
        try:
            generated = loaded.model.generate(
                **encoded,
                do_sample=False,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                pad_token_id=loaded.tokenizer.pad_token_id,
                eos_token_id=loaded.tokenizer.eos_token_id,
            )
        finally:
            handle.remove()
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
        tuple(row["prediction"]) if row["prediction"] is not None else None for row in records
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
    metrics["truncated_rate"] = sum(row["truncated"] for row in records) / len(records)
    return metrics, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", type=parse_cell, required=True)
    parser.add_argument("--scale", nargs="+", type=float, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--validation-ids", type=Path)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--model-revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-metric", default="invalid_zero_samples_f1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    if len({cell.block for cell in args.cell}) != len(args.cell):
        raise ValueError("Each block may appear only once")
    if not args.scale or not all(torch.isfinite(torch.tensor(args.scale))):
        raise ValueError("Scales must be finite")
    rows = select_rows(args.validation_file, args.validation_ids, args.limit)
    labels = tuple(args.labels)
    shifts: dict[int, torch.Tensor] = {}
    provenance: dict[int, dict[str, Any]] = {}
    for cell in args.cell:
        shift, contract, train_path, train_positions = mean_shift(cell)
        shifts[cell.block] = shift
        provenance[cell.block] = {
            "cache_contract": str((cell.cache_dir / "contract.json").resolve()),
            "cache_contract_sha256": sha256_file(cell.cache_dir / "contract.json"),
            "train_states": str(train_path.resolve()),
            "train_states_sha256": sha256_file(train_path),
            "condition_name": contract.get("condition_name"),
            "train_positions": train_positions,
        }
    loaded = load_condition(
        model_name=args.model_name,
        model_revision=args.model_revision,
        prompt_template=args.prompt_template,
        local_files_only=args.local_files_only,
    )
    depth = len(decoder_layers(loaded.model))
    if any(block < 0 or block >= depth for block in shifts):
        raise ValueError(f"Requested block is outside model depth {depth}")
    args.output_dir.mkdir(parents=True)
    grid_rows: list[dict[str, Any]] = []
    for block, shift in sorted(shifts.items()):
        for scale in args.scale:
            metrics, records = evaluate_cell(
                loaded=loaded,
                rows=rows,
                labels=labels,
                block=block,
                shift=shift,
                scale=scale,
                batch_size=args.batch_size,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
            )
            tag = f"block_{block:02d}_scale_{scale:g}".replace("-", "minus_").replace(".", "p")
            write_json(args.output_dir / f"{tag}_metrics.json", metrics)
            (args.output_dir / f"{tag}_records.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                encoding="utf-8",
            )
            grid_rows.append({"block": block, "scale": scale, **metrics})
    with (args.output_dir / "grid.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)
    if args.selection_metric not in grid_rows[0]:
        raise ValueError(f"Unknown selection metric: {args.selection_metric}")
    selected = max(
        grid_rows,
        key=lambda row: (float(row[args.selection_metric]), -int(row["block"]), -float(row["scale"])),
    )
    write_json(
        args.output_dir / "summary.json",
        {
            "status": "done",
            "scope": "validation-only recurrent mean-shift selection",
            "selection_metric": args.selection_metric,
            "selected": selected,
            "rows": len(rows),
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "labels": labels,
            "blocks": sorted(shifts),
            "scales": args.scale,
            "provenance": provenance,
            "validation_file": str(args.validation_file.resolve()),
            "validation_file_sha256": sha256_file(args.validation_file),
            "prompt_template_sha256": sha256_file(args.prompt_template),
        },
    )


if __name__ == "__main__":
    main()
