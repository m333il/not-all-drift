#!/usr/bin/env python3
"""Evaluate a DENSE+KL predictor with recurrent correction and real block skipping."""

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


class LoadedResidualLinear(torch.nn.Linear):
    def state_prediction(self, states: torch.Tensor) -> torch.Tensor:
        return states + self(states)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_from_contract(contract: dict[str, Any], split: str) -> list[dict[str, Any]]:
    spec = contract["provenance"]["splits"][split]
    rows = read_jsonl(Path(spec["source_path"]))
    by_id = {str(row["id"]): row for row in rows}
    return [by_id[str(value)] for value in spec["ids"]]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    config = read_json(args.config)
    contract = read_json(args.cache_dir / "contract.json")
    fit = read_json(args.fit_dir / "summary.json")
    if args.condition not in contract["conditions"] or fit["condition"] != args.condition:
        raise ValueError("Condition mismatch between cache, fit, and evaluation")
    rows = rows_from_contract(contract, "test")
    if args.limit is not None:
        rows = rows[: args.limit]
    labels = tuple(contract["provenance"]["labels"])
    loaded = load_condition(
        model_name=contract["model_name"],
        model_revision=contract.get("model_revision"),
        prompt_template=args.prompt_template,
        local_files_only=args.local_files_only,
    )
    state = load_file(str(args.fit_dir / "predictor.safetensors"), device="cpu")
    predictor = LoadedResidualLinear(state["weight"].shape[1], state["weight"].shape[0]).cuda().float()
    predictor.load_state_dict({key: value.float() for key, value in state.items()})
    predictor.eval().requires_grad_(False)
    source_block = int(contract["source_block"])
    layers = decoder_layers(loaded.model)
    total_blocks = len(layers)
    base = base_model(loaded.model)
    original_layers = layers
    base.model.layers = torch.nn.ModuleList(list(layers[: source_block + 1]))
    loaded.tokenizer.padding_side = "left"
    records = []
    diagnostics = []
    try:
        for start in range(0, len(rows), int(config["generation_batch_size"])):
            batch = rows[start : start + int(config["generation_batch_size"])]
            prompts = [
                render_prompt(loaded.tokenizer, loaded.prompt_template, str(row["text"]))
                for row in batch
            ]
            encoded = loaded.tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=int(config["max_input_length"]),
                return_tensors="pt",
            ).to(loaded.model.device)
            positions = torch.full(
                (len(batch),),
                encoded.input_ids.shape[1] - 1,
                dtype=torch.long,
                device=loaded.model.device,
            )
            hook = AutoregressiveDeltaHook(
                positions, predictor=predictor, mode="repredict_recurrent"
            )
            handle = base.model.layers[source_block].register_forward_hook(hook)
            try:
                generated = loaded.model.generate(
                    **encoded,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=int(config["max_new_tokens"]),
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
    finally:
        base.model.layers = original_layers
    targets = [canonicalize_labels(row["labels"], allowed_labels=labels) for row in rows]
    predictions = [
        tuple(record["prediction"]) if record["prediction"] is not None else None
        for record in records
    ]
    metrics = compute_multilabel_metrics(targets, predictions, labels=labels)
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
                "condition": args.condition,
                "objective": "normalized_dense_plus_kl",
                "source_block": source_block,
                "target_block": contract["target_block"],
                "total_blocks": total_blocks,
                "layers_kept": source_block + 1,
                "layers_skipped": total_blocks - source_block - 1,
                "mode": "repredict_recurrent",
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
