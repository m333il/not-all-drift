#!/usr/bin/env python3
"""Evaluate dependence on text instructions or learned virtual context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from prompt_optimization.attention import compute_prompt_spans, unmasked_position_ids
from prompt_optimization.civil_comments import (
    canonicalize_labels,
    compute_multilabel_metrics,
    parse_prediction,
    read_jsonl,
)
from prompt_optimization.conditions import base_model, load_condition, render_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--model-revision")
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--sample-ids", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def select_rows(rows: list[dict[str, Any]], path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload["ids"] if isinstance(payload, dict) else payload
    by_id = {str(row["id"]): row for row in rows}
    return [by_id[str(value)] for value in ids]


def instruction_mask(encoded: Any, offsets: list, prompts: list[str], rows: list[dict[str, Any]]):
    mask = encoded.attention_mask.clone()
    width = mask.shape[1]
    for index, row in enumerate(rows):
        left_padding = width - int(mask[index].sum())
        spans = compute_prompt_spans(
            prompts[index],
            str(row["text"]),
            offsets[index][left_padding:],
        )
        for left, right in spans["instruction"]:
            mask[index, left_padding + left : left_padding + right] = 0
    return mask


@torch.inference_mode()
def generate_batch(loaded: Any, rows: list[dict[str, Any]], args: argparse.Namespace, masked: bool):
    model, tokenizer = loaded.model, loaded.tokenizer
    prompts = [render_prompt(tokenizer, loaded.prompt_template, str(row["text"])) for row in rows]
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=args.max_input_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping").tolist()
    encoded = encoded.to(model.device)
    width = encoded.input_ids.shape[1]
    generation = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if not masked:
        sequences = model.generate(**encoded, **generation)
        continuation = sequences[:, width:]
    elif loaded.condition_type == "text":
        original = encoded.attention_mask.clone()
        encoded["attention_mask"] = instruction_mask(encoded, offsets, prompts, rows)
        encoded["position_ids"] = unmasked_position_ids(original)
        sequences = model.generate(**encoded, **generation)
        continuation = sequences[:, width:]
    else:
        peft_model = model
        raw = base_model(model)
        virtual = loaded.num_virtual_tokens
        ones = torch.ones(
            len(rows), virtual, dtype=encoded.attention_mask.dtype, device=model.device
        )
        full_mask = torch.cat([ones, encoded.attention_mask], dim=1)
        masked_attention = full_mask.clone()
        masked_attention[:, :virtual] = 0
        if loaded.condition_type == "prefix":
            prompt_cache = peft_model.get_prompt(batch_size=len(rows))
            if isinstance(prompt_cache, tuple):
                from transformers import DynamicCache

                prompt_cache = DynamicCache.from_legacy_cache(prompt_cache)
            sequences = raw.generate(
                input_ids=encoded.input_ids,
                attention_mask=masked_attention,
                past_key_values=prompt_cache,
                position_ids=unmasked_position_ids(full_mask)[:, virtual:],
                **generation,
            )
            continuation = sequences[:, width:]
        else:
            prompt_embeddings = peft_model.get_prompt(batch_size=len(rows))
            token_embeddings = raw.get_input_embeddings()(encoded.input_ids)
            sequences = raw.generate(
                inputs_embeds=torch.cat(
                    [prompt_embeddings.to(token_embeddings.dtype), token_embeddings], dim=1
                ),
                attention_mask=masked_attention,
                position_ids=unmasked_position_ids(full_mask),
                **generation,
            )
            continuation = sequences
    return tokenizer.batch_decode(continuation, skip_special_tokens=True)


def score(rows: list[dict[str, Any]], texts: list[str], labels: tuple[str, ...]):
    predictions = [parse_prediction(text, allowed_labels=labels) for text in texts]
    targets = [canonicalize_labels(row["labels"], allowed_labels=labels) for row in rows]
    return compute_multilabel_metrics(targets, predictions, labels=labels), predictions


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    rows = select_rows(read_jsonl(args.input_file), args.sample_ids)
    if args.limit is not None:
        rows = rows[: args.limit]
    labels = tuple(args.labels)
    loaded = load_condition(
        model_name=args.model_name,
        model_revision=args.model_revision,
        prompt_template=args.prompt_template,
        adapter_path=args.adapter_path,
        local_files_only=args.local_files_only,
    )
    loaded.tokenizer.padding_side = "left"
    texts = {"baseline": [], "masked": []}
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        for variant in texts:
            texts[variant].extend(generate_batch(loaded, batch, args, masked=variant == "masked"))
    metrics: dict[str, Any] = {}
    records = []
    for variant, generated in texts.items():
        metrics[variant], predictions = score(rows, generated, labels)
        for row, text, prediction in zip(rows, generated, predictions, strict=True):
            records.append(
                {
                    "id": str(row["id"]),
                    "variant": variant,
                    "gold": row["labels"],
                    "generated_text": text,
                    "prediction": list(prediction) if prediction is not None else None,
                }
            )
    metrics["delta_samples_f1"] = (
        metrics["masked"]["samples_f1"] - metrics["baseline"]["samples_f1"]
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (args.output_dir / "contract.json").write_text(
        json.dumps(
            {
                "condition_type": loaded.condition_type,
                "num_virtual_tokens": loaded.num_virtual_tokens,
                "samples": len(rows),
                "mask_target": "instruction" if loaded.condition_type == "text" else "virtual",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
