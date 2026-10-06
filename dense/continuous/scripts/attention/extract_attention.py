#!/usr/bin/env python3
"""Extract prompt/generation attention and residual states for one condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prompt_optimization.attention import (
    aggregate_segment_mass,
    compute_prompt_spans,
    key_segment_ids,
)
from prompt_optimization.civil_comments import read_jsonl
from prompt_optimization.conditions import load_condition, render_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--model-revision")
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--sample-ids", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def select_rows(rows: list[dict[str, Any]], sample_ids: Path | None) -> list[dict[str, Any]]:
    if sample_ids is None:
        return rows
    payload = json.loads(sample_ids.read_text(encoding="utf-8"))
    ids = payload["ids"] if isinstance(payload, dict) else payload
    by_id = {str(row["id"]): row for row in rows}
    missing = [str(value) for value in ids if str(value) not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} requested IDs are absent from the input file")
    return [by_id[str(value)] for value in ids]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    rows = select_rows(read_jsonl(args.input_file), args.sample_ids)
    if args.limit is not None:
        rows = rows[: args.limit]
    loaded = load_condition(
        model_name=args.model_name,
        model_revision=args.model_revision,
        prompt_template=args.prompt_template,
        adapter_path=args.adapter_path,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    )
    model, tokenizer = loaded.model, loaded.tokenizer
    tokenizer.padding_side = "left"
    eos_ids = model.generation_config.eos_token_id
    eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    eos.add(tokenizer.eos_token_id)
    eos.discard(None)
    index: dict[str, Any] = {}
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [render_prompt(tokenizer, loaded.prompt_template, str(row["text"])) for row in batch]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        encoded = encoded.to(model.device)
        prompt_width = encoded.input_ids.shape[1]
        outputs = model(
            **encoded,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        layer_count = len(outputs.attentions)
        key_count = outputs.attentions[0].shape[-1]
        expected_keys = prompt_width + loaded.num_virtual_tokens
        if key_count != expected_keys:
            raise RuntimeError(f"Attention keys={key_count}, expected {expected_keys}")
        for local, row in enumerate(batch):
            token_count = int(encoded.attention_mask[local].sum())
            left_padding = prompt_width - token_count
            real_offsets = offsets[local, left_padding:].tolist()
            spans = compute_prompt_spans(
                prompts[local],
                str(row["text"]),
                real_offsets,
                loaded.num_virtual_tokens,
            )
            segment_ids = key_segment_ids(
                spans,
                virtual_tokens=loaded.num_virtual_tokens,
                left_padding=left_padding,
                key_count=key_count,
            )
            last = torch.stack(
                [outputs.attentions[layer][local, :, -1, :] for layer in range(layer_count)]
            ).float()
            probabilities = last.clamp_min(1e-20)
            entropy = -(probabilities * probabilities.log()).sum(-1)
            hidden = torch.stack([state[local, -1] for state in outputs.hidden_states])
            token_ids = generated.sequences[local, prompt_width:].tolist()
            generated_steps = next(
                (step + 1 for step, token_id in enumerate(token_ids) if token_id in eos),
                len(token_ids),
            )
            generation_rows = np.zeros(
                (
                    generated_steps,
                    layer_count,
                    last.shape[1],
                    key_count + generated_steps,
                ),
                dtype=np.float16,
            )
            for step in range(generated_steps):
                values = torch.stack(
                    [generated.attentions[step][layer][local, :, -1, :] for layer in range(layer_count)]
                )
                generation_rows[step, :, :, : values.shape[-1]] = values.to(torch.float16).cpu().numpy()
            full_segment_ids = np.concatenate(
                [segment_ids, np.full(generated_steps, -1, dtype=np.int8)]
            )
            sample_id = str(row["id"])
            np.savez_compressed(
                args.output_dir / f"{sample_id}.npz",
                last_row=last.to(torch.float16).cpu().numpy(),
                segment_mass=aggregate_segment_mass(last.cpu().numpy(), segment_ids),
                entropy=entropy.float().cpu().numpy(),
                hidden_last=hidden.to(torch.float16).cpu().numpy(),
                generation_rows=generation_rows,
                generation_segment_mass=aggregate_segment_mass(
                    generation_rows.astype(np.float32), full_segment_ids
                ),
                generated_tokens=np.asarray(token_ids[:generated_steps], dtype=np.int32),
            )
            index[sample_id] = {
                "labels": row.get("labels"),
                "prompt_tokens": token_count,
                "attention_keys": key_count,
                "generated_tokens": generated_steps,
                "spans": spans,
                "generated_text": tokenizer.decode(token_ids, skip_special_tokens=True),
            }
        write_json(args.output_dir / "index.json", index)
    write_json(
        args.output_dir / "contract.json",
        {
            "condition_type": loaded.condition_type,
            "num_virtual_tokens": loaded.num_virtual_tokens,
            "samples": len(rows),
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "input_file": str(args.input_file),
            "prompt_template": str(args.prompt_template),
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        },
    )


if __name__ == "__main__":
    main()
