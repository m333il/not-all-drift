#!/usr/bin/env python3
"""Add baseline source-block states to aligned adapted-trajectory caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from prompt_optimization.civil_comments import read_jsonl, sha256_file
from prompt_optimization.conditions import decoder_layers, load_condition, render_prompt


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--target-cache must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate condition name: {name!r}")
        result[name] = Path(path)
    return result


def rows_from_contract(contract: dict[str, Any], split: str) -> list[dict[str, Any]]:
    spec = contract["splits"][split]
    path = Path(spec["source_path"])
    rows = read_jsonl(path)
    by_id = {str(row["id"]): row for row in rows}
    wanted = [str(value) for value in spec["ids"]]
    if any(value not in by_id for value in wanted):
        raise ValueError(f"Target cache references missing rows in {path}")
    return [by_id[value] for value in wanted]


def padded_trajectories(data: dict[str, torch.Tensor], start: int, end: int, pad_id: int):
    offsets = data["offsets"].long()
    lengths = offsets[start + 1 : end + 1] - offsets[start:end]
    width = int(lengths.max())
    tokens = torch.full((end - start, width), pad_id, dtype=torch.long)
    mask = torch.zeros((end - start, width), dtype=torch.bool)
    for local, (left, right) in enumerate(
        zip(offsets[start:end], offsets[start + 1 : end + 1], strict=True)
    ):
        values = data["token_ids"][int(left) : int(right)].long()
        tokens[local, : len(values)] = values
        mask[local, : len(values)] = True
    return tokens, mask


@torch.inference_mode()
def capture_source_states(loaded, rows, tokens, continuation_mask, source_block: int, max_length: int):
    tokenizer, model = loaded.tokenizer, loaded.model
    tokenizer.padding_side = "left"
    prompts = [render_prompt(tokenizer, loaded.prompt_template, str(row["text"])) for row in rows]
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    prompt_width = encoded.input_ids.shape[1]
    continuation_inputs = tokens[:, :-1]
    continuation_attention = continuation_mask[:, :-1]
    input_ids = torch.cat([encoded.input_ids, continuation_inputs], dim=1).to(model.device)
    attention_mask = torch.cat([encoded.attention_mask, continuation_attention], dim=1).to(model.device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    steps = tokens.shape[1]
    starts = torch.full(
        (len(rows),), prompt_width - 1, dtype=torch.long, device=model.device
    )
    captured = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        positions = starts[:, None] + torch.arange(steps, device=model.device)[None]
        batch_index = torch.arange(len(rows), device=model.device)[:, None]
        captured.append(hidden[batch_index, positions].detach().to(torch.bfloat16).cpu())

    handle = decoder_layers(model)[source_block].register_forward_hook(hook)
    try:
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Source-block hook did not run exactly once")
    return captured[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--target-cache", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    targets = named_paths(args.target_cache)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    args.output_dir.mkdir(parents=True)
    contracts = {name: read_json(path / "contract.json") for name, path in targets.items()}
    for name, contract in contracts.items():
        if contract.get("status") != "done":
            raise ValueError(f"Incomplete target cache: {name}")
    reference = next(iter(contracts.values()))
    rows = {split: rows_from_contract(reference, split) for split in ("train", "val", "test")}
    for name, contract in contracts.items():
        for split, values in rows.items():
            if [str(row["id"]) for row in values] != list(contract["splits"][split]["ids"]):
                raise ValueError(f"Target caches are not row-aligned for {name}/{split}")
    loaded = load_condition(
        model_name=config["model_name"],
        model_revision=config.get("model_revision"),
        prompt_template=args.prompt_template,
        local_files_only=args.local_files_only,
    )
    source_block = int(config["source_block"])
    total_blocks = len(decoder_layers(loaded.model))
    manifest: dict[str, Any] = {
        "status": "running",
        "model_name": config["model_name"],
        "model_revision": config.get("model_revision"),
        "source_block": source_block,
        "target_block": int(config["target_block"]),
        "total_blocks": total_blocks,
        "conditions": {},
        "provenance": {
            "labels": reference["labels"],
            "splits": reference["splits"],
        },
    }
    for name, root in targets.items():
        condition_manifest = {"splits": {}}
        for split, split_rows in rows.items():
            source_spec = contracts[name]["splits"][split]
            source_path = root / source_spec["path"]
            data = load_file(str(source_path), device="cpu")
            chunks = []
            batch_size = int(config["cache_batch_size"])
            for start in range(0, len(split_rows), batch_size):
                end = min(start + batch_size, len(split_rows))
                tokens, mask = padded_trajectories(
                    data, start, end, loaded.tokenizer.pad_token_id
                )
                values = capture_source_states(
                    loaded,
                    split_rows[start:end],
                    tokens,
                    mask,
                    source_block,
                    int(config["max_input_length"]),
                )
                chunks.append(values[mask])
            source_states = torch.cat(chunks).contiguous()
            if source_states.shape != data["baseline"].shape:
                raise RuntimeError(f"State shape mismatch for {name}/{split}")
            payload = {
                "source_states": source_states,
                "baseline_final": data["baseline"].contiguous(),
                "target_final": data["adapted"].contiguous(),
                "token_ids": data["token_ids"].contiguous(),
                "row_index": data["row_index"].contiguous(),
                "step_index": data["step_index"].contiguous(),
                "offsets": data["offsets"].contiguous(),
            }
            output_path = args.output_dir / name / f"{split}.safetensors"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(payload, str(output_path))
            condition_manifest["splits"][split] = {
                "path": str(output_path.relative_to(args.output_dir)),
                "sha256": sha256_file(output_path),
                "rows": len(split_rows),
                "positions": len(source_states),
            }
        manifest["conditions"][name] = condition_manifest
    manifest["status"] = "done"
    write_json(args.output_dir / "contract.json", manifest)


if __name__ == "__main__":
    main()
