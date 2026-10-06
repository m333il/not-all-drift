#!/usr/bin/env python3
"""Benchmark native conditions and true decoder-layer-skipping early exits.

Model loading, tokenisation, warm-up, and text decoding are outside the timed
region.  In an early-exit condition, blocks after ``source_block`` are removed;
the learned residual correction is applied after the final kept block at
prefill and every decode step, followed by the original final norm and LM head.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from prompt_optimization.civil_comments import read_jsonl, sha256_file
from prompt_optimization.conditions import base_model, decoder_layers, load_condition, render_prompt


class ResidualLinear(torch.nn.Linear):
    """Loadable residual map whose output is added to its input state."""


class FastRecurrentHook:
    def __init__(self, predictor: ResidualLinear) -> None:
        self.predictor = predictor
        self.positions: torch.Tensor | None = None
        self.calls = 0

    def reset(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.calls = 0

    def __call__(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if self.positions is None or hidden.ndim != 3:
            raise ValueError("Latency hook was not initialized")
        positions = (
            self.positions.to(hidden.device)
            if self.calls == 0
            else torch.full(
                (hidden.shape[0],), hidden.shape[1] - 1, device=hidden.device, dtype=torch.long
            )
        )
        self.calls += 1
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        original = hidden[rows, positions]
        hidden[rows, positions] = original + self.predictor(original.float()).to(original.dtype)
        return output


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def selected_rows(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    source = resolve(root, config["test_file"])
    assert source is not None
    rows = read_jsonl(source)
    if config.get("test_ids"):
        ids_path = resolve(root, config["test_ids"])
        assert ids_path is not None
        ids = list(map(str, read_json(ids_path)))
        by_id = {str(row["id"]): row for row in rows}
        rows = [by_id[value] for value in ids]
    rows = rows[: int(config["samples"])]
    if len(rows) != int(config["samples"]):
        raise ValueError("Insufficient selected test rows")
    return rows


def encode_batches(loaded: Any, rows: list[dict[str, Any]], config: dict[str, Any]):
    prompts = [
        render_prompt(loaded.tokenizer, loaded.prompt_template, str(row["text"])) for row in rows
    ]
    loaded.tokenizer.padding_side = "left"
    batches = []
    batch_size = int(config["batch_size"])
    for start in range(0, len(prompts), batch_size):
        batches.append(
            loaded.tokenizer(
                prompts[start : start + batch_size],
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=int(config["max_input_length"]),
                return_tensors="pt",
            ).to(loaded.model.device)
        )
    return batches


def generation_kwargs(tokenizer: Any, config: dict[str, Any], protocol: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if protocol == "fixed_tokens":
        count = int(config["fixed_decode_tokens"])
        values.update(min_new_tokens=count, max_new_tokens=count)
    elif protocol == "natural":
        values["max_new_tokens"] = int(config["natural_max_new_tokens"])
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    return values


def token_count(values: torch.Tensor, eos_id: int | None, pad_id: int | None) -> int:
    count = 0
    for row in values.tolist():
        for token in row:
            if pad_id is not None and token == pad_id:
                continue
            count += 1
            if eos_id is not None and token == eos_id:
                break
    return count


@torch.inference_mode()
def benchmark_condition(
    *,
    loaded: Any,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    condition: str,
    source_block: int | None = None,
    predictor: ResidualLinear | None = None,
) -> list[dict[str, Any]]:
    batches = encode_batches(loaded, rows, config)
    hook = FastRecurrentHook(predictor) if predictor is not None else None
    handle = None
    if hook is not None:
        if source_block is None:
            raise ValueError("source_block is required with a predictor")
        handle = decoder_layers(loaded.model)[source_block].register_forward_hook(hook)

    def generate(encoded: Any, kwargs: dict[str, Any]) -> torch.Tensor:
        if hook is not None:
            positions = (
                torch.arange(encoded.attention_mask.shape[1], device=loaded.model.device)
                .expand_as(encoded.attention_mask)
                .masked_fill(~encoded.attention_mask.bool(), -1)
                .max(1)
                .values
            )
            hook.reset(positions)
        return loaded.model.generate(**encoded, **kwargs)

    measurements: list[dict[str, Any]] = []
    try:
        for protocol in config["protocols"]:
            kwargs = generation_kwargs(loaded.tokenizer, config, protocol)
            for encoded in batches[: int(config["warmup_batches"])]:
                generate(encoded, kwargs)
            torch.cuda.synchronize()
            for repeat in range(int(config["repeats"])):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                continuations = []
                for encoded in batches:
                    generated = generate(encoded, kwargs)
                    continuations.append(generated[:, encoded.input_ids.shape[1] :])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                generated_tokens = sum(
                    token_count(
                        continuation,
                        loaded.tokenizer.eos_token_id,
                        loaded.tokenizer.pad_token_id,
                    )
                    for continuation in continuations
                )
                measurements.append(
                    {
                        "condition": condition,
                        "protocol": protocol,
                        "repeat": repeat,
                        "samples": len(rows),
                        "seconds": seconds,
                        "samples_per_second": len(rows) / seconds,
                        "generated_tokens": generated_tokens,
                        "generated_tokens_per_second": generated_tokens / seconds,
                        "milliseconds_per_sample": 1000.0 * seconds / len(rows),
                        "milliseconds_per_generated_token": 1000.0 * seconds / generated_tokens,
                        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    }
                )
    finally:
        if handle is not None:
            handle.remove()
    return measurements


def load_predictor(path: Path, device: torch.device) -> ResidualLinear:
    state = load_file(str(path), device="cpu")
    if set(state) != {"weight", "bias"}:
        raise ValueError(f"Expected affine checkpoint at {path}")
    predictor = ResidualLinear(state["weight"].shape[1], state["weight"].shape[0])
    predictor.load_state_dict({key: value.float() for key, value in state.items()})
    return predictor.to(device).float().eval().requires_grad_(False)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["condition"], row["protocol"]), []).append(row)
    output = []
    metrics = (
        "seconds",
        "samples_per_second",
        "generated_tokens_per_second",
        "milliseconds_per_sample",
        "milliseconds_per_generated_token",
        "peak_memory_bytes",
    )
    for (condition, protocol), values in sorted(groups.items()):
        record: dict[str, Any] = {
            "condition": condition,
            "protocol": protocol,
            "repeats": len(values),
        }
        for metric in metrics:
            observed = [float(value[metric]) for value in values]
            mean = sum(observed) / len(observed)
            sample_sd = (
                math.sqrt(sum((value - mean) ** 2 for value in observed) / (len(observed) - 1))
                if len(observed) > 1
                else 0.0
            )
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sample_sd"] = sample_sd
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    config = read_json(args.config)
    root = args.config.resolve().parent
    torch.set_num_threads(int(config.get("cpu_threads", 1)))
    torch.set_num_interop_threads(1)
    rows = selected_rows(config, root)
    raw: list[dict[str, Any]] = []
    for name, spec in config.get("native_conditions", {}).items():
        loaded = load_condition(
            model_name=config["model_name"],
            model_revision=config.get("model_revision"),
            prompt_template=resolve(root, spec["prompt_template"]),
            adapter_path=resolve(root, spec.get("adapter_path")),
            local_files_only=bool(config.get("local_files_only", False)),
        )
        loaded.model.config.use_cache = True
        raw.extend(benchmark_condition(loaded=loaded, rows=rows, config=config, condition=name))
        del loaded
        torch.cuda.empty_cache()
    for name, spec in config.get("early_exit_conditions", {}).items():
        loaded = load_condition(
            model_name=config["model_name"],
            model_revision=config.get("model_revision"),
            prompt_template=resolve(root, spec["prompt_template"]),
            local_files_only=bool(config.get("local_files_only", False)),
        )
        loaded.model.config.use_cache = True
        source_block = int(spec["source_block"])
        base = base_model(loaded.model)
        original_layers = base.model.layers
        base.model.layers = torch.nn.ModuleList(list(original_layers[: source_block + 1]))
        checkpoint = resolve(root, spec["predictor"])
        assert checkpoint is not None
        predictor = load_predictor(checkpoint, loaded.model.device)
        try:
            raw.extend(
                benchmark_condition(
                    loaded=loaded,
                    rows=rows,
                    config=config,
                    condition=name,
                    source_block=source_block,
                    predictor=predictor,
                )
            )
        finally:
            base.model.layers = original_layers
        del predictor, loaded
        torch.cuda.empty_cache()
    if not raw:
        raise ValueError("Config contains no benchmark conditions")
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "raw_timing.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    summary = summarize(raw)
    with (args.output_dir / "timing_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "contract.json").write_text(
        json.dumps(
            {
                "status": "done",
                "config": str(args.config.resolve()),
                "config_sha256": sha256_file(args.config),
                "samples": len(rows),
                "protocols": config["protocols"],
                "timed_region": "synchronized model.generate calls only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
