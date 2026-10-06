#!/usr/bin/env python3
"""Extract exact decoder-block anchor states for a shared-SAE shift audit."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from prompt_optimization.civil_comments import canonicalize_labels, sha256_file
from prompt_optimization.frozen_text_prompt import load_prompt_template
from prompt_optimization.gemma_scope import (
    hidden_from_decoder_output,
    resolve_gemma_decoder_layer,
)
from prompt_optimization.prefix_cache import install_gemma_prefix_cache_compatibility

from scripts.sae.evaluate_civil_comments_sae_fidelity import (
    encode_prompts,
    sample_ids_from_manifest,
    select_dataset_by_ids,
)
from scripts.probing.extract_civil_comments_activations import load_fixed_datasets, load_model


CONDITIONS = ("manual", "prompt", "prefix", "gepa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument(
        "--frozen-prompt-file",
        type=Path,
        help=(
            "Prompt template with one {text} placeholder. Required for --condition=gepa "
            "and forbidden for the other conditions."
        ),
    )
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("probe_train", "probe_val", "test", "intervention_val"),
        default="test",
    )
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-input-length", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument(
        "--storage-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def extract_states(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    *,
    layers: tuple[int, ...],
    labels: tuple[str, ...],
    setup: str,
    max_input_length: int,
    batch_size: int,
    hidden_offset: int,
    storage_dtype: torch.dtype,
    frozen_prompt_template: str | None,
) -> torch.Tensor:
    captures: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(layer: int) -> Any:
        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if layer in captures:
                raise RuntimeError(f"Decoder block {layer} ran more than once in one forward")
            captures[layer] = hidden_from_decoder_output(output).detach()

        return capture

    for layer in layers:
        handles.append(resolve_gemma_decoder_layer(model, layer).register_forward_hook(make_hook(layer)))

    chunks: list[torch.Tensor] = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    model.eval()
    try:
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]
            captures.clear()
            encoded = encode_prompts(
                tokenizer,
                batch,
                labels=labels,
                setup=setup,
                max_length=max_input_length,
                device=model.device,
                frozen_prompt_template=frozen_prompt_template,
            )
            with torch.inference_mode():
                model(**encoded, return_dict=True, use_cache=False)
            if set(captures) != set(layers):
                raise RuntimeError(
                    f"Expected captures for {layers}, observed {sorted(captures)}"
                )
            positions = (
                encoded["attention_mask"].sum(dim=1, dtype=torch.long)
                + hidden_offset
                - 1
            )
            batch_indices = torch.arange(len(positions), device=model.device)
            selected: list[torch.Tensor] = []
            for layer in layers:
                hidden = captures[layer]
                expected_length = encoded["attention_mask"].shape[1] + hidden_offset
                if hidden.shape[1] != expected_length:
                    raise RuntimeError(
                        f"Layer {layer} sequence length {hidden.shape[1]} != "
                        f"expected {expected_length}"
                    )
                selected.append(hidden[batch_indices, positions.to(hidden.device)])
            chunks.append(
                torch.stack(selected, dim=1).to(device="cpu", dtype=storage_dtype)
            )
    finally:
        tokenizer.padding_side = old_padding_side
        for handle in handles:
            handle.remove()
    return torch.cat(chunks, dim=0).contiguous()


def main() -> None:
    args = parse_args()
    if (args.condition == "gepa") != (args.frozen_prompt_file is not None):
        raise ValueError(
            "--frozen-prompt-file is required exactly when --condition=gepa"
        )
    layers = tuple(args.layers)
    if not layers or min(layers) < 0 or len(set(layers)) != len(layers):
        raise ValueError("--layers must contain unique non-negative indices")
    if layers != tuple(sorted(layers)):
        raise ValueError("--layers must be sorted")
    for name in ("batch_size", "cpu_threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.max_input_length is not None and args.max_input_length <= 0:
        raise ValueError("--max-input-length must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Anchor extraction requires CUDA")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(args.cpu_threads)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    tensor_path = args.output_dir / "states.safetensors"
    metadata_path = args.output_dir / "metadata.json"
    if any(path.exists() for path in (summary_path, tensor_path, metadata_path)):
        raise FileExistsError(f"Refusing to overwrite artifacts in {args.output_dir}")

    config = read_json(args.reference_run / "config.json")
    datasets, split_contract, labels, setup = load_fixed_datasets(
        config,
        splits=(args.split,),
    )
    requested_ids = sample_ids_from_manifest(args.sample_manifest, args.split)
    if args.max_samples is not None:
        requested_ids = requested_ids[: args.max_samples]
    dataset = select_dataset_by_ids(datasets[args.split], requested_ids)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        revision=config.get("model_revision"),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    frozen_prompt_template = (
        load_prompt_template(args.frozen_prompt_file)
        if args.frozen_prompt_file is not None
        else None
    )
    loader_condition = "adapted" if args.condition in {"prompt", "prefix"} else "manual"
    model, hidden_offset, adapter_path, peft_type, seed_equivalence = load_model(
        config,
        condition=loader_condition,
        reference_run=args.reference_run,
        seed_adapter_run=None,
        equivalent_seed_adapter_runs=(),
    )
    prefix_compatibility = None
    if peft_type == "PREFIX_TUNING":
        prefix_compatibility = install_gemma_prefix_cache_compatibility(
            model,
            num_virtual_tokens=int(config["num_virtual_tokens"]),
        )

    max_input_length = int(args.max_input_length or config["max_length"])
    storage_dtype = torch.bfloat16 if args.storage_dtype == "bfloat16" else torch.float32
    torch.cuda.reset_peak_memory_stats()
    started_at = time.time()
    states = extract_states(
        model,
        tokenizer,
        dataset,
        layers=layers,
        labels=labels,
        setup=setup,
        max_input_length=max_input_length,
        batch_size=args.batch_size,
        hidden_offset=hidden_offset,
        storage_dtype=storage_dtype,
        frozen_prompt_template=frozen_prompt_template,
    )
    peak_cuda_memory = int(torch.cuda.max_memory_allocated())
    save_file(
        {"states": states},
        str(tensor_path),
        metadata={
            "condition": args.condition,
            "split": args.split,
            "layers": json.dumps(layers),
            "shape": json.dumps(list(states.shape)),
            "anchor": "last_common_textual_prompt_token",
        },
    )
    rows = [
        {
            "id": str(example_id),
            "labels": list(canonicalize_labels(row_labels, allowed_labels=labels)),
        }
        for example_id, row_labels in zip(dataset["id"], dataset["labels"], strict=True)
    ]
    write_json(metadata_path, {"rows": rows})
    write_json(
        summary_path,
        {
            "status": "done",
            "condition": args.condition,
            "frozen_prompt_file": (
                str(args.frozen_prompt_file.resolve())
                if args.frozen_prompt_file is not None
                else None
            ),
            "frozen_prompt_sha256": (
                sha256_file(args.frozen_prompt_file)
                if args.frozen_prompt_file is not None
                else None
            ),
            "split": args.split,
            "samples": len(dataset),
            "layers": list(layers),
            "shape": list(states.shape),
            "storage_dtype": args.storage_dtype,
            "anchor": "last_common_textual_prompt_token",
            "hook_source": "decoder_block_output_before_final_norm",
            "reference_run": str(args.reference_run.resolve()),
            "adapter_path": adapter_path,
            "peft_type": peft_type,
            "prepended_virtual_tokens": hidden_offset,
            "seed_adapter_equivalence": seed_equivalence,
            "prefix_cache_compatibility": prefix_compatibility,
            "max_input_length": max_input_length,
            "sample_selection": {
                "manifest": str(args.sample_manifest.resolve()),
                "ids": requested_ids,
            },
            "states_file": {
                "path": str(tensor_path.resolve()),
                "sha256": sha256_file(tensor_path),
                "size_bytes": tensor_path.stat().st_size,
            },
            "metadata_file": {
                "path": str(metadata_path.resolve()),
                "sha256": sha256_file(metadata_path),
                "size_bytes": metadata_path.stat().st_size,
            },
            "split_contract": split_contract[args.split],
            "elapsed_seconds": time.time() - started_at,
            "resource_usage": {"peak_cuda_memory_bytes": peak_cuda_memory},
            "environment": {
                "git_revision": git_revision(),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "cuda": torch.version.cuda,
                "visible_devices": visible,
            },
        },
    )
    del model, states
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"status": "done", "condition": args.condition, "samples": len(dataset)}))


if __name__ == "__main__":
    main()
