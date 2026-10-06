#!/usr/bin/env python3
"""Extract last-prompt-token hidden states on fixed Civil Comments probe splits."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_optimization.civil_comments import (
    canonicalize_labels,
    load_manifest,
    prepended_virtual_tokens,
    read_jsonl,
    sha256_directory,
    sha256_file,
    split_file_name,
    validate_rows,
)
from prompt_optimization.probing import select_last_prompt_states

from scripts.adapters.train_civil_comments_peft import evaluate, render_prompt

CONDITIONS = ("manual", "seed", "adapted")
SPLITS = ("probe_train", "probe_val", "test")
AVAILABLE_SPLITS = (*SPLITS, "intervention_val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument(
        "--seed-adapter-run",
        type=Path,
        help="Run containing initial_adapter; defaults to reference-run.",
    )
    parser.add_argument(
        "--equivalent-seed-adapter-runs",
        type=Path,
        nargs="*",
        default=(),
        help="Runs whose initial_adapter directories must hash identically before reuse.",
    )
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--storage-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--evaluate-output", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def peft_type_name(model: Any) -> str | None:
    if not hasattr(model, "peft_config"):
        return None
    peft_type = model.peft_config["default"].peft_type
    return str(getattr(peft_type, "value", peft_type))


def load_model(
    config: dict[str, Any],
    *,
    condition: str,
    reference_run: Path,
    seed_adapter_run: Path | None,
    equivalent_seed_adapter_runs: tuple[Path, ...],
) -> tuple[Any, int, str | None, str | None, dict[str, Any] | None]:
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        revision=config.get("model_revision"),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    base_model.config.use_cache = False
    if condition == "manual":
        return base_model, 0, None, None, None
    adapter_path = (
        reference_run / "best_adapter"
        if condition == "adapted"
        else (seed_adapter_run or reference_run) / "initial_adapter"
    )
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
    seed_equivalence = None
    if condition == "seed":
        candidate_paths = sorted(
            {
                adapter_path.resolve(),
                *(run.resolve() / "initial_adapter" for run in equivalent_seed_adapter_runs),
            }
        )
        fingerprints = {str(path): sha256_directory(path) for path in candidate_paths}
        if len(set(fingerprints.values())) != 1:
            raise ValueError("Initial adapters expected to be reusable are not byte-identical")
        seed_equivalence = {
            "verified": True,
            "sha256": next(iter(fingerprints.values())),
            "adapter_directories": list(fingerprints),
        }
    model = PeftModel.from_pretrained(base_model, adapter_path)
    num_virtual_tokens = int(model.peft_config["default"].num_virtual_tokens)
    current_peft_type = peft_type_name(model)
    hidden_state_offset = prepended_virtual_tokens(current_peft_type, num_virtual_tokens)
    return (
        model,
        hidden_state_offset,
        str(adapter_path.resolve()),
        current_peft_type,
        seed_equivalence,
    )


def load_fixed_datasets(
    config: dict[str, Any],
    *,
    splits: tuple[str, ...] = SPLITS,
) -> tuple[dict[str, Dataset], dict[str, Any], tuple[str, ...], str]:
    split_root = Path(config["split_root"])
    manifest = load_manifest(split_root)
    labels = tuple(manifest["labels"])
    configured_labels = tuple(config.get("labels", labels))
    if configured_labels != labels:
        raise ValueError("Run labels differ from the dataset manifest")
    setup = str(
        manifest.get("setup", manifest.get("contract", {}).get("setup", "multilabel"))
    )
    if str(config.get("setup", setup)) != setup:
        raise ValueError("Run setup differs from the dataset manifest")
    binary_output_format = str(manifest["contract"].get("binary_output_format", "safe-toxic"))
    if str(config.get("binary_output_format", "safe-toxic")) != binary_output_format:
        raise ValueError("Run binary output format differs from the dataset manifest")
    datasets: dict[str, Dataset] = {}
    contract: dict[str, Any] = {}
    manifest_contract = manifest["contract"]
    expected_sizes = {
        "probe_train": int(manifest_contract["probe_train_size"]),
        "probe_val": int(manifest_contract["probe_val_size"]),
        "test": int(manifest_contract["test_size"]),
    }
    if "intervention_val" in splits:
        expected_sizes["intervention_val"] = int(
            manifest_contract["intervention_val_size"]
        )
    unknown_splits = set(splits) - set(AVAILABLE_SPLITS)
    if unknown_splits:
        raise ValueError(f"Unknown fixed splits: {sorted(unknown_splits)}")
    if not splits:
        raise ValueError("At least one fixed split is required")
    for split in splits:
        filename = split_file_name(split)
        path = split_root / filename
        rows = read_jsonl(path)
        validate_rows(
            rows,
            expected_size=expected_sizes[split],
            allowed_labels=labels,
            setup=setup,
        )
        datasets[split] = Dataset.from_list(rows)
        contract[split] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size": len(rows),
            "ids": [row["id"] for row in rows],
        }
    split_ids = {name: set(payload["ids"]) for name, payload in contract.items()}
    split_names = tuple(split_ids)
    if any(
        split_ids[left] & split_ids[right]
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    ):
        raise ValueError("Requested fixed splits overlap by id")
    return datasets, contract, labels, setup


def extract_split(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    *,
    batch_size: int,
    max_length: int,
    hidden_state_offset: int,
    storage_dtype: torch.dtype,
    labels: tuple[str, ...],
    setup: str,
    binary_output_format: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    model.eval()
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    chunks: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    try:
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]
            prompts = [
                render_prompt(
                    tokenizer,
                    text,
                    labels,
                    setup,
                    binary_output_format,
                )
                for text in batch["text"]
            ]
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                outputs = model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            selected = select_last_prompt_states(
                outputs.hidden_states,
                encoded["attention_mask"],
                num_virtual_tokens=hidden_state_offset,
            )
            chunks.append(selected.to(device="cpu", dtype=storage_dtype))
            for index, (example_id, row_labels) in enumerate(
                zip(batch["id"], batch["labels"], strict=True)
            ):
                row = {
                    "id": example_id,
                    "labels": list(
                        canonicalize_labels(row_labels, allowed_labels=labels)
                    ),
                }
                if setup == "binary":
                    row["binary_label"] = str(batch["binary_label"][index])
                rows.append(row)
    finally:
        tokenizer.padding_side = old_padding_side
    return torch.cat(chunks, dim=0).contiguous(), rows


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.evaluation_batch_size is not None and args.evaluation_batch_size <= 0:
        raise ValueError("--evaluation-batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; extraction must not run on CPU")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.condition}_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")

    config = read_json(args.reference_run / "config.json")
    datasets, split_contract, labels, setup = load_fixed_datasets(config)
    binary_output_format = str(config.get("binary_output_format", "safe-toxic"))
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], revision=config.get("model_revision")
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, hidden_state_offset, adapter_path, current_peft_type, seed_equivalence = load_model(
        config,
        condition=args.condition,
        reference_run=args.reference_run,
        seed_adapter_run=args.seed_adapter_run,
        equivalent_seed_adapter_runs=tuple(args.equivalent_seed_adapter_runs),
    )
    storage_dtype = torch.bfloat16 if args.storage_dtype == "bfloat16" else torch.float32
    started_at = time.time()
    split_summaries: dict[str, Any] = {}
    for split in args.splits:
        dataset = datasets[split]
        if args.max_samples is not None:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        tensor_path = args.output_dir / f"{args.condition}_{split}.safetensors"
        metadata_path = args.output_dir / f"{args.condition}_{split}.json"
        if tensor_path.exists() or metadata_path.exists():
            raise FileExistsError(f"Refusing to overwrite activation split: {split}")
        states, rows = extract_split(
            model,
            tokenizer,
            dataset,
            batch_size=args.batch_size,
            max_length=int(config["max_length"]),
            hidden_state_offset=hidden_state_offset,
            storage_dtype=storage_dtype,
            labels=labels,
            setup=setup,
            binary_output_format=binary_output_format,
        )
        save_file(
            {"states": states},
            tensor_path,
            metadata={
                "condition": args.condition,
                "split": split,
                "shape": json.dumps(list(states.shape)),
                "peft_type": current_peft_type or "none",
            },
        )
        write_json(metadata_path, {"rows": rows})
        split_summaries[split] = {
            "samples": len(rows),
            "shape": list(states.shape),
            "tensor": tensor_path.name,
            "metadata": metadata_path.name,
        }

    output_metrics = None
    if args.evaluate_output:
        output_metrics, predictions = evaluate(
            model,
            tokenizer,
            datasets["test"],
            batch_size=(
                args.evaluation_batch_size
                if args.evaluation_batch_size is not None
                else int(config["eval_batch_size"])
            ),
            max_length=int(config["max_length"]),
            max_new_tokens=int(config["max_new_tokens"]),
            labels=labels,
            setup=setup,
            binary_output_format=binary_output_format,
        )
        write_json(args.output_dir / f"{args.condition}_test_predictions.json", predictions)

    write_json(
        summary_path,
        {
            "status": "done",
            "condition": args.condition,
            "labels": list(labels),
            "setup": setup,
            "binary_output_format": binary_output_format,
            "reference_run": str(args.reference_run.resolve()),
            "adapter_path": adapter_path,
            "peft_type": current_peft_type,
            "num_virtual_tokens": int(config["num_virtual_tokens"]) if adapter_path else 0,
            "hidden_state_position_offset": hidden_state_offset,
            "seed_adapter_equivalence": seed_equivalence,
            "splits": split_summaries,
            "split_contract": split_contract,
            "output_test": output_metrics,
            "elapsed_seconds": time.time() - started_at,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "peft": importlib.metadata.version("peft"),
                "cuda": torch.version.cuda,
                "visible_devices": visible,
            },
        },
    )


if __name__ == "__main__":
    main()
