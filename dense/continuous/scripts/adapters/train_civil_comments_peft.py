#!/usr/bin/env python3
"""Train Prompt Tuning or Prefix Tuning on fixed Civil Comments splits."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import (
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_optimization.civil_comments import (
    BINARY_OUTPUT_FORMATS,
    binary_target,
    binary_target_text,
    canonical_target,
    canonicalize_labels,
    compute_binary_metrics,
    compute_multilabel_metrics,
    load_manifest,
    parse_binary_prediction,
    parse_prediction,
    parse_prediction_strict,
    optimizer_validation_samples,
    prompt_sha256,
    prompt_init_text,
    prompt_template,
    read_jsonl,
    sha256_file,
    split_file_name,
    use_gradient_checkpointing,
    validate_rows,
)
from prompt_optimization.prefix_cache import install_gemma_prefix_cache_compatibility
from prompt_optimization.scheduler_sweep import (
    CheckpointTracker,
    SCHEDULER_PROFILES,
    lr_multiplier,
)

LOGGER = logging.getLogger("train_civil_comments_peft")
METHODS = ("prompt_tuning", "prefix_tuning")
SELECTION_METRICS = ("samples_f1", "macro_f1", "micro_f1")


@dataclass(frozen=True)
class ExperimentConfig:
    method: str
    model_name: str
    model_revision: str | None
    split_root: str
    train_samples: int
    labels: tuple[str, ...]
    setup: str
    binary_output_format: str
    split_seed: int
    training_seed: int
    num_virtual_tokens: int
    prompt_init: str
    prefix_projection: bool
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    lr_scheduler: str
    warmup_ratio: float
    min_lr_ratio: float
    max_epochs: int
    patience: int
    min_delta: float
    max_new_tokens: int
    selection_metric: str
    checkpoint_epochs: tuple[int, ...]
    gradient_checkpointing: bool
    disable_early_stopping: bool
    skip_test: bool
    evaluate_both_checkpoints: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--split-root",
        type=Path,
        required=True,
        help="Directory containing manifest.json and the fixed JSONL splits.",
    )
    parser.add_argument("--train-samples", type=int, required=True)
    parser.add_argument("--split-seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--num-virtual-tokens", type=int, required=True)
    parser.add_argument("--prompt-init", choices=("text", "random"), default="text")
    parser.add_argument("--prefix-projection", action="store_true")
    parser.add_argument("--binary-output-format", choices=BINARY_OUTPUT_FORMATS)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--lr-scheduler",
        choices=SCHEDULER_PROFILES,
        default="constant",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Fraction of optimizer updates used for warmup when the profile has warmup.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.0,
        help="Final LR divided by base LR for cosine decay profiles.",
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default="samples_f1",
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        nargs="*",
        default=(1, 2, 5, 10, 20, 50, 100),
        help="Milestone epochs at which to retain a full intermediate adapter.",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Run exactly max_epochs; used by scheduler comparisons.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not inspect the common test split during hyperparameter screening.",
    )
    parser.add_argument("--evaluate-both-checkpoints", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_trainable_parameter_state(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_parameter_state(model: Any, state: dict[str, torch.Tensor]) -> None:
    current = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if current.keys() != state.keys():
        raise ValueError("Trainable checkpoint keys differ from the current model")
    with torch.no_grad():
        for name, parameter in current.items():
            checkpoint = state[name]
            if checkpoint.shape != parameter.shape:
                raise ValueError(f"Trainable checkpoint shape differs for {name}")
            parameter.copy_(checkpoint.to(device=parameter.device, dtype=parameter.dtype))


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_history_csv(path: Path, history: list[dict[str, float | int]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def render_prompt(
    tokenizer: Any,
    text: str,
    labels: tuple[str, ...],
    setup: str = "multilabel",
    binary_output_format: str = "safe-toxic",
) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt_template(
                labels,
                setup=setup,
                binary_output_format=binary_output_format,
            ).format(text=text),
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_training_example(
    example: dict[str, Any],
    *,
    tokenizer: Any,
    max_length: int,
    labels: tuple[str, ...],
    setup: str = "multilabel",
    binary_output_format: str = "safe-toxic",
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(
        render_prompt(
            tokenizer,
            example["text"],
            labels,
            setup,
            binary_output_format,
        ),
        add_special_tokens=False,
    )["input_ids"]
    target = (
        binary_target_text(example, output_format=binary_output_format)
        if setup == "binary"
        else canonical_target(example["labels"], allowed_labels=labels)
    )
    target_ids = tokenizer(
        target + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]
    if len(target_ids) >= max_length:
        raise ValueError("max_length is too small to hold the target labels")
    prompt_ids = prompt_ids[-(max_length - len(target_ids)) :]
    return {
        "input_ids": prompt_ids + target_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


class CausalClassificationCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(example["input_ids"]) for example in examples)
        fields: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for example in examples:
            padding = width - len(example["input_ids"])
            fields["input_ids"].append(example["input_ids"] + [self.pad_token_id] * padding)
            fields["attention_mask"].append(example["attention_mask"] + [0] * padding)
            fields["labels"].append(example["labels"] + [-100] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in fields.items()}


def build_peft_config(config: ExperimentConfig) -> PromptTuningConfig | PrefixTuningConfig:
    if config.method == "prompt_tuning":
        init = PromptTuningInit.TEXT if config.prompt_init == "text" else PromptTuningInit.RANDOM
        return PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=config.num_virtual_tokens,
            prompt_tuning_init=init,
            prompt_tuning_init_text=(
                prompt_init_text(
                    config.labels,
                    setup=config.setup,
                    binary_output_format=config.binary_output_format,
                )
                if init == PromptTuningInit.TEXT
                else None
            ),
            tokenizer_name_or_path=(config.model_name if init == PromptTuningInit.TEXT else None),
        )
    if config.method == "prefix_tuning":
        return PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=config.num_virtual_tokens,
            prefix_projection=config.prefix_projection,
        )
    raise ValueError(f"Unsupported PEFT method: {config.method}")


def predict(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    *,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    labels: tuple[str, ...],
    setup: str = "multilabel",
    binary_output_format: str = "safe-toxic",
) -> tuple[list[Any], list[Any | None], list[str]]:
    targets: list[Any] = []
    predictions: list[Any | None] = []
    raw_outputs: list[str] = []
    model.eval()
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
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
                output_ids = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            continuation = output_ids[:, encoded["input_ids"].shape[1] :]
            generated = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            raw_outputs.extend(generated)
            if setup == "binary":
                predictions.extend(
                    parse_binary_prediction(
                        text,
                        output_format=binary_output_format,
                    )
                    for text in generated
                )
                targets.extend(str(row) for row in batch["binary_label"])
            else:
                predictions.extend(
                    parse_prediction(text, allowed_labels=labels) for text in generated
                )
                targets.extend(
                    canonicalize_labels(row, allowed_labels=labels)
                    for row in batch["labels"]
                )
    finally:
        tokenizer.padding_side = old_padding_side
    return targets, predictions, raw_outputs


def evaluate(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    *,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    labels: tuple[str, ...],
    setup: str = "multilabel",
    binary_output_format: str = "safe-toxic",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    targets, predictions, raw_outputs = predict(
        model,
        tokenizer,
        dataset,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        labels=labels,
        setup=setup,
        binary_output_format=binary_output_format,
    )
    metrics = (
        compute_binary_metrics(targets, predictions)
        if setup == "binary"
        else compute_multilabel_metrics(targets, predictions, labels=labels)
    )
    strict_predictions = (
        predictions
        if setup == "binary"
        else tuple(
            parse_prediction_strict(text, allowed_labels=labels)
            for text in raw_outputs
        )
    )
    if setup == "multilabel":
        strict_metrics = compute_multilabel_metrics(
            targets, strict_predictions, labels=labels
        )
        metrics.update({f"set_{key}": value for key, value in metrics.items()})
        metrics.update({f"strict_{key}": value for key, value in strict_metrics.items()})
    rows = [
        {
            "id": example_id,
            "target": target if setup == "binary" else list(target),
            "target_text": (
                (
                    "Yes" if target == "toxic" else "No"
                )
                if setup == "binary" and binary_output_format == "yes-no"
                else target
                if setup == "binary"
                else canonical_target(target, allowed_labels=labels)
            ),
            "prediction": (
                prediction
                if setup == "binary"
                else (list(prediction) if prediction is not None else None)
            ),
            "strict_prediction": (
                strict_prediction
                if setup == "binary"
                else (
                    list(strict_prediction)
                    if strict_prediction is not None
                    else None
                )
            ),
            "raw_output": raw_output,
            "valid_format": prediction is not None,
            "valid_strict_format": strict_prediction is not None,
            "exact_match": prediction == target,
            "strict_exact_match": strict_prediction == target,
        }
        for example_id, target, prediction, strict_prediction, raw_output in zip(
            dataset["id"],
            targets,
            predictions,
            strict_predictions,
            raw_outputs,
            strict=True,
        )
    ]
    return metrics, rows


def evaluate_token_loss(model: Any, loader: DataLoader) -> float:
    """Compute teacher-forced CE weighted by the number of target tokens."""
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {name: tensor.to(model.device) for name, tensor in batch.items()}
            outputs = model(**batch)
            token_count = int((batch["labels"][:, 1:] != -100).sum().item())
            total_loss += float(outputs.loss.item()) * token_count
            total_tokens += token_count
    if total_tokens == 0:
        raise RuntimeError("Validation dataset has no supervised target tokens")
    return total_loss / total_tokens


def trainable_parameter_stats(
    model: Any,
    initial_parameters: dict[str, torch.Tensor],
) -> dict[str, float | int]:
    total_norm_sq = torch.zeros((), device=model.device, dtype=torch.float32)
    initial_norm_sq = torch.zeros((), device=model.device, dtype=torch.float32)
    delta_norm_sq = torch.zeros((), device=model.device, dtype=torch.float32)
    dot = torch.zeros((), device=model.device, dtype=torch.float32)
    count = 0
    with torch.inference_mode():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            current = parameter.detach().float()
            initial = initial_parameters[name].float()
            total_norm_sq += current.square().sum()
            initial_norm_sq += initial.square().sum()
            delta_norm_sq += (current - initial).square().sum()
            dot += (current * initial).sum()
            count += parameter.numel()
    denominator = (total_norm_sq.sqrt() * initial_norm_sq.sqrt()).clamp_min(1e-12)
    return {
        "trainable_parameters": count,
        "parameter_l2": float(total_norm_sq.sqrt().item()),
        "delta_from_initial_l2": float(delta_norm_sq.sqrt().item()),
        "cosine_to_initial": float((dot / denominator).item()),
    }


def load_fixed_datasets(
    config: ExperimentConfig,
) -> tuple[Dataset, Dataset, Dataset, dict[str, Any]]:
    split_root = Path(config.split_root)
    manifest = load_manifest(split_root)
    if tuple(manifest["labels"]) != config.labels:
        raise ValueError("Experiment labels differ from the dataset manifest")
    manifest_binary_output_format = str(
        manifest.get("contract", {}).get("binary_output_format", "safe-toxic")
    )
    if config.binary_output_format != manifest_binary_output_format:
        raise ValueError("Experiment binary output format differs from the dataset manifest")
    validation_samples = optimizer_validation_samples(manifest, config.train_samples)
    train_name = split_file_name(
        "optimizer_train",
        split_seed=config.split_seed,
        train_samples=config.train_samples,
    )
    validation_name = split_file_name(
        "optimizer_val",
        split_seed=config.split_seed,
        train_samples=validation_samples,
    )
    test_name = split_file_name("test")
    names = (train_name, validation_name, test_name)
    rows = [read_jsonl(split_root / name) for name in names]
    expected_sizes = tuple(
        int(manifest["splits"][Path(name).stem]["size"]) for name in names
    )
    if expected_sizes[0] != config.train_samples:
        raise ValueError("Train split size differs from train_samples")
    for split_rows, expected_size in zip(rows, expected_sizes, strict=True):
        validate_rows(
            split_rows,
            expected_size=expected_size,
            allowed_labels=config.labels,
            setup=config.setup,
        )
    train_ids = {row["id"] for row in rows[0]}
    validation_ids = {row["id"] for row in rows[1]}
    test_ids = {row["id"] for row in rows[2]}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Fixed train/validation/test splits overlap by id")
    split_contract = {
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["dataset_revision"],
        "manifest_sha256": sha256_file(split_root / "manifest.json"),
        "prompt_sha256": prompt_sha256(
            config.labels,
            setup=config.setup,
            binary_output_format=config.binary_output_format,
        ),
        "setup": config.setup,
        "binary_output_format": config.binary_output_format,
        "threshold": manifest["contract"]["threshold"],
        "labels": list(config.labels),
        "files": {
            name: {
                "sha256": sha256_file(split_root / name),
                "size": len(split_rows),
                "ids": [row["id"] for row in split_rows],
            }
            for name, split_rows in zip(names, rows, strict=True)
        },
    }
    return (
        Dataset.from_list(rows[0]),
        Dataset.from_list(rows[1]),
        Dataset.from_list(rows[2]),
        split_contract,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; this experiment must not run on CPU")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly on the shared server")
    if "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("Expose exactly one GPU per Civil Comments training process")
    disable_cudnn_sdpa = os.environ.get("CIVIL_COMMENTS_DISABLE_CUDNN_SDPA") == "1"
    if disable_cudnn_sdpa:
        torch.backends.cuda.enable_cudnn_sdp(False)
    checkpoint_epochs = tuple(sorted(set(args.checkpoint_epochs)))
    if any(epoch <= 0 for epoch in checkpoint_epochs):
        raise ValueError("checkpoint epochs must be positive")
    if args.num_virtual_tokens <= 0:
        raise ValueError("num_virtual_tokens must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if args.lr_scheduler in {"linear_warmup_constant", "linear_warmup_cosine", "smooth_cosine"} and args.warmup_ratio <= 0:
        raise ValueError(f"{args.lr_scheduler} requires warmup_ratio > 0")
    if args.skip_test and args.evaluate_both_checkpoints:
        raise ValueError("evaluate_both_checkpoints cannot be combined with skip_test")
    dataset_manifest = load_manifest(args.split_root.resolve())
    labels = tuple(dataset_manifest["labels"])
    setup = str(
        dataset_manifest.get(
            "setup", dataset_manifest.get("contract", {}).get("setup", "multilabel")
        )
    )
    manifest_binary_output_format = str(
        dataset_manifest.get("contract", {}).get(
            "binary_output_format",
            "safe-toxic",
        )
    )
    binary_output_format = args.binary_output_format or manifest_binary_output_format
    if binary_output_format != manifest_binary_output_format:
        raise ValueError(
            "--binary-output-format differs from the fixed dataset manifest contract"
        )

    config = ExperimentConfig(
        method=args.method,
        model_name=args.model_name,
        model_revision=args.model_revision,
        split_root=str(args.split_root.resolve()),
        labels=labels,
        setup=setup,
        binary_output_format=binary_output_format,
        train_samples=args.train_samples,
        split_seed=args.split_seed,
        training_seed=args.training_seed,
        num_virtual_tokens=args.num_virtual_tokens,
        prompt_init=args.prompt_init,
        prefix_projection=args.prefix_projection,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        max_new_tokens=args.max_new_tokens,
        selection_metric=args.selection_metric,
        checkpoint_epochs=checkpoint_epochs,
        gradient_checkpointing=use_gradient_checkpointing(args.method),
        disable_early_stopping=args.disable_early_stopping,
        skip_test=args.skip_test,
        evaluate_both_checkpoints=args.evaluate_both_checkpoints,
    )
    set_seed(config.training_seed)
    out_dir = args.output_dir
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "train.log", encoding="utf-8"),
        ],
    )
    write_json(out_dir / "config.json", asdict(config))
    (out_dir / "prompt.txt").write_text(
        prompt_template(
            config.labels,
            setup=config.setup,
            binary_output_format=config.binary_output_format,
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(out_dir / "status.json", {"status": "running", "started_at": time.time()})

    try:
        train_dataset, validation_dataset, test_dataset, split_contract = load_fixed_datasets(
            config
        )
        write_json(out_dir / "split_contract.json", split_contract)
        write_json(
            out_dir / "environment.json",
            {
                "git_revision": git_revision(),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "peft": importlib.metadata.version("peft"),
                "datasets": importlib.metadata.version("datasets"),
                "cuda": torch.version.cuda,
                "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "cudnn_sdpa_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
            },
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            revision=config.model_revision,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            revision=config.model_revision,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        model.config.use_cache = False
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model = get_peft_model(model, build_peft_config(config))
        prefix_cache_compatibility = None
        if config.method == "prefix_tuning":
            prefix_cache_compatibility = install_gemma_prefix_cache_compatibility(
                model,
                num_virtual_tokens=config.num_virtual_tokens,
            )
        model.print_trainable_parameters()
        model.save_pretrained(out_dir / "initial_adapter")
        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        encoded_train = train_dataset.map(
            lambda example: encode_training_example(
                example,
                tokenizer=tokenizer,
                max_length=config.max_length,
                labels=config.labels,
                setup=config.setup,
                binary_output_format=config.binary_output_format,
            ),
            remove_columns=train_dataset.column_names,
        )
        loader = DataLoader(
            encoded_train,
            batch_size=config.train_batch_size,
            shuffle=True,
            collate_fn=CausalClassificationCollator(tokenizer.pad_token_id),
            generator=torch.Generator().manual_seed(config.training_seed),
            num_workers=0,
        )
        encoded_validation = validation_dataset.map(
            lambda example: encode_training_example(
                example,
                tokenizer=tokenizer,
                max_length=config.max_length,
                labels=config.labels,
                setup=config.setup,
                binary_output_format=config.binary_output_format,
            ),
            remove_columns=validation_dataset.column_names,
        )
        validation_loader = DataLoader(
            encoded_validation,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=CausalClassificationCollator(tokenizer.pad_token_id),
            num_workers=0,
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        updates_per_epoch = (
            len(loader) + config.gradient_accumulation_steps - 1
        ) // config.gradient_accumulation_steps
        total_optimizer_steps = config.max_epochs * updates_per_epoch
        warmup_steps = min(
            round(total_optimizer_steps * config.warmup_ratio),
            total_optimizer_steps - 1,
        )
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda optimizer_step: lr_multiplier(
                config.lr_scheduler,
                min(optimizer_step, total_optimizer_steps),
                total_optimizer_steps,
                warmup_steps,
                config.min_lr_ratio,
            ),
        )

        history: list[dict[str, float | int]] = []
        tracker = CheckpointTracker()
        best_f1_state: dict[str, torch.Tensor] | None = None
        best_loss_state: dict[str, torch.Tensor] | None = None
        early_stop_best = float("-inf")
        stale_epochs = 0
        optimizer_steps = 0
        stopped_early = False
        started_at = time.time()
        for epoch in range(1, config.max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss_sum = 0.0
            epoch_target_tokens = 0
            for step, batch in enumerate(loader, start=1):
                batch = {name: tensor.to(model.device) for name, tensor in batch.items()}
                raw_loss = model(**batch).loss
                group_start = (
                    (step - 1) // config.gradient_accumulation_steps
                ) * config.gradient_accumulation_steps + 1
                group_size = min(
                    config.gradient_accumulation_steps,
                    len(loader) - group_start + 1,
                )
                (raw_loss / group_size).backward()
                token_count = int((batch["labels"][:, 1:] != -100).sum().item())
                epoch_loss_sum += raw_loss.item() * token_count
                epoch_target_tokens += token_count
                if step % config.gradient_accumulation_steps == 0 or step == len(loader):
                    optimizer.step()
                    scheduler.step()
                    optimizer_steps += 1
                    optimizer.zero_grad(set_to_none=True)

            validation_loss = evaluate_token_loss(model, validation_loader)
            validation_metrics, _ = evaluate(
                model,
                tokenizer,
                validation_dataset,
                batch_size=config.eval_batch_size,
                max_length=config.max_length,
                max_new_tokens=config.max_new_tokens,
                labels=config.labels,
                setup=config.setup,
                binary_output_format=config.binary_output_format,
            )
            parameter_stats = trainable_parameter_stats(model, initial_parameters)
            record: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": epoch_loss_sum / epoch_target_tokens,
                "val_loss": validation_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "optimizer_steps": optimizer_steps,
                "elapsed_seconds": time.time() - started_at,
                **{f"val_{key}": value for key, value in validation_metrics.items()},
                **parameter_stats,
            }
            history.append(record)
            write_json(out_dir / "history.json", history)
            write_history_csv(out_dir / "history.csv", history)
            LOGGER.info("epoch=%d metrics=%s", epoch, record)

            if epoch in config.checkpoint_epochs:
                model.save_pretrained(out_dir / "intermediate_adapters" / f"epoch_{epoch:04d}")

            score = validation_metrics[config.selection_metric]
            update_f1, update_loss = tracker.observe(
                epoch=epoch, samples_f1=score, val_loss=validation_loss
            )
            if update_f1 or update_loss:
                checkpoint_state = capture_trainable_parameter_state(model)
                if update_f1:
                    best_f1_state = checkpoint_state
                    model.save_pretrained(out_dir / "best_f1_adapter")
                    model.save_pretrained(out_dir / "best_adapter")
                if update_loss:
                    best_loss_state = checkpoint_state
                    model.save_pretrained(out_dir / "best_loss_adapter")
            write_json(
                out_dir / "checkpoint_selection.json",
                {
                    "selection_metric": config.selection_metric,
                    "best_f1_epoch": tracker.best_f1_epoch,
                    "best_f1": tracker.best_f1,
                    "loss_at_best_f1": tracker.loss_at_best_f1,
                    "best_loss_epoch": tracker.best_loss_epoch,
                    "best_loss": tracker.best_loss,
                    "f1_at_best_loss": tracker.f1_at_best_loss,
                },
            )

            if not config.disable_early_stopping:
                if score > early_stop_best + config.min_delta:
                    early_stop_best = score
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if stale_epochs >= config.patience:
                    stopped_early = True
                    LOGGER.info("Early stopping after epoch %d", epoch)
                    break

        if best_f1_state is None or best_loss_state is None:
            raise RuntimeError("Training finished without both required checkpoints")
        test_metrics_by_checkpoint: dict[str, dict[str, float]] = {}
        if not config.skip_test:
            restore_trainable_parameter_state(model, best_f1_state)
            test_metrics, predictions = evaluate(
                model,
                tokenizer,
                test_dataset,
                batch_size=config.eval_batch_size,
                max_length=config.max_length,
                max_new_tokens=config.max_new_tokens,
                labels=config.labels,
                setup=config.setup,
                binary_output_format=config.binary_output_format,
            )
            write_json(out_dir / "test_predictions_best_f1.json", predictions)
            write_json(out_dir / "test_predictions.json", predictions)
            test_metrics_by_checkpoint["best_f1"] = test_metrics
            if config.evaluate_both_checkpoints:
                restore_trainable_parameter_state(model, best_loss_state)
                loss_metrics, loss_predictions = evaluate(
                    model,
                    tokenizer,
                    test_dataset,
                    batch_size=config.eval_batch_size,
                    max_length=config.max_length,
                    max_new_tokens=config.max_new_tokens,
                    labels=config.labels,
                    setup=config.setup,
                    binary_output_format=config.binary_output_format,
                )
                write_json(out_dir / "test_predictions_best_loss.json", loss_predictions)
                test_metrics_by_checkpoint["best_loss"] = loss_metrics
        summary = {
            "status": "done",
            "method": config.method,
            "setup": config.setup,
            "binary_output_format": config.binary_output_format,
            "train_samples": config.train_samples,
            "split_seed": config.split_seed,
            "training_seed": config.training_seed,
            "num_virtual_tokens": config.num_virtual_tokens,
            "selection_metric": config.selection_metric,
            "lr_scheduler": config.lr_scheduler,
            "warmup_ratio": config.warmup_ratio,
            "min_lr_ratio": config.min_lr_ratio,
            "best_epoch": tracker.best_f1_epoch,
            "best_validation_score": tracker.best_f1,
            "validation_loss_at_best_f1": tracker.loss_at_best_f1,
            "best_loss_epoch": tracker.best_loss_epoch,
            "best_validation_loss": tracker.best_loss,
            "validation_score_at_best_loss": tracker.f1_at_best_loss,
            "test": test_metrics_by_checkpoint.get("best_f1"),
            "test_by_checkpoint": test_metrics_by_checkpoint,
            "epochs_completed": len(history),
            "optimizer_steps": optimizer_steps,
            "planned_optimizer_steps": total_optimizer_steps,
            "warmup_steps": warmup_steps,
            "stopped_early": stopped_early,
            "elapsed_seconds": time.time() - started_at,
            "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "prefix_cache_compatibility": prefix_cache_compatibility,
        }
        write_json(out_dir / "summary.json", summary)
        write_json(out_dir / "status.json", summary)
        LOGGER.info("Final test metrics by checkpoint: %s", test_metrics_by_checkpoint)
    except BaseException as error:
        write_json(
            out_dir / "status.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
