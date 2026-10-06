#!/usr/bin/env python3
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import torch
from peft import get_peft_model
from transformers import get_scheduler

from mrd.jsonl import read_jsonl
from mrd.chat import CHAT_DATE, REASONING_EFFORT, encode_final_content
from mrd.models.loading import load_causal_lm, model_backends
from mrd.models.registry import MODEL_SPECS
from mrd.peft_training import PeftTrainConfig, _build_peft_config, _encode
from mrd.training import checkpoint_native_attention, train_encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-spec", choices=["qwen3-2507", "gpt-oss-20b"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["prompt", "prefix-projected", "prefix"], required=True,
                        help="prefix-projected is the prefix tuning of the paper; prefix learns the key/value prefix directly, without the MLP")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--virtual-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lr-scheduler", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--checkpoint-attention", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--epoch-adapter-only", action="store_true")
    parser.add_argument("--extend-from", type=Path,
                        help="Completed training checkpoint; preserve Adam/data state and restart LR in a new output directory")
    parser.add_argument("--loss-scope", choices=["native-turn", "final-content"], default="native-turn")
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1)")
    extension = None
    if args.extend_from:
        if args.lr_scheduler != "linear" or args.warmup_ratio:
            parser.error("Extension only supports the original linear schedule without warmup")
        if args.out_dir.resolve() == args.extend_from.parent.resolve():
            parser.error("An extension requires a separate output directory")
        if not (args.extend_from / "COMPLETE").is_file():
            parser.error("Extension checkpoint is incomplete")
        parent_file = args.extend_from / "training.pt"
        parent_state = torch.load(parent_file, map_location="cpu", weights_only=False)["state"]
        if args.epochs <= parent_state["epoch"]:
            parser.error("--epochs is the total horizon and must exceed the completed parent epochs")
        extension = {"start_step": parent_state["step"], "completed_epochs": parent_state["epoch"],
                     "parent_checkpoint_sha256": hashlib.sha256(parent_file.read_bytes()).hexdigest(),
                     "parent_steps_sha256": hashlib.sha256((args.extend_from.parent / "steps.jsonl").read_bytes()).hexdigest(),
                     "policy": "preserve optimizer and RNG/data cursor; restart initial LR then linearly decay over remaining steps"}
    started = time.monotonic()
    spec = MODEL_SPECS[args.model_spec]
    rows = read_jsonl(args.targets)
    systems = {row["system"] for row in rows}
    if not rows or len(systems) != 1:
        raise ValueError("Need nonempty targets with one shared system prompt")
    system_prompt = systems.pop()
    if not isinstance(system_prompt, str):
        raise ValueError("Target system prompt must be text")
    torch.manual_seed(args.seed)
    base, tokenizer = load_causal_lm(spec, args.model_dir)
    encoded = [_encode(tokenizer, row, system_prompt) for row in rows] if args.loss_scope == "native-turn" else []
    if args.loss_scope == "final-content":
        for row in rows:
            ids, positions = encode_final_content(tokenizer, row["system"], row["input_text"],
                                                   row["target_text"], row.get("thinking"))
            terminators = [i for i in range(max(positions) + 1, len(ids)) if ids[i] == tokenizer.eos_token_id]
            if len(terminators) != 1:
                raise ValueError("Final-content supervision requires one native end-of-turn token")
            supervised = set(positions + terminators)
            encoded.append((ids, [token if i in supervised else -100 for i, token in enumerate(ids)]))
    lengths = [len(item[0]) + (args.virtual_tokens if args.method == "prompt" else 0) for item in encoded]
    if max(lengths) > args.max_sequence_length:
        raise ValueError(f"Target sequence {max(lengths)} exceeds the shared training contract; filter the common pool before adaptation")
    method = {"prompt": "prompt_tuning", "prefix-projected": "prefix_tuning", "prefix": "p_tuning_v2"}[args.method]
    cfg = PeftTrainConfig(method=method,
                          sft_targets=args.targets, num_virtual_tokens=args.virtual_tokens)
    model = get_peft_model(base, _build_peft_config(cfg, base, spec.repo_id))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(math.ceil(len(rows) / args.batch_size) / args.accumulation) * args.epochs
    start_step = extension["start_step"] if extension else 0
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    if extension:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0.0, min(1.0, 1 - (step - start_step) / (total_steps - start_step))))
    else:
        scheduler = get_scheduler(args.lr_scheduler, optimizer, num_warmup_steps=warmup_steps,
                                  num_training_steps=total_steps)
    contract = {"model": spec.repo_id, "revision": spec.revision, "method": args.method, "seed": args.seed,
                "backends": model_backends(base),
                "targets_sha256": hashlib.sha256(args.targets.read_bytes()).hexdigest(),
                "keys": [row["key"] for row in rows], "system": system_prompt,
                "epochs": args.epochs, "virtual_tokens": args.virtual_tokens, "batch_size": args.batch_size,
                "accumulation": args.accumulation, "optimizer": "AdamW", "lr": args.lr, "weight_decay": 0.0,
                "schedule": "linear decay, no warmup", "total_steps": total_steps, "gradient_clip": 1.0,
                "max_sequence_length": args.max_sequence_length, "max_actual_length": max(lengths),
                "loss": "native assistant target token mean per accumulation window, including EOS and analysis when present",
                "prefix_projection": None if args.method == "prompt" else args.method == "prefix-projected",
                "attention_checkpointing": args.checkpoint_attention,
                "trainable_parameters": sum(p.numel() for p in parameters), "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"), "peft": importlib.metadata.version("peft"),
                "date_in_template": CHAT_DATE, "reasoning_effort": REASONING_EFFORT,
                "device": torch.cuda.get_device_name(), "precision": "bf16 frozen base, fp32 adapter"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_every != 1:
        contract["checkpoint_every"] = args.checkpoint_every
    if args.epoch_adapter_only:
        contract["epoch_adapter_only"] = True
    if args.lr_scheduler != "linear" or args.warmup_ratio:
        contract.update(schedule=args.lr_scheduler + " decay with linear warmup",
                        warmup_ratio=args.warmup_ratio, warmup_steps=warmup_steps)
    if args.loss_scope != "native-turn":
        contract.update(loss_scope=args.loss_scope,
                        loss="final-content tokens and native EOS only; assistant metadata and analysis masked")
    if extension:
        contract.update(extension=extension, schedule="parent linear decay; explicit LR restart and remaining-horizon linear decay")
    manifest = args.out_dir / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError("Resume manifest differs")
    manifest.write_text(json.dumps(contract, indent=2) + "\n")
    checkpoints = sorted(path for path in args.out_dir.glob("step_*") if (path / "COMPLETE").is_file())
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with checkpoint_native_attention(model) if args.checkpoint_attention else nullcontext():
        state = train_encoded(model, encoded, contract["keys"], optimizer, scheduler, args.out_dir,
                              pad_id=pad_id, epochs=args.epochs, batch_size=args.batch_size,
                              accumulation=args.accumulation, seed=args.seed, contract=contract,
                              resume=checkpoints[-1] if checkpoints else None, stop_after=args.stop_after,
                              checkpoint_every=args.checkpoint_every,
                              extend_from=args.extend_from if not checkpoints else None,
                              epoch_adapter_only=args.epoch_adapter_only)
    summary = {"status": "COMPLETE" if state["step"] == total_steps else "PREFLIGHT_PAUSED",
               "steps": state["step"], "epochs_completed": state["epoch"],
               "training_seconds_to_checkpoint": state["elapsed_seconds"],
               "this_process_seconds": time.monotonic() - started,
               "peak_memory_bytes": torch.cuda.max_memory_allocated()}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
