#!/usr/bin/env python3
"""Turn a trained adaptation run into an arm directory the probes can load.

The input is the output of ``moe/adaptation``: the training run with its
``step_NNNNNN/adapter`` checkpoints, the validation ``generations.jsonl`` and
the ``selection.json`` written by ``select_checkpoints.py``. The step
is the one ``selection.json`` names, chosen on validation quality; ``--step 0``
gives the untrained initialization instead.

The arm directory holds ``adapter/``, ``selection.json``, ``receipt.json`` and
``contract_sample.json``. The probes read the PEFT type and the number of
virtual tokens from the receipt, and use the contract sample to check that
their prompt rendering reproduces the token ids the adapter was trained under.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True,
                        help="selection.json from select_checkpoints.py")
    parser.add_argument("--generations", type=Path, required=True,
                        help="Validation generations.jsonl the selection was scored on")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-index", type=int, default=0,
                        help="Which training run of selection.json to take")
    parser.add_argument("--step", type=int,
                        help="Training step to use instead of the selected one; 0 is the "
                             "untrained initialization")
    parser.add_argument("--baseline-arm", default="baseline",
                        help="Arm name of the unadapted rows in generations.jsonl")
    parser.add_argument("--sample-size", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text())
    run = selection["selected"][args.run_index]
    step = run["best"]["step"] if args.step is None else args.step
    source = Path(run["training_run"]) / f"step_{step:06d}" / "adapter"
    if not (source / "adapter_config.json").is_file():
        raise FileNotFoundError(f"{source} is not a PEFT adapter directory")

    adapter = args.out / "adapter"
    if adapter.exists():
        raise FileExistsError(f"{adapter} already exists")
    shutil.copytree(source, adapter)
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(adapter.iterdir()) if path.is_file()}

    sample = []
    with args.generations.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["arm"] == args.baseline_arm:
                sample.append({"key": row["key"], "input_ids": row["input_ids"]})
            if len(sample) == args.sample_size:
                break
    if not sample:
        raise ValueError(f"No rows of arm {args.baseline_arm!r} in {args.generations}")

    config = json.loads((adapter / "adapter_config.json").read_text())
    receipt = {
        "training_run": run["training_run"],
        "selected_step": step,
        "selection_score": run["best"]["score"] if args.step is None else None,
        "step_overridden": args.step is not None,
        "selection_split": selection.get("selection_split"),
        "selection_n": selection.get("n"),
        "base_model": selection.get("model"),
        "base_revision": selection.get("revision"),
        "peft_type": config["peft_type"],
        "num_virtual_tokens": config.get("num_virtual_tokens"),
        "prompt_tuning_init": config.get("prompt_tuning_init"),
        "adapter_files": files,
    }
    shutil.copyfile(args.selection, args.out / "selection.json")
    (args.out / "contract_sample.json").write_text(json.dumps(sample, indent=2) + "\n")
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("selected_step", "peft_type", "num_virtual_tokens")}))


if __name__ == "__main__":
    main()
