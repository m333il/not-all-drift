#!/usr/bin/env python3
"""Fit binary or multilabel probes on one fixed Civil Comments resample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from prompt_optimization.civil_comments import (
    labels_to_matrix,
    load_manifest,
    read_jsonl,
    split_file_name,
)
from prompt_optimization.multilabel_probing import (
    fit_torch_multilabel_probe_bank,
    score_multilabel_probe_bank,
)
from prompt_optimization.probing import ProbeBank

CONDITIONS = ("manual", "seed", "adapted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--shared-manual-activation-dir", type=Path)
    parser.add_argument("--shared-seed-activation-dir", type=Path)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--probe-seed", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--l2-values", type=float, nargs="+", default=(0.0, 1e-5, 1e-4, 1e-3, 1e-2))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--optimization-seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_split(
    directory: Path,
    condition: str,
    split: str,
    labels: tuple[str, ...],
    setup: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    states = load_file(directory / f"{condition}_{split}.safetensors", device="cpu")["states"]
    rows = read_json(directory / f"{condition}_{split}.json")["rows"]
    ids = [row["id"] for row in rows]
    if setup == "binary":
        missing = [row["id"] for row in rows if "binary_label" not in row]
        if missing:
            raise ValueError(
                f"Binary activation metadata is missing targets for {len(missing)} rows"
            )
        label_matrix = torch.tensor(
            [[float(row["binary_label"] == "toxic")] for row in rows],
            dtype=torch.float32,
        )
    else:
        label_matrix = torch.from_numpy(
            labels_to_matrix([row["labels"] for row in rows], labels=labels)
        )
    if len(states) != len(rows) or len(set(ids)) != len(ids):
        raise ValueError(f"Invalid activation metadata for {condition}/{split}")
    return states, label_matrix, ids


def load_probe_contract(split_root: Path) -> tuple[tuple[str, ...], int, str]:
    manifest = load_manifest(split_root)
    labels = tuple(manifest["labels"])
    contract = manifest["contract"]
    setup = str(manifest.get("setup", contract.get("setup", "multilabel")))
    fraction = float(contract["probe_fraction"])
    if not 0 < fraction <= 1:
        raise ValueError("probe_fraction must be in (0, 1]")
    expected_subset_size = round(int(contract["probe_train_size"]) * fraction)
    return labels, expected_subset_size, setup


def select_ids(
    payload: tuple[torch.Tensor, torch.Tensor, list[str]],
    ordered_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    states, labels, ids = payload
    positions = {example_id: index for index, example_id in enumerate(ids)}
    missing = [example_id for example_id in ordered_ids if example_id not in positions]
    if missing:
        raise ValueError(f"Probe subset has {len(missing)} IDs missing from activations")
    indices = torch.tensor([positions[example_id] for example_id in ordered_ids])
    return states[indices], labels[indices], ordered_ids


def save_bank(path: Path, bank: ProbeBank) -> None:
    np.savez_compressed(
        path,
        class_names=np.asarray(bank.class_names),
        means=bank.means,
        scales=bank.scales,
        weights=bank.weights,
        intercepts=bank.intercepts,
        selected_regularization=bank.selected_regularization,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; probes must not train on CPU")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    labels, expected_subset_size, setup = load_probe_contract(args.split_root)
    class_names = ("toxic",) if setup == "binary" else labels

    subset_path = args.split_root / split_file_name(
        "probe_train_subset", split_seed=args.probe_seed
    )
    subset_rows = read_jsonl(subset_path)
    ordered_subset_ids = [row["id"] for row in subset_rows]
    if (
        len(ordered_subset_ids) != expected_subset_size
        or len(set(ordered_subset_ids)) != expected_subset_size
    ):
        raise ValueError(
            f"Expected a unique fixed {expected_subset_size}-example probe train subset"
        )

    loaded: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor, list[str]]]] = {}
    for condition in CONDITIONS:
        source = (
            args.shared_manual_activation_dir
            if condition == "manual" and args.shared_manual_activation_dir is not None
            else (
                args.shared_seed_activation_dir
                if condition == "seed" and args.shared_seed_activation_dir is not None
                else args.activation_dir
            )
        )
        loaded[condition] = {
            split: load_split(source, condition, split, labels, setup)
            for split in ("probe_train", "probe_val", "test")
        }
        loaded[condition]["probe_train"] = select_ids(
            loaded[condition]["probe_train"], ordered_subset_ids
        )

    for split in ("probe_train", "probe_val", "test"):
        reference_labels, reference_ids = loaded["manual"][split][1:]
        for condition in CONDITIONS[1:]:
            if loaded[condition][split][2] != reference_ids:
                raise ValueError(f"Example IDs differ for {condition}/{split}")
            if not torch.equal(loaded[condition][split][1], reference_labels):
                raise ValueError(f"Labels differ for {condition}/{split}")

    results: dict[str, Any] = {
        "status": "running",
        "conditions": list(CONDITIONS),
        "activation_dir": str(args.activation_dir.resolve()),
        "shared_manual_activation_dir": (
            str(args.shared_manual_activation_dir.resolve())
            if args.shared_manual_activation_dir is not None
            else None
        ),
        "shared_seed_activation_dir": (
            str(args.shared_seed_activation_dir.resolve())
            if args.shared_seed_activation_dir is not None
            else None
        ),
        "setup": setup,
        "class_names": list(class_names),
        "probe_seed": args.probe_seed,
        "probe_subset": str(subset_path.resolve()),
        "probe_train_samples": len(ordered_subset_ids),
        "probe_training": {
            "l2_values": args.l2_values,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimization_seed": args.optimization_seed,
        },
        "scores": {},
    }
    device = torch.device("cuda:0")
    for train_condition in CONDITIONS:
        train_states, train_labels, _ = loaded[train_condition]["probe_train"]
        validation_states, validation_labels, _ = loaded[train_condition]["probe_val"]
        bank = fit_torch_multilabel_probe_bank(
            train_states,
            train_labels,
            validation_states,
            validation_labels,
            class_names=class_names,
            l2_values=args.l2_values,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.optimization_seed + args.probe_seed,
        )
        save_bank(args.output_dir / f"probe_bank_{train_condition}.npz", bank)
        results["scores"][train_condition] = {}
        for test_condition in CONDITIONS:
            test_states, test_labels, _ = loaded[test_condition]["test"]
            results["scores"][train_condition][test_condition] = score_multilabel_probe_bank(
                bank, test_states.float().numpy(), test_labels.numpy()
            )
        write_json(args.output_dir / "summary.partial.json", results)

    results["status"] = "done"
    write_json(args.output_dir / "summary.json", results)
    (args.output_dir / "summary.partial.json").unlink()


if __name__ == "__main__":
    main()
