#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from mrd.civil_v2_contract import (
    CONTRACT_SOURCE_COMMIT,
    CONTRACT_SOURCE_REPOSITORY,
    PROMPT_CONTRACT_ID,
    SEED_INSTRUCTION,
    format_labels,
    render_user_prompt,
)
from mrd.jsonl import read_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolved-dataset-revision", required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise ValueError(f"Output already exists: {args.out_dir}")
    source_manifest_path = args.splits_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    files = {}
    args.out_dir.mkdir(parents=True)
    for seed in (42, 43, 44):
        source_train = args.splits_dir / f"optimizer_train_seed{seed}_n1000.jsonl"
        source_validation = args.splits_dir / f"optimizer_val_seed{seed}_n200.jsonl"
        train = read_jsonl(source_train)
        validation = read_jsonl(source_validation)
        if {row["id"] for row in train} & {row["id"] for row in validation}:
            raise ValueError(f"Train/validation overlap for seed {seed}")
        directory = args.out_dir / f"seed{seed}"
        directory.mkdir()

        def example(row: dict) -> dict:
            prompt = render_user_prompt(SEED_INSTRUCTION, row["text"])
            return {
                **row,
                "key": row["id"],
                "split_group": row["id"],
                "prompt": prompt,
                "target": format_labels(row["labels"]),
            }

        train_examples = [example(row) for row in train]
        validation_examples = [example(row) for row in validation]
        targets = [
            {
                "key": row["key"],
                "split_group": row["split_group"],
                "input_text": row["prompt"],
                "target_text": row["target"],
                "system": "",
                "supervision": "dataset gold labels; no generated reasoning",
            }
            for row in train_examples
        ]
        outputs = {
            "train.jsonl": train_examples,
            "train_targets.jsonl": targets,
            "validation.jsonl": validation_examples,
        }
        for name, rows in outputs.items():
            path = directory / name
            write_jsonl(path, rows)
            files[str(path.relative_to(args.out_dir))] = {"sha256": sha256(path), "n": len(rows)}
    manifest = {
        "status": "PASS",
        "prompt_contract_id": PROMPT_CONTRACT_ID,
        "placement": "first and only user message; no system message",
        "seed_instruction": SEED_INSTRUCTION,
        "contract_source_repository": CONTRACT_SOURCE_REPOSITORY,
        "contract_source_commit": CONTRACT_SOURCE_COMMIT,
        "split_manifest_sha256": sha256(source_manifest_path),
        "declared_dataset_revision": source_manifest["dataset_revision"],
        "resolved_dataset_revision": args.resolved_dataset_revision,
        "sources": source_manifest["sources"],
        "files": files,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
