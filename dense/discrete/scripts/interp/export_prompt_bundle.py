"""Export the frozen GEPA prompt corpus as a self-contained bundle.

Every field is copied from the prompt registry: instruction, rendered prompt, and the
hashes needed to check that the same text was run against the same splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from interpretability_gepa.artifacts import sha256_path
from interpretability_gepa.errors import ArtifactError
from interpretability_gepa.prompt_registry import PromptRegistry, load_prompt_registry
from interpretability_gepa.prompts import render_messages

TEXT_PLACEHOLDER = "{text}"


def _readme(registry: PromptRegistry, splits_dir: Path) -> str:
    labels = ", ".join(registry.labels)
    adapted = [item for item in registry.prompts if item.kind == "C_adapt"]
    return f"""# Frozen prompt corpus: {registry.dataset_id}

{len(registry.prompts)} prompts: one seed condition and {len(adapted)} adapted
conditions produced by GEPA, one per (optimizer seed, optimizer train size) cell.

## Model these prompts were optimized against

| field | value |
|---|---|
| model | `{registry.model_id}` |
| revision | `{registry.model_revision}` |
| tokenizer revision | `{registry.tokenizer_revision}` |
| non-thinking | `{registry.non_thinking}` |

The adapted prompts were selected by scoring this exact checkpoint. Running them
against a different model or revision measures something else.

## Splits

`split_manifest_sha256` = `{registry.split_manifest_sha256}`

That is the SHA-256 of `manifest.json` in the frozen split directory
(`{splits_dir}`). Verify it before probing; a prompt frozen against one split
contract is not comparable to activations extracted under another.

## Output contract

`prompt_contract_id` = `{registry.prompt_contract_id}`

Label schema order, which is **not** alphabetical:

    {labels}

The model must return every applicable label in that order, separated by
commas, and exactly `NONE` when no label applies. Nothing else. A response in
any other shape (a JSON array, alphabetical order, prose around the answer) is
a parse failure, not a wrong prediction, and the two must be counted apart.

## Layout

    prompts.jsonl                     one row per prompt, all fields inline
    prompt_registry.json              the registry this bundle was cut from
    prompts/<prompt_id>/instruction.txt   the optimized instruction alone
    prompts/<prompt_id>/prompt.txt        the full rendered prompt
    prompts/<prompt_id>/meta.json         provenance for that one prompt
    SHA256SUMS                        checksums for every file above

`prompt.txt` contains the literal placeholder `{TEXT_PLACEHOLDER}` where the
example text goes. Substitute it verbatim; do not strip or reflow the
surrounding text, since the instruction hashes cover it.

## Verifying what you received

    sha256sum -c SHA256SUMS

Each `meta.json` also carries `instruction_sha256`, which is the SHA-256 of
`instruction.txt` on its own. `registry_sha256` in `prompt_registry.json` covers
the whole corpus.
"""


def _prompt_rows(registry: PromptRegistry) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in registry.prompts:
        rendered = render_messages(
            record.instruction, registry.labels, TEXT_PLACEHOLDER
        ).messages[0]["content"]
        rows.append(
            {
                "prompt_id": record.prompt_id,
                "kind": record.kind,
                "gepa_seed": record.gepa_seed,
                "gepa_train_size": record.gepa_train_size,
                "instruction": record.instruction,
                "instruction_sha256": record.instruction_sha256,
                "prompt": rendered,
                "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "optimized_prompt_sha256": record.optimized_prompt_sha256,
                "source_run": record.source_run,
                "gepa_git_commit": record.gepa_git_commit,
                "gepa_config_sha256": record.gepa_config_sha256,
                "optimizer_train_sha256": record.optimizer_train_sha256,
                "optimizer_val_sha256": record.optimizer_val_sha256,
            }
        )
    return rows


def _write_checksums(bundle: Path) -> None:
    lines = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        lines.append(f"{sha256_path(path)}  {path.relative_to(bundle).as_posix()}")
    (bundle / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_bundle(registry_path: Path, splits_dir: Path, output: Path) -> Path:
    registry = load_prompt_registry(registry_path)
    registry.validate()

    manifest = splits_dir / "manifest.json"
    if not manifest.exists():
        raise ArtifactError(f"missing split manifest: {manifest}")
    actual = sha256_path(manifest)
    if actual != registry.split_manifest_sha256:
        raise ArtifactError(
            f"split manifest {actual} does not match registry {registry.split_manifest_sha256}"
        )

    if output.exists():
        raise ArtifactError(f"bundle already exists: {output}")
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "prompts").mkdir(parents=True)

    try:
        rows = _prompt_rows(registry)
        for row in rows:
            directory = staging / "prompts" / str(row["prompt_id"])
            directory.mkdir()
            (directory / "instruction.txt").write_text(row["instruction"], encoding="utf-8")
            (directory / "prompt.txt").write_text(row["prompt"], encoding="utf-8")
            inline = {"instruction", "prompt"}
            meta = {key: value for key, value in row.items() if key not in inline}
            (directory / "meta.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        (staging / "prompts.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        shutil.copyfile(registry_path, staging / "prompt_registry.json")
        (staging / "README.md").write_text(_readme(registry, splits_dir), encoding="utf-8")
        _write_checksums(staging)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _tar(bundle: Path) -> Path:
    archive = bundle.with_name(bundle.name + ".tar.gz")
    if archive.exists():
        raise ArtifactError(f"archive already exists: {archive}")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(bundle, arcname=bundle.name)
    return archive


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tarball", action="store_true", help="also write <output>.tar.gz")
    args = parser.parse_args(argv)

    bundle = export_bundle(args.registry, args.splits_dir, args.output)
    print(bundle)
    if args.tarball:
        print(_tar(bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
