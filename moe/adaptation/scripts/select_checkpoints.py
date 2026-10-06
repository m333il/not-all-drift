#!/usr/bin/env python3
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


def select_checkpoints(manifest, scores):
    keys = manifest["keys"]
    arms = {arm["name"]: arm for arm in manifest["arms"]}
    if len(set(keys)) != len(keys) or len(arms) != len(manifest["arms"]):
        raise ValueError("Duplicate validation IDs or arm names")
    if set(scores["arms"]) != set(arms):
        raise ValueError("Scored arms differ from the evaluation manifest")
    candidates = defaultdict(list)
    controls = []
    for name, result in scores["arms"].items():
        rows = result["per_example"]
        if len(rows) != len(keys) or {row["key"] for row in rows} != set(keys):
            raise ValueError("Every arm needs exactly one score per validation ID")
        value = sum(row["score"] for row in rows) / len(rows)
        if not math.isfinite(value) or not math.isclose(value, result["score"], abs_tol=1e-12, rel_tol=0):
            raise ValueError("Aggregate primary score differs from per-example scores")
        arm = arms[name]
        row = {"arm": name, "score": value}
        if "adapter" not in arm:
            controls.append(row)
            continue
        checkpoint = Path(arm["adapter"]).parent
        match = re.fullmatch(r"step_(\d+)", checkpoint.name)
        if not match or Path(arm["adapter"]).name != "adapter":
            raise ValueError("Expected a step_NNNNNN/adapter checkpoint")
        row.update(step=int(match[1]), checkpoint=str(checkpoint), adapter=arm["adapter"],
                   adapter_hashes=arm["adapter_hashes"])
        if row["step"] == 0:
            controls.append(row)
        else:
            candidates[str(checkpoint.parent)].append(row)
    if not candidates:
        raise ValueError("No trained checkpoints were evaluated")
    selected = []
    for run, rows in sorted(candidates.items()):
        rows.sort(key=lambda row: (-row["score"], row["step"]))
        selected.append({"training_run": run, "best": rows[0], "evaluated": rows})
    return {"selected": selected, "controls": controls}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.evaluation_dir / "manifest.json"
    generations = args.evaluation_dir / "generations.jsonl"
    manifest = json.loads(manifest_path.read_text())
    scores = json.loads(args.scores.read_text())
    validation_hash = hashlib.sha256(args.validation.read_bytes()).hexdigest()
    generation_hash = hashlib.sha256(generations.read_bytes()).hexdigest()
    if validation_hash != manifest["examples_sha256"] or validation_hash != scores["examples_sha256"]:
        raise ValueError("Validation source hashes differ")
    if generation_hash != scores["generations_sha256"]:
        raise ValueError("Scores do not describe these saved generations")
    result = {"model": manifest["model"], "revision": manifest["revision"],
              "selection_split": "validation", "n": len(manifest["keys"]),
              "primary": scores["primary"], "tie_break": "earliest trained step",
              "initialization_is_candidate": False,
              "validation_sha256": validation_hash, "generations_sha256": generation_hash,
              "evaluation_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
              **select_checkpoints(manifest, scores)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([row["best"] for row in result["selected"]], indent=2))


if __name__ == "__main__":
    main()
