#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel

from mrd.jsonl import read_jsonl
from mrd.models.loading import load_causal_lm, model_backends
from mrd.models.registry import MODEL_SPECS, build_adapter
from mrd.prompts import SEED_SYSTEM_PROMPT
from mrd.generation import generate_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-spec", choices=["qwen3-2507", "gpt-oss-20b"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    spec = MODEL_SPECS[args.model_spec]
    examples = read_jsonl(args.examples)
    arms = json.loads(args.arms.read_text())
    if len({arm["name"] for arm in arms}) != len(arms):
        raise ValueError("Duplicate arm names")
    for arm in arms:
        if "system_file" in arm and "system" in arm:
            raise ValueError("An arm cannot set both system and system_file")
        if "system_file" in arm:
            arm["system"] = Path(arm["system_file"]).read_text()
        elif "system" not in arm:
            arm["system"] = SEED_SYSTEM_PROMPT
        elif not isinstance(arm["system"], str):
            raise ValueError("Arm system prompt must be text")
        if "adapter" in arm:
            path = Path(arm["adapter"])
            arm["adapter_hashes"] = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                                     for name in ("adapter_config.json", "adapter_model.safetensors")}
    manifest = {"model": spec.repo_id, "revision": spec.revision,
                "examples_sha256": hashlib.sha256(args.examples.read_bytes()).hexdigest(),
                "keys": [r["key"] for r in examples], "arms": arms, "max_tokens": args.max_tokens,
                "decoding": "native deterministic greedy; free generation", "torch": torch.__version__}
    base, tokenizer = load_causal_lm(spec, args.model_dir)
    manifest["backends"] = model_backends(base)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Evaluation resume contract differs")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    output = args.out_dir / "generations.jsonl"
    previous = read_jsonl(output) if output.exists() else []
    completed = {(r["arm"], r["key"]) for r in previous}
    with output.open("a", buffering=1) as stream:
        for arm in arms:
            model = PeftModel.from_pretrained(base, arm["adapter"]).eval() if "adapter" in arm else base
            adapter = build_adapter(model)
            for row in examples:
                if (arm["name"], row["key"]) in completed:
                    continue
                record, _ = generate_record(model, tokenizer, adapter, arm["system"], row["prompt"], args.max_tokens)
                stream.write(json.dumps({"key": row["key"], "arm": arm["name"], **record}, ensure_ascii=False) + "\n")
                print(json.dumps({"key": row["key"], "arm": arm["name"], "tokens": record["completion_tokens"],
                                  "finished": record["finished"], "seconds": record["elapsed_seconds"]}), flush=True)
            del model, adapter
            gc.collect()
            torch.cuda.empty_cache()
    print("EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
