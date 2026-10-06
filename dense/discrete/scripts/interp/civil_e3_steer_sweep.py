"""Mean-shift steering (E3) with norm-matched random controls.

For each layer l the mean seed-relative shift

    v_l = mean_i [ h_l(x_i | C_target) - h_l(x_i | C_seed) ]

is added as alpha * v_l to the seed run, and the recovered fraction is reported:

    R = (F1(C_seed + alpha v_l) - F1(C_seed)) / (F1(C_target) - F1(C_seed)).

Cells with too many parse failures or inflated prediction counts are marked as degraded.
Per-example scores go to ``per_example.jsonl`` for ``e3_bootstrap_recovery.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Cell:
    layer: int
    alpha: float
    kind: str
    fraction: float | None = None


def _mean_shift(seed_store: Path, target_store: Path) -> tuple[np.ndarray, np.ndarray]:
    from interpretability_gepa.activations import load_activation_store

    seed = load_activation_store(seed_store)
    target = load_activation_store(target_store)
    if seed.example_ids != target.example_ids:
        raise SystemExit(
            "activation stores are not row-aligned; a mean shift would pair different examples"
        )
    if seed.last_prompt.shape != target.last_prompt.shape:
        raise SystemExit("activation stores disagree on shape")
    # Stores are float16; deep-layer shift norms overflow it, so widen first.
    seed_states = seed.last_prompt.astype(np.float32)
    shift = (target.last_prompt.astype(np.float32) - seed_states).mean(axis=0)
    return shift, seed_states


def _score(
    outputs: list[str], gold: list[tuple[str, ...]], labels: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate metrics plus the per-example rows they average over."""
    from interpretability_gepa.metrics import empty_aware_sample_f1_rows, multilabel_matrix
    from interpretability_gepa.prompts import ParseError, parse_labels

    predicted: list[tuple[str, ...]] = []
    parse_ok: list[bool] = []
    for text in outputs:
        try:
            # Set parser: we score the decision, not the output order.
            predicted.append(parse_labels(text, labels, enforce_order=False))
            parse_ok.append(True)
        except ParseError:
            predicted.append(())
            parse_ok.append(False)
    rows = empty_aware_sample_f1_rows(
        multilabel_matrix(gold, labels), multilabel_matrix(predicted, labels)
    )
    rows[~np.asarray(parse_ok, dtype=bool)] = 0.0
    aggregate = {
        "f1_samples": float(rows.mean()),
        "parse_failure_rate": 1.0 - float(np.mean(parse_ok)),
        "avg_predictions": float(np.mean([len(p) for p in predicted])),
    }
    per_example = {
        "f1": [round(float(x), 6) for x in rows],
        "parse_ok": [bool(x) for x in parse_ok],
        "predicted": [list(p) for p in predicted],
    }
    return aggregate, per_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--seed-store", type=Path, required=True)
    parser.add_argument("--target-store", type=Path, required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--target-instruction", type=Path)
    parser.add_argument("--target-adapter", type=Path)
    parser.add_argument("--layers", required=True, help="comma-separated layer indices")
    parser.add_argument("--alphas", default="0.5,1,2,4")
    parser.add_argument(
        "--relative-norms",
        help="comma-separated fractions of the mean state norm at the edited layer; "
        "overrides --alphas",
    )
    parser.add_argument("--random-controls", type=int, default=5)
    parser.add_argument("--positions", default="prompt_last")
    parser.add_argument("--model-key", default="gemma2_2b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--parse-failure-limit", type=float, default=0.10)
    parser.add_argument("--prediction-multiplier-limit", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    from interpretability_gepa.causal import (
        EditSpec,
        block_for_residual_index,
        generate_with_edit,
        norm_matched_random,
        relative_alpha,
    )
    from interpretability_gepa.config import load_config
    from interpretability_gepa.datasets import labels_for_dataset, load_jsonl_split
    from interpretability_gepa.modeling import load_hf_model
    from interpretability_gepa.prompts import (
        SEED_INSTRUCTIONS,
        apply_chat_template,
        render_condition_messages,
    )

    cfg = load_config(args.config)
    model_cfg = cfg.model(args.model_key)
    labels = labels_for_dataset(cfg.dataset.id)
    examples = load_jsonl_split(args.split_file)
    gold = [tuple(example.labels) for example in examples]
    shift, seed_states = _mean_shift(args.seed_store, args.target_store)
    layers = [int(value) for value in args.layers.split(",")]
    fractions = (
        [float(value) for value in args.relative_norms.split(",")] if args.relative_norms else None
    )
    alphas = [float(value) for value in args.alphas.split(",")]

    def cells_for(layer: int) -> list[Cell]:
        """One cell per strength; relative strengths are comparable across layers and methods."""
        if fractions is None:
            return [Cell(layer, alpha, "shift") for alpha in alphas]
        reference = seed_states[:, layer, :]
        return [
            Cell(layer, relative_alpha(shift[layer], reference, fraction=f), "shift", f)
            for f in fractions
        ]

    def applied_relative(layer: int, alpha: float, vector: np.ndarray) -> float:
        reference = seed_states[:, layer, :]
        reference_norm = float(np.linalg.norm(reference, axis=1).mean())
        return alpha * float(np.linalg.norm(vector)) / reference_norm

    model, tokenizer = load_hf_model(model_cfg)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    seed_instruction = SEED_INSTRUCTIONS[cfg.dataset.id]

    def prompts_for(instruction: str) -> list[str]:
        return [
            apply_chat_template(
                tokenizer,
                render_condition_messages("C_seed", instruction, labels, example.text),
                non_thinking=model_cfg.non_thinking,
            )
            for example in examples
        ]

    def generate(prompts: list[str], spec: EditSpec | None, value: Any, model_: Any) -> list[str]:
        if spec is None:
            zero = torch.zeros(shift.shape[-1], dtype=torch.float32)
            spec = EditSpec(layer=0, mode="add", positions=args.positions, alpha=0.0)
            value = zero
        return generate_with_edit(
            model=model_,
            tokenizer=tokenizer,
            prompts=prompts,
            spec=spec,
            value=value,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

    seed_prompts = prompts_for(seed_instruction)
    rows: list[dict[str, Any]] = []
    per_example: list[dict[str, Any]] = []

    def keep(cell: str, scores: dict[str, Any]) -> None:
        per_example.append({"cell": cell, **scores})

    baseline, baseline_rows = _score(generate(seed_prompts, None, None, model), gold, labels)
    keep("C_seed", baseline_rows)
    rows.append({"cell": "C_seed", "kind": "baseline", "layer": None, "alpha": None, **baseline})
    print(
        f"E3_BASELINE f1={baseline['f1_samples']:.4f}"
        f" parse_fail={baseline['parse_failure_rate']:.4f}"
    )

    # Target run: GEPA differs only in the instruction, prompt/prefix tuning needs the adapter.
    if args.target_adapter is not None:
        from peft import PeftModel

        target_model = PeftModel.from_pretrained(
            model, str(args.target_adapter), local_files_only=True
        )
        target_model.eval()
        target_outputs = generate(seed_prompts, None, None, target_model)
        del target_model
        torch.cuda.empty_cache()
    else:
        instruction = args.target_instruction.read_text(encoding="utf-8")
        target_outputs = generate(prompts_for(instruction), None, None, model)
    target, target_rows = _score(target_outputs, gold, labels)
    keep(args.target_label, target_rows)
    rows.append(
        {"cell": args.target_label, "kind": "target", "layer": None, "alpha": None, **target}
    )
    print(f"E3_TARGET {args.target_label} f1={target['f1_samples']:.4f}")

    gap = target["f1_samples"] - baseline["f1_samples"]
    if gap <= 0:
        print(
            f"E3_WARN target does not beat the seed on this split (gap={gap:.4f}); "
            "recovery is undefined and will be reported as null"
        )

    def record(cell: Cell, value: np.ndarray) -> dict[str, Any]:
        # Store index 0 is the embedding output, so store index l is written by block l - 1.
        block = block_for_residual_index(cell.layer)
        spec = EditSpec(layer=block, mode="add", positions=args.positions, alpha=cell.alpha)
        tensor = torch.as_tensor(value, dtype=torch.float32)
        scored, scored_rows = _score(generate(seed_prompts, spec, tensor, model), gold, labels)
        recovery = None if gap <= 0 else (scored["f1_samples"] - baseline["f1_samples"]) / gap
        degraded = bool(
            scored["parse_failure_rate"] > args.parse_failure_limit
            or scored["avg_predictions"]
            > args.prediction_multiplier_limit * max(baseline["avg_predictions"], 1e-9)
        )
        tag = f"r{cell.fraction:g}" if cell.fraction is not None else f"a{cell.alpha:g}"
        row = {
            "cell": f"{cell.kind}_L{cell.layer}_{tag}",
            "kind": cell.kind,
            "layer": cell.layer,
            "block": block,
            "alpha": cell.alpha,
            "relative_norm": cell.fraction,
            "applied_relative": applied_relative(cell.layer, cell.alpha, value),
            "recovery": recovery,
            "degraded": degraded,
            **scored,
        }
        keep(row["cell"], scored_rows)
        marker = " DEGRADED" if degraded else ""
        shown = "n/a" if recovery is None else f"{recovery:+.3f}"
        print(
            f"E3_CELL {row['cell']} alpha={cell.alpha:.4f}"
            f" rel={row['applied_relative']:.4f}"
            f" f1={scored['f1_samples']:.4f} R={shown}{marker}"
        )
        return row

    for layer in layers:
        for cell in cells_for(layer):
            rows.append(record(cell, shift[layer]))

    # Norm-matched random directions at the best non-degraded cell.
    live = [
        r for r in rows if r["kind"] == "shift" and not r["degraded"] and r["recovery"] is not None
    ]
    if live and args.random_controls:
        best = max(live, key=lambda r: r["recovery"])
        draws = norm_matched_random(shift[best["layer"]], args.random_controls)
        for index, draw in enumerate(draws):
            rows.append(
                record(
                    Cell(best["layer"], best["alpha"], f"random{index}", best["relative_norm"]),
                    draw,
                )
            )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cells.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    # Written last so that a crashed run leaves no partial file.
    with (args.output / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for entry in per_example:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    summary = {
        "target_label": args.target_label,
        "split": str(args.split_file),
        "n_examples": len(examples),
        "example_ids": [example.id for example in examples],
        "gold": [list(row) for row in gold],
        "positions": args.positions,
        "strength_mode": "relative_norms" if fractions else "absolute_alphas",
        "relative_norms": fractions,
        "alphas": None if fractions else alphas,
        "baseline_f1": baseline["f1_samples"],
        "target_f1": target["f1_samples"],
        "gap": gap,
        "best": max(live, key=lambda r: r["recovery"]) if live else None,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"E3_SWEEP_COMPLETE {args.output}")


if __name__ == "__main__":
    main()
