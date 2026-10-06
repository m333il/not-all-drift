"""Which experts each layer loses - computed from the map, not from a finished run.

The selection is a function of the routing map and the level alone: the sweep
does the same call before it loads any weights. So a checkpoint whose quality
run has not finished yet still has a fully determined pruning set, and there is
no reason to wait for the GPU to report it.

Written next to the run-derived renderer on purpose: the two must agree, and
`--verify` checks that they do on every cell that already has results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from mrd_pruning.frequency import (  # noqa: E402
    load_counts_npz,
    pruned_set_stats,
    resolve_levels,
    select_pruned,
)

LEVEL_TITLES = {0: "model", 50: "−50%", 75: "−75%"}
ARM_KEYS = {
    "base": "base",
    "gepa": "gepa",
    "prompt": "prompt_tuning",
    "prefix": "prefix_tuning",
}


def arm_key_for(cell: str) -> str:
    if cell == "base":
        return "base"
    if cell.startswith("gepa"):
        return "gepa"
    if cell.startswith("prompt"):
        return "prompt_tuning"
    if cell.startswith("prefix"):
        return "prefix_tuning"
    raise ValueError(f"cannot infer the arm of cell {cell}")


def ranges(ids: list[int]) -> str:
    if not ids:
        return " - "
    out, start, prev = [], ids[0], ids[0]
    for value in ids[1:] + [None]:
        if value is not None and value == prev + 1:
            prev = value
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if value is not None:
            start = prev = value
    return ", ".join(out)


def render(model: str, cell: str, map_path: Path, levels: list[str], top_k: int) -> str:
    arm = arm_key_for(cell)
    counts = load_counts_npz(map_path, arm, "__all__")
    matrix = np.asarray(counts.counts, dtype=np.float64)
    resolved = resolve_levels(levels, counts.n_experts)

    lines = [
        f"The cut experts -- {model}/{cell}",
        "",
        f"Frequency map: `{map_path}`, stage ` __all__ `, arm `{arm}`  ",
        "Selection: `per_layer` - in each layer are cut experts, who "
        "**This layer is the least used  ",
        f"Sloev: {counts.n_layers}, experts in the layer: {counts.n_experts}, "
        f"top-k router: {top_k}",
        "",
        "Pruning – mask on the router: logit cut expert put in "
        "`finfo(dtype).min` , no weights to be touched. The set is determined by the map and "
        "level, the measurement of quality does not affect it.",
        "",
    ]

    sets: dict[int, dict[int, set[int]]] = {}
    for spec, n_prune in zip(levels, resolved):
        pct = 0 if n_prune == 0 else round(100 * n_prune / counts.n_experts)
        title = LEVEL_TITLES.get(pct, f"−{pct}%")
        lines += [f"## Level {title} ({n_prune} from {counts.n_experts} layered", ""]
        if n_prune == 0:
            lines += ["Nothing is cut; this is the control level.", ""]
            continue

        pruned = select_pruned(counts, n_prune, top_k=top_k, selection="per_layer")
        sets[pct] = {layer: set(ids) for layer, ids in pruned.items()}
        stats = pruned_set_stats(pruned, counts)
        lines += [
            f"Cut everything:{n_prune * len(pruned)}** experts in "
            f"{len(pruned)} Share of traffic under the mask: average by layer "
            f"{stats.get('mass_pruned_mean', 0.0):.2%}maximum "
            f"{stats.get('mass_pruned_max', 0.0):.2%}.",
            "",
            "| layer | cut | traffic cut | experts |",
            "|---|---|---|---|",
        ]
        for layer in sorted(pruned):
            ids = sorted(int(e) for e in pruned[layer])
            row = matrix[layer]
            total = row.sum()
            share = row[ids].sum() / total if total else 0.0
            lines += [f"| {layer} | {len(ids)} | {share:.2%} | `{ranges(ids)}` |"]
        lines += [""]

    if 50 in sets and 75 in sets:
        nested = all(sets[50].get(layer, set()) <= experts
                     for layer, experts in sets[75].items())
        lines += [
            "## Embedded levels",
            "",
            ("-75% contains the whole set of -50% in all layers: yes."
             if nested else
             "-75% **n** contains a set of -50% completely - the selection is recalculated."),
            "",
        ]

    lines += ["---", "",
              "Counted from the map by `scripts/plan_pruned_experts.py` script."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default="routing_maps_s42_native")
    parser.add_argument("--out", default="/tmp/pruned_experts_planned")
    parser.add_argument("--levels", default="0,50%,75%")
    parser.add_argument("--results", default="results/prune_eval_s42",
                        help="for --verify: where are pruned_experts.json from runs")
    parser.add_argument("--verify", action="store_true",
                        help="Check against what the sweep recorded.")
    args = parser.parse_args()

    levels = args.levels.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    maps = Path(args.maps)
    results = Path(args.results)

    written = mismatches = verified = 0
    for model_dir in sorted(maps.iterdir()):
        if not model_dir.is_dir():
            continue
        top_k = 4 if model_dir.name == "gpt-oss" else 8
        for cell_dir in sorted(model_dir.iterdir()):
            map_path = cell_dir / "expert_counts.npz"
            if not map_path.is_file():
                continue
            model, cell = model_dir.name, cell_dir.name
            text = render(model, cell, map_path, levels, top_k)
            (out / f"{model}__{cell}.md").write_text(text)
            written += 1

            if args.verify:
                counts = load_counts_npz(map_path, arm_key_for(cell), "__all__")
                for spec, n_prune in zip(levels, resolve_levels(levels, counts.n_experts)):
                    if n_prune == 0:
                        continue
                    ours = {int(k): sorted(int(e) for e in v)
                            for k, v in select_pruned(counts, n_prune, top_k=top_k).items()}
                    found = list((results / model / cell).glob(f"*prune{n_prune:03d}"))
                    if not found:
                        continue
                    theirs_path = found[0] / "pruned_experts.json"
                    if not theirs_path.is_file():
                        continue
                    theirs = {int(k): sorted(int(e) for e in v)
                              for k, v in json.loads(theirs_path.read_text()).items()}
                    verified += 1
                    if ours != theirs:
                        mismatches += 1
                        print(f"mismatch: {model}/{cell} level {spec}")

    print(f"files: {written}")
    if args.verify:
        print(f"levels checked against finished runs: {verified}, mismatches: {mismatches}")


if __name__ == "__main__":
    main()
