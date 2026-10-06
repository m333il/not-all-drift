"""Paired bootstrap CIs for the recovered fraction of the mean-shift steering sweep.

Resamples evaluation examples once and recomputes all means on the same draw.
``recovery`` resamples the gap as well; ``recovery_fixed_gap`` divides by the point gap
and is more stable when the gap is small (see ``gap_le_zero_frac``). Seeds of one method
share the evaluation split, so pooling across seeds is paired too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOOTSTRAP_SAMPLES = 1000
CI = (2.5, 97.5)


def load_sweep(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    rows: dict[str, np.ndarray] = {}
    with (path / "per_example.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            rows[entry["cell"]] = np.asarray(entry["f1"], dtype=np.float64)
    if "example_ids" not in summary:
        raise SystemExit(f"{path}: summary predates per-example output; rerun the sweep")
    return summary, rows


def mode_of(cell: str) -> str:
    """Cells are named `<kind>_L<layer>_<strength>`, so the kind is everything before `_L`."""
    return cell.rsplit("_L", 1)[0] if "_L" in cell else cell


def interval(samples: np.ndarray) -> dict[str, float]:
    low, high = np.percentile(samples, CI)
    return {"mean": float(samples.mean()), "lo": float(low), "hi": float(high)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", type=Path, required=True, help="directory of sweep outputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", help="comma-separated sweep names; default is every one")
    parser.add_argument(
        "--contrasts",
        default="shift:random0",
        help="comma-separated left:right kind pairs to difference on the shared rows",
    )
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = (
        args.labels.split(",")
        if args.labels
        # Smoke runs use a different draw and would fail the pairing check.
        else sorted(
            p.name
            for p in args.sweeps.iterdir()
            if (p / "per_example.jsonl").exists() and not p.name.startswith("smoke")
        )
    )
    if not names:
        raise SystemExit(f"no sweep with per_example.jsonl under {args.sweeps}")

    sweeps = {name: load_sweep(args.sweeps / name) for name in names}
    reference = sweeps[names[0]][0]["example_ids"]
    for name, (summary, _) in sweeps.items():
        if summary["example_ids"] != reference:
            raise SystemExit(f"{name}: evaluation rows differ from {names[0]}; cannot pair")
    n = len(reference)

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, n, size=(args.samples, n))

    def resample(vector: np.ndarray) -> np.ndarray:
        """Bootstrap distribution of the mean, on the shared row draws."""
        return vector[draws].mean(axis=1)

    results: dict[str, dict] = {"sweeps": {}}
    per_sweep = results["sweeps"]
    branch_delta: dict[str, dict[str, np.ndarray]] = {}
    for name, (summary, rows) in sweeps.items():
        seed_rows = rows["C_seed"]
        target_rows = rows[summary["target_label"]]
        seed_boot = resample(seed_rows)
        gap_boot = resample(target_rows) - seed_boot
        gap_point = float(target_rows.mean() - seed_rows.mean())
        cells = {}
        for cell, cell_rows in rows.items():
            if cell in {"C_seed", summary["target_label"]}:
                continue
            delta_boot = resample(cell_rows) - seed_boot
            # R is undefined when the resampled gap vanishes.
            usable = np.abs(gap_boot) > 1e-9
            ratio = np.where(usable, delta_boot / np.where(usable, gap_boot, 1.0), np.nan)
            cells[cell] = {
                "d_f1": interval(delta_boot),
                "recovery": interval(ratio[np.isfinite(ratio)]),
                "recovery_fixed_gap": interval(delta_boot / gap_point),
                "gap_le_zero_frac": float((gap_boot <= 0).mean()),
                "p_delta_le_zero": float((delta_boot <= 0).mean()),
                "point_d_f1": float(cell_rows.mean() - seed_rows.mean()),
                "point_recovery": float((cell_rows.mean() - seed_rows.mean()) / gap_point),
            }
            branch_delta.setdefault(name, {})[cell] = delta_boot
        per_sweep[name] = {
            "gap": {"point": gap_point, **interval(gap_boot)},
            "baseline_f1": float(seed_rows.mean()),
            "target_f1": float(target_rows.mean()),
            "cells": cells,
        }
        best = max(cells.items(), key=lambda kv: kv[1]["point_recovery"])
        print(
            f"{name:20s} gap={gap_point:+.4f} [{per_sweep[name]['gap']['lo']:+.4f},"
            f"{per_sweep[name]['gap']['hi']:+.4f}]  best={best[0]}"
            f" R={best[1]['point_recovery']:+.3f}"
            f" dF1={best[1]['point_d_f1']:+.4f}"
            f" [{best[1]['d_f1']['lo']:+.4f},{best[1]['d_f1']['hi']:+.4f}]",
            flush=True,
        )

    # Pool cells of one kind (paired, same rows).
    modes: dict[str, dict[str, np.ndarray]] = {}
    for name, cells_of in branch_delta.items():
        for cell, delta_boot in cells_of.items():
            modes.setdefault(name, {}).setdefault(mode_of(cell), []).append(delta_boot)
    pooled: dict[str, dict[str, np.ndarray]] = {
        name: {mode: np.mean(draws, axis=0) for mode, draws in by_mode.items()}
        for name, by_mode in modes.items()
    }
    results["modes"] = {}
    for name in sorted(pooled):
        results["modes"][name] = {
            mode: {
                **interval(samples),
                "p_le_zero": float((samples <= 0).mean()),
                "n_cells": len(modes[name][mode]),
                "recovery_fixed_gap": float(samples.mean() / per_sweep[name]["gap"]["point"]),
            }
            for mode, samples in pooled[name].items()
        }
        for mode in sorted(pooled[name]):
            row = results["modes"][name][mode]
            print(
                f"MODE {name:20s} {mode:15s} n={row['n_cells']:2d}"
                f" dF1={row['mean']:+.5f} [{row['lo']:+.5f},{row['hi']:+.5f}]"
                f" R_fixed={row['recovery_fixed_gap']:+.3f} p(<=0)={row['p_le_zero']:.3f}",
                flush=True,
            )

    # Random controls exist only at the best cell, so compare them with that cell alone.
    results["matched_control"] = {}
    matched: dict[str, np.ndarray] = {}
    for name, cells_of in branch_delta.items():
        controls = [cell for cell in cells_of if mode_of(cell).startswith("random")]
        if not controls:
            continue
        suffix = controls[0].split("_", 1)[1]
        partner = f"shift_{suffix}"
        if partner not in cells_of or any(c.split("_", 1)[1] != suffix for c in controls):
            continue
        against = np.mean([cells_of[cell] for cell in controls], axis=0)
        contrast = cells_of[partner] - against
        matched[name] = contrast
        gap_point = per_sweep[name]["gap"]["point"]
        results["matched_control"][name] = {
            "cell": partner,
            "controls": sorted(controls),
            **interval(contrast),
            "p_le_zero": float((contrast <= 0).mean()),
            "recovery_fixed_gap": float(contrast.mean() / gap_point),
        }
        row = results["matched_control"][name]
        print(
            f"MATCHED {name:20s} {partner:18s} shift-random={row['mean']:+.5f}"
            f" [{row['lo']:+.5f},{row['hi']:+.5f}] p(<=0)={row['p_le_zero']:.3f}",
            flush=True,
        )
    for prefix in sorted({name.rsplit("_s", 1)[0] for name in matched}):
        members = [name for name in matched if name.rsplit("_s", 1)[0] == prefix]
        if len(members) < 2:
            continue
        # Gaps differ per seed: convert to recovery before pooling.
        pooled_r = np.mean(
            [matched[name] / per_sweep[name]["gap"]["point"] for name in members], axis=0
        )
        results["matched_control"][f"branch_{prefix}"] = {
            "seeds": members,
            "recovery": interval(pooled_r),
            "p_le_zero": float((pooled_r <= 0).mean()),
        }
        row = results["matched_control"][f"branch_{prefix}"]["recovery"]
        print(
            f"MATCHED BRANCH {prefix:14s} R(shift-random) = {row['mean']:+.3f}"
            f" [{row['lo']:+.3f},{row['hi']:+.3f}]",
            flush=True,
        )

    results["contrasts"] = {}
    for pair in args.contrasts.split(","):
        left, right = pair.split(":")
        for name in sorted(pooled):
            if left not in pooled[name] or right not in pooled[name]:
                continue
            contrast = pooled[name][left] - pooled[name][right]
            results["contrasts"].setdefault(name, {})[pair] = {
                **interval(contrast),
                "p_le_zero": float((contrast <= 0).mean()),
            }
            row = results["contrasts"][name][pair]
            print(
                f"CONTRAST {name:20s} {pair:32s} {row['mean']:+.5f}"
                f" [{row['lo']:+.5f},{row['hi']:+.5f}] p(<=0)={row['p_le_zero']:.3f}",
                flush=True,
            )

    branches: dict[str, dict] = {}
    for prefix in sorted({name.rsplit("_s", 1)[0] for name in pooled}):
        members = [name for name in pooled if name.rsplit("_s", 1)[0] == prefix]
        if len(members) < 2:
            continue
        shared = sorted(set.intersection(*(set(pooled[name]) for name in members)))
        branches[prefix] = {"seeds": members, "modes": {}}
        for mode in shared:
            samples = np.mean([pooled[name][mode] for name in members], axis=0)
            recovery = np.mean(
                [pooled[name][mode] / per_sweep[name]["gap"]["point"] for name in members],
                axis=0,
            )
            branches[prefix]["modes"][mode] = {
                **interval(samples),
                "recovery": interval(recovery),
                "p_le_zero": float((samples <= 0).mean()),
            }
            row = branches[prefix]["modes"][mode]
            print(
                f"BRANCH {prefix:16s} {mode:15s} over {len(members)} seeds:"
                f" dF1={row['mean']:+.5f} [{row['lo']:+.5f},{row['hi']:+.5f}]"
                f" R={row['recovery']['mean']:+.3f}"
                f" [{row['recovery']['lo']:+.3f},{row['recovery']['hi']:+.3f}]"
                f" p(<=0)={row['p_le_zero']:.3f}",
                flush=True,
            )

    results["branches"] = branches
    results["bootstrap"] = {"samples": args.samples, "seed": args.seed, "n_examples": n}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"E3_BOOTSTRAP_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
