#!/usr/bin/env python3
"""Per-layer tables for both measurements, in one place.

The summary tables average over layers, which is right for a headline number and
wrong for reading depth structure: a mean of 0.63 covers layers ranging from 0.4
to 0.9. This writes the layer axis out.

Two outputs per measurement:

* a **full CSV** with every layer of every cell, for anyone who wants to work
  with the numbers;
* a **markdown table at sampled depths**, because 48 layers x 17 cells does not
  fit in a note anyone reads.

Routing metrics follow the project's existing conventions so the numbers can be
compared with earlier measurements: per-layer normalisation, ``n_eff`` as the
inverse Simpson index ``1/sum(p^2)``, ``frac_dead`` at ``p < 1e-4``.

    uv run scripts/export_per_layer_tables.py --out results-files
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mrd_pruning.routing_stats import layer_stats  # noqa: E402

logger = logging.getLogger("per_layer_tables")

CELL = re.compile(r"^(prompt|prefix-projected|prefix)-m(\d+)-s(\d+)$")
ALIAS = {"prompt_tuning": "prompt", "prefix_tuning": "prefix",
         "prefix-projected": "prefix"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--routing", type=Path, nargs="*",
                   default=[Path("routing_maps_grid")])
    p.add_argument("--attention", type=Path, nargs="*",
                   default=[Path("attention_maps_grid")])
    p.add_argument("--stage", default="__all__")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def label(name: str) -> tuple[str, int | None]:
    m = CELL.match(name)
    raw = m.group(1) if m else name
    return ALIAS.get(raw, raw), (int(m.group(2)) if m else None)


def sample_layers(n: int, want: int = 9) -> list[int]:
    """Evenly spaced depths including the first and last."""
    if n <= want:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (want - 1))) for i in range(want)})


def routing_rows(roots, stage: str) -> list[dict]:
    rows = []
    for root in roots:
        for cell in sorted(Path(root).glob("*/*")):
            npz = cell / "expert_counts.npz"
            if not npz.exists():
                continue
            z = np.load(npz)
            key = next((k for k in z.files if k.endswith(f"|{stage}")), None)
            if key is None:
                continue
            counts = z[key]
            base_npz = cell.parent / "base" / "expert_counts.npz"
            base = None
            if base_npz.exists() and cell.name != "base":
                zb = np.load(base_npz)
                bk = next((k for k in zb.files if k.endswith(f"|{stage}")), None)
                base = zb[bk] if bk else None
            method, m = label(cell.name)
            n_experts = counts.shape[1]
            for layer in range(counts.shape[0]):
                p = counts[layer]
                if p.sum() <= 0:
                    continue
                p = p / p.sum()
                row = {"model": cell.parent.name, "method": method, "m": m or "",
                       "layer": layer, "n_experts": n_experts, **layer_stats(p)}
                if base is not None and base[layer].sum() > 0:
                    b = base[layer] / base[layer].sum()
                    row["drift_l1"] = float(np.abs(p - b).sum())
                rows.append(row)
    return rows


def attention_rows(roots) -> list[dict]:
    rows = []
    for root in roots:
        for cell in sorted(Path(root).glob("*/*/attention.json")):
            j = json.loads(cell.read_text())
            method, m = label(cell.parent.name)
            series = (j["adapter_mass_by_layer"] if j["n_adapter_positions"]
                      else j["system_mass_by_layer"])
            kinds = j.get("layer_types") or []
            row_sum = j.get("row_sum_by_layer") or [1.0] * len(series)
            for layer, value in enumerate(series):
                rows.append({
                    "model": cell.parent.parent.name, "method": method, "m": m or "",
                    "layer": layer, "n_layers": j["n_layers"],
                    "block_mass": float(value),
                    "row_sum": float(row_sum[layer]),
                    "sink_mass": float(1.0 - row_sum[layer]),
                    "layer_type": kinds[layer] if layer < len(kinds) else "full_attention",
                })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list({k for r in rows for k in r})
    order = ["model", "method", "m", "layer"]
    keys = order + [k for k in keys if k not in order]
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def md_routing(rows: list[dict]) -> str:
    out = []
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        n_layers = max(r["layer"] for r in sub) + 1
        picks = sample_layers(n_layers)
        cells = sorted({(r["method"], r["m"]) for r in sub},
                       key=lambda x: (x[0], x[1] if x[1] != "" else 0))
        out.append(f"\n#### {model} - Gini by layers of \n")
        out.append("| arm | " + " | ".join(f"L{p}" for p in picks) + " | mean |")
        out.append("|" + "---|" * (len(picks) + 2))
        for method, m in cells:
            vals = {r["layer"]: r["gini"] for r in sub
                    if r["method"] == method and r["m"] == m}
            mean = np.mean(list(vals.values()))
            name = method if m == "" else f"{method} m{m}"
            out.append(f"| {name} | "
                       + " | ".join(f"{vals.get(p, float('nan')):.3f}" for p in picks)
                       + f" | **{mean:.3f}** |")
        out.append(f"\n#### {model} Load offset from base (L1) over layers of \n")
        out.append("| arm | " + " | ".join(f"L{p}" for p in picks) + " | mean |")
        out.append("|" + "---|" * (len(picks) + 2))
        for method, m in cells:
            if method == "base":
                continue
            vals = {r["layer"]: r.get("drift_l1") for r in sub
                    if r["method"] == method and r["m"] == m}
            got = [v for v in vals.values() if v is not None]
            if not got:
                continue
            name = f"{method} m{m}" if m != "" else method
            out.append(f"| {name} | "
                       + " | ".join(f"{vals.get(p) or float('nan'):.3f}" for p in picks)
                       + f" | **{np.mean(got):.3f}** |")
    return "\n".join(out)


def md_attention(rows: list[dict]) -> str:
    out = []
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        n_layers = max(r["layer"] for r in sub) + 1
        picks = sample_layers(n_layers)
        windowed = {r["layer"] for r in sub if r["layer_type"] != "full_attention"}
        cells = sorted({(r["method"], r["m"]) for r in sub},
                       key=lambda x: (x[0], x[1] if x[1] != "" else 0))
        note = (" (Layers with an asterisk window: block outside the window, zero structural)"
                if windowed else "")
        out.append(f"\n#### {model} Share of attention on the block by layers{note}\n")
        head = [f"L{p}" + ("*" if p in windowed else "") for p in picks]
        out.append("| arm | " + " | ".join(head) + " | mean (full) |")
        out.append("|" + "---|" * (len(picks) + 2))
        for method, m in cells:
            vals = {r["layer"]: r["block_mass"] for r in sub
                    if r["method"] == method and r["m"] == m}
            full = [v for layer, v in vals.items() if layer not in windowed]
            name = method if m == "" else f"{method} m{m}"
            out.append(f"| {name} | "
                       + " | ".join(f"{vals.get(p, float('nan')):.3f}" for p in picks)
                       + f" | **{np.mean(full):.3f}** |")
        if windowed:
            out.append(f"\n#### {model}: attention mass on the sink, by layer\n")
            out.append("| arm | " + " | ".join(head) + " | mean |")
            out.append("|" + "---|" * (len(picks) + 2))
            for method, m in cells:
                vals = {r["layer"]: r["sink_mass"] for r in sub
                        if r["method"] == method and r["m"] == m}
                name = method if m == "" else f"{method} m{m}"
                out.append(f"| {name} | "
                           + " | ".join(f"{vals.get(p, float('nan')):.3f}" for p in picks)
                           + f" | **{np.mean(list(vals.values())):.3f}** |")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rr = routing_rows(args.routing, args.stage)
    ar = attention_rows(args.attention)
    logger.info("routing: %d layer-rows | attention: %d layer-rows", len(rr), len(ar))

    write_csv(rr, args.out / "routing_per_layer.csv")
    write_csv(ar, args.out / "attention_per_layer.csv")
    (args.out / "routing_per_layer.md").write_text(md_routing(rr) + "\n")
    (args.out / "attention_per_layer.md").write_text(md_attention(ar) + "\n")
    for name in ("routing_per_layer.csv", "attention_per_layer.csv",
                 "routing_per_layer.md", "attention_per_layer.md"):
        logger.info("wrote %s", args.out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
