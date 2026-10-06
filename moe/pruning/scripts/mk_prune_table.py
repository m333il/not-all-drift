#!/usr/bin/env python3
"""Pruning results, one row per (checkpoint, level).

Scores come from the sweep's own summary. The degeneration columns are computed
here from `results.jsonl`, reusing `peek_answers`' definitions rather than a
second set: what breaks at the larger cut is generation, and a score without
those columns beside it describes two different failures with one curve.
"""
import argparse
import csv, json, pathlib, re, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from peek_answers import END_MARKERS, repeats, said

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--maps", type=pathlib.Path, default=pathlib.Path("routing_maps_final"),
                help="Routing maps as <maps>/<model>/<cell>/expert_counts.npz")
ap.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results/prune"),
                help="Finished pruning cells as <results>/<model>/<cell>/<arm>_pruneNNN/")
ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("expert_tables"))
args = ap.parse_args()
OUT = args.out; OUT.mkdir(parents=True, exist_ok=True)
SHORT = {"base": "base", "prefix-projected-m100-s42": "prefix-100",
         "prefix-projected-m200-s42": "prefix-200", "prefix-projected-m500-s42": "prefix-500",
         "prompt-m100-s42": "prompt-100", "prompt-m200-s42": "prompt-200",
         "prompt-m500-s42": "prompt-500"}
def short(c): return SHORT.get(c, "GEPA" if c.startswith("gepa-") else c)

rows = []
for f in sorted(args.results.rglob("summary.json")):
    if ".no_routing" in str(f): continue
    d = json.loads(f.read_text())
    cell = f.parent.parent.name; model = f.parent.parent.parent.name
    lvl = int(re.search(r"prune(\d+)$", f.parent.name).group(1))
    nE = 32 if model == "gpt-oss" else 128
    pm = d.get("pruned_mass") or {}
    gen = d.get("generation") or {}

    # Degeneration, from the answers themselves.
    deg = {}
    rj = f.parent / "results.jsonl"
    if rj.is_file():
        rr = [json.loads(l) for l in rj.open()]
        raw = [r.get("raw_response") or r.get("response") or "" for r in rr]
        body = [said(r) for r in rr]
        lens = [len(re.sub(r"<\|[a-z_]+\|>", "", t)) for t in body]
        deg = dict(
            no_end_marker=round(sum(1 for t in raw
                                    if not any(m in t for m in END_MARKERS)) / len(rr), 4),
            looping=round(sum(1 for t in body[:300] if repeats(t)) / min(300, len(rr)), 4),
            chars_median=int(np.median(lens)), chars_p99=int(np.percentile(lens, 99)))

    rows.append(dict(
        model=model, arm=short(cell), cell=cell,
        level_pct=round(100 * lvl / nE), experts_cut=lvl,
        experts_kept=nE - lvl, n_experts=nE,
        mass_removed_mean=round(100 * (pm.get("mass_pruned_mean") or 0), 2),
        mass_removed_max=round(100 * (pm.get("mass_pruned_max") or 0), 2),
        f1=round(d.get("f1_mean") or 0, 4),
        exact=round(d.get("exact_mean") or 0, 4),
        empty_rate=round(100 * (d.get("empty_pred_rate") or 0), 2),
        unparsable_rate=round(100 * (d.get("unparsable_rate") or 0), 2),
        mixed_none_rate=round(100 * (d.get("mixed_none_rate") or 0), 2),
        no_end_marker=round(100 * deg.get("no_end_marker", float("nan")), 2) if deg else "",
        looping=round(100 * deg.get("looping", float("nan")), 2) if deg else "",
        chars_median=deg.get("chars_median", ""), chars_p99=deg.get("chars_p99", ""),
        n=d.get("n"), max_new_tokens=gen.get("max_new_tokens"),
        reasoning_effort=gen.get("reasoning_effort", ""),
        moe_kernel=d.get("moe_kernel", ""), seconds=int(d.get("seconds") or 0)))

rows.sort(key=lambda r: (r["model"], r["arm"], r["level_pct"]))
with (OUT / "pruning_results.csv").open("w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"pruning_results.csv: {len(rows)} rows, cells {len({(r['model'],r['arm']) for r in rows})}")
missing = 32 - len(rows)
if missing:
    print(f"{missing} of 32 levels missing")
