#!/usr/bin/env python3
"""Expert-load tables from the frozen routing maps and the pruned counters.

Four files, from coarse to fine:
  expert_load_per_checkpoint.csv  one row per cell
  gini_per_layer.csv              one row per (cell, layer)
  expert_token_distribution.csv   one row per (cell, layer, expert)
  gini_under_mask.csv             survivors before and after the mask

The per-layer Gini is the honest unit here, because the mask is chosen per
layer and a single number over the whole checkpoint averages layers that were
treated differently. The checkpoint table therefore carries a median and the
worst layer beside it rather than a mean alone.
"""
import argparse
import csv, json, pathlib, re, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mrd_pruning.frequency import load_counts_npz

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--maps", type=pathlib.Path, default=pathlib.Path("routing_maps_final"),
                help="Routing maps as <maps>/<model>/<cell>/expert_counts.npz")
ap.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results/prune"),
                help="Finished pruning cells as <results>/<model>/<cell>/<arm>_pruneNNN/")
ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("expert_tables"))
args = ap.parse_args()
OUT = args.out; OUT.mkdir(parents=True, exist_ok=True)

def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64)); n = x.size
    if x.sum() <= 0: return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))

def arm_of(c):
    if c == "base": return "base"
    if c.startswith("gepa-"): return "gepa"
    if c.startswith("prompt-"): return "prompt_tuning"
    return "prefix_tuning"

SHORT = {"base": "base", "prefix-projected-m100-s42": "prefix-100",
         "prefix-projected-m200-s42": "prefix-200", "prefix-projected-m500-s42": "prefix-500",
         "prompt-m100-s42": "prompt-100", "prompt-m200-s42": "prompt-200",
         "prompt-m500-s42": "prompt-500"}
def short(c): return SHORT.get(c, "GEPA" if c.startswith("gepa-") else c)

cells = []
for model in ("qwen", "gpt-oss"):
    root = args.maps / model
    if not root.is_dir(): continue
    for d in sorted(root.iterdir()):
        f = d / "expert_counts.npz"
        if f.is_file(): cells.append((model, d.name, f))

per_ckpt, per_layer, dist = [], [], []
for model, cell, f in cells:
    c = load_counts_npz(f, arm=arm_of(cell), stage="__all__")
    k = np.asarray(c.counts, dtype=np.float64)          # [layers, experts]
    nL, nE = k.shape
    share = k / k.sum(axis=1, keepdims=True).clip(min=1)
    g = np.array([gini(r) for r in k])
    top1 = share.max(axis=1)
    topd = np.sort(share, axis=1)[:, -max(1, nE // 10):].sum(axis=1)
    dead = (k == 0).sum(axis=1)
    per_ckpt.append(dict(
        model=model, arm=short(cell), cell=cell, n_layers=nL, n_experts=nE,
        gini_median=round(float(np.median(g)), 4),
        gini_mean=round(float(g.mean()), 4),
        gini_worst_layer=round(float(g.max()), 4),
        worst_layer_index=int(g.argmax()),
        top1_expert_share_median=round(float(np.median(top1)), 4),
        top10pct_share_median=round(float(np.median(topd)), 4),
        unused_experts_per_layer=round(float(dead.mean()), 2),
        total_assignments=int(k.sum())))
    for l in range(nL):
        per_layer.append(dict(
            model=model, arm=short(cell), layer=l, n_experts=nE,
            gini=round(float(g[l]), 4),
            top1_expert_share=round(float(top1[l]), 4),
            top10pct_share=round(float(topd[l]), 4),
            unused_experts=int(dead[l]),
            assignments=int(k[l].sum())))
        order = np.argsort(share[l])[::-1]
        for rank, e in enumerate(order):
            if share[l, e] <= 0 and rank > 0: continue
            dist.append(dict(model=model, arm=short(cell), layer=l,
                             expert=int(e), rank=rank + 1,
                             assignments=int(k[l, e]),
                             share=round(float(share[l, e]), 8)))

# --- survivors under the mask ---
under = []
for f in sorted(args.results.rglob("expert_counts_pruned.npz")):
    lvl = int(re.search(r"prune(\d+)$", f.parent.name).group(1))
    cell = f.parent.parent.name; model = f.parent.parent.parent.name
    cut_map = json.loads((f.parent / "pruned_experts.json").read_text())
    z = np.load(f); after = z[z.files[0]]
    cal = load_counts_npz(args.maps/model/cell/"expert_counts.npz",
                          arm=arm_of(cell), stage="__all__")
    before = np.asarray(cal.counts, dtype=np.float64)
    gb, ga, deads = [], [], []
    for l in range(after.shape[0]):
        cut = set(cut_map.get(str(l), cut_map.get(l, [])))
        surv = [e for e in range(after.shape[1]) if e not in cut]
        if not surv or after[l, surv].sum() <= 0: continue
        gb.append(gini(before[l, surv])); ga.append(gini(after[l, surv]))
        deads.append(float((after[l, surv] == 0).sum()))
    if not gb: continue
    under.append(dict(
        model=model, arm=short(cell), level_experts_cut=lvl,
        level_pct=round(100 * lvl / after.shape[1]),
        survivors=after.shape[1] - lvl,
        gini_before_median=round(float(np.median(gb)), 4),
        gini_after_median=round(float(np.median(ga)), 4),
        delta=round(float(np.median(ga) - np.median(gb)), 4),
        gini_after_worst=round(float(np.max(ga)), 4),
        unused_survivors_per_layer=round(float(np.mean(deads)), 2)))

def dump(name, rows):
    if not rows: return
    with (OUT / name).open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{name}: {len(rows)} rows")

dump("expert_load_per_checkpoint.csv", per_ckpt)
dump("gini_per_layer.csv", per_layer)
dump("expert_token_distribution.csv", dist)
dump("gini_under_mask.csv", under)
