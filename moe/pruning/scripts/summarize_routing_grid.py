#!/usr/bin/env python3
"""Expert-usage maps: table, per-cell stage figures, and a depth comparison.

Each cell holds counts of shape ``[layers, experts]`` per prompt stage. Three
things are produced from them.

* **A table.** Concentration (Gini, dead experts, effective experts) and
  displacement from the seed, per cell and per stage. Concentration is computed
  **per layer and then averaged**: each layer routes independently, and pooling
  the counts first washes the per-layer peaks out entirely.
* **A stage map per cell.** A heat map of load over layers and experts at each
  stage. Layers are kept separate for the same reason the table keeps them
  separate: pooling them washes the per-layer peaks out. Experts are ordered
  once, by the base arm's usage, so the same column means the same expert in
  every panel.
* **A depth profile.** Displacement from the seed by layer, which is what can be
  read against the attention profiles.

Displacement uses the L1 distance between usage shares, which equals twice the
total variation and is bounded by 2 whatever the expert count, so Qwen's 128
experts and gpt-oss's 32 sit on the same scale.

    uv run scripts/summarize_routing_grid.py --roots routing_maps_grid \\
        --flat "qwen=routing_maps" --out figures/routing
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mrd_pruning.routing_stats import gini, layer_shares  # noqa: E402

logger = logging.getLogger("summarize_routing")

CELL = re.compile(r"^(prompt|prefix-projected|prefix)-m(\d+)-s(\d+)$")
ALIAS = {"prompt_tuning": "prompt", "prefix_tuning": "prefix",
         "prefix-projected": "prefix"}
STAGE_ORDER = ["system", "virtual", "comment", "question", "template", "answer",
               "__all__"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", type=Path, nargs="*", default=[])
    p.add_argument("--flat", nargs="*", default=[], metavar="LABEL=PATH")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def _row(model: str, name: str, cell: Path, source: str = "v24") -> dict | None:
    npz = cell / "expert_counts.npz"
    meta_path = cell / "meta.json"
    if not (npz.exists() and meta_path.exists()):
        return None
    z = np.load(npz)
    meta = json.loads(meta_path.read_text())
    stages = {}
    for key in z.files:
        if "|" not in key:
            continue
        _, stage = key.split("|", 1)
        stages[stage] = z[key]
    m = CELL.match(name)
    raw = m.group(1) if m else name
    return {
        "model": model, "cell": name, "method": ALIAS.get(raw, raw),
        "m": int(m.group(2)) if m else meta.get("n_virtual") or None,
        "seed": int(m.group(3)) if m else None,
        "source": source,
        "stages": stages, "meta": meta,
    }


# Arms trained on our own pool are far weaker than the published grid - their
# prompt reaches exact 0.700 on validation where ours reaches 0.251 - so putting
# both in one table or one panel invites reading a training-budget gap as a
# method difference. Only GEPA is kept from that pool, because no archived GEPA
# arm exists to replace it.
KEEP_FROM_OUR_POOL = {"gepa"}


def walk(roots, flat) -> list[dict]:
    rows, seen = [], set()
    for root in roots:
        for cell in sorted(Path(root).glob("*/*")):
            if not cell.is_dir():
                continue
            r = _row(cell.parent.name, cell.name, cell)
            if r and (r["model"], r["cell"]) not in seen:
                seen.add((r["model"], r["cell"]))
                rows.append(r)
    for spec in flat:
        label, _, path = spec.partition("=")
        for cell in sorted(Path(path).iterdir()):
            if not cell.is_dir():
                continue
            r = _row(label, cell.name, cell, source="our r1000")
            if r and r["method"] not in KEEP_FROM_OUR_POOL:
                continue
            if r and (r["model"], r["cell"]) not in seen:
                seen.add((r["model"], r["cell"]))
                rows.append(r)
    return rows


shares = layer_shares


def summarise(row: dict, base: dict | None) -> dict:
    """Concentration and displacement, computed per layer and then averaged.

    Concentration is a per-layer property: every layer routes independently and
    has its own hot experts. Pooling the counts across layers first averages
    those peaks away - on Qwen it turns a Gini of 0.730 into 0.234 and a dead
    share of 24.5% into 0.0%, because an expert idle in one layer is busy in
    another. The pooled number is not a weaker version of the right one, it
    answers a different question.
    """
    counts = row["stages"].get("__all__")
    s = shares(counts)
    n_experts = counts.shape[1]

    per_gini, per_dead, per_hot, per_neff = [], [], [], []
    for layer in range(counts.shape[0]):
        p = s[layer]
        if p.sum() <= 0:
            continue
        per_gini.append(gini(p))
        # Threshold from the project's metric reference. Note it does not
        # reproduce the earlier reported 7.88% for Qwen base on `comment` - that
        # figure matches a far stricter cut (~3e-6) or a plain count==0 at
        # n=500. Recorded as an open discrepancy rather than tuned to match.
        per_dead.append(float((p < 1e-4).mean()))
        per_hot.append(float(p.max() * n_experts))
        # Inverse Simpson index, the convention the earlier measurements used.
        # Entropy-based exp(H) is a different number on the same data (45.9 vs
        # 32.6 for Qwen base on `comment`), so mixing the two silently breaks
        # comparison with anything measured before.
        per_neff.append(float(1.0 / np.square(p).sum()))

    out = {
        "gini": float(np.mean(per_gini)),
        "dead_pct": 100.0 * float(np.mean(per_dead)),
        "n_eff": float(np.mean(per_neff)),
        "n_experts": n_experts,
        "hottest": float(np.mean(per_hot)),
    }
    if base is not None and base is not row:
        b = shares(base["stages"]["__all__"])
        out["drift_all"] = float(np.abs(s - b).sum(axis=1).mean())
        out["drift_by_layer"] = np.abs(s - b).sum(axis=1)
        for stage in ("comment", "answer"):
            if stage in row["stages"] and stage in base["stages"]:
                bb = shares(base["stages"][stage])
                ss = shares(row["stages"][stage])
                out[f"drift_{stage}"] = float(np.abs(ss - bb).sum(axis=1).mean())
    return out


def table(rows: list[dict], bases: dict) -> str:
    head = ("| model | method | m | sid | expert layers | Gini | dead % | "
            "n_eff | hot/plain. | drift `__all__` | drift comment | drift answer |")
    out = [head, "|" + "---|" * 13]
    for r in sorted(rows, key=lambda x: (x["model"], x["method"], x["m"] or 0)):
        st = summarise(r, bases.get(r["model"]))
        d = lambda k: f"{st[k]:.3f}" if k in st else " - "  # noqa: E731
        out.append(
            f"| {r['model']} | {r['method']} | {r['m'] or ' - '} | {r['seed'] or ' - '} "
            f"| {st['n_experts']} | {r['stages']['__all__'].shape[0]} "
            f"| {st['gini']:.3f} | {st['dead_pct']:.1f} | {st['n_eff']:.1f} "
            f"| {st['hottest']:.1f} | {d('drift_all')} | {d('drift_comment')} "
            f"| {d('drift_answer')} |")
    return "\n".join(out)


def stage_map(row: dict, base: dict, out: Path) -> Path:
    """Load per layer and expert, one panel per stage.

    A heat map rather than a pooled curve. Summing the counts over layers first
    would hide what the map is for: each layer has its own hot experts, and on
    Qwen pooling turns a per-layer peak of 11.2x uniform into 2.4x. The same
    mistake as in the table, and just as invisible in the output.

    Experts are ordered once, by the seed's pooled usage, so a column means the
    same expert in every panel and in every arm. Ordering each panel by its own
    usage would make every arm look identical.
    """
    stages = [s for s in STAGE_ORDER if s in row["stages"] and s != "__all__"]
    b = base["stages"]["__all__"].sum(axis=0)
    order = np.argsort(-b)
    n_layers, n_experts = row["stages"]["__all__"].shape

    fig, axes = plt.subplots(1, len(stages), figsize=(2.9 * len(stages), 3.8),
                             sharey=True, squeeze=False)
    im = None
    for ax, stage in zip(axes[0], stages):
        counts = row["stages"][stage]
        per = shares(counts)[:, order] * n_experts  # load relative to uniform
        im = ax.imshow(per, aspect="auto", origin="lower", cmap="magma",
                       vmin=0, vmax=4, interpolation="nearest")
        ax.set_title(stage, fontsize=9)
        ax.set_xlabel("Expert \n (base order)", fontsize=7.5)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("layer", fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.01,
                 label="loading/even")
    fig.suptitle(f"{row['model']} · {row['cell']} - loading of experts on layers and "
                 f"stages (n=){row['meta']['n_examples']}, 1.0 = uniform)",
                 fontsize=9)
    path = out / f"stages_{row['model']}_{row['cell']}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def gini_by_depth(rows: list[dict], out: Path) -> Path:
    """Gini against relative depth, one panel per model.

    The summary table gives one number per arm, which hides that concentration
    is not flat with depth: on gpt-oss the base rises from 0.28 at the first
    layer to 0.73 at the last. An arm that looks like a small mean shift can be
    doing all of it in the last third.

    Depth is relative so 48 and 24 layers can be read side by side; the y range
    is shared so the two panels are comparable.
    """
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(6.4 * len(models), 4.2),
                             sharey=True, squeeze=False)
    colors = {"prompt": "#d95f02", "prefix": "#1f78b4", "gepa": "#1b9e77",
              "base": "#000000"}
    dashes = {100: "-", 200: "--", 500: ":"}
    for ax, model in zip(axes[0], models):
        for r in sorted((x for x in rows if x["model"] == model),
                        key=lambda x: (x["method"] != "base", x["method"], x["m"] or 0)):
            counts = r["stages"]["__all__"]
            sh = shares(counts)
            y = np.array([gini(sh[i]) if sh[i].sum() > 0 else np.nan
                          for i in range(counts.shape[0])])
            x = np.arange(len(y)) / (len(y) - 1)
            if r["method"] == "base":
                ax.plot(x, y, color="#000000", linewidth=2.4, label="base", zorder=3)
                continue
            if r["method"] == "gepa":
                ax.plot(x, y, color=colors["gepa"], linewidth=1.9,
                        label="gepa (our pool)", alpha=0.9)
                continue
            label = r["method"] if r["m"] is None else f"{r['method']} m{r['m']}"
            ax.plot(x, y, color=colors.get(r["method"], "#777777"),
                    linestyle=dashes.get(r["m"], "-"), linewidth=1.7, alpha=0.9,
                    label=label)
        n_layers = next(r["stages"]["__all__"].shape[0]
                        for r in rows if r["model"] == model)
        ax.set_title(f"{model} - {n_layers} layering", fontsize=10)
        ax.set_xlabel("Relative depth (0 = first layer, 1 = last)",
                      fontsize=9)
        ax.axhline(0.5, color="#bbbbbb", linewidth=1, linestyle="--", zorder=0)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    axes[0][0].set_ylabel("Gini download experts \n (0 = uniform)", fontsize=9)
    fig.suptitle("Uneven loading depth, stage `__all__`, n = 2000 \n"
                 "gray dotted 0.5 = \"half the experts divide everything equally.\" "
                 "half idle", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = out / "gini_by_depth.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def gini_by_depth_per_stage(rows: list[dict], out: Path) -> Path:
    """Gini against depth, one row of panels per model, one column per stage.

    The pooled ``__all__`` view mixes token populations that route differently:
    a prompt block, the user's text and the generated answer are not the same
    thing, and a mixture of peaked distributions is flatter than any of them. On
    Qwen base that alone moves the mean from 0.730 on `comment` to 0.632 on
    `__all__`. Splitting by stage is what shows where an arm actually changes
    the spread.

    Stages absent from an arm are simply missing: prefix tuning has no
    ``virtual`` block because its states occupy no sequence positions, and the
    PEFT arms have no ``system`` block because they were trained without one.
    """
    models = sorted({r["model"] for r in rows})
    stages = [s for s in STAGE_ORDER
              if any(s in r["stages"] for r in rows) and s != "__all__"]
    stages.append("__all__")
    colors = {"prompt": "#d95f02", "prefix": "#1f78b4", "gepa": "#1b9e77"}
    dashes = {100: "-", 200: "--", 500: ":"}

    fig, axes = plt.subplots(len(models), len(stages),
                             figsize=(2.7 * len(stages), 3.5 * len(models)),
                             sharey=True, squeeze=False)
    for row, model in enumerate(models):
        cells = sorted((r for r in rows if r["model"] == model),
                       key=lambda x: (x["method"] != "base", x["method"], x["m"] or 0))
        for col, stage in enumerate(stages):
            ax = axes[row][col]
            drawn = False
            for r in cells:
                counts = r["stages"].get(stage)
                if counts is None or counts.sum() <= 0:
                    continue
                sh = shares(counts)
                y = np.array([gini(sh[i]) if sh[i].sum() > 0 else np.nan
                              for i in range(counts.shape[0])])
                x = np.arange(len(y)) / (len(y) - 1)
                if r["method"] == "base":
                    ax.plot(x, y, color="#000000", linewidth=2.2, label="base",
                            zorder=3)
                elif r["method"] == "gepa":
                    ax.plot(x, y, color=colors["gepa"], linewidth=1.7, label="gepa")
                else:
                    ax.plot(x, y, color=colors.get(r["method"], "#777777"),
                            linestyle=dashes.get(r["m"], "-"), linewidth=1.5,
                            alpha=0.9, label=f"{r['method']} m{r['m']}")
                drawn = True
            ax.set_title(f"{model} · {stage}" if col == 0 else stage, fontsize=8.5)
            ax.axhline(0.5, color="#cccccc", linewidth=0.9, linestyle="--", zorder=0)
            ax.set_xlim(0, 1)
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=7)
            if row == len(models) - 1:
                ax.set_xlabel("depth", fontsize=7.5)
            if not drawn:
                ax.text(0.5, 0.5, "There is no \n in these arms.", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8,
                        color="#999999")
        axes[row][0].set_ylabel(f"{model}\n Gini download", fontsize=8.5)
        axes[row][-1].legend(frameon=False, fontsize=6.5, ncol=1,
                             loc="lower right")
    fig.suptitle("Uneven loading in depth, separately in stages "
                 "Prompt (n=2000) \n gray dotted 0.5 = \"half experts\" "
                 "He divides everything equally, half stands idle.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / "gini_by_depth_per_stage.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def expert_histograms(rows: list[dict], bases: dict, out: Path) -> list[Path]:
    """Token load per expert, one grid per model: arms down, stages across.

    **Experts are ordered by the seed, once per stage.** On a stage the seed also
    has, the order comes from the seed's own load there; on a stage it does not
    have - ``virtual`` exists only for prompt tuning - the order falls back to
    the seed's ``__all__``. Ordering each panel by its own load would sort every
    arm into the same decreasing curve and make them indistinguishable; a shared
    order is what lets a column be read as "this expert".

    The bars pool assignments over layers, which is what a distribution over
    experts means. Pooling flattens per-layer peaks badly - on Qwen base's
    `comment` the pooled Gini is 0.23 while the per-layer mean is 0.73 - so each
    panel carries both numbers. Annotating only the per-layer one beside a
    pooled picture reads as a contradiction.
    """
    written = []
    stages = [s for s in STAGE_ORDER if any(s in r["stages"] for r in rows)]
    for model in sorted({r["model"] for r in rows}):
        base = bases.get(model)
        if base is None:
            continue
        cells = sorted((r for r in rows if r["model"] == model),
                       key=lambda x: (x["method"] != "base", x["method"], x["m"] or 0))
        present = [st for st in stages
                   if any(st in r["stages"] and r["stages"][st].sum() > 0 for r in cells)]
        n_experts = base["stages"]["__all__"].shape[1]

        orders = {}
        for st in present:
            ref = base["stages"].get(st)
            if ref is None or ref.sum() <= 0:
                ref = base["stages"]["__all__"]
            orders[st] = np.argsort(-ref.sum(axis=0))

        fig, axes = plt.subplots(len(cells), len(present),
                                 figsize=(2.35 * len(present), 1.75 * len(cells)),
                                 sharex=True, sharey=True, squeeze=False)
        for i, r in enumerate(cells):
            for j, st in enumerate(present):
                ax = axes[i][j]
                # Titles go on before the missing-stage shortcut below: the
                # first row is the seed, and the seed has no `virtual` block, so
                # setting them inside the drawing branch leaves that column
                # unlabelled.
                if i == 0:
                    ax.set_title(st, fontsize=8)
                counts = r["stages"].get(st)
                if counts is None or counts.sum() <= 0:
                    ax.text(0.5, 0.5, "stage", ha="center", va="center",
                            transform=ax.transAxes, fontsize=7, color="#bbbbbb")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for side in ("top", "right", "left", "bottom"):
                        ax.spines[side].set_visible(False)
                    continue
                pooled = counts.sum(axis=0)
                pooled = pooled / pooled.sum()
                sh = layer_shares(counts)
                per_layer_gini = float(np.mean(
                    [gini(sh[k]) for k in range(len(sh)) if sh[k].sum() > 0]))
                # Both numbers, because they describe different objects and
                # annotating only the per-layer one next to a pooled picture
                # reads as a contradiction: the bars look near-uniform (Gini
                # ~0.25) while the label says 0.81. Pooling mixes layers whose
                # hot experts differ, so the pooled curve is always flatter.
                pooled_gini = gini(pooled)
                colour = ("#000000" if r["method"] == "base"
                          else {"prompt": "#d95f02", "prefix": "#1f78b4",
                                "gepa": "#1b9e77"}.get(r["method"], "#777777"))
                ax.fill_between(np.arange(n_experts), pooled[orders[st]] * n_experts,
                                step="mid", color=colour, alpha=0.85, linewidth=0)
                ax.axhline(1.0, color="#d62728", linewidth=0.8, linestyle="--")
                ax.text(0.97, 0.90,
                        f"pool {pooled_gini:.2f} • layers {per_layer_gini:.2f}",
                        ha="right", va="top", transform=ax.transAxes,
                        fontsize=6.2, color="#555555")
                ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.tick_params(labelsize=6)
                if i == len(cells) - 1:
                    ax.set_xlabel("expert (base order)", fontsize=6.5)
            name = (r["method"] if r["m"] is None else f"{r['method']}\nm{r['m']}")
            axes[i][0].set_ylabel(name, fontsize=7.5)
        fig.suptitle(f"{model}: loading of stage experts, n = 2000 \n"
                     f"columns are sorted by base, dotted = uniform "
                     f"loading; in the corner of Gini: pool by layers · average by layers",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = out / f"expert_histograms_{model}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written


def depth_figure(rows: list[dict], bases: dict, out: Path) -> Path:
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.0),
                             squeeze=False)
    colors = {"prompt": "#d95f02", "prefix": "#1f78b4"}
    dashes = {100: "-", 200: "--", 500: ":"}
    for ax, model in zip(axes[0], models):
        base = bases.get(model)
        for r in sorted((x for x in rows if x["model"] == model),
                        key=lambda x: (x["method"], x["m"] or 0)):
            if r is base:
                continue
            st = summarise(r, base)
            if "drift_by_layer" not in st:
                continue
            y = st["drift_by_layer"]
            ax.plot(np.arange(len(y)) / (len(y) - 1), y,
                    color=colors.get(r["method"], "#666666"),
                    linestyle=dashes.get(r["m"], "-"), linewidth=1.8,
                    label=f"{r['method']} m{r['m']}")
        ax.set_title(f"{model} Displace the load from the base", fontsize=10)
        ax.set_xlabel("depth")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8, ncol=2)
    axes[0][0].set_ylabel("L1- distance of shares (0..2)")
    fig.suptitle("How much adaptation rearranges the load of experts, by layers",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / "routing_drift_by_depth.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args(argv)
    rows = walk(args.roots, args.flat)
    if not rows:
        raise SystemExit("no expert_counts.npz found")
    args.out.mkdir(parents=True, exist_ok=True)

    bases = {r["model"]: r for r in rows if r["method"] == "base"}
    missing = {r["model"] for r in rows} - set(bases)
    if missing:
        logger.warning("no base arm for %s - drift columns will be empty", missing)

    md = table(rows, bases)
    (args.out / "routing_table.md").write_text(md + "\n")
    print(md)

    for r in rows:
        base = bases.get(r["model"])
        if base is None:
            continue
        stage_map(r, base, args.out)
    logger.info("wrote %d stage maps", len(rows))
    logger.info("wrote %s", gini_by_depth(rows, args.out))
    logger.info("wrote %s", gini_by_depth_per_stage(rows, args.out))
    for path in expert_histograms(rows, bases, args.out):
        logger.info("wrote %s", path)
    logger.info("wrote %s", depth_figure(rows, bases, args.out))
    logger.info("wrote %s", args.out / "routing_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
