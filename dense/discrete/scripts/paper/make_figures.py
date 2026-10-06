"""Build the dense-section figures of the paper.

Inputs: ``--figdata`` (lens_curves.csv, selected_layer.json, dense_appendix_e3.json
from ``export_appendix_data.py``) and ``--attention`` (attention and masking JSON files).
Table values for the quality and steering panels are inlined below.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import numpy as np
from scipy.stats import gaussian_kde

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
FIGDATA = HERE / "figdata"
FIGURES = HERE / "figures_dense"
ATTENTION = HERE / "attention"

# ---------------------------------------------------------------- style ----

COLORS = {
    "seed": "#6E6E6E",
    "discrete": "#27AE60",
    "prompt": "#F2994A",
    "prefix": "#2F80ED",
    "placebo": "#7B61A8",
    "control": "#EB5757",
}
MARKERS = {"seed": "o", "discrete": "s", "prompt": "^", "prefix": "D"}
LABELS = {
    "seed": "unadapted seed",
    "discrete": "GEPA",
    "prompt": "prompt tuning",
    "prefix": "prefix tuning",
}
GRID = dict(linestyle=(0, (3, 2)), linewidth=0.45, color="#D8CFC2", alpha=0.78)

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.65,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "legend.fontsize": 7.0,
    "legend.frameon": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def dress(ax, title=None):
    """Apply the shared axis decoration."""
    if title is not None:
        ax.set_title(title, fontsize=9.5, fontweight="semibold", pad=4)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#B9B1A8")
        spine.set_linewidth(0.7)
    ax.tick_params(axis="both", which="major", color="#9A8F84", labelcolor="0.15")


def frame(legend):
    legend.get_frame().set_facecolor("#FFFBFB")
    legend.get_frame().set_edgecolor("#D0D0D0")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_alpha(1.0)
    return legend


def save(fig, name, rect=None):
    fig.tight_layout(pad=0.25, rect=rect)
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    logger.info("wrote %s", path)


# ----------------------------------------------------------------- data ----

# Held-out test performance, 3000 rows, empty-aware samples F1 (tab:quality).
QUALITY = {
    "discrete": {200: (0.6077, 0.6604), 500: (0.5888, 0.6526), 1000: (0.6190, 0.6598)},
    "prompt": {200: (0.7192, 0.7336), 500: (0.7662, 0.7717), 1000: (0.8003, 0.8017)},
    "prefix": {200: (0.7443, 0.7445), 500: (0.7951, 0.7952), 1000: (0.8117, 0.8118)},
}
SEED_QUALITY = (0.2837, 0.6092)

# Recovered fraction R against a norm-matched random direction (tab:steering).
STEERING = {
    "prefix": {"shift": (0.019, -0.009, 0.052), "random": (0.019, -0.004, 0.043)},
    "discrete": {"shift": (0.023, -0.117, 0.174), "random": (0.024, -0.083, 0.127)},
}

BRANCH_OF_PREFIX = {"C_adapt": "discrete", "C_prompt": "prompt", "C_prefix": "prefix"}


def lens_curves():
    """condition -> per-layer tuned-lens margin, from the 64 scored runs."""
    curves: dict[str, np.ndarray] = {}
    raw: dict[str, dict[int, float]] = defaultdict(dict)
    with (FIGDATA / "lens_curves.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw[row["condition"]][int(row["layer"])] = float(row["margin"])
    for condition, by_layer in raw.items():
        curves[condition] = np.array([by_layer[k] for k in sorted(by_layer)])
    return curves


def onset(curve: np.ndarray, seed: np.ndarray):
    """First layer from which the curve stays above the seed to the top."""
    for layer in range(len(curve)):
        if np.all(curve[layer:] > seed[layer:]):
            return layer
    return None


def by_branch(curves):
    grouped = defaultdict(list)
    for condition, curve in curves.items():
        for prefix, branch in BRANCH_OF_PREFIX.items():
            if condition.startswith(prefix):
                grouped[branch].append(curve)
    return {k: np.vstack(v) for k, v in grouped.items()}


# -------------------------------------------------------------- figures ----

def figure_quality():
    """Strict against set parser: how much of each gain is format."""
    fig, ax = plt.subplots(figsize=(3.9, 2.6))
    x = [0, 1]
    for branch in ("discrete", "prompt", "prefix"):
        for index, pool in enumerate((200, 500, 1000)):
            strict, loose = QUALITY[branch][pool]
            ax.plot(
                x, [strict, loose],
                color=COLORS[branch],
                marker=MARKERS[branch],
                linewidth=1.45,
                markersize=4.3,
                markerfacecolor=COLORS[branch],
                markeredgecolor="0.15",
                markeredgewidth=0.45,
                alpha=0.55 + 0.2 * index,
                label=LABELS[branch] if index == 2 else None,
                zorder=3,
            )
    ax.plot(
        x, list(SEED_QUALITY),
        color=COLORS["seed"], marker=MARKERS["seed"], linestyle="--",
        linewidth=1.45, markersize=4.3, markerfacecolor=COLORS["seed"],
        markeredgecolor="0.15", markeredgewidth=0.45,
        label=LABELS["seed"], zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["strict parser", "set parser"])
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylabel("samples $F_1$", labelpad=2)
    dress(ax)
    frame(ax.legend(loc="lower right", handlelength=1.6, handletextpad=0.35,
                    labelspacing=0.38, borderpad=0.42))
    save(fig, "quality_format.pdf")


def figure_lens():
    """Where the label margin starts to exceed the seed."""
    curves = lens_curves()
    seed = curves["C_seed"]
    grouped = by_branch(curves)
    layers = np.arange(len(seed))

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    ax = axes[0]
    for branch in ("discrete", "prompt", "prefix"):
        block = grouped[branch]
        ax.fill_between(layers, block.min(0), block.max(0),
                        color=COLORS[branch], alpha=0.16, linewidth=0, zorder=2)
        ax.plot(layers, block.mean(0), color=COLORS[branch], linewidth=1.45,
                marker=MARKERS[branch], markevery=6, markersize=4.0,
                markerfacecolor=COLORS[branch], markeredgecolor="0.15",
                markeredgewidth=0.45, label=LABELS[branch], zorder=3)
    ax.plot(layers, seed, color=COLORS["seed"], linestyle="--", linewidth=1.45,
            marker=MARKERS["seed"], markevery=6, markersize=4.0,
            markerfacecolor=COLORS["seed"], markeredgecolor="0.15",
            markeredgewidth=0.45, label=LABELS["seed"], zorder=4)
    ax.set_xlabel("layer", labelpad=2)
    ax.set_ylabel(r"label margin $\mu_\ell$ (nats)", labelpad=2)
    ax.set_xlim(0, len(seed) - 1)
    dress(ax)
    frame(ax.legend(loc="upper left", handlelength=1.6, handletextpad=0.35,
                    labelspacing=0.35, borderpad=0.4))

    ax = axes[1]
    width = 0.30
    for index, branch in enumerate(("discrete", "prompt", "prefix")):
        onsets = [onset(c, seed) for c in grouped[branch]]
        onsets = [o for o in onsets if o is not None]
        counts = np.bincount(onsets, minlength=len(seed)).astype(float)
        counts /= counts.sum()
        shown = np.arange(len(seed))
        ax.bar(shown + (index - 1) * width, counts, width=width,
               color=COLORS[branch], edgecolor="0.15", linewidth=0.4,
               alpha=0.9, label=LABELS[branch], zorder=3)
    ax.set_xlabel("onset layer", labelpad=2)
    ax.set_ylabel("fraction of cells", labelpad=2)
    ax.set_xlim(2.5, 23.5)
    dress(ax)
    frame(ax.legend(loc="upper left", handlelength=1.6, handletextpad=0.35,
                    labelspacing=0.35, borderpad=0.4))

    save(fig, "lens_onset.pdf")


def figure_probe():
    """Linear decodability by layer, against what the model actually emits."""
    payload = json.loads((FIGDATA / "selected_layer.json").read_text())
    scores = payload["mean_validation_scores"]
    selected = payload["selected_layer"]
    layers = np.array(sorted(int(k) for k in scores))
    values = np.array([scores[str(k)] for k in layers])

    fig, ax = plt.subplots(figsize=(3.9, 2.6))
    ax.plot(layers, values, color=COLORS["prefix"], linewidth=1.45, marker="o",
            markevery=3, markersize=4.0, markerfacecolor=COLORS["prefix"],
            markeredgecolor="0.15", markeredgewidth=0.45,
            label="linear probe", zorder=4)
    ax.axvline(selected, color="0.35", linestyle=(0, (1, 2)), linewidth=0.9, zorder=2)
    ax.scatter([selected], [scores[str(selected)]], s=34, zorder=5,
               facecolor="#FFFBFB", edgecolor="0.15", linewidth=0.8)
    ax.annotate(rf"$\ell^\star={selected}$, {scores[str(selected)]:.4f}",
                xy=(selected, scores[str(selected)]), xytext=(-6, -34),
                textcoords="offset points", ha="right", fontsize=7.2, color="0.2",
                arrowprops=dict(arrowstyle="-", linewidth=0.6, color="0.45",
                                shrinkA=1.0, shrinkB=3.0))
    ax.axhline(SEED_QUALITY[1], color=COLORS["seed"], linestyle="--",
               linewidth=1.1, zorder=2, label="model output, set parser")
    ax.axhline(SEED_QUALITY[0], color=COLORS["control"], linestyle="-.",
               linewidth=1.1, zorder=2, label="model output, strict parser")
    ax.set_xlabel("layer", labelpad=2)
    ax.set_ylabel("samples $F_1$", labelpad=2)
    ax.set_xlim(0, layers.max())
    dress(ax)
    frame(ax.legend(loc="lower right", handlelength=1.6, handletextpad=0.35,
                    labelspacing=0.38, borderpad=0.42))
    save(fig, "probe_layers.pdf")


def figure_attention():
    """What the model reads: the discrete instruction and the trainable vectors."""
    profile = json.loads((ATTENTION / "attn_gepa_instruction_civil2_2b.json").read_text())
    segments = json.loads((ATTENTION / "attn_segmass_civil2_2b.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    ax = axes[0]
    styles = {"L12": (0, (1, 1.6)), "L19": (0, (4, 1.8)), "L25": "-"}
    for layer, style in styles.items():
        values = np.asarray(profile["gepa"][layer], dtype=float)
        position = np.linspace(0.0, 1.0, len(values))
        cumulative = np.cumsum(values) / values.sum()
        ax.plot(position, cumulative, color=COLORS["discrete"], linestyle=style,
                linewidth=1.5, label=f"layer {layer[1:]}", zorder=3)
    ax.axvspan(0.18, 0.82, color="#D8CFC2", alpha=0.42, linewidth=0, zorder=1)
    ax.text(0.50, 0.11, "per-label criteria\n($\\approx$60% of tokens)",
            ha="center", va="center", fontsize=6.8, color="0.3", zorder=4)
    ax.set_xlabel("position along the optimized instruction", labelpad=2)
    ax.set_ylabel(r"cumulative attention mass $M_\ell(k)$", labelpad=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    dress(ax)
    frame(ax.legend(loc="upper left", handlelength=1.9, handletextpad=0.35,
                    labelspacing=0.35, borderpad=0.4))

    ax = axes[1]
    for branch in ("prefix", "prompt"):
        values = np.asarray(segments["vt"][branch], dtype=float)
        ax.plot(np.arange(len(values)), values, color=COLORS[branch],
                linewidth=1.45, marker=MARKERS[branch], markevery=5,
                markersize=4.0, markerfacecolor=COLORS[branch],
                markeredgecolor="0.15", markeredgewidth=0.45,
                label=LABELS[branch], zorder=3)
    ax.set_xlabel("layer", labelpad=2)
    ax.set_ylabel("mass on trainable vectors", labelpad=2)
    ax.set_xlim(0, len(segments["vt"]["prefix"]) - 1)
    ax.set_ylim(0, 1.0)
    dress(ax)
    frame(ax.legend(loc="center left", handlelength=1.6, handletextpad=0.35,
                    labelspacing=0.38, borderpad=0.42))

    save(fig, "attention.pdf")


def figure_steering():
    """Recovered fraction against a norm-matched random direction."""
    fig, ax = plt.subplots(figsize=(4.3, 2.2))
    rows = [("prefix", "shift"), ("prefix", "random"),
            ("discrete", "shift"), ("discrete", "random")]
    ticks, names = [], []
    for index, (branch, kind) in enumerate(rows):
        y = len(rows) - 1 - index
        point, low, high = STEERING[branch][kind]
        color = COLORS[branch] if kind == "shift" else COLORS["control"]
        ax.plot([low, high], [y, y], color=color, linewidth=1.5, zorder=3,
                solid_capstyle="butt")
        ax.plot([low, low, high, high], [y - 0.1, y + 0.1, y + 0.1, y - 0.1],
                ls="none", zorder=3)
        for bound in (low, high):
            ax.plot([bound, bound], [y - 0.11, y + 0.11], color=color,
                    linewidth=1.2, zorder=3)
        ax.plot([point], [y], marker="o" if kind == "shift" else "s",
                color=color, markersize=4.6, markerfacecolor=color,
                markeredgecolor="0.15", markeredgewidth=0.45, zorder=4)
        ticks.append(y)
        names.append(f"{LABELS[branch]}\n{'mean shift' if kind == 'shift' else 'random direction'}")
    ax.axvline(0.0, color="0.45", linewidth=0.8, zorder=2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(names, fontsize=7.2)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-0.2, 0.24)
    ax.set_xlabel("recovered fraction $R$ [95% CI]", labelpad=2)
    dress(ax)
    save(fig, "steering.pdf")


def figure_masking():
    """Subset accuracy before and after masking the adapted context."""
    payload = json.loads((ATTENTION / "causal_masking.json").read_text())["civil2"]
    order = [("prefix", "prefix"), ("prompt", "prompt"), ("gepa", "discrete")]
    zero_shot = payload["zs"]["baseline"]["subset_accuracy"]

    fig, ax = plt.subplots(figsize=(3.9, 2.4))
    width = 0.34
    for index, (key, branch) in enumerate(order):
        base = payload[key]["baseline"]["subset_accuracy"]
        masked = payload[key]["masked"]["subset_accuracy"]
        ax.bar(index - width / 2, base, width=width, color=COLORS[branch],
               edgecolor="0.15", linewidth=0.4, alpha=0.92, zorder=3,
               label="adapted" if index == 0 else None)
        ax.bar(index + width / 2, masked, width=width, color=COLORS[branch],
               edgecolor="0.15", linewidth=0.4, alpha=0.32, hatch="////",
               zorder=3, label="context masked" if index == 0 else None)
        ax.annotate(f"{masked:.4f}", xy=(index + width / 2, masked),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=6.8, color="0.2")
    ax.axhline(zero_shot, color=COLORS["seed"], linestyle="--", linewidth=1.2,
               zorder=2, label=f"unadapted seed ({zero_shot:.4f})")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS[b] for _, b in order], fontsize=7.6)
    ax.set_ylabel("subset accuracy", labelpad=2)
    ax.set_ylim(0, 0.86)
    dress(ax)
    frame(ax.legend(loc="upper right", handlelength=1.6, handletextpad=0.35,
                    labelspacing=0.38, borderpad=0.42))
    save(fig, "causal_masking.pdf")



DEPTHS = ("L12", "L19", "L25")
DEPTH_ALPHA = {"L12": 0.38, "L19": 0.68, "L25": 1.0}


def figure_attention_cumulative():
    """Cumulative attention along the instruction and along the virtual tokens."""
    payload = json.loads((ATTENTION / "attn_cum_civil2_2b.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6))

    ax = axes[0]
    for depth in DEPTHS:
        values = np.asarray(payload["instruction"]["gepa"][depth], dtype=float)
        ax.plot(np.linspace(0.0, 1.0, len(values)), values,
                color=COLORS["discrete"], alpha=DEPTH_ALPHA[depth],
                linewidth=1.5, label=f"GEPA {depth}", zorder=3)
    ax.set_xlabel("relative position in instruction", labelpad=2)
    ax.set_ylabel("cumulative attention mass", labelpad=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    dress(ax)
    frame(ax.legend(loc="upper left", handlelength=1.7, handletextpad=0.35,
                    labelspacing=0.32, borderpad=0.4))

    ax = axes[1]
    for branch in ("prefix", "prompt"):
        for depth in DEPTHS:
            values = np.asarray(payload["vt_avg"][branch][depth], dtype=float)
            ax.plot(np.linspace(0.0, 1.0, len(values)), values,
                    color=COLORS[branch], alpha=DEPTH_ALPHA[depth],
                    linewidth=1.5, label=f"{branch} {depth}", zorder=3)
    ax.set_xlabel("relative position among virtual tokens", labelpad=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    dress(ax)
    handles, names = ax.get_legend_handles_labels()
    frame(ax.legend(handles, names, loc="upper left", ncol=2, handlelength=1.7,
                    handletextpad=0.35, labelspacing=0.32, columnspacing=0.8,
                    borderpad=0.4))

    save(fig, "attention_cumulative.pdf")


def appendix_payload():
    """Load the compact E3 export used only by appendix figures."""
    return json.loads((FIGDATA / "dense_appendix_e3.json").read_text())


def figure_shift_cosine_distributions():
    """Per-example cosine geometry for matched prompt and prefix capacities."""
    payload = appendix_payload()["cosine_distributions"]
    pair_styles = {
        "prefix_prompt": ("prefix–prompt", COLORS["prompt"], "-"),
        "gepa_prompt": ("GEPA–prompt", COLORS["discrete"], (0, (4, 1.8))),
        "gepa_prefix": ("GEPA–prefix", COLORS["prefix"], (0, (1, 1.6))),
    }
    grid = np.linspace(-0.42, 0.92, 420)
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.85), sharex=True)

    for layer_index, (ax, block) in enumerate(zip(axes, payload["block_labels"], strict=True)):
        for pair, (label, color, linestyle) in pair_styles.items():
            values = np.asarray(payload["pairs"][pair][layer_index], dtype=float)
            density = gaussian_kde(values)(grid)
            ax.fill_between(grid, density, color=color, alpha=0.08, linewidth=0)
            ax.plot(grid, density, color=color, linestyle=linestyle,
                    linewidth=1.35, label=label, zorder=3)
        ax.axvline(0.0, color="0.50", linewidth=0.65, zorder=2)
        ax.set_title(f"block {block}", fontsize=8.2, fontweight="semibold", pad=3)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_xticks([-0.25, 0.25, 0.75])
        ax.tick_params(axis="y", labelleft=False)
        dress(ax)
    axes[0].set_ylabel("density", labelpad=2)
    axes[2].set_xlabel("cosine between seed-relative shifts", labelpad=2)
    handles, names = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, names, loc="upper center", ncol=3,
                        bbox_to_anchor=(0.5, 1.08), handlelength=1.9,
                        handletextpad=0.4, columnspacing=1.0, borderpad=0.38)
    frame(legend)
    save(fig, "shift_cosine_distributions.pdf", rect=(0, 0, 1, 0.84))


def figure_shift_structure():
    """Mean-shift energy and rank of the input-dependent remainder."""
    payload = appendix_payload()["rank_geometry"]
    layers = np.asarray(payload["layers"], dtype=int)
    styles = {
        "gepa": ("GEPA", COLORS["discrete"], MARKERS["discrete"], "-"),
        "prompt": ("prompt", COLORS["prompt"], MARKERS["prompt"], "-"),
        "prefix": ("prefix", COLORS["prefix"], MARKERS["prefix"], "-"),
        "padding": ("padding", COLORS["placebo"], "P", (0, (4, 1.8))),
    }
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), sharex=True)
    metrics = (
        ("eta", r"mean-shift energy $\eta_\ell$", (0.35, 1.01)),
        ("erank_centred", "effective rank of the\ncentered remainder", (320, 710)),
    )
    for ax, (metric, ylabel, ylim) in zip(axes, metrics, strict=True):
        for condition, (label, color, marker, linestyle) in styles.items():
            summaries = payload["conditions"][condition][metric]
            mean = np.asarray([row["mean"] for row in summaries])
            low = np.asarray([row["min"] for row in summaries])
            high = np.asarray([row["max"] for row in summaries])
            if np.any(high > low):
                ax.fill_between(layers, low, high, color=color, alpha=0.10,
                                linewidth=0, zorder=2)
            ax.plot(layers, mean, color=color, marker=marker, linestyle=linestyle,
                    markevery=5, linewidth=1.35, markersize=3.5,
                    markerfacecolor=color, markeredgecolor="0.15",
                    markeredgewidth=0.35, label=label, zorder=3)
        ax.set_xlabel("layer", labelpad=2)
        ax.set_ylabel(ylabel, labelpad=2, linespacing=1.15,
                      multialignment="center")
        ax.set_xlim(1, layers.max())
        ax.set_ylim(*ylim)
        dress(ax)
    handles, names = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, names, loc="upper center", ncol=4,
                        bbox_to_anchor=(0.5, 1.09), fontsize=6.6,
                        handlelength=1.5, handletextpad=0.3,
                        columnspacing=0.7, borderpad=0.35)
    frame(legend)
    save(fig, "shift_structure_controls.pdf")


def figure_movement_utility():
    """Contrast representation distance with held-out set-based quality."""
    payload = appendix_payload()["movement_geometry"]
    scores = {
        "seed": 0.6092,
        "gepa": 0.6598,
        "prompt": 0.8097621693,
        "prefix": 0.8088484127,
        "padding": 0.5575,
    }
    styles = {
        "seed": ("seed", COLORS["seed"], MARKERS["seed"]),
        "gepa": ("GEPA", COLORS["discrete"], MARKERS["discrete"]),
        "prompt": ("prompt", COLORS["prompt"], MARKERS["prompt"]),
        "prefix": ("prefix", COLORS["prefix"], MARKERS["prefix"]),
        "padding": ("padding", COLORS["placebo"], "P"),
    }
    offsets = {
        "seed": (5, 1),
        "gepa": (5, 1),
        "prompt": (-5, 7),
        "prefix": (-5, -11),
        "padding": (5, 1),
    }
    fig, ax = plt.subplots(figsize=(3.65, 2.6))
    for condition, (label, color, marker) in styles.items():
        if condition == "seed":
            point = low = high = 0.0
        else:
            summary = payload["conditions"][condition]
            point, low, high = summary["mean"], summary["min"], summary["max"]
        ax.errorbar(
            point,
            scores[condition],
            xerr=[[point - low], [high - point]],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.0,
            markersize=5.0,
            markeredgecolor="0.15",
            markeredgewidth=0.45,
            zorder=4,
        )
        ax.annotate(label, (point, scores[condition]), xytext=offsets[condition],
                    textcoords="offset points", fontsize=7.0,
                    ha="left" if offsets[condition][0] > 0 else "right")
    ax.set_xlabel(r"centered distance from seed, $1-\cos$", labelpad=2)
    ax.set_ylabel(r"held-out set samples $F_1$", labelpad=2)
    ax.set_xlim(-0.03, 0.94)
    ax.set_ylim(0.53, 0.84)
    dress(ax)
    save(fig, "movement_utility.pdf")



# --------------------------------------------------- half-column variants ---
# Rendered at subfigure size (0.49 of a 5.5in column) so the text is not scaled down.
HALF = (2.68, 2.10)


def figure_quality_compact():
    fig, ax = plt.subplots(figsize=HALF)
    x = [0, 1]
    for branch in ("discrete", "prompt", "prefix"):
        for index, pool in enumerate((200, 500, 1000)):
            strict, loose = QUALITY[branch][pool]
            ax.plot(x, [strict, loose], color=COLORS[branch],
                    marker=MARKERS[branch], linewidth=1.3, markersize=3.6,
                    markerfacecolor=COLORS[branch], markeredgecolor="0.15",
                    markeredgewidth=0.4, alpha=0.55 + 0.2 * index,
                    label=LABELS[branch] if index == 2 else None, zorder=3)
    ax.plot(x, list(SEED_QUALITY), color=COLORS["seed"], marker=MARKERS["seed"],
            linestyle="--", linewidth=1.3, markersize=3.6,
            markerfacecolor=COLORS["seed"], markeredgecolor="0.15",
            markeredgewidth=0.4, label="seed", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["strict", "set"])
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylabel("samples $F_1$", labelpad=2)
    dress(ax)
    frame(ax.legend(loc="lower right", fontsize=6.0, handlelength=1.3,
                    handletextpad=0.3, labelspacing=0.28, borderpad=0.35))
    save(fig, "quality_format_half.pdf")


def figure_probe_compact():
    payload = json.loads((FIGDATA / "selected_layer.json").read_text())
    scores, selected = payload["mean_validation_scores"], payload["selected_layer"]
    layers = np.array(sorted(int(k) for k in scores))
    values = np.array([scores[str(k)] for k in layers])
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(layers, values, color=COLORS["prefix"], linewidth=1.3, marker="o",
            markevery=4, markersize=3.4, markerfacecolor=COLORS["prefix"],
            markeredgecolor="0.15", markeredgewidth=0.4, label="probe", zorder=4)
    ax.axvline(selected, color="0.35", linestyle=(0, (1, 2)), linewidth=0.8, zorder=2)
    ax.scatter([selected], [scores[str(selected)]], s=26, zorder=5,
               facecolor="#FFFBFB", edgecolor="0.15", linewidth=0.7)
    ax.annotate(rf"$\ell^\star={selected}$", xy=(selected, scores[str(selected)]),
                xytext=(-4, -12), textcoords="offset points", ha="right",
                fontsize=6.4, color="0.2")
    ax.axhline(SEED_QUALITY[1], color=COLORS["seed"], linestyle="--",
               linewidth=1.0, zorder=2, label="set parser")
    ax.axhline(SEED_QUALITY[0], color=COLORS["control"], linestyle="-.",
               linewidth=1.0, zorder=2, label="strict parser")
    ax.set_xlabel("layer", labelpad=2)
    ax.set_ylabel("samples $F_1$", labelpad=2)
    ax.set_xlim(0, layers.max())
    dress(ax)
    frame(ax.legend(loc="lower right", fontsize=6.0, handlelength=1.3,
                    handletextpad=0.3, labelspacing=0.28, borderpad=0.35))
    save(fig, "probe_layers_half.pdf")


def figure_masking_compact():
    payload = json.loads((ATTENTION / "causal_masking.json").read_text())["civil2"]
    order = [("prefix", "prefix"), ("prompt", "prompt"), ("gepa", "discrete")]
    zero_shot = payload["zs"]["baseline"]["subset_accuracy"]
    fig, ax = plt.subplots(figsize=HALF)
    width = 0.34
    for index, (key, branch) in enumerate(order):
        base = payload[key]["baseline"]["subset_accuracy"]
        masked = payload[key]["masked"]["subset_accuracy"]
        ax.bar(index - width / 2, base, width=width, color=COLORS[branch],
               edgecolor="0.15", linewidth=0.35, alpha=0.92, zorder=3,
               label="adapted" if index == 0 else None)
        ax.bar(index + width / 2, masked, width=width, color=COLORS[branch],
               edgecolor="0.15", linewidth=0.35, alpha=0.32, hatch="////",
               zorder=3, label="masked" if index == 0 else None)
    ax.axhline(zero_shot, color=COLORS["seed"], linestyle="--", linewidth=1.0,
               zorder=2, label=f"unadapted seed ({zero_shot:.3f})")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["prefix", "prompt", "GEPA"], fontsize=7.0)
    ax.set_ylabel("subset accuracy", labelpad=2)
    ax.set_ylim(0, 0.92)
    dress(ax)
    frame(ax.legend(loc="upper right", fontsize=6.0, handlelength=1.3,
                    handletextpad=0.3, labelspacing=0.28, borderpad=0.35))
    save(fig, "causal_masking_half.pdf")


def figure_steering_compact():
    fig, ax = plt.subplots(figsize=(2.68, 1.75))
    rows = [("prefix", "shift"), ("prefix", "random"),
            ("discrete", "shift"), ("discrete", "random")]
    ticks, names = [], []
    for index, (branch, kind) in enumerate(rows):
        y = len(rows) - 1 - index
        point, low, high = STEERING[branch][kind]
        color = COLORS[branch] if kind == "shift" else COLORS["control"]
        ax.plot([low, high], [y, y], color=color, linewidth=1.3, zorder=3)
        for bound in (low, high):
            ax.plot([bound, bound], [y - 0.1, y + 0.1], color=color,
                    linewidth=1.0, zorder=3)
        ax.plot([point], [y], marker="o" if kind == "shift" else "s",
                color=color, markersize=3.8, markerfacecolor=color,
                markeredgecolor="0.15", markeredgewidth=0.4, zorder=4)
        ticks.append(y)
        names.append(("prefix" if branch == "prefix" else "GEPA")
                     + (" shift" if kind == "shift" else " random"))
    ax.axvline(0.0, color="0.45", linewidth=0.7, zorder=2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(names, fontsize=6.4)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-0.16, 0.21)
    ax.set_xlabel("recovered fraction $R$", labelpad=2)
    dress(ax)
    save(fig, "steering_half.pdf")


def main():
    global FIGDATA, FIGURES, ATTENTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figdata", type=Path, default=FIGDATA)
    parser.add_argument("--attention", type=Path, default=ATTENTION)
    parser.add_argument("--output", type=Path, default=FIGURES)
    args = parser.parse_args()
    FIGDATA, ATTENTION, FIGURES = args.figdata, args.attention, args.output
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_quality()
    figure_lens()
    figure_probe()
    figure_attention()
    figure_attention_cumulative()
    figure_shift_cosine_distributions()
    figure_shift_structure()
    figure_movement_utility()
    figure_steering()
    figure_masking()
    figure_quality_compact()
    figure_probe_compact()
    figure_masking_compact()
    figure_steering_compact()


if __name__ == "__main__":
    main()
