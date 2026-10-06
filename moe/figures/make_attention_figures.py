"""Attention-share figures of the super-expert subsection and its appendix.

Two inputs, both produced by ``moe/super-experts/scripts`` and passed with
``--logs`` and ``--panels`` (or ``MOE_ATTENTION_LOGS`` and ``MOE_SINK_PANELS``):

* ``MOE_ATTENTION_LOGS``: a directory of ``attn-<model>-<arm>.txt`` files, the
  stdout of ``probe_mechanism.py`` run with ``--conditions intact,ablated``.
  ``<model>`` is ``q`` for Qwen3-30B-A3B and ``g`` for gpt-oss-20b.
* ``MOE_SINK_PANELS``: per-layer panels of ``probe_expert_channel.py`` (intact,
  super expert removed, random control) on Civil Comments and WikiText-2.

Figures are written to ``--out``.
"""
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import make_sink_figures as sink  # noqa: E402

OUT = Path(os.environ.get("MOE_OUT", "."))
LOGS = Path(os.environ.get("MOE_ATTENTION_LOGS", "results/attention/logs"))
QWEN_LAYERS = list(range(1, 6))
GPT_OSS_LAYERS = list(range(7, 18, 2))
METHODS = [("base", "base"), ("gepa", "GEPA"), ("prompt-m500", "prompt 500"), ("prefix-m500", "prefix 500")]


def read_log(model: str, arm: str) -> dict[tuple[str, int], dict]:
    """Per (condition, layer) attention shares from the ATTN lines of one log."""
    rows = {}
    for line in (LOGS / f"attn-{model}-{arm}.txt").read_text().splitlines():
        if not line.startswith(f"ATTN arm={arm} "):
            continue
        fields = dict(part.split("=", 1) for part in line.split()[1:])
        rows[fields["condition"], int(fields["layer"])] = {
            "prefix": float(fields["virtual"]), "pos0": float(fields["key0"]),
            "pos1": float(fields["key1"]), "pos2": float(fields["key2"]),
            # Qwen has no learned sink; its residual mass there is bf16 rounding of the row sums.
            "sink": float(fields["learned_sink"]) if model == "g" else 0.0,
            "rest": float(fields["rest"])}
    return rows


def grid_panel(model: str, arm: str, condition: str, layers: list[int]) -> list[dict]:
    rows = read_log(model, arm)
    missing = [layer for layer in layers if (condition, layer) not in rows]
    if missing:
        raise ValueError(f"{model}/{arm}/{condition}: no ATTN rows for layers {missing}")
    return [rows[condition, layer] for layer in layers]


def titled(ax, title: str, note: str) -> None:
    ax.set_title(title, pad=9, fontsize=7.5)
    ax.text(0.5, 1.02, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.2,
            color="#444444")


def fig_method_grid(model: str, columns: list[tuple[str, str]], layers: list[int], name: str) -> Path:
    """Columns are arms; the top row is intact, the bottom row has the super experts pruned."""
    fig, axes = plt.subplots(2, len(columns), figsize=(6.6, 3.1), squeeze=False)
    labels = [f"L{layer}" for layer in layers]
    for column, (arm, title) in enumerate(columns):
        for row, (condition, note) in enumerate((("intact", "intact"), ("ablated", "super experts pruned"))):
            ax = axes[row, column]
            sink.stacked(ax, grid_panel(model, arm, condition, layers), labels)
            titled(ax, title, note)
            if column:
                ax.set_yticklabels([])
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=1.3, rect=(0, 0.06, 1, 1))
    keys = [key for key in sink.ORDER if key != "sink" or model == "g"]
    sink.legend(fig, keys, y=0.0)
    path = OUT / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_qwen_channel(corpus: str) -> Path:
    """Qwen, layers 1 to 5: intact, the super expert's channel removed, and a random control."""
    data = sink.panels()
    arms = [("base", "frozen base"), ("init", "prefix 500, untrained"), ("trained", "prefix 500, LR 3e-5")]
    conditions = [("intact", "intact"), ("remove", "super expert removed"),
                  ("control", "random direction, same norm")]
    fig, axes = plt.subplots(3, 3, figsize=(6.6, 3.8))
    labels = [f"L{layer}" for layer in QWEN_LAYERS]
    for i, (arm, arm_label) in enumerate(arms):
        for j, (condition, condition_label) in enumerate(conditions):
            ax = axes[i, j]
            sink.stacked(ax, data[f"{arm}-{corpus}"][condition], labels)
            if i == 0:
                ax.set_title(condition_label, pad=4, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(arm_label, fontsize=7)
            else:
                ax.set_yticklabels([])
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.9, rect=(0, 0.05, 1, 1))
    sink.legend(fig, ["prefix", "pos0", "pos1", "pos2", "rest"], y=0.0)
    path = OUT / f"moe_attention_qwen_{corpus}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


GPT_OSS_ARMS = [
    ("base", "base", "SE present"),
    ("prompt-m500", "prompt 500", "SE present, trigger on virtual 0"),
    ("prefix-m100", "prefix 100", "SE suppressed"),
    ("prefix-m200", "prefix 200", "SE present, peak on position 2"),
    ("prefix-m500", "prefix 500", "SE suppressed"),
    ("prefix-m500-init", "prefix 500, untrained", "SE present"),
]


def fig_gpt_oss_arms(layers: tuple[int, ...] = (0, 1, 2, 3)) -> Path:
    """gpt-oss: min-max over layers 0 to 3 of the learned sink's and the prefix keys' share."""
    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    for i, (arm, label, state) in enumerate(GPT_OSS_ARMS):
        rows = read_log("g", arm)
        y = len(GPT_OSS_ARMS) - 1 - i
        for key, offset in (("sink", 0.17), ("prefix", -0.17)):
            values = [100 * rows["intact", layer][key] for layer in layers]
            if key == "prefix" and not arm.startswith("prefix"):
                continue
            lo, hi = min(values), max(values)
            ax.plot([lo, hi], [y + offset] * 2, color=sink.COLORS[key], linewidth=5,
                    solid_capstyle="round")
            ax.text(hi + 1.2, y + offset, f"{lo:.0f}-{hi:.0f}", va="center", fontsize=6)
        ax.text(101, y, state, va="center", fontsize=6.2, color="#444444")
    ax.set_yticks(range(len(GPT_OSS_ARMS)))
    ax.set_yticklabels([label for _, label, _ in GPT_OSS_ARMS][::-1], fontsize=6.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of attention, min-max over layers 0-3 (%)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3, rect=(0, 0.1, 0.83, 1))
    fig.legend(handles=[plt.Line2D([], [], color=sink.COLORS["sink"], linewidth=5, label="learned sink"),
                        plt.Line2D([], [], color=sink.COLORS["prefix"], linewidth=5, label="prefix keys")],
               loc="lower center", ncol=2, frameon=False, fontsize=6.5)
    path = OUT / "moe_attention_gptoss_arms.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, default=LOGS, help="Directory of attn-<model>-<arm>.txt")
    parser.add_argument("--panels", type=Path, help="panels.json of the expert-channel probe")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    LOGS, OUT = args.logs, args.out
    if args.panels:
        os.environ["MOE_SINK_PANELS"] = str(args.panels)
    OUT.mkdir(parents=True, exist_ok=True)
    lr_columns = [("prefix-m500-init", "prefix 500, untrained"), ("prefix-m500", "prefix 500, LR 3e-5"),
                  ("prefix-m500-lr1e-4", "prefix 500, LR 1e-4")]
    for path in (
        fig_method_grid("q", METHODS, QWEN_LAYERS, "moe_attention_grid_qwen"),
        fig_method_grid("g", METHODS, GPT_OSS_LAYERS, "moe_attention_grid_gptoss"),
        fig_method_grid("q", lr_columns, QWEN_LAYERS, "moe_attention_prefix_lr"),
        fig_qwen_channel("civil"),
        fig_qwen_channel("wiki"),
        fig_gpt_oss_arms(),
    ):
        print(path)
