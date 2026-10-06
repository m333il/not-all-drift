"""Shared drawing code for the attention-share figures.

A panel is a list of rows, one per layer, each holding the share of attention
that queries from position 3 on give to the virtual tokens or prefix keys
(``prefix``), to real positions 0, 1 and 2, to the learned sink of gpt-oss
(``sink``) and to the rest of the sequence. Shares are drawn as horizontal
stacked bars, one bar per layer, with the larger components labelled.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch

ORDER = ["prefix", "pos0", "pos1", "pos2", "sink", "rest"]
LABEL = {
    "prefix": "virtual tokens or prefix keys",
    "pos0": "position 0",
    "pos1": "position 1",
    "pos2": "position 2",
    "sink": "learned sink (gpt-oss)",
    "rest": "rest of the sequence",
}
# Okabe-Ito, with the rest of the sequence in light grey.
COLORS = {"prefix": "#009E73", "pos0": "#E69F00", "pos1": "#0072B2", "pos2": "#D55E00",
          "sink": "#CC79A7", "rest": "#DDDDDD"}
MIN_LABELLED = 0.10


def panels() -> dict:
    """Panels produced from the expert-channel probe (``MOE_SINK_PANELS``).

    The file maps ``"<arm>-<corpus>"`` to a condition (``intact``, ``remove``,
    ``control``) and then to the per-layer rows described in the module docstring.
    """
    path = Path(os.environ.get("MOE_SINK_PANELS", "results/attention/panels.json"))
    if not path.is_file():
        raise FileNotFoundError(f"Attention panels are absent: {path}. Set MOE_SINK_PANELS.")
    return json.loads(path.read_text())


def stacked(ax, rows: list[dict], layers: list[str]) -> None:
    if len(rows) != len(layers):
        raise ValueError("Attention panel row count differs from its layer labels")
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows), dtype=float)
    for key in ORDER:
        values = np.asarray([row.get(key, 0.0) for row in rows], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Attention component {key} must contain finite nonnegative shares")
        ax.barh(y, values, left=left, color=COLORS[key], height=0.62, linewidth=0)
        for yi, x0, value in zip(y, left, values):
            if value >= MIN_LABELLED:
                dark = key in ("prefix", "pos1", "pos2", "sink")
                ax.text(x0 + 0.015, yi, f"{100 * value:.0f}", va="center", ha="left",
                        fontsize=6, color="white" if dark else "#333333")
        left += values
    if (left > 1.01).any():
        raise ValueError("Attention components exceed unit mass")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50", "100%"])
    ax.set_yticks(y)
    ax.set_yticklabels(layers)
    ax.tick_params(labelsize=6.5, length=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend(fig, keys: list[str], y: float = 0.0) -> None:
    fig.legend(handles=[Patch(facecolor=COLORS[key], label=LABEL[key]) for key in keys],
               loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(keys),
               frameon=False, fontsize=6.5)
