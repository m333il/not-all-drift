"""The published Super-Expert criterion, kept bit-for-bit.

Equation 6 of arXiv:2507.23279 marks expert ``e`` of layer ``l`` a Super Expert
when its maximum ``down_proj`` output magnitude ``a[l,e]`` satisfies

    a[l,e] > P99.5   and   a[l,e] > a_max / 10   and   l in L

where ``L`` is the prefix of layers in which massive activations form. Upstream
implements ``L`` as the first ``include_fraction`` of decoder layers, computes
the percentile over the experts that survive that filter (not over all experts),
and writes the ratio test with floor division. All three details are reproduced
here rather than tidied, because changing any of them changes which experts the
published criterion returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QUANTILE = 99.5
RATIO = 10
INCLUDE_FRACTION = 0.75


@dataclass(frozen=True)
class SuperExpert:
    layer: int
    expert: int
    output_max: float
    rank: int


def identify(
    output_max: dict[tuple[int, int], float],
    total_layers: int,
    include_fraction: float = INCLUDE_FRACTION,
    quantile: float = QUANTILE,
    ratio: int = RATIO,
) -> list[SuperExpert]:
    """Apply the criterion to ``{(layer, expert): max |down_proj output|}``.

    ``total_layers`` is the model's decoder-layer count, which is not the same
    as the number of MoE layers whenever a model starts with dense layers.
    """
    include_layers = round(total_layers * include_fraction)
    candidates = {key: value for key, value in output_max.items() if key[0] < include_layers}
    if not candidates:
        return []
    values = np.array(list(candidates.values()), dtype=np.float64)
    percentile = np.percentile(values, quantile)
    floor = np.max(values) // ratio
    selected = [
        SuperExpert(layer=layer, expert=expert, output_max=value, rank=0)
        for (layer, expert), value in candidates.items()
        if value > percentile and value > floor
    ]
    selected.sort(key=lambda row: row.output_max, reverse=True)
    return [SuperExpert(row.layer, row.expert, row.output_max, rank) for rank, row in enumerate(selected, start=1)]


def output_max_map(records) -> dict[tuple[int, int], float]:
    return {key: record.output_max for key, record in records.items()}
