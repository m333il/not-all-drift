from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import DecisionConfig
from .metrics import Interval

World = Literal["A", "B", "C", "mixed_inconclusive"]


@dataclass(frozen=True)
class WorldEvidence:
    world: World
    reason: str


def classify_world(
    *,
    probe_change: Interval,
    seed_gap: Interval,
    target_gap: Interval,
    elicitation_fraction: float,
    config: DecisionConfig,
    selectivity_change: float = 0.0,
    cross_condition_fraction: float = 1.0,
) -> WorldEvidence:
    if (
        probe_change.estimate >= config.practical_f1
        and probe_change.low > 0
        and selectivity_change >= 0
    ):
        return WorldEvidence("B", "probe accessibility increased materially")
    probe_equivalent = abs(probe_change.estimate) <= config.equivalence_margin
    if (
        probe_equivalent
        and seed_gap.estimate >= config.latent_gap
        and seed_gap.low > 0
        and elicitation_fraction >= config.elicitation_fraction
        and cross_condition_fraction >= 0.8
    ):
        return WorldEvidence("A", "stable probe signal and material latent gap closure")
    if (
        abs(seed_gap.estimate) < config.latent_gap
        and abs(probe_change.estimate) < config.practical_f1
    ):
        return WorldEvidence("C", "small latent gap and no material probe reorganization")
    return WorldEvidence("mixed_inconclusive", "confirmatory margins do not identify one world")
