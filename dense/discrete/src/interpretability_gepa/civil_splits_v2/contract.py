"""Split contracts for the v2 Civil Comments setups (multilabel and binary)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

CIVIL_LABELS_V2: tuple[str, ...] = (
    "toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
)
"""Frozen label schema. ``toxicity`` must stay first: it is the near-superset."""

UMBRELLA_INDEX = 0
SUBTYPE_INDICES: tuple[int, ...] = tuple(range(1, len(CIVIL_LABELS_V2)))
MASK_COUNT = 1 << len(CIVIL_LABELS_V2)

DATASET_REVISION = "61864fdf071fdb5a8fd8c74f5d23271c6a9fc65c"

Setup = Literal["multilabel", "binary"]


class SplitGenerationError(ValueError):
    """Raised when the requested split contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class SplitContractV2:
    """Split geometry of one setup.

    ``ladder`` is one nested chain per seed, so the N=1000 step is the same set of rows
    for GEPA and for the continuous methods.
    """

    setup: Setup = "multilabel"
    threshold: float = 0.5
    empty_fraction: float = 0.33
    ladder: tuple[int, ...] = (200, 500, 1000, 10000, 20000)
    optimizer_val_ladder: tuple[int, ...] = (200, 1000)
    optimizer_seeds: tuple[int, ...] = (42, 43, 44)
    test_size: int = 3000
    probe_train_size: int = 8000
    probe_val_size: int = 2000
    intervention_val_size: int = 2000
    probe_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    probe_fraction: float = 0.8
    # Share of the positive half per label; unlisted labels keep the corpus rate.
    # threat: 2.9% -> 5%.
    enrichment: Mapping[str, float] = field(default_factory=lambda: {"threat": 0.05})
    # Exact-mask floors (half of the natural contribution) give exclusive support.
    mask_floor_fraction: float = 0.5
    mask_floor_absolute: int = 5
    pairwise_weight: float = 0.25
    test_seed: int = 1729
    auxiliary_seed: int = 9917
    seed_salt: int = 0
    dataset_revision: str = DATASET_REVISION

    def positive_size(self, size: int) -> int:
        """Rows in the positive half of a split of ``size`` rows."""
        return size - self.empty_size(size)

    def empty_size(self, size: int) -> int:
        return int(round(size * self.empty_fraction))

    def quota_masks(self) -> tuple[int, ...]:
        """Exact masks that carry a floor: ``{toxicity}`` and every ``{toxicity, X}``."""
        umbrella = 1 << UMBRELLA_INDEX
        return (umbrella, *(umbrella | (1 << index) for index in SUBTYPE_INDICES))

    def validate(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise SplitGenerationError("threshold must be in (0, 1]")
        if not 0.0 < self.empty_fraction < 1.0:
            raise SplitGenerationError("empty_fraction must be in (0, 1)")
        if self.setup not in ("multilabel", "binary"):
            raise SplitGenerationError(f"unknown setup: {self.setup}")
        ladders = (("ladder", self.ladder), ("optimizer_val_ladder", self.optimizer_val_ladder))
        for name, ladder in ladders:
            if not ladder:
                raise SplitGenerationError(f"{name} must not be empty")
            if tuple(sorted(set(ladder))) != tuple(ladder):
                raise SplitGenerationError(f"{name} must be strictly increasing and unique")
            if any(size <= 0 for size in ladder):
                raise SplitGenerationError(f"{name} sizes must be positive")
        seed_groups = (
            (self.optimizer_seeds, "optimizer_seeds"),
            (self.probe_seeds, "probe_seeds"),
        )
        for seeds, name in seed_groups:
            if len(set(seeds)) != len(seeds):
                raise SplitGenerationError(f"{name} must be unique")
        sizes = (
            *self.ladder,
            *self.optimizer_val_ladder,
            self.test_size,
            self.probe_train_size,
            self.probe_val_size,
            self.intervention_val_size,
        )
        for size in sizes:
            if size <= 0:
                raise SplitGenerationError(f"split size must be positive: {size}")
            if self.positive_size(size) <= 0 or self.empty_size(size) <= 0:
                raise SplitGenerationError(
                    f"split size {size} leaves an empty half at this fraction"
                )
        for label, share in self.enrichment.items():
            if label not in CIVIL_LABELS_V2:
                raise SplitGenerationError(f"enrichment names an unknown label: {label}")
            if not 0.0 < share < 1.0:
                raise SplitGenerationError(f"enrichment share for {label} must be in (0, 1)")
        if self.setup == "binary" and self.enrichment:
            raise SplitGenerationError(
                "the binary setup makes no per-label claim, so it takes no enrichment"
            )
        if not 0.0 <= self.mask_floor_fraction < 1.0:
            raise SplitGenerationError("mask_floor_fraction must be in [0, 1)")
        if self.mask_floor_absolute < 0:
            raise SplitGenerationError("mask_floor_absolute must be non-negative")
        if not math.isfinite(self.pairwise_weight) or self.pairwise_weight < 0:
            raise SplitGenerationError("pairwise_weight must be finite and non-negative")
        subset_size = self.probe_train_size * self.probe_fraction
        if not 0.0 < self.probe_fraction < 1.0 or not float(subset_size).is_integer():
            raise SplitGenerationError("probe_fraction must produce an integer subset size")
        if not self.dataset_revision.strip():
            raise SplitGenerationError("dataset_revision must not be empty")


MULTILABEL_CONTRACT = SplitContractV2()
BINARY_CONTRACT = SplitContractV2(
    setup="binary",
    empty_fraction=0.5,
    enrichment={},
    # The binary setup needs no floors.
    mask_floor_fraction=0.0,
    mask_floor_absolute=0,
    seed_salt=0x8121,
)
