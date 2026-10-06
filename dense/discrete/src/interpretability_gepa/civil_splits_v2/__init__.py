"""Civil Comments split generator (v2): five labels, configurable empty fraction,
exact-mask quotas that scale with split size, mirrored enrichment.
"""

from __future__ import annotations

from .allocation import allocate_positive_counts, mask_floors, select_split_indices
from .contract import (
    BINARY_CONTRACT,
    CIVIL_LABELS_V2,
    MULTILABEL_CONTRACT,
    SplitContractV2,
    SplitGenerationError,
)
from .corpus import (
    enriched_mask_distribution,
    label_marginals,
    natural_mask_distribution,
)
from .generator import generate_splits_v2, labels_from_scores

__all__ = [
    "BINARY_CONTRACT",
    "CIVIL_LABELS_V2",
    "MULTILABEL_CONTRACT",
    "SplitContractV2",
    "SplitGenerationError",
    "allocate_positive_counts",
    "enriched_mask_distribution",
    "generate_splits_v2",
    "label_marginals",
    "labels_from_scores",
    "mask_floors",
    "natural_mask_distribution",
    "select_split_indices",
]
