from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


def stable_example_id(dataset: str, text: str, source_id: str | int) -> str:
    raw = f"{dataset}\0{source_id}\0{text}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    dataset: str
    text: str
    labels: tuple[str, ...]
    source_id: str
    group_id: str | None = None
    scores: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "text": self.text,
            "labels": list(self.labels),
            "source_id": self.source_id,
            "group_id": self.group_id,
            "scores": dict(self.scores),
        }


@dataclass(frozen=True, slots=True)
class Prediction:
    example_id: str
    condition: str
    raw_response: str
    labels: tuple[str, ...]
    parse_ok: bool
    parse_error: str | None = None
    # Set-parser result of the same response.
    lenient_labels: tuple[str, ...] = ()
    order_violation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "condition": self.condition,
            "raw_response": self.raw_response,
            "labels": list(self.labels),
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
            "lenient_labels": list(self.lenient_labels),
            "order_violation": self.order_violation,
        }
