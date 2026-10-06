from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError


class Preregistration:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.selection_splits = tuple(str(x) for x in payload.get("selection_splits", ()))
        self.final_splits = tuple(str(x) for x in payload.get("final_splits", ()))
        overlap = set(self.selection_splits) & set(self.final_splits)
        if overlap:
            raise ConfigurationError(f"selection/final split overlap: {sorted(overlap)}")

    def content_hash(self) -> str:
        encoded = json.dumps(self.payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def assert_selection_split(self, split: str) -> None:
        if split not in self.selection_splits:
            raise ConfigurationError(f"hyperparameter selection is forbidden on split: {split}")

    def assert_final_split(self, split: str) -> None:
        if split not in self.final_splits:
            raise ConfigurationError(f"confirmatory reporting requires a final split: {split}")


def load_preregistration(path: Path) -> Preregistration:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigurationError("preregistration must be a YAML mapping")
    return Preregistration(payload)


__all__ = ["Preregistration", "load_preregistration"]
