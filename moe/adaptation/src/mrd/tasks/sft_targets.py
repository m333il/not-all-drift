"""Cross-entropy training targets for the prompt- and prefix-tuning arms."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class SFTExample(TypedDict):
    input_text: str
    target_text: str


def write_jsonl(rows: list[SFTExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[SFTExample]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]
