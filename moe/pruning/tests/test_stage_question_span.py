"""The question segment has to exist, and under both prompt contracts.

`label_prompt_stages` matched `"\\n\\nLabels (use exact names):"` - a blank line
before the marker. The v25 contract renders a single newline there, so the span
never matched, the instruction fell into `template`, and every measurement
built on these labels reported `question = 0`. The sixteen-cell attention
delivery shows exactly that in all sixteen summaries.

The spans are checked against the real rendered prompt rather than a fixture,
because the bug was a mismatch between the two.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("torch")


class _Tok:
    """Character-offset tokenizer: one token per word, offsets exact."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offs, ids, i = [], [], 0
        for part in text.split(" "):
            offs.append((i, i + len(part)))
            ids.append(len(ids))
            i += len(part) + 1
        return {"input_ids": ids, "offset_mapping": offs}


def _labels(user_prompt, comment, monkeypatch):
    import measure_routing_map as mrm

    tok = _Tok(user_prompt)
    monkeypatch.setattr(mrm, "render_chat",
                        lambda *a, **k: user_prompt)
    ids = tok(user_prompt)["input_ids"]
    return mrm.label_prompt_stages(tok, None, comment, user_prompt, ids)


def test_one_newline_before_the_marker_still_makes_a_question(monkeypatch):
    """The v25 rendering. This is the case that silently produced nothing."""
    comment = "you are a fool"
    prompt = ("Classify the text.\nLabels (use exact names): toxicity, insult\n"
              "Text:\n" + comment + "\nAnswer:")
    names = _labels(prompt, comment, monkeypatch)
    assert "question" in names, "The problem is not there again."


def test_a_blank_line_before_the_marker_works_too(monkeypatch):
    """The v24 rendering must keep working."""
    comment = "you are a fool"
    prompt = ("Classify the text.\n\nLabels (use exact names): toxicity\n"
              "Text:\n" + comment + "\nAnswer:")
    names = _labels(prompt, comment, monkeypatch)
    assert "question" in names


def test_the_comment_is_not_swallowed_by_the_question(monkeypatch):
    """The question runs to the end of the prompt, so it must start after it."""
    comment = "you are a fool"
    prompt = ("Labels (use exact names): toxicity\nText:\n" + comment +
              "\nAnswer:")
    names = _labels(prompt, comment, monkeypatch)
    assert "comment" in names and "question" in names


def test_a_prompt_without_the_marker_has_no_question(monkeypatch):
    comment = "you are a fool"
    prompt = "Say something about:\n" + comment + "\nAnswer:"
    names = _labels(prompt, comment, monkeypatch)
    assert "question" not in names
