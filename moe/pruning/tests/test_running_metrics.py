"""A cell must say how it is doing before it is over.

The run that prompted this took ten hours to report F1 0.3300 with every answer
unparsable - ten hours to learn what the first sixty-four rows already knew, on
a card that could have been computing something else. Speed was no substitute:
a degenerate arm babbling to the ceiling is slow, one answering instantly with
nothing is fast, and neither shows up in seconds-per-example.

So the progress line now carries F1, empty and unparsable over the rows already
finished. These tests pin that it appears, that it is computed from the real
answers rather than a placeholder, and that passing no golds leaves the old
behaviour untouched.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")

from mrd_pruning.evaluate import GenerationConfig, generate_responses  # noqa: E402


class _Tok:
    """The slice of a tokenizer this function touches."""

    pad_token_id = 0
    eos_token_id = 0
    # `render_chat` refuses a tokenizer without one rather than hand-rolling a
    # prompt format that would not match the trained arms.
    chat_template = "{{ messages }}"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * (len(text) % 3 + 2)}

    def apply_chat_template(self, messages, tokenize=False, **kw):
        return messages[-1]["content"]

    def batch_decode(self, tokens, skip_special_tokens=False):
        return [self._answer for _ in range(len(tokens))]


class _Model:
    device = "cpu"

    def __init__(self, tok):
        self._tok = tok

    def generate(self, input_ids, **kw):
        extra = torch.ones((input_ids.shape[0], 2), dtype=torch.long)
        return torch.cat([input_ids, extra], dim=1)


def run(answer: str, golds, caplog, **kw):
    tok = _Tok()
    tok._answer = answer
    model = _Model(tok)
    prompts = [f"line {i}" for i in range(8)]
    cfg = GenerationConfig(max_new_tokens=4, batch_size=2, log_every=1)
    with caplog.at_level(logging.INFO, logger="mrd_pruning.evaluate"):
        generate_responses(model, tok, prompts, None, config=cfg,
                           golds=golds, **kw)
    return [r.getMessage() for r in caplog.records]


def test_progress_line_carries_the_running_score(caplog):
    golds = [["toxicity"]] * 8
    lines = run("toxicity", golds, caplog)
    running = [l for l in lines if "intermediate" in l]
    assert running, "There is no intermediate metric in the log"
    assert "F1 1.0000" in running[-1], running[-1]


def test_a_silent_arm_shows_as_empty_immediately(caplog):
    """The case that cost ten hours: every answer unparsable, F1 at the floor."""
    golds = [["toxicity"]] * 8
    lines = run("", golds, caplog)
    running = [l for l in lines if "intermediate" in l]
    assert running
    assert "F1 0.0000" in running[-1], running[-1]
    assert "empty 100.0%" in running[-1], running[-1]


def test_the_score_grows_with_the_rows(caplog):
    """It is a running total, not a fixed sample: n rises across the marks."""
    golds = [["toxicity"]] * 8
    lines = run("toxicity", golds, caplog)
    counts = [int(l.split("on ")[1].split(" rows")[0])
              for l in lines if "intermediate" in l]
    assert counts == sorted(counts)
    assert counts[-1] == 8
    assert len(counts) > 1, "There was only one mark - the height was not checked."


def test_without_golds_nothing_changes(caplog):
    """The old signature still works and prints no running score."""
    lines = run("toxicity", None, caplog)
    assert not [l for l in lines if "intermediate" in l]
    assert [l for l in lines if "generated" in l], "Progress disappeared with the metric."
