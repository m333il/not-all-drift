"""Every arm must be scored on the same target positions.

Without an adapter the causal shift already drops the first token's target.
PEFT's prompt tuning prepends its own -100 block, so the last virtual position
predicts the window's first real token and that arm alone would score one extra
target -- a small bias, and one that favours exactly the arms whose virtual block
is longest. Masking position 0 removes it.

CPU, tiny random model, randomly initialised adapter -- a wiring test.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from test_profiler import tiny_model

pytest.importorskip("peft")

VIRTUAL = 6


def scored_targets(labels_row):
    return int((labels_row != -100).sum())


def test_masking_position_zero_equalises_the_target_count():
    from peft import PromptTuningConfig, TaskType, get_peft_model

    model, _config = tiny_model("eager")
    ids = torch.arange(1, 17).unsqueeze(0)
    labels = ids.clone()
    labels[:, 0] = -100

    config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=VIRTUAL)
    wrapped = get_peft_model(model, config)

    # What PEFT will hand the base model: its own -100 block, then our labels.
    prefixed = torch.cat([torch.full((1, VIRTUAL), -100), labels], dim=1)

    # The causal shift scores a target when the position before it is real, so the
    # count of scorable targets is the number of non -100 labels in both cases.
    assert scored_targets(prefixed[0]) == scored_targets(labels[0]) == ids.shape[1] - 1

    output = wrapped(input_ids=ids, labels=labels)
    assert torch.isfinite(output.loss)


def test_without_the_mask_the_prompt_arm_would_score_one_more():
    unmasked = torch.arange(1, 17).unsqueeze(0).clone()
    prefixed = torch.cat([torch.full((1, VIRTUAL), -100), unmasked], dim=1)
    assert scored_targets(prefixed[0]) == unmasked.shape[1]
    # The base arm's shift drops the first target, so it would score one fewer.
    assert scored_targets(prefixed[0]) == unmasked.shape[1] - 1 + 1
