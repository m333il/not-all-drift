import numpy as np
import pytest

from prompt_optimization.residual_geometry import (
    behavior_group,
    compute_geometry,
    hidden_state_index_for_block,
    select_examples,
)


@pytest.mark.parametrize(
    ("manual", "prompt", "prefix", "expected"),
    [
        (False, True, True, "both_rescue"),
        (False, True, False, "prompt_only_rescue"),
        (False, False, True, "prefix_only_rescue"),
        (False, False, False, "both_still_wrong"),
        (True, True, True, "both_preserve"),
        (True, False, True, "prompt_only_breaks"),
        (True, True, False, "prefix_only_breaks"),
        (True, False, False, "both_break"),
    ],
)
def test_behavior_group_covers_all_patterns(
    manual: bool, prompt: bool, prefix: bool, expected: str
) -> None:
    assert behavior_group(manual, prompt, prefix) == expected


def test_block_zero_is_hidden_state_one() -> None:
    assert hidden_state_index_for_block(0, 27) == 1
    assert hidden_state_index_for_block(25, 27) == 26
    with pytest.raises(ValueError):
        hidden_state_index_for_block(26, 27)


def test_compute_geometry_uses_displacements_from_manual() -> None:
    manual = np.zeros((1, 3, 2), dtype=np.float32)
    prompt = manual.copy()
    prefix = manual.copy()
    manual[0, 1] = [1.0, 0.0]
    prompt[0, 1] = [2.0, 0.0]
    prefix[0, 1] = [1.0, 1.0]

    result = compute_geometry(
        manual,
        prompt,
        prefix,
        ids=["sample"],
        labels=[["toxicity"]],
        behaviors={"sample": "both_rescue"},
        blocks=[0],
        seed=42,
    )

    cross = result.cross_method_rows[0]
    assert cross["hidden_state_index"] == 1
    assert cross["delta_cosine"] == pytest.approx(0.0)
    assert cross["delta_angle_deg"] == pytest.approx(90.0)
    prompt_row = next(
        row for row in result.method_rows if row["method"] == "prompt_tuning"
    )
    assert prompt_row["delta_norm"] == pytest.approx(1.0)
    assert prompt_row["relative_delta_norm"] == pytest.approx(1.0)


def test_selection_covers_groups_labels_and_cardinalities() -> None:
    records = [
        {"id": "a", "labels": [], "behavior_group": "both_rescue", "cardinality_bucket": "0"},
        {"id": "b", "labels": ["toxicity"], "behavior_group": "both_preserve", "cardinality_bucket": "1"},
        {"id": "c", "labels": ["threat", "insult"], "behavior_group": "prefix_only_rescue", "cardinality_bucket": "2"},
        {"id": "d", "labels": ["toxicity", "threat", "insult"], "behavior_group": "both_break", "cardinality_bucket": "3+"},
    ]

    selected = select_examples(
        records,
        label_names=["toxicity", "threat", "insult"],
        samples_per_group=1,
        selection_seed=42,
    )

    assert set(selected) == {"a", "b", "c", "d"}
