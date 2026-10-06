import pytest

from prompt_optimization.teacher_forcing import (
    build_teacher_forcing_example,
    left_pad_teacher_forcing,
)


def test_teacher_forcing_positions_predict_all_target_tokens():
    example = build_teacher_forcing_example([10, 11, 12], [20, 21, 22], max_length=8)
    assert example.input_ids == (10, 11, 12, 20, 21)
    assert example.target_ids == (20, 21, 22)
    assert example.prediction_positions == (2, 3, 4)


def test_prompt_is_left_truncated_to_fit_the_target():
    example = build_teacher_forcing_example(list(range(10)), [20, 21, 22], max_length=6)
    assert example.input_ids == (7, 8, 9, 20, 21)
    assert example.prediction_positions == (2, 3, 4)


def test_left_padding_shifts_prediction_positions():
    short = build_teacher_forcing_example([1, 2], [7, 8], max_length=8)
    long = build_teacher_forcing_example([1, 2, 3, 4], [9], max_length=8)
    inputs, mask, positions, targets = left_pad_teacher_forcing([short, long], 0)
    assert inputs.tolist() == [[0, 1, 2, 7], [1, 2, 3, 4]]
    assert mask.tolist() == [[0, 1, 1, 1], [1, 1, 1, 1]]
    assert positions[0].tolist() == [2, 3]
    assert positions[1].tolist() == [3]
    assert [value.tolist() for value in targets] == [[7, 8], [9]]


@pytest.mark.parametrize("prompt,target", [([], [1]), ([1], [])])
def test_teacher_forcing_rejects_empty_sequences(prompt, target):
    with pytest.raises(ValueError):
        build_teacher_forcing_example(prompt, target, max_length=8)
