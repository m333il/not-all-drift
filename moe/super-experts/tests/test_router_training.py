import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from se_gepa.router_training import lr_factor, paired, select_epochs, token_weighted_windows


def test_lr_factor_warms_up_then_decays_to_zero():
    assert lr_factor(0, 4, 10) == 0.25
    assert lr_factor(3, 4, 10) == 1.0
    assert lr_factor(4, 4, 10) == 1.0
    assert lr_factor(10, 4, 10) == 0.0


def test_windows_weight_by_supervised_tokens():
    windows = list(token_weighted_windows([2, 0, 1], [1, 3, 4], 2))
    assert windows == [[(2, 4 / 5), (0, 1 / 5)], [(1, 1.0)]]


def test_selection_can_decline_training_and_breaks_ties_early():
    assert select_epochs({0: 0.8, 1: 0.7, 2: 0.75}) == (2, 0)
    assert select_epochs({0: 0.5, 1: 0.7, 2: 0.7}) == (1, 1)
    assert select_epochs({0: 0.5}) == (None, 0)


def test_paired_difference():
    result = paired([1.0, 0.5, 0.0], [0.5, 0.5, 0.5])
    assert result["n"] == 3 and result["changed"] == 2
    assert abs(result["mean"]) < 1e-12
