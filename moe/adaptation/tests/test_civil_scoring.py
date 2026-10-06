import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("score_civil", Path(__file__).parents[1] / "scripts/score_civil.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_invalid_empty_and_unfinished_are_distinct():
    examples = [{"key": "invalid", "labels": []}, {"key": "empty", "labels": []},
                {"key": "unfinished", "labels": ["toxicity"]}]
    rows = [{"key": key, "final": text, "finished": finished, "completion_tokens": 3, "elapsed_seconds": 1}
            for key, text, finished in [("invalid", "garbage", True), ("empty", "NONE", True), ("unfinished", "toxicity", False)]]
    result = runner.score_arm(examples, rows)
    assert result["label_f1"] == 2 / 3
    assert result["score"] == 1 / 3
    assert result["exact_label_set"] == 2 / 3
    assert result["valid"] == 2 / 3
    assert result["per_label"]["toxicity"] == {"tp": 0, "fp": 0, "fn": 1, "f1": 0.0}
    assert result["macro_label_f1"] == 0


def test_duplicates_cannot_replace_a_missing_example():
    with pytest.raises(ValueError, match="exactly one"):
        runner.score_arm([{"key": "a"}, {"key": "b"}], [{"key": "a"}, {"key": "a"}])
