import pytest

from mrd.civil_multilabel_task import exact_match, parse_response, score_example


@pytest.mark.parametrize("response", ["", " ", "garbage", "NONE, toxicity", "toxicity, unknown", "I think NONE"])
@pytest.mark.parametrize("labels", [[], ["toxicity"]])
def test_invalid_outputs_never_receive_credit(response, labels):
    assert parse_response(response) is None
    assert score_example({"labels": labels}, response)[0] == 0
    assert exact_match({"labels": labels}, response) == 0


def test_explicit_none_and_valid_partial_label_set():
    assert score_example({"labels": []}, "NONE")[0] == 1
    assert exact_match({"labels": []}, "Answer: none.") == 1
    assert score_example({"labels": ["toxicity", "threat"]}, "Toxicity")[0] == pytest.approx(2 / 3)
    assert exact_match({"labels": ["toxicity", "threat"]}, "threat, toxicity") == 1
