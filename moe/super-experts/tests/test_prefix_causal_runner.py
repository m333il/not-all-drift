import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_prefix_causal import calibrate, evaluate, select_rows
from test_attention_contributions import wrapped


def test_split_rejects_id_and_text_leakage():
    rows = [{"id": str(i), "text": f"example {i}"} for i in range(6)]
    assert select_rows(rows, 2, 3, 2) == (rows[:2], rows[3:5])
    for field in ("id", "text"):
        changed = [dict(r) for r in rows]
        changed[3][field] = rows[0][field]
        with pytest.raises(ValueError, match="overlap"):
            select_rows(changed, 2, 3, 2)
    with pytest.raises(ValueError, match="disjoint"):
        select_rows(rows, 3, 2, 2)


@pytest.mark.parametrize("eos", [None, 0])
def test_real_model_calibration_and_all_conditions_with_scoring(tmp_path, eos):
    pytest.importorskip("sklearn")
    from se_gepa.arms import load_contract
    model, _ = wrapped("prefix", seed=61)
    if eos == 0:
        with torch.no_grad():
            model.get_base_model().lm_head.weight.zero_()
    vectors, stats = calibrate(model, [[3, 2, 8], [5, 7]], [1], "last")
    assert vectors[1].shape == (32,) and stats[1]["examples"] == 2
    args = SimpleNamespace(layers=[1], scope="last", max_new_tokens=2, seed=42, out=tmp_path)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = eos

        def decode(self, _ids, skip_special_tokens=False):
            return "NONE"

    stream = io.StringIO()
    summaries, random = evaluate(model, Tokenizer(), {"name": "toy-prefix"}, [[4, 3, 2]],
        [{"id": "eval", "labels": []}], load_contract(), vectors, args, stream)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [r["condition"] for r in rows] == ["intact", "zero", "restore", "constant", "random"]
    assert len(summaries) == 5
    assert rows[0]["generation_scores_sha256"] == rows[2]["generation_scores_sha256"]
    assert rows[2]["first_token_kl_from_intact"] == 0
    for row in rows:
        assert row["valid"] and row["parsed_score"] == 1
        assert row["truncated"] is (eos is None)
        assert row["score"] == (0 if eos is None else 1)
        if row["condition"] != "intact":
            assert row["intervention_calls"] == row["completion_tokens"]
    assert torch.tensor(random["1"]).norm() == pytest.approx(float(vectors[1].norm()), rel=1e-6)
