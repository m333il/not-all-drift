import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_prefix_civil import evaluate
from se_gepa.arms import load_contract
from se_gepa.prefix_intervention import PrefixKeyMask
from test_attention_contributions import tiny_model, wrapped


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_masked_cached_generation_matches_uncached_replay(dtype):
    from contextlib import ExitStack
    model, _ = wrapped("prefix", seed=81)
    model.to(dtype)
    ids = torch.tensor([[2, 5, 6]])
    with torch.no_grad(), ExitStack() as stack:
        hooks = [stack.enter_context(PrefixKeyMask(model, layer)) for layer in range(3)]
        cached = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=4,
                                do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=0,
                                return_dict_in_generate=True, output_scores=True)
        assert all(h.calls == 4 for h in hooks)
        replay = ids.clone()
        for score in cached.scores:
            logits = model(input_ids=replay, attention_mask=torch.ones_like(replay), use_cache=False).logits[:, -1]
            tolerance = .004 if dtype == torch.bfloat16 else 1e-6
            assert torch.allclose(logits.float(), score.float(), atol=tolerance, rtol=tolerance)
            replay = torch.cat((replay, logits.argmax(-1, keepdim=True)), dim=1)
        assert torch.equal(replay, cached.sequences)


@pytest.mark.parametrize("prefix,eos", [(False, None), (True, None), (True, 0)])
def test_civil_all_conditions_scoring_and_restore(tmp_path, prefix, eos):
    pytest.importorskip("sklearn")
    model, _ = wrapped("prefix", seed=82) if prefix else tiny_model(seed=82)
    if eos == 0:
        with torch.no_grad():
            model.get_base_model().lm_head.weight.zero_()

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = eos

        def decode(self, _ids, skip_special_tokens=False):
            return "NONE"

    summaries = evaluate(model, Tokenizer(), {"name": "test", "kind": "peft" if prefix else "text"},
                         [[2, 5, 6], [7, 8]], [{"id": "a", "labels": []}, {"id": "b", "labels": []}],
                         load_contract(), [0, 1, 2], 3, tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    conditions = ["intact", "zero_values", "mask_keys"] if prefix else ["intact"]
    assert [r["condition"] for r in summaries] == conditions and len(rows) == 2 * len(conditions)
    for row in rows:
        assert row["parsed_score"] == 1 and row["valid"]
        assert row["finished"] == (eos is not None)
        assert row["truncated"] == (eos is None)
        assert row["score"] == (0 if eos is None else 1)
        if row["condition"] != "intact":
            assert set(row["intervention_calls"]) == {"0", "1", "2"}
            assert all(count == row["completion_tokens"] for count in row["intervention_calls"].values())
        if row["condition"] == "mask_keys":
            assert all(check["calls"] == row["completion_tokens"] for check in row["mask_checks"].values())
    restore = json.loads((tmp_path / "restore-checks.json").read_text())
    assert len(restore) == (2 if prefix else 0)
    assert all(r["generation_exact"] for r in restore)
