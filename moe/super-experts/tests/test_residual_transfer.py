import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from se_gepa.residual_shift import EarlyResidualShift
from probe_residual_transfer import capture, controls, evaluate, select_civil, CONDITIONS
from test_attention_contributions import tiny_model, wrapped
from se_gepa.arms import load_contract


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_shift_matches_full_replay_and_zero_identity(dtype):
    base, _ = tiny_model(seed=93)
    base.to(dtype)
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    vector = torch.randn(3, 32) * .05
    options = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=4,
                   do_sample=False, eos_token_id=None, pad_token_id=0, use_cache=True,
                   return_dict_in_generate=True, output_scores=True)
    with torch.no_grad():
        native = base.generate(**options)
        with EarlyResidualShift(base, 1, "add", torch.zeros_like(vector)) as zero:
            identity = base.generate(**options)
        assert zero.calls == 4 and zero.patched == 3
        assert all(torch.equal(a, b) for a, b in zip(native.scores, identity.scores))
        with EarlyResidualShift(base, 1, "add", vector) as shift:
            cached = base.generate(**options)
        assert shift.calls == 4 and shift.patched == 3
        assert not torch.equal(native.scores[0], cached.scores[0])
        replay = ids.clone()
        for score in cached.scores:
            with EarlyResidualShift(base, 1, "add", vector):
                logits = base(input_ids=replay, use_cache=False).logits[:, -1]
            tolerance = .004 if dtype == torch.bfloat16 else 1e-6
            assert torch.allclose(logits.float(), score.float(), atol=tolerance, rtol=tolerance)
            replay = torch.cat((replay, score.argmax(-1, keepdim=True)), dim=1)
        assert torch.equal(replay, cached.sequences)
    assert not base.model.layers[1]._forward_hooks


def test_donor_patch_and_current_layer_cache_boundary():
    teacher, _ = wrapped("prefix", seed=94)
    base = teacher.get_base_model()
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    donor = capture(teacher, ids[0].tolist(), 1, check=True)
    state = capture(base, ids[0].tolist(), 1, check=True)
    with torch.no_grad():
        native = base(input_ids=ids, use_cache=True)
        with EarlyResidualShift(base, 1, "replace", donor) as patch:
            actual = base(input_ids=ids, use_cache=True)
        assert torch.equal(torch.cat(patch.after), donor)
        assert not torch.equal(native.logits, actual.logits)
        for layer in (0, 1):
            assert torch.equal(native.past_key_values.layers[layer].values, actual.past_key_values.layers[layer].values)
        assert not torch.equal(native.past_key_values.layers[2].values, actual.past_key_values.layers[2].values)
        with EarlyResidualShift(base, 1, "add", donor - state) as additive:
            base(input_ids=ids, use_cache=True)
        assert torch.allclose(torch.cat(additive.after), donor, atol=1e-7)
        assert torch.allclose(controls(donor).norm(dim=-1), donor.norm(dim=-1))


def test_fixed_early_tokens_are_causally_independent_of_later_text():
    teacher, _ = wrapped("prefix", seed=95)
    for model in (teacher, teacher.get_base_model()):
        a = capture(model, [2, 3, 4, 5, 6], 1)
        b = capture(model, [2, 3, 4, 8, 9], 1)
        assert torch.allclose(a, b, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("task", ["civil", "wiki"])
def test_full_runner_conditions_and_receipts(tmp_path, task):
    pytest.importorskip("sklearn")
    teacher, _ = wrapped("prefix", seed=96)
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = None
        def decode(self, _ids, skip_special_tokens=False):
            return "NONE"
    summaries = evaluate(teacher, [[2, 3, 4, 5], [4, 5, 6, 7]],
        [{"id": "a", "labels": []}, {"id": "b", "labels": []}], [[2, 3, 4, 8], [4, 3, 2, 8]],
        task, Tokenizer(), load_contract(), tmp_path, layer=1, max_new_tokens=3)
    assert [r["condition"] for r in summaries] == list(CONDITIONS)
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 14
    for row in rows:
        if row["condition"] not in {"base", "teacher"}:
            assert row["patched_positions"] == 3
        if row["condition"] == "donor_state":
            assert row["donor_state_max_error"] == 0
        if task == "civil":
            assert row["score"] == 0 and row["truncated"] and row["parsed_score"] == 1
    assert all(r.get("zero_shift_exact", r.get("base_after_teacher_exact")) for r in json.loads((tmp_path / "gates.json").read_text()))


def test_real_frozen_split_is_disjoint_and_overlap_rejected():
    root = Path(__file__).resolve().parents[1]
    a, b = [[json.loads(line) for line in (root / "data" / name).read_text().splitlines()]
            for name in ("civil_v2_val_seed42_n200.jsonl", "civil_v2_test_head2000.jsonl")]
    cal, val = select_civil(a, b, 32, 1800, 200)
    assert len(cal) == 32 and len(val) == 200
    assert not {r["id"] for r in a} & {r["id"] for r in val}
    with pytest.raises(ValueError, match="overlapping"):
        select_civil(a, a, 32, 0, 32)
