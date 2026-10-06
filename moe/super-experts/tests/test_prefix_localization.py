from contextlib import ExitStack
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_prefix_civil import evaluate
from probe_prefix_wikitext import measure
from se_gepa.arms import load_contract
from se_gepa.prefix_intervention import PrefixKeyMask, localization_designs
from test_attention_contributions import wrapped, VIRTUAL


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scope", ["early3", "after3"])
@pytest.mark.parametrize("length", [1, 5])
def test_position_mask_cached_matches_full_replay(dtype, scope, length):
    model, _ = wrapped("prefix", seed=91)
    model.to(dtype)
    ids = torch.tensor([[2, 5, 6, 3, 8]])[:, :length]
    with torch.no_grad(), ExitStack() as stack:
        hooks = [stack.enter_context(PrefixKeyMask(model, layer, scope=scope)) for layer in range(3)]
        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=4,
                                   do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=0,
                                   return_dict_in_generate=True, output_scores=True)
        assert all(h.calls == 4 for h in hooks)
        expected = min(3, length + 3) if scope == "early3" else max(0, length + 3 - 3)
        assert all(h.masked_queries == expected for h in hooks)
    replay = ids.clone()
    with torch.no_grad():
        for score in generated.scores:
            with ExitStack() as stack:
                for layer in range(3):
                    stack.enter_context(PrefixKeyMask(model, layer, scope=scope))
                logits = model(input_ids=replay, attention_mask=torch.ones_like(replay), use_cache=False).logits[:, -1]
            tolerance = .004 if dtype == torch.bfloat16 else 1e-6
            assert torch.allclose(logits.float(), score.float(), atol=tolerance, rtol=tolerance)
            replay = torch.cat((replay, logits.argmax(-1, keepdim=True)), dim=1)
    assert torch.equal(replay, generated.sequences)


@pytest.mark.parametrize("scope", ["early3", "after3"])
def test_only_selected_queries_change_at_target_layer(scope):
    model, _ = wrapped("prefix", seed=92)
    ids = torch.tensor([[5, 3, 9, 7, 2]])
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=True, output_attentions=True)
        with PrefixKeyMask(model, 1, scope=scope) as hook:
            actual = model(input_ids=ids, use_cache=True, output_attentions=True)
    selected = torch.arange(5) < 3 if scope == "early3" else torch.arange(5) >= 3
    assert torch.equal(ref.attentions[1][..., ~selected, :], actual.attentions[1][..., ~selected, :])
    assert torch.count_nonzero(actual.attentions[1][..., selected, :VIRTUAL]) == 0
    real = ref.attentions[1][..., selected, VIRTUAL:]
    assert torch.allclose(actual.attentions[1][..., selected, VIRTUAL:], real / real.sum(-1, keepdim=True), atol=2e-7)
    assert torch.equal(ref.past_key_values.layers[1].keys, actual.past_key_values.layers[1].keys)
    assert torch.equal(ref.past_key_values.layers[1].values, actual.past_key_values.layers[1].values)
    assert hook.masked_queries == int(selected.sum())


@pytest.mark.parametrize("panel", ["layers", "positions"])
def test_localization_civil_and_wiki_runners(tmp_path, panel):
    pytest.importorskip("sklearn")
    designs = localization_designs(panel)
    assert len(designs) == (7 if panel == "layers" else 4)
    designs = [dict(d, layers=[l for l in d["layers"] if l < 3]) for d in designs
               if d["name"] not in {"mask_L3", "mask_L4", "mask_L5"}]
    model, _ = wrapped("prefix", seed=93)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = None

        def decode(self, _ids, skip_special_tokens=False):
            return "NONE"

    civil = tmp_path / "civil"; civil.mkdir()
    evaluate(model, Tokenizer(), {"name": "test", "kind": "peft"}, [[2], [3, 4, 8, 7, 6]],
             [{"id": "a", "labels": []}, {"id": "b", "labels": []}], load_contract(),
             [0, 1, 2], 4, civil, designs)
    rows = [json.loads(line) for line in (civil / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * len(designs)
    for row in rows:
        design = next(d for d in designs if d["name"] == row["condition"])
        assert set(row["mask_checks"]) == {str(l) for l in design["layers"]}
        for check in row["mask_checks"].values():
            total = (1 if row["key"] == "a" else 5) + 3
            expected = min(3, total) if design["scope"] == "early3" else total - 3 if design["scope"] == "after3" else total
            assert check["masked_queries"] == expected
    wiki = tmp_path / "wiki"; wiki.mkdir()
    summaries, gates = measure(model, torch.tensor([[2, 3, 4, 5, 6]]), [0, 1, 2], [(1, 2)], wiki, designs)
    assert [r["condition"] for r in summaries] == [d["name"] for d in designs]
    assert all(gates[d["name"]]["observation_logits_exact"] for d in designs)
    assert gates["restore"]["logits_exact"]
