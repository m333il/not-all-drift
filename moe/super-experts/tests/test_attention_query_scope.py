import pytest
import torch

from test_attention_contributions import tiny_model, wrapped, VIRTUAL
from se_gepa.attention_contributions import AttentionContributionProbe


@pytest.mark.parametrize('kind', ['base', 'prefix', 'prompt'])
def test_late_query_readouts_match_native_attention(kind):
    model, _ = tiny_model(73) if kind == 'base' else wrapped(kind, 73)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    virtual_queries = VIRTUAL if kind == 'prompt' else 0
    prefix_keys = VIRTUAL if kind == 'prefix' else 0
    with torch.no_grad():
        native = model(input_ids=ids, use_cache=False, output_attentions=True)
        with AttentionContributionProbe(model, [1], virtual_query_tokens=virtual_queries,
                prefix_key_tokens=prefix_keys, real_query_start=3) as probe:
            probe.begin_example(0, 6)
            observed = model(input_ids=ids, use_cache=False)
    assert torch.equal(native.logits, observed.logits)
    row = probe.records[0]
    assert row['real_queries'] == 3 and row['real_query_start'] == 3
    virtual = virtual_queries + prefix_keys
    weights = native.attentions[1][0, :, virtual_queries + 3:]
    for i in range(3):
        expected = weights[:, :, virtual + i].float().mean(-1)
        assert row['per_head_attention_mass'][f'real_{i}'] == pytest.approx(expected.tolist(), abs=1e-6)
    assert row['reconstruction']['after_o_proj']['relative_l2'] < 1e-5
