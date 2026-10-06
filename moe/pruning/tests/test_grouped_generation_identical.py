"""Same prompts, same weights, two kernels: do they generate the same tokens?

The block-level check says the batched form is a correct implementation, and
the 2000-row A/B says the answers move on 1.45% of rows. Neither runs in a test
suite: one needs a 62 GB card, the other four hours.

This closes the gap cheaply. A Qwen3-MoE small enough to hold in memory, real
`generate()` with the real mask hooks, greedy decoding, token ids compared
exactly. At float64 the two orderings agree to 1e-15, so any difference in the
ids is the code, not the arithmetic - which makes this a regression test rather
than a measurement.

The bf16 case is the one that matters in production and is deliberately not
asserted equal: there the orderings differ in the last bits and a near-tie can
flip, which is what the A/B measured. It is reported, not enforced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mrd_pruning.grouped_moe import group_qwen3_moe  # noqa: E402


def tiny_model(dtype):
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_experts=16, num_experts_per_tok=4, norm_topk_prob=True,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, vocab_size=128, decoder_sparse_step=1, mlp_only_layers=[],
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).to(dtype).eval()
    return model


def generate(model, ids):
    with torch.inference_mode():
        return model.generate(ids, max_new_tokens=24, do_sample=False,
                              pad_token_id=0, use_cache=True)


class _Holder:
    def __init__(self, model):
        self._m = model

    def modules(self):
        return list(self._m.modules())


def test_float64_generation_is_token_for_token_identical():
    """At float64 the two orderings agree to 1e-15, so the ids must match."""
    torch.manual_seed(7)
    ids = torch.randint(0, 128, (3, 8))

    ref = tiny_model(torch.float64)
    want = generate(ref, ids)

    fast = tiny_model(torch.float64)          # same seed, same weights
    assert group_qwen3_moe(fast, token_budget=4096) == 4, "All layers must be restructured."
    got = generate(fast, ids)

    assert torch.equal(got, want), (
        "Butch core issued other tokens where the arithmetic coincides to 1e-15"
    )


def test_the_conversion_touched_every_moe_layer():
    model = tiny_model(torch.float64)
    n = group_qwen3_moe(model)
    assert n == 4
    # Nothing left holding the old per-expert Linears.
    for mod in model.modules():
        if type(mod).__name__ == "Qwen3MoeSparseMoeBlock":
            assert not isinstance(mod.experts, torch.nn.ModuleList)


def test_bfloat16_agreement_is_reported_not_required(capsys):
    """bf16 may flip a near-tie - the A/B measured 1.45% of rows on the real model.

    Asserting equality here would be asserting that rounding does not exist.
    What is worth guarding is that the two stay *close*: a wholesale divergence
    means something other than rounding.
    """
    torch.manual_seed(7)
    ids = torch.randint(0, 128, (4, 8))

    want = generate(tiny_model(torch.bfloat16), ids)
    fast = tiny_model(torch.bfloat16)
    group_qwen3_moe(fast, token_budget=4096)
    got = generate(fast, ids)

    same = (got == want).float().mean().item()
    print(f"bf16: Coincidence of tokens {100 * same:.1f}%")
    assert same > 0.5, f"bf16 matched only {100 * same:.0f}The percentage of tokens is not rounding"
