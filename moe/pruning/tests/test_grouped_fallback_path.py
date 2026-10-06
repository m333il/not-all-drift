"""The fallback path is not the tested path, and in production it is the busy one.

`_grouped_forward` hands off to `_loop_forward` whenever a call carries more
tokens than the budget (256 by default). At batch 128 that means every prefill
goes down the fallback and only the decode steps use the batched form - so the
fallback decides most of what the model reads before it writes anything.

The existing generation test pins the batched form with `token_budget=4096`,
which never reaches the fallback at all. This file covers the other half, and
the masked case besides: pruning works by driving router logits to
`finfo.min`, so the shape the sweep actually runs is "fallback path, most
experts masked off".

float64 throughout: there the two summation orders agree to 1e-15, so any
difference in the tokens is the code rather than the arithmetic.
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
    return Qwen3MoeForCausalLM(cfg).to(dtype).eval()


def generate(model, ids):
    with torch.inference_mode():
        return model.generate(ids, max_new_tokens=24, do_sample=False,
                              pad_token_id=0, use_cache=True)


def mask_experts(model, drop):
    """Pruning as the sweep does it: router logits driven to the dtype floor.

    Returns the handles so a caller could remove them; the tests build a fresh
    model each time and do not bother.
    """
    handles = []
    for module in model.modules():
        if type(module).__name__ != "Qwen3MoeSparseMoeBlock":
            continue

        def hook(_mod, _inp, out, drop=drop):
            out[:, drop] = torch.finfo(out.dtype).min
            return out

        handles.append(module.gate.register_forward_hook(hook))
    return handles


def test_fallback_path_generates_the_same_tokens():
    """Budget 1 forces every call through `_loop_forward`."""
    torch.manual_seed(7)
    ids = torch.randint(0, 128, (3, 8))

    want = generate(tiny_model(torch.float64), ids)

    fast = tiny_model(torch.float64)
    assert group_qwen3_moe(fast, token_budget=1) == 4
    got = generate(fast, ids)

    assert torch.equal(got, want), (
        "The backup path issued other tokens where the arithmetic coincides to 1e-15")


def test_mixed_budget_matches_too():
    """The real shape: prefill over the budget, decode under it.

    With a budget of 8 and a prompt of 8 rows × 8 tokens, the prefill (24
    tokens per sequence in one call) takes the fallback and each decode step
    takes the batched form - both inside one `generate`.
    """
    torch.manual_seed(11)
    ids = torch.randint(0, 128, (2, 8))

    want = generate(tiny_model(torch.float64), ids)

    fast = tiny_model(torch.float64)
    assert group_qwen3_moe(fast, token_budget=8) == 4
    got = generate(fast, ids)

    assert torch.equal(got, want)


def test_masked_router_fallback_matches():
    """Pruning plus the fallback - the combination the sweep actually ran."""
    torch.manual_seed(3)
    ids = torch.randint(0, 128, (2, 8))
    drop = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])      # half the experts

    ref = tiny_model(torch.float64)
    mask_experts(ref, drop)
    want = generate(ref, ids)

    fast = tiny_model(torch.float64)
    assert group_qwen3_moe(fast, token_budget=1) == 4
    mask_experts(fast, drop)
    got = generate(fast, ids)

    assert torch.equal(got, want), "under the mask of a spare path diverges from the stock"


def test_masked_router_batched_matches():
    """Same mask, batched form."""
    torch.manual_seed(3)
    ids = torch.randint(0, 128, (2, 8))
    drop = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    ref = tiny_model(torch.float64)
    mask_experts(ref, drop)
    want = generate(ref, ids)

    fast = tiny_model(torch.float64)
    assert group_qwen3_moe(fast, token_budget=4096) == 4
    mask_experts(fast, drop)
    got = generate(fast, ids)

    assert torch.equal(got, want), "under the mask of the batch path diverges from the stock"


def test_padded_rows_do_not_change_the_answer():
    """The sweep pads batches; a padded neighbour must not move a row's tokens.

    Every level in this run was generated with `--pad-batches`, and the cell
    under suspicion ran at batch 128 against the loop cell's 64. If the batched
    kernel were sensitive to what shares its batch, that difference alone would
    explain a collapse - so it is worth pinning rather than assuming.
    """
    torch.manual_seed(5)
    ids = torch.randint(1, 128, (1, 8))

    fast = tiny_model(torch.float64)
    group_qwen3_moe(fast, token_budget=4096)
    alone = generate(fast, ids)

    crowd = torch.cat([ids, torch.randint(1, 128, (3, 8))], dim=0)
    together = generate(fast, crowd)

    assert torch.equal(together[0], alone[0]), (
        "The line changed from who else was in the batch.")
