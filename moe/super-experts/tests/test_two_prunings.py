"""Zeroing down_proj and masking the router are different interventions.

The paper prunes by "skipping the experts when selected by the router": the
router still picks them, they still take their share of the gate weights, and
they return nothing, so the block's output is attenuated by that share. Deleting
an expert from a shipped model removes its row from the gate instead, the freed
slot goes to the next expert and the weights renormalise over the winners.

The test that matters asserts on the router's **indices**. An earlier version
asserted that the mask had been written into the logits the router returns -- and
passed while masking nothing, because on this runtime the block discards those
logits and uses the indices the router already chose. Checking the write is not
checking the effect.

CPU, tiny random model -- a wiring test.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from se_gepa.ablation import ExpertAblation, RouterMask
from test_profiler import tiny_model

LAYER = 1


def routing(model, ids):
    """The indices and weights the block actually receives."""
    seen = {}

    def hook(_module, _inputs, output):
        _logits, scores, indices = output
        seen["scores"] = scores.detach().clone()
        seen["indices"] = indices.detach().clone()

    mlp = model.model.layers[LAYER].mlp
    router = getattr(mlp, "gate", None) or mlp.router
    handle = router.register_forward_hook(hook)
    try:
        model(input_ids=ids)
    finally:
        handle.remove()
    return seen


def busiest(model, ids):
    count = getattr(model.config, "num_experts", None) or model.config.num_local_experts
    counts = torch.bincount(routing(model, ids)["indices"].reshape(-1),
                            minlength=count)
    return int(counts.argmax())


def test_router_mask_keeps_the_expert_out_of_every_top_k():
    model, _config = tiny_model("eager")
    ids = torch.arange(1, 13).unsqueeze(0)
    target = busiest(model, ids)
    assert (routing(model, ids)["indices"] == target).any(), "target must be selected without the mask"

    with RouterMask(model, {(LAYER, target)}):
        masked = routing(model, ids)
    assert not (masked["indices"] == target).any()


def test_router_mask_keeps_the_mixture_at_full_weight():
    # The measured model sets norm_topk_prob, so the freed slot is refilled and the
    # weights renormalise over the winners rather than the token losing that share.
    model, config = tiny_model("eager")
    assert config.norm_topk_prob is not None
    model.config.norm_topk_prob = True
    for layer in model.model.layers:
        layer.mlp.gate.norm_topk_prob = True
    ids = torch.arange(1, 13).unsqueeze(0)
    with RouterMask(model, {(LAYER, busiest(model, ids))}):
        weights = routing(model, ids)["scores"]
    total = weights.float().sum(dim=-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-2)


def test_zeroing_down_proj_loses_that_share_of_the_mixture():
    # The other half of the contrast: the expert keeps its weight in the sum and
    # returns nothing, so the block's output at that token is attenuated by it.
    model, _config = tiny_model("eager")
    ids = torch.arange(1, 13).unsqueeze(0)
    target = busiest(model, ids)
    scores = routing(model, ids)["scores"]
    indices = routing(model, ids)["indices"]
    lost = scores[indices == target]
    assert lost.numel() > 0 and float(lost.float().mean()) > 0


def test_zeroing_down_proj_leaves_the_routing_untouched():
    model, _config = tiny_model("eager")
    ids = torch.arange(1, 13).unsqueeze(0)
    target = busiest(model, ids)
    plain = routing(model, ids)
    with ExpertAblation(model, {(LAYER, target)}):
        ablated = routing(model, ids)
    assert torch.equal(plain["indices"], ablated["indices"])
    assert torch.equal(plain["scores"], ablated["scores"])


def test_the_two_give_different_outputs():
    model, _config = tiny_model("eager")
    ids = torch.arange(1, 13).unsqueeze(0)
    target = busiest(model, ids)
    with ExpertAblation(model, {(LAYER, target)}):
        by_weights = model(input_ids=ids).logits.detach().clone()
    with RouterMask(model, {(LAYER, target)}):
        by_router = model(input_ids=ids).logits.detach().clone()
    assert not torch.allclose(by_weights, by_router)


def test_both_restore_the_model():
    model, _config = tiny_model("eager")
    ids = torch.arange(1, 13).unsqueeze(0)
    target = busiest(model, ids)
    before = model(input_ids=ids).logits.detach().clone()
    for intervention in (ExpertAblation, RouterMask):
        with intervention(model, {(LAYER, target)}):
            pass
        assert torch.equal(model(input_ids=ids).logits, before)


def test_router_mask_preserves_gpt_oss_selected_softmax():
    from transformers import GptOssConfig, GptOssForCausalLM

    config = GptOssConfig(
        vocab_size=97, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_local_experts=8,
        num_experts_per_tok=2, max_position_embeddings=2048,
        experts_implementation="grouped_mm",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    model = GptOssForCausalLM(config).to(torch.float32).eval()
    ids = torch.arange(1, 13).unsqueeze(0)
    target = busiest(model, ids)
    with RouterMask(model, {(LAYER, target)}):
        masked = routing(model, ids)
    assert not (masked["indices"] == target).any()
    assert torch.allclose(masked["scores"].sum(dim=-1), torch.ones(ids.numel()))
