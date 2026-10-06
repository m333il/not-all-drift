from __future__ import annotations

import torch
from transformers import Gemma2Config

from prompt_optimization.prefix_cache import build_gemma_prefix_cache


def test_build_gemma_prefix_cache_consumes_per_layer_stacked_key_values() -> None:
    config = Gemma2Config(
        num_hidden_layers=2,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["sliding_attention", "full_attention"],
    )
    per_layer = tuple(torch.randn(2, 1, 1, 3, 4) for _ in range(2))
    cache = build_gemma_prefix_cache(
        per_layer,
        base_config=config,
        num_virtual_tokens=3,
    )
    assert len(cache.layers) == 2
    assert cache.get_seq_length(0) == 3
    assert cache.get_seq_length(1) == 3
