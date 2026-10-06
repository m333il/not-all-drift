"""Compatibility support for PEFT Prefix Tuning on Gemma 2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from peft import TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING
from transformers import DynamicCache


def build_gemma_prefix_cache(
    per_layer_past_key_values: Sequence[torch.Tensor],
    *,
    base_config: Any,
    num_virtual_tokens: int,
) -> DynamicCache:
    """Convert PEFT's per-layer stacked K/V tensors to a Gemma DynamicCache.

    PEFT 0.19.1 splits the prefix into one ``[2, batch, heads, tokens, dim]``
    tensor per layer, but its Gemma branch indexes the first tensor as if it
    contained all layers. This follows the corrected upstream PEFT iteration.
    """
    if not per_layer_past_key_values:
        raise ValueError("Prefix cache must contain at least one layer")
    cache = DynamicCache(config=base_config)
    cache_position = torch.arange(
        num_virtual_tokens,
        device=per_layer_past_key_values[0].device,
    )
    for layer_index, layer_past in enumerate(per_layer_past_key_values):
        if layer_past.shape[0] != 2:
            raise ValueError(
                "Expected per-layer prefix tensor with stacked key/value dimension of size 2"
            )
        key_states, value_states = layer_past
        cache.update(
            key_states,
            value_states,
            layer_index,
            cache_kwargs={"cache_position": cache_position},
        )
    return cache


def install_gemma_prefix_cache_compatibility(
    peft_model: Any,
    *,
    num_virtual_tokens: int,
) -> str | None:
    """Install a process-local Gemma postprocessor when Prefix Tuning needs it."""
    base_model = peft_model.get_base_model()
    base_config = base_model.config
    model_type = str(getattr(base_config, "model_type", ""))
    if model_type != "gemma2":
        return None

    def postprocess(per_layer_past_key_values: Sequence[torch.Tensor]) -> DynamicCache:
        return build_gemma_prefix_cache(
            per_layer_past_key_values,
            base_config=base_config,
            num_virtual_tokens=num_virtual_tokens,
        )

    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[model_type] = postprocess
    return "peft-0.19.1-gemma2-per-layer-dynamic-cache"
