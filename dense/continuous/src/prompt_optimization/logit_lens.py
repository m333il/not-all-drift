"""Vanilla Logit Lens token-probability readouts."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .tuned_lens import (
    decoder_dtype,
    project_final_states,
    project_intermediate_states,
)


@torch.inference_mode()
def direct_token_probabilities(
    states: torch.Tensor,
    *,
    token_ids: Sequence[int],
    final_norm: nn.Module,
    lm_head: nn.Module,
    device: torch.device,
    batch_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return selected-token probabilities as ``[samples, depth, tokens]``.

    Intermediate residual states pass through the frozen final normalization
    and unembedding. The stored final state is already normalized and is sent
    directly to the unembedding, preventing a second normalization.
    """
    if states.ndim != 3 or len(states) == 0:
        raise ValueError("states must have shape [samples, depth, hidden_size]")
    if not token_ids or len(set(token_ids)) != len(token_ids) or min(token_ids) < 0:
        raise ValueError("token_ids must be unique non-negative integers")
    if batch_size <= 0 or temperature <= 0:
        raise ValueError("batch_size and temperature must be positive")
    indices = torch.tensor(token_ids, device=device, dtype=torch.long)
    result = torch.empty(
        len(states), states.shape[1], len(token_ids), dtype=torch.float32
    )
    for start in range(0, len(states), batch_size):
        stop = min(start + batch_size, len(states))
        for depth in range(states.shape[1]):
            values = states[start:stop, depth].to(device=device, dtype=torch.float32)
            logits = (
                project_final_states(values.to(dtype=decoder_dtype(lm_head)), lm_head)
                if depth == states.shape[1] - 1
                else project_intermediate_states(
                    values, final_norm=final_norm, lm_head=lm_head
                )
            )
            if int(indices.max()) >= logits.shape[-1]:
                raise ValueError("a selected token ID is outside the model vocabulary")
            probabilities = (logits.float() / temperature).softmax(-1)
            result[start:stop, depth] = probabilities[:, indices].cpu()
    return result
