from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .activations import forward_with_residuals
from .modeling import model_family_factory, readout_logits


def token_sequence_logprobs(
    logits: np.ndarray, token_ids: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Sum teacher-forced log probabilities over every token in a label string."""
    if logits.ndim != 3 or token_ids.shape != logits.shape[:2] or mask.shape != token_ids.shape:
        raise ValueError("expected logits [batch,tokens,vocab] and aligned token IDs/mask")
    log_probabilities = logits - logsumexp(logits, axis=-1, keepdims=True)
    selected = np.take_along_axis(log_probabilities, token_ids[..., None], axis=-1)[..., 0]
    return np.where(mask, selected, 0.0).sum(axis=1)


def layerwise_teacher_forced_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    family: str,
    prompts: Sequence[str],
    candidate_strings: Sequence[str],
    include_eos: bool = False,
    length_normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Score complete candidate serializations at every residual layer.

    Returns ``(scores, token_counts)`` with scores shaped [examples, candidates, layers].
    With ``length_normalize`` the score is the mean log-probability per candidate token.
    """
    import torch

    adapter = model_family_factory(family)
    norm = adapter.final_norm(model)
    device = next(model.parameters()).device
    rows: list[np.ndarray] = []
    counts: list[int] = []
    model.eval()
    for prompt in prompts:
        candidate_rows: list[np.ndarray] = []
        for candidate in candidate_strings:
            complete = prompt + candidate
            encoded = tokenizer(
                complete,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            complete_ids = list(encoded.input_ids)
            target_positions = [
                index
                for index, (_start, end) in enumerate(encoded.offset_mapping)
                if end > len(prompt)
            ]
            if include_eos and tokenizer.eos_token_id is not None:
                complete_ids.append(tokenizer.eos_token_id)
                target_positions.append(len(complete_ids) - 1)
            if not complete_ids or not target_positions or target_positions[0] == 0:
                raise ValueError("candidate serialization has no causally scoreable tokens")
            input_ids = torch.as_tensor([complete_ids], device=device)
            # Keep the readout under no_grad (lm_head parameters require grad).
            with torch.no_grad():
                _, hidden_states = forward_with_residuals(
                    model, input_ids=input_ids, return_dict=True, use_cache=False
                )
                layer_scores = []
                # Prompt tuning shifts hidden states by its virtual tokens; prefix tuning does not.
                offset = int(hidden_states[0].shape[1]) - int(input_ids.shape[1])
                if offset < 0:
                    raise ValueError("residual stream is shorter than the input sequence")
                state_positions = torch.as_tensor(
                    [position - 1 + offset for position in target_positions], device=device
                )
                targets = torch.as_tensor(
                    [complete_ids[position] for position in target_positions], device=device
                )
                for hidden in hidden_states:
                    states = hidden[:, state_positions]
                    logits = readout_logits(model, norm(states) if norm is not None else states)[0]
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    total = log_probs.gather(-1, targets[:, None]).sum()
                    layer_scores.append(
                        float(total / len(target_positions)) if length_normalize else float(total)
                    )
            candidate_rows.append(np.asarray(layer_scores, dtype=np.float32))
            counts.append(len(target_positions))
        rows.append(np.stack(candidate_rows))
    return np.stack(rows), np.asarray(counts[: len(candidate_strings)], dtype=np.int32)


def label_margins(scores: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Gold-vs-nongold margin for multilabel candidate scores."""
    if scores.shape[:2] != gold.shape or scores.ndim != 3:
        raise ValueError("scores must be [examples,labels,layers] with aligned gold labels")
    margins = np.empty((scores.shape[0], scores.shape[2]), dtype=np.float32)
    for index in range(scores.shape[0]):
        positive = gold[index].astype(bool)
        if not np.any(positive) or np.all(positive):
            margins[index] = np.nan
            continue
        margins[index] = scores[index, positive].mean(0) - scores[index, ~positive].mean(0)
    return margins


__all__ = [
    "label_margins",
    "layerwise_teacher_forced_logprobs",
    "token_sequence_logprobs",
]
