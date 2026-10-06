"""PEFT configuration, target encoding and batching for the continuous arms.

Prompt tuning learns virtual tokens at the input embeddings (Lester et al.,
2021). Prefix tuning learns per-layer key and value prefixes through PEFT's
MLP reparameterization (Li and Liang, 2021). The training loop itself is
``mrd.training.train_encoded``.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from mrd.chat import encode_assistant_turn
from mrd.prompts import SEED_SYSTEM_PROMPT
from mrd.tasks.sft_targets import SFTExample

log = logging.getLogger(__name__)

Method = Literal["prompt_tuning", "prefix_tuning", "p_tuning_v2"]


@dataclass
class PeftTrainConfig:
    method: Method
    sft_targets: Path | None = None
    num_virtual_tokens: int = 100
    # Prompt tuning only: "sample_vocab", "text" or "random".
    prompt_init: str = "sample_vocab"
    prompt_init_text: str | None = None
    # Prefix tuning always uses the MLP projection; p_tuning_v2 makes it optional.
    prefix_projection: bool = False


def kv_prefix_shape(model) -> tuple[int, int]:
    """``(num_kv_heads, head_dim)`` this model's attention actually caches.

    Needed because PEFT provisions the prefix as
    ``[batch, num_attention_heads, n_virtual, token_dim // num_attention_heads]``
    and only slices it down to the true KV shape for models that expose
    ``layer_types``/``per_layer_config`` (Gemma-style). For a grouped-query
    model without them the provisioned prefix would be too wide and every
    layer's ``cache.update`` would reject it; PEFT reports that as "Prefix
    tuning skipped every layer".

    Passing these back through ``num_attention_heads``/``token_dim`` is exactly
    the override PEFT's own error message prescribes.
    """
    cfg = model.config
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    return num_kv_heads, head_dim


def _build_peft_config(cfg: PeftTrainConfig, model, tokenizer_name: str):
    from peft import PrefixTuningConfig, PromptTuningConfig, PromptTuningInit, TaskType

    if cfg.method == "prompt_tuning":
        init = cfg.prompt_init.lower()
        if init == "text":
            if not cfg.prompt_init_text:
                raise ValueError("prompt_init='text' requires prompt_init_text")
            return PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=cfg.num_virtual_tokens,
                prompt_tuning_init=PromptTuningInit.TEXT,
                prompt_tuning_init_text=cfg.prompt_init_text,
                tokenizer_name_or_path=tokenizer_name,
            )
        if init == "sample_vocab":
            # Lester samples from the 5000 most frequent tokens; PEFT samples
            # uniformly over the whole vocabulary. Same family, coarser prior -
            # noted rather than worked around, since PEFT exposes no frequency
            # cutoff.
            return PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=cfg.num_virtual_tokens,
                prompt_tuning_init=PromptTuningInit.SAMPLE_VOCAB,
            )
        if init == "random":
            log.warning(
                "prompt_init='random' is the weakest option in Lester et al.'s own "
                "ablation and ships in no config of their repo - use it only as a control"
            )
            return PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=cfg.num_virtual_tokens,
                prompt_tuning_init=PromptTuningInit.RANDOM,
            )
        raise ValueError(f"unknown prompt_init {cfg.prompt_init!r}")

    if cfg.method in ("prefix_tuning", "p_tuning_v2"):
        num_kv_heads, head_dim = kv_prefix_shape(model)
        log.info("prefix KV geometry: num_kv_heads=%d head_dim=%d (token_dim=%d)",
                 num_kv_heads, head_dim, num_kv_heads * head_dim)
        return PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=cfg.num_virtual_tokens,
            prefix_projection=(cfg.method == "prefix_tuning" or cfg.prefix_projection),
            num_attention_heads=num_kv_heads,
            token_dim=num_kv_heads * head_dim,
        )
    raise ValueError(f"unknown method {cfg.method!r}")


def _encode(tokenizer, example: SFTExample, system_prompt=SEED_SYSTEM_PROMPT) -> tuple[list[int], list[int]]:
    """Mask the generation prefix; supervise the native assistant turn and terminator."""
    if getattr(tokenizer, "chat_template", None):
        return encode_assistant_turn(
            tokenizer, system_prompt, example["input_text"], example["target_text"],
            thinking=example.get("thinking"),
        )
    else:
        context = f"{system_prompt}\n\n" if system_prompt else ""
        prompt_ids = tokenizer(f"{context}{example['input_text']}\n", add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(example["target_text"], add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = list(target_ids) + [tokenizer.eos_token_id]
    return list(prompt_ids) + list(target_ids), [-100] * len(prompt_ids) + list(target_ids)


def _pad_batch(encoded: list[tuple[list[int], list[int]]], pad_id: int, device):
    """``[(input_ids, labels), ...]`` -> right-padded ``(ids, attention_mask, labels)``.

    Right padding, not left: the sequence is ``[virtual tokens][prompt][target]``
    and peft prepends the virtual block itself, so padding on the right leaves
    every real token at the same relative position it had at batch size 1.
    ``attention_mask`` zeroes the pad columns and ``labels`` is ``-100`` there,
    so pad tokens contribute to neither attention nor loss.
    """
    max_len = max(len(ids) for ids, _ in encoded)
    ids_rows, mask_rows, label_rows = [], [], []
    for ids, labels in encoded:
        pad = max_len - len(ids)
        ids_rows.append(ids + [pad_id] * pad)
        mask_rows.append([1] * len(ids) + [0] * pad)
        label_rows.append(labels + [-100] * pad)
    return (
        torch.tensor(ids_rows, device=device),
        torch.tensor(mask_rows, device=device),
        torch.tensor(label_rows, device=device),
    )


def _length_bucketed_batches(
    encoded: list[tuple[list[int], list[int]]], batch_size: int, rng: random.Random,
) -> list[list[int]]:
    """Indices grouped into batches of similar length, batch order shuffled.

    Sorting by length before batching keeps padding small. Randomness is kept at
    the batch level (order of batches) rather than within batches.
    """
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    batches = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches
