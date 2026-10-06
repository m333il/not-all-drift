"""Tests for the PEFT config the gradient arms are built with.

The GQA case is the reason this file exists. PEFT provisions a prefix shaped
``[batch, num_attention_heads, n_virtual, token_dim // num_attention_heads]``
and only slices it down to a layer's real KV shape for models exposing
``layer_types``/``per_layer_config``. Ling exposes neither and is GQA
(16 attention heads over 4 KV heads), so the default config would provision a
4x-too-wide prefix, every layer would reject it, and P-Tuning-v2 would die with
"Prefix tuning skipped every layer". ``kv_prefix_shape`` computes the override.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mrd.peft_training import (
    PeftTrainConfig,
    _build_peft_config,
    kv_prefix_shape,
)


def _model(**config_kwargs):
    return SimpleNamespace(config=SimpleNamespace(**config_kwargs))


def test_gqa_model_reports_kv_heads_not_attention_heads():
    """Ling-mini-2.0's real numbers."""
    model = _model(
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        hidden_size=2048,
    )

    num_kv_heads, head_dim = kv_prefix_shape(model)

    assert num_kv_heads == 4, "must follow the KV cache, not the query heads"
    assert head_dim == 128
    # what gets handed to PrefixTuningConfig
    assert num_kv_heads * head_dim == 512


def test_mha_model_is_unchanged_by_the_override():
    """Without GQA the override is a no-op - same shape either way."""
    model = _model(
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=64,
        hidden_size=1024,
    )

    num_kv_heads, head_dim = kv_prefix_shape(model)

    assert (num_kv_heads, head_dim) == (16, 64)
    assert num_kv_heads * head_dim == model.config.hidden_size


def test_falls_back_to_attention_heads_when_kv_heads_absent():
    """Older configs predate num_key_value_heads entirely."""
    model = _model(num_attention_heads=8, hidden_size=512)

    num_kv_heads, head_dim = kv_prefix_shape(model)

    assert num_kv_heads == 8
    assert head_dim == 64, "derived from hidden_size // num_attention_heads"


def test_head_dim_derived_when_absent_but_kv_heads_present():
    model = _model(num_attention_heads=32, num_key_value_heads=8, hidden_size=4096)

    num_kv_heads, head_dim = kv_prefix_shape(model)

    assert num_kv_heads == 8
    assert head_dim == 128, "hidden_size // num_attention_heads, not // num_kv_heads"


@pytest.mark.parametrize("explicit_head_dim", [64, 128, 256])
def test_explicit_head_dim_wins_over_derivation(explicit_head_dim):
    """head_dim is not always hidden_size // num_attention_heads (Ling: 2048/16
    happens to equal 128, but that coincidence must not be relied on)."""
    model = _model(
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=explicit_head_dim,
        hidden_size=2048,
    )

    _, head_dim = kv_prefix_shape(model)

    assert head_dim == explicit_head_dim


# ── per-paper defaults ───────────────────────────────────────────────────────
# These pin the published recipes so a future edit cannot quietly drift back to
# library defaults. Sources: Lester et al. 2104.08691 (Adafactor, constant LR
# 0.3, prompt length 100, sampled-vocab/class-label init) and THUDM/P-tuning-v2
# run_script/*.sh (AdamW, lr 5e-3 on RTE/SQuAD).


def _cfg(method, **kw):
    return PeftTrainConfig(method=method, sft_targets=Path("unused.jsonl"), **kw)


def test_prefix_tuning_uses_the_original_mlp_projection():
    cfg = _cfg("prefix_tuning")
    peft_cfg = _build_peft_config(
        cfg,
        _model(
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=128,
            hidden_size=2048,
        ),
        "unused",
    )
    assert peft_cfg.prefix_projection is True


def test_default_prompt_length_is_lesters_hundred():
    assert _cfg("prompt_tuning").num_virtual_tokens == 100
    assert _cfg("prefix_tuning").num_virtual_tokens == 100
    assert _cfg("p_tuning_v2").num_virtual_tokens == 100


def test_prompt_init_defaults_away_from_random():
    """PEFT's default is RANDOM, which is the weakest option in Lester's own
    ablation and ships in no config of their repo."""
    assert _cfg("prompt_tuning").prompt_init == "sample_vocab"


def test_prefix_projection_defaults_off_but_is_configurable():
    """Task-dependent in the paper's ablation, not a defining feature - so it
    must be a knob, not a constant."""
    assert _cfg("p_tuning_v2").prefix_projection is False
    assert _cfg("p_tuning_v2", prefix_projection=True).prefix_projection is True
