"""Portable loading utilities for text, Prompt Tuning, and Prefix Tuning conditions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prefix_cache import install_gemma_prefix_cache_compatibility


@dataclass(frozen=True)
class LoadedCondition:
    model: Any
    tokenizer: Any
    prompt_template: str
    hidden_state_offset: int
    condition_type: str
    num_virtual_tokens: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_template(path: Path) -> str:
    template = path.read_text(encoding="utf-8").strip()
    if template.count("{text}") != 1:
        raise ValueError("Prompt template must contain exactly one {text} placeholder")
    return template


def peft_type_name(model: Any) -> str | None:
    if not hasattr(model, "peft_config"):
        return None
    value = model.peft_config["default"].peft_type
    return str(getattr(value, "value", value)).upper()


def prepended_hidden_tokens(peft_type: str | None, num_virtual_tokens: int) -> int:
    """Return virtual residual positions visible in hidden states.

    Prompt Tuning prepends embeddings to the residual stream. Prefix Tuning
    contributes key/value states only, so it adds no residual token positions.
    """
    return num_virtual_tokens if peft_type == "PROMPT_TUNING" else 0


def load_condition(
    *,
    model_name: str,
    prompt_template: Path,
    adapter_path: Path | None = None,
    model_revision: str | None = None,
    device_map: dict[str, int] | str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | None = None,
    local_files_only: bool = False,
) -> LoadedCondition:
    """Load a frozen base model and, optionally, a PEFT Prompt/Prefix adapter."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "revision": model_revision,
        "local_files_only": local_files_only,
        "torch_dtype": torch_dtype,
        "device_map": device_map if device_map is not None else {"": 0},
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    template = load_prompt_template(prompt_template)
    if adapter_path is None:
        model = base
        condition_type = "text"
        virtual_tokens = 0
        hidden_offset = 0
    else:
        model = PeftModel.from_pretrained(base, adapter_path)
        peft_type = peft_type_name(model)
        if peft_type not in {"PROMPT_TUNING", "PREFIX_TUNING"}:
            raise ValueError(f"Unsupported PEFT condition: {peft_type}")
        virtual_tokens = int(model.peft_config["default"].num_virtual_tokens)
        hidden_offset = prepended_hidden_tokens(peft_type, virtual_tokens)
        condition_type = "prompt" if peft_type == "PROMPT_TUNING" else "prefix"
        if condition_type == "prefix":
            install_gemma_prefix_cache_compatibility(
                model,
                num_virtual_tokens=virtual_tokens,
            )
    model.eval()
    model.config.use_cache = False
    return LoadedCondition(
        model=model,
        tokenizer=tokenizer,
        prompt_template=template,
        hidden_state_offset=hidden_offset,
        condition_type=condition_type,
        num_virtual_tokens=virtual_tokens,
    )


def render_prompt(tokenizer: Any, template: str, text: str) -> str:
    content = template.format(text=text)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def base_model(model: Any) -> Any:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def decoder_layers(model: Any) -> Any:
    base = base_model(model)
    if not hasattr(base, "model") or not hasattr(base.model, "layers"):
        raise TypeError("Expected a causal LM with model.layers decoder blocks")
    return base.model.layers


def final_readout(model: Any, states: torch.Tensor) -> torch.Tensor:
    """Apply the model's final normalization and LM head to residual states."""
    base = base_model(model)
    dtype = next(base.parameters()).dtype
    hidden = base.model.norm(states.to(device=base.device, dtype=dtype))
    logits = base.lm_head(hidden)
    cap = getattr(base.config, "final_logit_softcapping", None)
    if cap is not None:
        logits = cap * torch.tanh(logits / cap)
    return logits.float()
