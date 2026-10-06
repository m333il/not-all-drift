from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from .config import ModelConfig
from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class LayerMetadata:
    index: int
    normalized_depth: float
    attention: str


class ModelFamilyAdapter:
    family: str

    def layer_metadata(self, layers: int) -> tuple[LayerMetadata, ...]:
        raise NotImplementedError

    def output_is_non_thinking(self, output: str) -> bool:
        del output
        return True

    @staticmethod
    def transformer_layers(model: Any) -> Any:
        return transformer_layers(model)

    @staticmethod
    def final_norm(model: Any) -> Any | None:
        base = getattr(unwrap_hf_model(model), "model", unwrap_hf_model(model))
        norm = getattr(base, "norm", None)
        return norm if norm is not None else getattr(base, "final_layernorm", None)


def readout_logits(model: Any, states: Any) -> Any:
    """Apply the model's output head, including Gemma 2 final logit softcapping."""
    # Unwrap PEFT so the readout is identical with and without an adapter.
    base = unwrap_hf_model(model)
    logits = base.lm_head(states)
    cap = getattr(getattr(base, "config", None), "final_logit_softcapping", None)
    if cap:
        import torch

        logits = torch.tanh(logits / cap) * cap
    return logits


def unwrap_hf_model(model: Any) -> Any:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def transformer_layers(model: Any) -> Any:
    causal_model = unwrap_hf_model(model)
    base = getattr(causal_model, "model", causal_model)
    for candidate in (
        getattr(base, "layers", None),
        getattr(getattr(base, "model", None), "layers", None),
        getattr(getattr(base, "transformer", None), "h", None),
    ):
        if candidate is not None:
            return candidate
    raise TypeError("unsupported decoder architecture: cannot locate transformer blocks")


T = TypeVar("T", bound=type[ModelFamilyAdapter])
MODEL_FAMILY_FACTORY: dict[str, type[ModelFamilyAdapter]] = {}


def register_model_family(name: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        if name in MODEL_FAMILY_FACTORY:
            raise RuntimeError(f"model family already registered: {name}")
        MODEL_FAMILY_FACTORY[name] = cls
        return cls

    return decorator


def model_family_factory(name: str) -> ModelFamilyAdapter:
    try:
        return MODEL_FAMILY_FACTORY[name]()
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_FAMILY_FACTORY))
        raise ConfigurationError(f"unknown model family {name!r}; available: {available}") from exc


@register_model_family("gemma2")
class Gemma2Adapter(ModelFamilyAdapter):
    family = "gemma2"

    def layer_metadata(self, layers: int) -> tuple[LayerMetadata, ...]:
        return tuple(
            LayerMetadata(
                index,
                (index + 1) / layers,
                "local" if index % 2 == 0 else "global",
            )
            for index in range(layers)
        )


@register_model_family("qwen3")
class Qwen3Adapter(ModelFamilyAdapter):
    family = "qwen3"

    def layer_metadata(self, layers: int) -> tuple[LayerMetadata, ...]:
        return tuple(
            LayerMetadata(index, (index + 1) / layers, "global") for index in range(layers)
        )

    def output_is_non_thinking(self, output: str) -> bool:
        without_empty_blocks = re.sub(r"<think>\s*</think>", "", output, flags=re.IGNORECASE)
        return "<think>" not in without_empty_blocks.casefold()


@register_model_family("synthetic")
class SyntheticAdapter(ModelFamilyAdapter):
    family = "synthetic"

    def layer_metadata(self, layers: int) -> tuple[LayerMetadata, ...]:
        return tuple(
            LayerMetadata(index, (index + 1) / layers, "synthetic") for index in range(layers)
        )


def load_hf_model(model_config: ModelConfig, *, device: str = "cuda:0") -> tuple[Any, Any]:
    """Load a pinned model on the visible device. Never downloads."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ConfigurationError("install the 'models' extra for HF model loading") from exc
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[model_config.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.id,
        revision=model_config.tokenizer_revision or model_config.revision,
        local_files_only=True,
    )
    model = (
        cast(Any, AutoModelForCausalLM)
        .from_pretrained(
            model_config.id,
            revision=model_config.revision,
            torch_dtype=dtype,
            local_files_only=True,
        )
        .to(device)
    )
    return model, tokenizer


__all__ = [
    "LayerMetadata",
    "MODEL_FAMILY_FACTORY",
    "ModelFamilyAdapter",
    "load_hf_model",
    "model_family_factory",
    "register_model_family",
    "readout_logits",
    "transformer_layers",
    "unwrap_hf_model",
]
