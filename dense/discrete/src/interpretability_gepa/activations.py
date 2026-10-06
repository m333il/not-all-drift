from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .modeling import transformer_layers


def _text_span(prompt: str, text: str) -> tuple[int, int]:
    marker = "INPUT_TEXT:\n"
    marker_index = prompt.rfind(marker)
    start = marker_index + len(marker) if marker_index >= 0 else prompt.rfind(text)
    if start < 0 or prompt[start : start + len(text)] != text:
        raise ValueError("input text missing from its rendered prompt field")
    return start, start + len(text)


@dataclass(frozen=True)
class ActivationBatch:
    example_ids: tuple[str, ...]
    last_prompt: np.ndarray
    mean_text: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        expected = len(self.example_ids)
        if self.last_prompt.ndim != 3 or self.mean_text.ndim != 3:
            raise ValueError("activation arrays must have shape [examples, layers, hidden]")
        if self.last_prompt.shape != self.mean_text.shape or self.last_prompt.shape[0] != expected:
            raise ValueError("activation arrays and example IDs are misaligned")


def forward_with_residuals(model: Any, **model_inputs: Any) -> tuple[Any, tuple[Any, ...]]:
    """Run one forward pass and capture embedding plus every raw post-block residual."""
    layers = transformer_layers(model)
    captured: list[Any | None] = [None] * (len(layers) + 1)
    handles = [
        layers[0].register_forward_pre_hook(
            lambda _module, inputs: captured.__setitem__(0, inputs[0])
        )
    ]
    for index, layer in enumerate(layers, start=1):
        handles.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, index=index: captured.__setitem__(
                    index, output[0] if isinstance(output, tuple) else output
                )
            )
        )
    try:
        output = model(**model_inputs)
    finally:
        for handle in handles:
            handle.remove()
    if any(hidden is None for hidden in captured):
        raise RuntimeError("residual hooks did not fire exactly once for every model block")
    return output, tuple(captured)


def logical_last_positions(attention_mask: np.ndarray) -> np.ndarray:
    if attention_mask.ndim != 2:
        raise ValueError("attention mask must have shape [batch, sequence]")
    lengths = attention_mask.astype(bool).sum(axis=1)
    if np.any(lengths == 0):
        raise ValueError("empty prompt in activation batch")
    return lengths - 1


def select_hidden_states(
    hidden_states: Sequence[Any],
    attention_mask: Any,
    text_mask: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Select logical positions from a Transformers hidden-state tuple."""
    import torch

    mask_np = attention_mask.detach().cpu().numpy()
    last = logical_last_positions(mask_np)
    # Prompt tuning prepends virtual tokens to the hidden states but not to the mask;
    # drop them (prefix tuning uses the KV cache and has no offset).
    offset = int(hidden_states[0].shape[1]) - int(attention_mask.shape[1])
    if offset < 0:
        raise ValueError("residual stream is shorter than the attention mask")
    last_layers, mean_layers = [], []
    for full_hidden in hidden_states:
        hidden = full_hidden[:, offset:] if offset else full_hidden
        selected = hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            torch.as_tensor(last, device=hidden.device),
        ]
        weights = text_mask.to(hidden.device, hidden.dtype).unsqueeze(-1)
        means = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        last_layers.append(selected.float().cpu().numpy())
        mean_layers.append(means.float().cpu().numpy())
    return np.stack(last_layers, axis=1).astype(np.float16), np.stack(mean_layers, axis=1).astype(
        np.float16
    )


def save_activation_store(path: Path, batch: ActivationBatch, *, compress: bool = True) -> None:
    import zarr

    from .artifacts import sha256_path

    batch.validate()
    path.mkdir(parents=True, exist_ok=False)
    group = zarr.open_group(path / "acts.zarr", mode="w")
    chunks = (min(128, len(batch.example_ids)), 1, batch.last_prompt.shape[-1])
    compressor = None
    if compress:
        from numcodecs import Blosc

        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    group.create_dataset(
        "last_prompt",
        data=batch.last_prompt,
        shape=batch.last_prompt.shape,
        chunks=chunks,
        compressor=compressor,
    )
    group.create_dataset(
        "mean_text",
        data=batch.mean_text,
        shape=batch.mean_text.shape,
        chunks=chunks,
        compressor=compressor,
    )
    meta = {
        **batch.metadata,
        "example_ids": list(batch.example_ids),
        "shape": list(batch.last_prompt.shape),
        "dtype": "float16",
    }
    encoded = json.dumps(meta, sort_keys=True, indent=2)
    (path / "meta.json").write_text(encoded, encoding="utf-8")
    checksum = {
        "metadata": hashlib.sha256(encoded.encode()).hexdigest(),
        "tensors": sha256_path(path / "acts.zarr"),
    }
    (path / "sha256.txt").write_text(json.dumps(checksum, sort_keys=True) + "\n", encoding="utf-8")


def load_activation_store(path: Path) -> ActivationBatch:
    import zarr

    from .artifacts import sha256_path

    encoded = (path / "meta.json").read_text(encoding="utf-8")
    checksum = json.loads((path / "sha256.txt").read_text(encoding="utf-8"))
    if checksum.get("metadata") != hashlib.sha256(encoded.encode()).hexdigest():
        raise ValueError("activation metadata checksum mismatch")
    if checksum.get("tensors") != sha256_path(path / "acts.zarr"):
        raise ValueError("activation tensor checksum mismatch")
    meta = json.loads(encoded)
    group = zarr.open_group(path / "acts.zarr", mode="r")
    result = ActivationBatch(
        tuple(meta.pop("example_ids")),
        np.asarray(group["last_prompt"]),
        np.asarray(group["mean_text"]),
        meta,
    )
    result.validate()
    return result


def induced_shift(target: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if target.shape != seed.shape or target.ndim != 3:
        raise ValueError("target and seed activations must share [examples,layers,hidden]")
    per_example = target.astype(np.float32) - seed.astype(np.float32)
    return per_example, per_example.mean(axis=0)


def induced_shift_batches(
    seed: ActivationBatch,
    target: ActivationBatch,
    *,
    position: str = "last_prompt",
) -> tuple[np.ndarray, np.ndarray]:
    seed.validate()
    target.validate()
    if seed.example_ids != target.example_ids:
        raise ValueError("activation example ordering differs between conditions")
    for key in (
        "model",
        "model_revision",
        "tokenizer_revision",
        "split",
        "example_id_hash",
    ):
        if seed.metadata.get(key) != target.metadata.get(key):
            raise ValueError(f"activation metadata differs for {key}")
    if position not in {"last_prompt", "mean_text"}:
        raise ValueError("position must be last_prompt or mean_text")
    seed_values = getattr(seed, position)
    target_values = getattr(target, position)
    return induced_shift(target_values, seed_values)


def verify_final_logits(model: Any, hidden: Any, expected_logits: Any, atol: float = 2e-3) -> None:
    """Verify the architecture-specific final norm/head convention."""
    import torch

    causal_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    base = getattr(causal_model, "model", causal_model)
    norm = getattr(base, "norm", None)
    if norm is None:
        norm = getattr(base, "final_layernorm", None)
    projected = causal_model.lm_head(norm(hidden) if norm is not None else hidden)
    softcap = getattr(getattr(causal_model, "config", None), "final_logit_softcapping", None)
    if softcap is not None:
        projected = projected / float(softcap)
        projected = torch.tanh(projected) * float(softcap)
    if not torch.allclose(projected, expected_logits, atol=atol, rtol=atol):
        raise AssertionError("hidden-state convention does not reproduce model logits")


def extract_hf(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    texts: Sequence[str],
    example_ids: Sequence[str],
    batch_size: int = 8,
    metadata: dict[str, Any] | None = None,
    verify_logits: bool = False,
) -> ActivationBatch:
    """Extract all residual layers without retaining token-level tensors."""
    import torch

    if len(prompts) != len(texts) or len(prompts) != len(example_ids):
        raise ValueError("prompts, texts, and example IDs must align")
    tokenizer.padding_side = "right"
    last_parts, mean_parts = [], []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = encoded.pop("offset_mapping")
        masks = []
        for prompt, text, row_offsets in zip(
            batch_prompts, texts[start : start + batch_size], offsets, strict=True
        ):
            char_start, char_end = _text_span(prompt, text)
            masks.append([(int(a) < char_end and int(b) > char_start) for a, b in row_offsets])
        text_mask = torch.as_tensor(masks, dtype=torch.bool)
        if not bool(text_mask.any(dim=1).all()):
            raise ValueError("rendered prompt has no tokens overlapping the input text")
        device = next(model.parameters()).device
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            # Only the residual stream is needed; skip the full-sequence vocab projection.
            output, hidden_states = forward_with_residuals(
                model, **model_inputs, return_dict=True, use_cache=False, logits_to_keep=1
            )
        if verify_logits and start == 0:
            verify_final_logits(model, hidden_states[-1][:, -1:], output.logits)
        last, mean = select_hidden_states(hidden_states, encoded["attention_mask"], text_mask)
        last_parts.append(last)
        mean_parts.append(mean)
    return ActivationBatch(
        tuple(example_ids), np.concatenate(last_parts), np.concatenate(mean_parts), metadata or {}
    )
