"""Qwen3 contributions for virtual keys, real keys 0/1/2, and later keys."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


GROUPS = ("virtual", "real_0", "real_1", "real_2", "real_remaining")


def _repeat_kv(states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return states
    batch, heads, length, width = states.shape
    return states[:, :, None, :, :].expand(batch, heads, repeats, length, width).reshape(
        batch, heads * repeats, length, width
    )


def _errors(actual: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - reconstructed.float()).flatten()
    reference = actual.float().flatten()
    return {
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "relative_l2": float(torch.linalg.vector_norm(delta) /
                             torch.linalg.vector_norm(reference).clamp_min(1e-30)),
    }


def _conditional_virtual_stats(virtual_weights: torch.Tensor) -> dict:
    virtual_mass = virtual_weights.sum(dim=-1, keepdim=True)
    conditional = virtual_weights / virtual_mass.clamp_min(1e-30)
    entropy = -(conditional * conditional.clamp_min(1e-30).log()).sum(dim=-1)
    top1 = conditional.amax(dim=-1)
    top5 = conditional.topk(min(5, virtual_weights.shape[-1]), dim=-1).values.sum(dim=-1)
    eligible = virtual_mass.squeeze(-1) > 0

    def means(values):
        return [float(values[head, eligible[head]].mean()) if eligible[head].any() else None
                for head in range(values.shape[0])]

    return {
        "eligible_real_queries": eligible.sum(dim=-1).tolist(),
        "conditional_entropy": means(entropy),
        "conditional_effective_count": means(entropy.exp()),
        "conditional_top1_mass": means(top1),
        "conditional_top5_mass": means(top5),
    }


class AttentionContributionProbe:
    """Observe selected eager-attention layers without changing their outputs.

    The hook uses the attention weights already computed by eager attention and
    captures the value projection/cache plus the native input to ``o_proj``. It
    never requests or retains the model-wide ``output_attentions`` tuple.
    """

    def __init__(self, model, layers: list[int], *, virtual_query_tokens: int = 0,
                 prefix_key_tokens: int = 0, emit=None, audit_vector_dims: int = 0,
                 reconstruction_rtol: float = 1e-5, real_query_start: int = 0) -> None:
        if virtual_query_tokens and prefix_key_tokens:
            raise ValueError("An arm cannot contain prompt queries and prefix-only keys together")
        if virtual_query_tokens < 0 or prefix_key_tokens < 0:
            raise ValueError("Virtual token counts must be non-negative")
        if audit_vector_dims < 0:
            raise ValueError("audit_vector_dims must be non-negative")
        if real_query_start < 0:
            raise ValueError("real_query_start must be non-negative")

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        config = base.config
        if config.model_type not in {"qwen3", "qwen3_moe"}:
            raise TypeError(f"Only Qwen3 is supported, got model_type={config.model_type!r}")
        if getattr(config, "_attn_implementation", None) != "eager":
            raise ValueError("Attention contributions require attn_implementation='eager'")
        self._layers = list(base.model.layers)
        if len(set(layers)) != len(layers) or any(index < 0 or index >= len(self._layers) for index in layers):
            raise ValueError(f"Selected layers must be unique indices in [0, {len(self._layers)})")
        self.layers = list(layers)
        for index in self.layers:
            attention = self._layers[index].self_attn
            if not attention.__class__.__name__.startswith("Qwen3"):
                raise TypeError(f"Layer {index} uses unsupported attention {attention.__class__.__name__}")

        self.virtual_query_tokens = virtual_query_tokens
        self.prefix_key_tokens = prefix_key_tokens
        self.audit_vector_dims = audit_vector_dims
        self.reconstruction_rtol = reconstruction_rtol
        self.real_query_start = real_query_start
        self.records: list[dict] = []
        self._emit = emit if emit is not None else self.records.append
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._captures: dict[int, dict[str, torch.Tensor]] = {}
        self._example: int | None = None
        self._real_tokens: int | None = None
        self._sequence_hash: str | None = None

    def begin_example(self, index: int, real_tokens: int, sequence_hash: str | None = None) -> None:
        if real_tokens <= 0:
            raise ValueError("The prefill must contain at least one real token")
        if real_tokens <= self.real_query_start:
            raise ValueError("Selected real-query range is empty")
        self._example = index
        self._real_tokens = real_tokens
        self._sequence_hash = sequence_hash
        self._captures.clear()

    def __enter__(self) -> "AttentionContributionProbe":
        for index in self.layers:
            attention = self._layers[index].self_attn
            self._handles.append(attention.v_proj.register_forward_hook(self._capture_value(index)))
            self._handles.append(attention.o_proj.register_forward_pre_hook(self._capture_o_input(index)))
            self._handles.append(attention.register_forward_hook(self._observe(index), with_kwargs=True))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captures.clear()

    def _capture_value(self, layer: int):
        def hook(_module, _inputs, output):
            self._captures.setdefault(layer, {})["value_projection"] = output
        return hook

    def _capture_o_input(self, layer: int):
        def hook(_module, inputs):
            self._captures.setdefault(layer, {})["o_input"] = inputs[0]
        return hook

    def _observe(self, layer: int):
        @torch.no_grad()
        def hook(module, _args, kwargs, output):
            if self._example is None or self._real_tokens is None:
                raise RuntimeError("Call begin_example(index, real_tokens) before each forward pass")
            if module.training:
                raise RuntimeError("Attention contribution probing requires eval mode")
            weights = output[1]
            if weights is None:
                raise RuntimeError("Eager Qwen3 attention did not return attention weights")
            if weights.shape[0] != 1:
                raise RuntimeError("Attention contribution attribution requires batch size 1")

            virtual_keys = self.virtual_query_tokens + self.prefix_key_tokens
            expected_queries = self.virtual_query_tokens + self._real_tokens
            expected_keys = virtual_keys + self._real_tokens
            if weights.shape[-2:] != (expected_queries, expected_keys):
                raise RuntimeError(
                    "Prefill-only probe received unsupported query/key geometry: "
                    f"got q={weights.shape[-2]} k={weights.shape[-1]}, expected "
                    f"q={expected_queries} k={expected_keys}. Cached decoding is unsupported."
                )
            if not torch.isfinite(weights).all():
                raise RuntimeError(f"Layer {layer} produced non-finite attention weights")

            captures = self._captures.get(layer, {})
            if self.prefix_key_tokens:
                cache = kwargs.get("past_key_values")
                if cache is None or layer >= len(cache.layers):
                    raise RuntimeError("Prefix tuning did not expose the expected Qwen3 DynamicCache")
                values = cache.layers[layer].values
            else:
                raw = captures.get("value_projection")
                if raw is None:
                    raise RuntimeError("The Qwen3 value projection was not captured")
                values = raw.view(raw.shape[0], raw.shape[1], -1, module.head_dim).transpose(1, 2)
            if values.shape[0] != 1 or values.shape[-2:] != (expected_keys, module.head_dim):
                raise RuntimeError(
                    f"Layer {layer} value geometry {tuple(values.shape)} does not match {expected_keys} keys"
                )
            values = _repeat_kv(values, module.num_key_value_groups)
            if values.shape[1] != weights.shape[1]:
                raise RuntimeError("GQA value heads do not align with attention heads")

            query_start = self.virtual_query_tokens + self.real_query_start
            measured_queries = self._real_tokens - self.real_query_start
            real_weights = weights[0, :, query_start:, :]
            real_native = output[0][0, query_start:, :]
            native_o_input = captures.get("o_input")
            if native_o_input is None:
                raise RuntimeError("The native Qwen3 o_proj input was not captured")
            native_o_input = native_o_input[0, query_start:, :]

            spans = {
                "virtual": (0, virtual_keys),
                "real_0": (virtual_keys, min(virtual_keys + 1, expected_keys)),
                "real_1": (min(virtual_keys + 1, expected_keys), min(virtual_keys + 2, expected_keys)),
                "real_2": (min(virtual_keys + 2, expected_keys), min(virtual_keys + 3, expected_keys)),
                "real_remaining": (min(virtual_keys + 3, expected_keys), expected_keys),
            }
            mass = {}
            projected = {}
            pre_outputs = []
            contribution = {}
            weight_by_head = module.o_proj.weight.view(
                module.o_proj.weight.shape[0], weights.shape[1], module.head_dim
            ).float()
            for name, (start, stop) in spans.items():
                if start == stop:
                    group_mass = torch.zeros((weights.shape[1], measured_queries), device=weights.device)
                    pre = torch.zeros((weights.shape[1], measured_queries, module.head_dim),
                                      device=values.device, dtype=values.dtype)
                else:
                    group_weights = real_weights[..., start:stop]
                    group_mass = group_weights.float().sum(dim=-1)
                    pre = torch.matmul(group_weights, values[0, :, start:stop, :])
                mass[name] = group_mass.mean(dim=-1).tolist()
                flat = pre.transpose(0, 1).reshape(measured_queries, -1)
                group_projected = F.linear(flat, module.o_proj.weight, None)
                projected[name] = group_projected
                pre_outputs.append(flat)
                norms = torch.linalg.vector_norm(group_projected.float(), dim=-1)
                mean_vector = group_projected.float().mean(dim=0)
                mean_pre = pre.float().mean(dim=1)
                head_mean_vectors = torch.einsum("ohd,hd->ho", weight_by_head, mean_pre).float()
                head_mean_norms = torch.linalg.vector_norm(head_mean_vectors, dim=-1)
                head_mean_total = torch.linalg.vector_norm(head_mean_vectors.sum(dim=0))
                head_norm_sum = head_mean_norms.sum()
                summary = {
                    "mean_l2": float(norms.mean()),
                    "rms_l2": float(torch.sqrt(torch.mean(norms.square()))),
                    "mean_vector_l2": float(torch.linalg.vector_norm(mean_vector)),
                    "mean_squared_l2_about_mean": float(
                        (group_projected.float() - mean_vector).square().sum(dim=-1).mean()
                    ),
                    "per_head_mean_vector_l2": head_mean_norms.tolist(),
                    "mean_head_cancellation_fraction": (
                        float(1.0 - head_mean_total / head_norm_sum) if head_norm_sum > 0 else None
                    ),
                }
                if self.audit_vector_dims:
                    summary["mean_vector"] = mean_vector[:self.audit_vector_dims].tolist()
                contribution[name] = summary

            reconstructed_pre = torch.stack(pre_outputs).sum(dim=0)
            reconstructed_projected = torch.stack([projected[name] for name in GROUPS]).sum(dim=0)
            if module.o_proj.bias is not None:
                reconstructed_projected = reconstructed_projected + module.o_proj.bias
            group_norm_sum = torch.stack([
                torch.linalg.vector_norm(projected[name].float(), dim=-1) for name in GROUPS
            ]).sum(dim=0)
            total_norm = torch.linalg.vector_norm(
                torch.stack([projected[name].float() for name in GROUPS]).sum(dim=0), dim=-1
            )
            cancellation_eligible = group_norm_sum > 0
            cancellation = 1.0 - total_norm[cancellation_eligible] / group_norm_sum[cancellation_eligible]

            virtual_distribution = None
            if virtual_keys:
                virtual_weights = real_weights[..., :virtual_keys].float()
                virtual_distribution = _conditional_virtual_stats(virtual_weights)

            row = {
                "example": self._example,
                "sequence_sha256": self._sequence_hash,
                "layer": layer,
                "real_queries": measured_queries,
                "attention_heads": weights.shape[1],
                "key_value_heads": values.shape[1] // module.num_key_value_groups,
                "head_dim": module.head_dim,
                "virtual_query_tokens": self.virtual_query_tokens,
                "virtual_key_tokens": virtual_keys,
                "prefix_only_key_tokens": self.prefix_key_tokens,
                "per_head_attention_mass": mass,
                "within_virtual": virtual_distribution,
                "projected_contribution": contribution,
                "between_key_group_cancellation": {
                    "eligible_real_queries": int(cancellation_eligible.sum()),
                    "mean_fraction": float(cancellation.mean()) if cancellation.numel() else None,
                    "max_fraction": float(cancellation.max()) if cancellation.numel() else None,
                },
                "reconstruction": {
                    "before_o_proj": _errors(native_o_input, reconstructed_pre),
                    "after_o_proj": _errors(real_native, reconstructed_projected),
                },
            }
            if self.real_query_start:
                row["real_query_start"] = self.real_query_start
                row["total_real_tokens"] = self._real_tokens
            for stage, errors in row["reconstruction"].items():
                if not math.isfinite(errors["relative_l2"]) or errors["relative_l2"] > self.reconstruction_rtol:
                    raise RuntimeError(
                        f"Layer {layer} {stage} decomposition relative L2 error "
                        f"{errors['relative_l2']:.6g} exceeds {self.reconstruction_rtol:.6g}"
                    )
            self._emit(row)
            self._captures.pop(layer, None)
        return hook
