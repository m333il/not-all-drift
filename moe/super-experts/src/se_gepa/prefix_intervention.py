"""Intervene on a prefix's direct value contribution, keeping attention intact."""
import torch
import torch.nn.functional as F

from se_gepa.attention_contributions import _repeat_kv


class PrefixValueIntervention:
    def __init__(self, model, layer, *, mode="observe", vector=None, scope="last"):
        config = model.peft_config[model.active_adapter]
        if config.peft_type != "PREFIX_TUNING":
            raise ValueError("This intervention supports PREFIX_TUNING only")
        base = model.get_base_model()
        if base.config.model_type != "qwen3_moe" or base.config._attn_implementation != "eager":
            raise ValueError("This intervention requires eager Qwen3-MoE attention")
        if mode not in {"observe", "zero", "restore", "constant"} or scope not in {"last", "all"}:
            raise ValueError("Unknown intervention mode or query scope")
        if layer < 0 or layer >= len(base.model.layers):
            raise ValueError("Invalid intervention layer")
        if mode == "constant":
            if vector is None or tuple(vector.shape) != (base.config.hidden_size,) or not torch.isfinite(vector).all():
                raise ValueError("Constant replacement must be a finite hidden-size vector")
        self.layer = layer
        self.prefix_tokens = config.num_virtual_tokens
        self.attention = base.model.layers[layer].self_attn
        self.mode, self.vector, self.scope = mode, vector, scope
        self.calls = 0
        self.decode_calls = 0
        self.last_contribution = None
        self._handle = None

    def __enter__(self):
        self._handle = self.attention.register_forward_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, *_exc):
        self._handle.remove()

    @torch.no_grad()
    def _hook(self, module, _args, kwargs, output):
        if module.training:
            raise RuntimeError("Prefix intervention requires eval mode")
        weights = output[1]
        cache = kwargs.get("past_key_values")
        if weights is None or weights.shape[0] != 1 or cache is None:
            raise RuntimeError("Expected batch-one eager attention with prefix DynamicCache")
        values = cache.layers[self.layer].values
        if values.shape[-2] != weights.shape[-1] or values.shape[-2] <= self.prefix_tokens:
            raise RuntimeError("Prefix cache and attention key positions do not align")
        values = _repeat_kv(values[:, :, :self.prefix_tokens, :], module.num_key_value_groups)
        query = slice(-1, None) if self.scope == "last" else slice(None)
        prefix_weights = weights[:, :, query, :self.prefix_tokens]
        pre = (prefix_weights @ values).transpose(1, 2).reshape(1, prefix_weights.shape[2], -1)
        contribution = F.linear(pre, module.o_proj.weight, None)
        self.last_contribution = contribution[0].detach()
        if self.calls:
            self.decode_calls += 1
        self.calls += 1
        if self.mode == "observe":
            return None
        replacement = (contribution.float() if self.mode == "restore" else
                       self.vector.to(contribution.device, torch.float32) if self.mode == "constant" else
                       torch.zeros_like(contribution, dtype=torch.float32))
        # Form the correction first so exact-donor restoration is bit-identical.
        correction = replacement - contribution.float()
        changed = output[0].clone()
        changed[:, query, :] = (changed[:, query, :].float() + correction).to(changed.dtype)
        return (changed, *output[1:])


class PrefixKeyMask:
    """Exclude prefix keys before softmax without changing positions or cache."""

    def __init__(self, model, layer, *, scope="all"):
        if scope not in {"all", "early3", "after3"}:
            raise ValueError("Unknown key-mask query scope")
        observer = PrefixValueIntervention(model, layer)
        self.attention = observer.attention
        self.prefix_tokens = observer.prefix_tokens
        self.scope = scope
        self.selected_queries = None
        self.masked_queries = 0
        self.calls = 0

    def __enter__(self):
        self._handle = self.attention.register_forward_pre_hook(self._mask, with_kwargs=True)
        return self

    def __exit__(self, *_exc):
        self._handle.remove()

    def _mask(self, _module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None or mask.ndim != 4 or not mask.is_floating_point():
            raise RuntimeError("Prefix key masking requires the eager prefill additive mask")
        if mask.shape[-1] <= self.prefix_tokens:
            raise RuntimeError("No real keys remain after prefix masking")
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        queries = hidden.shape[1]
        start = mask.shape[-1] - self.prefix_tokens - queries
        if start < 0 or mask.shape[-2] not in {1, queries}:
            raise RuntimeError("Query positions do not align with the dynamic prefix cache")
        positions = torch.arange(start, start + queries, device=mask.device)
        selected = (positions < 3 if self.scope == "early3" else positions >= 3 if self.scope == "after3"
                    else torch.ones_like(positions, dtype=torch.bool))
        self.selected_queries = selected
        self.masked_queries += int(selected.sum())
        self.calls += 1
        if not selected.any():
            return None
        mask = mask.expand(*mask.shape[:-2], queries, mask.shape[-1]).clone()
        mask[..., selected, :self.prefix_tokens] = float("-inf")
        return args, {**kwargs, "attention_mask": mask}


def localization_designs(panel):
    intact = {"name": "intact", "mode": "intact", "layers": [], "scope": "all"}
    if panel == "layers":
        return [intact] + [{"name": f"mask_L{layer}", "mode": "mask_keys", "layers": [layer], "scope": "all"}
                           for layer in range(6)]
    if panel == "positions":
        return [intact] + [{"name": f"mask_{scope}", "mode": "mask_keys", "layers": list(range(6)), "scope": scope}
                           for scope in ("all", "early3", "after3")]
    raise ValueError("Unknown localization panel")
