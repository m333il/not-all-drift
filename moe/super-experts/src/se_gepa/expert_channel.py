import hashlib

import torch
import torch.nn.functional as F


def _target(model, layer, expert, length):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    block = base.model.layers[layer].mlp
    if not isinstance(length, int) or length < 1:
        raise ValueError("Original prefill length must be positive")
    if not 0 <= expert < block.experts.num_experts:
        raise ValueError("Expert index out of range")
    return block, block.experts


def _weights(module):
    transposed = bool(module.is_transposed)
    down = module.down_proj.transpose(-1, -2) if transposed else module.down_proj
    up = module.gate_up_proj.transpose(-1, -2) if transposed else module.gate_up_proj
    expected = (module.num_experts, module.hidden_dim, module.intermediate_dim)
    if down.shape != expected or up.shape != (expected[0], 2 * expected[2], expected[1]):
        raise ValueError("Unexpected expert projection layout")
    if module.has_bias or not module.is_concatenated or not module.has_gate:
        raise ValueError("Expected bias-free concatenated Qwen gated experts")
    return up, down


def _routing(indices, weights, expert, count):
    hits = indices[:count] == expert
    return {"hit": hits.any(-1).cpu().tolist(),
            "weight": (weights[:count].float() * hits).sum(-1).cpu().tolist()}


class EarlyExpertContribution:
    """Observe a selected expert's routed early-token output, recomputed locally."""

    def __init__(self, model, layer, expert, length):
        self.block, self.module = _target(model, layer, expert, length)
        self.expert, self.length = expert, length
        self.calls = 0
        self.contribution = self.routing = None
        self.audit = {}

    def _shape(self, module, args):
        if args[0].shape[0] != 1:
            raise ValueError("Expert observation requires batch size one")
        if self.calls == 0 and args[0].shape[1] < self.length:
            raise ValueError("First call must contain the original prefill")

    def __enter__(self):
        self.shape_handle = self.block.register_forward_pre_hook(self._shape)
        self.handle = self.module.register_forward_hook(self._capture)
        return self

    def __exit__(self, *_):
        self.handle.remove()
        self.shape_handle.remove()

    @torch.no_grad()
    def _capture(self, module, args, output):
        self.calls += 1
        if self.calls != 1:
            return
        hidden, indices, weights = args
        count = min(3, self.length)
        up, down = _weights(module)
        gate, value = F.linear(hidden[:count], up[self.expert]).chunk(2, -1)
        projected = F.linear(module.act_fn(gate) * value, down[self.expert])
        routing_weight = (weights[:count] * (indices[:count] == self.expert)).sum(-1)
        self.contribution = (projected * routing_weight[:, None]).detach().float().cpu()
        self.routing = _routing(indices, weights, self.expert, count)
        self.audit = {"routing": self.routing, "positions": list(range(count)),
                      "is_transposed": bool(module.is_transposed),
                      "finite": bool(torch.isfinite(self.contribution).all()),
                      "method": "native_dtype_selected_expert_linear_recomputation",
                      "norm": self.contribution.norm(dim=-1).tolist()}


class ExpertChannelIntervention:
    """Replay the native backend with one expert output row zeroed, then restore."""

    def __init__(self, model, layer, expert, channel, mode="remove", *, length, seed=42):
        self.block, self.module = _target(model, layer, expert, length)
        if mode not in {"remove", "control", "rescue", "observe"}:
            raise ValueError("Unknown expert-channel intervention")
        if not 0 <= channel < self.module.hidden_dim:
            raise ValueError("Output channel out of range")
        self.expert, self.channel, self.mode, self.length = expert, channel, mode, length
        self.seed = seed
        generator = torch.Generator(device="cpu").manual_seed(seed)
        direction = torch.randn(self.module.hidden_dim, generator=generator, dtype=torch.float32)
        direction[channel] = 0
        self.direction = direction / direction.norm()
        self.calls = self.patched_positions = 0
        self.weight_restored = True
        self.audit = {}

    def _shape(self, module, args):
        if args[0].shape[0] != 1:
            raise ValueError("Expert intervention requires batch size one")
        if self.calls == 0 and args[0].shape[1] < self.length:
            raise ValueError("First call must contain the original prefill")

    def __enter__(self):
        self.shape_handle = self.block.register_forward_pre_hook(self._shape)
        self.had_forward = "forward" in self.module.__dict__
        self.original = self.module.forward
        self.module.forward = self._forward
        return self

    def __exit__(self, *_):
        if self.had_forward:
            self.module.forward = self.original
        else:
            del self.module.forward
        self.shape_handle.remove()

    @torch.no_grad()
    def _forward(self, hidden_states, top_k_index, top_k_weights):
        self.calls += 1
        native = self.original(hidden_states, top_k_index, top_k_weights)
        if self.calls != 1:
            return native
        _, down = _weights(self.module)
        row = down[self.expert, self.channel]
        saved = row.clone()
        self.weight_restored = False
        try:
            row.zero_()
            masked = self.original(hidden_states, top_k_index, top_k_weights)
        finally:
            row.copy_(saved)
            self.weight_restored = bool(torch.equal(row, saved))
        repeated = self.original(hidden_states, top_k_index, top_k_weights)
        native_repeat_exact = bool(torch.equal(native, repeated))
        if not native_repeat_exact:
            raise RuntimeError("Native expert replay is not exact after weight restoration")
        other = torch.arange(native.shape[-1], device=native.device) != self.channel
        unrouted = ~(top_k_index == self.expert).any(-1)
        locality = {"other_channels_exact": bool(torch.equal(native[:, other], masked[:, other])),
                    "unrouted_rows_exact": bool(torch.equal(native[unrouted], masked[unrouted]))}
        if not all(locality.values()):
            raise RuntimeError("Masked native replay changed unrelated expert outputs")
        count = min(3, self.length)
        before = native[:count].float()
        scalar = before[:, self.channel] - masked[:count, self.channel].float()
        intended = torch.zeros_like(before)
        changed = native
        if self.mode in {"remove", "control"}:
            changed = native.clone()
            self.patched_positions = count
            if self.mode == "remove":
                intended[:, self.channel] = -scalar
                changed[:count, self.channel] = masked[:count, self.channel]
            else:
                intended = -scalar[:, None] * self.direction.to(native.device)[None, :]
                changed[:count] = (before + intended).to(native.dtype)
        after = changed[:count].float()
        realized = after - before
        intended_norm, realized_norm = intended.norm(dim=-1), realized.norm(dim=-1)
        denominator = intended_norm * realized_norm
        cosine = torch.where(denominator > 0, (intended * realized).sum(-1) / denominator.clamp_min(1e-30), 0)
        self.audit = {"mode": self.mode, "positions": list(range(count)),
                      "is_transposed": bool(self.module.is_transposed),
                      "native_repeat_exact": native_repeat_exact,
                      "control_seed": self.seed,
                      "control_direction_sha256": hashlib.sha256(self.direction.numpy().tobytes()).hexdigest(),
                      "control_direction_norm": float(self.direction.norm()),
                      "control_direction_selected_channel": float(self.direction[self.channel]),
                      "weight_restored": self.weight_restored, **locality,
                      "routing": _routing(top_k_index, top_k_weights, self.expert, count),
                      "signed_contribution": scalar.cpu().tolist(),
                      "before_norm": before.norm(dim=-1).cpu().tolist(),
                      "after_norm": after.norm(dim=-1).cpu().tolist(),
                      "before_max_abs": before.abs().amax(-1).cpu().tolist(),
                      "after_max_abs": after.abs().amax(-1).cpu().tolist(),
                      "before_other_max_abs": before[:, other].abs().amax(-1).cpu().tolist(),
                      "after_other_max_abs": after[:, other].abs().amax(-1).cpu().tolist(),
                      "before_channel": before[:, self.channel].cpu().tolist(),
                      "after_channel": after[:, self.channel].cpu().tolist(),
                      "intended_norm": intended_norm.cpu().tolist(),
                      "realized_norm": realized_norm.cpu().tolist(),
                      "correction_cosine": cosine.cpu().tolist(),
                      "correction_error_norm": (realized - intended).norm(dim=-1).cpu().tolist(),
                      "later_positions_exact": bool(torch.equal(native[count:], changed[count:])),
                      "finite": bool(torch.isfinite(changed).all())}
        return changed
