import torch


class PrefillStatePatch:
    """Capture or replace original prefill states after one decoder layer."""

    def __init__(self, model, layer, length, scope="all", donor=None, alpha=1.0):
        if not isinstance(length, int) or length < 1:
            raise ValueError("Prefill length must be a positive integer")
        if scope not in {"all", "after3", "last"}:
            raise ValueError("Unknown prefill scope")
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        if donor is not None and donor.shape != (length, base.config.hidden_size):
            raise ValueError("Expected donor shape [prefill length, hidden size]")
        self.layer = base.model.layers[layer]
        self.length, self.scope = length, scope
        self.donor, self.alpha = donor, alpha
        self.calls = self.patched_positions = 0
        self.before = self.after = None

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self.hook)
        return self

    def __exit__(self, *_):
        self.handle.remove()

    @torch.no_grad()
    def hook(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != 1:
            raise ValueError("Prefill patch requires batch size one")
        self.calls += 1
        if self.calls > 1:
            return None
        if hidden.shape[1] < self.length:
            raise ValueError("First call must contain the original prefill")
        self.before = hidden[0, :self.length].detach().float().cpu().clone()
        if self.donor is None:
            return None
        start = {"all": 0, "after3": min(3, self.length), "last": self.length - 1}[self.scope]
        self.patched_positions = self.length - start
        if self.alpha == 0 or not self.patched_positions:
            self.after = self.before.clone()
            return None
        donor = self.donor[start:self.length].to(device=hidden.device, dtype=torch.float32)
        changed = hidden.clone()
        if self.alpha == 1:
            replacement = donor
        else:
            replacement = ((1 - self.alpha) * hidden[0, start:self.length].float()
                           + self.alpha * donor)
        changed[0, start:self.length] = replacement.to(hidden.dtype)
        self.after = changed[0, :self.length].detach().float().cpu().clone()
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
