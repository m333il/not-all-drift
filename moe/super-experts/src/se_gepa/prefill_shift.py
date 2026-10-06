import torch


class PrefillShift:
    """Add one vector to original prefill states after the first three positions."""

    def __init__(self, model, layer, length, vector):
        if not isinstance(length, int) or length < 1:
            raise ValueError("Prefill length must be a positive integer")
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        if vector.shape != (base.config.hidden_size,):
            raise ValueError("Expected one vector with shape [hidden size]")
        self.layer = base.model.layers[layer]
        self.length, self.vector = length, vector
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
            raise ValueError("Prefill shift requires batch size one")
        self.calls += 1
        if self.calls > 1:
            return None
        if hidden.shape[1] < self.length:
            raise ValueError("First call must contain the original prefill")
        self.before = hidden[0, :self.length].detach().float().cpu().clone()
        self.patched_positions = max(self.length - 3, 0)
        if not self.patched_positions or not torch.count_nonzero(self.vector):
            self.after = self.before.clone()
            return None
        vector = self.vector.to(device=hidden.device, dtype=torch.float32)
        changed = hidden.clone()
        changed[0, 3:self.length] = (hidden[0, 3:self.length].float() + vector).to(hidden.dtype)
        self.after = changed[0, :self.length].detach().float().cpu().clone()
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
