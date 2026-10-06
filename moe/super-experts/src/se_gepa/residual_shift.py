import torch


class EarlyResidualShift:
    """Capture or patch the first three real states after one decoder layer."""

    def __init__(self, model, layer, mode="observe", vector=None):
        if mode not in {"observe", "add", "replace"}:
            raise ValueError("Unknown residual intervention")
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.layer = base.model.layers[layer]
        self.mode, self.vector = mode, vector
        if mode != "observe" and (vector is None or vector.shape != (3, base.config.hidden_size)):
            raise ValueError("Expected three position-specific residual vectors")
        self.calls = self.seen = self.patched = 0
        self.before, self.after = [], []

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self.hook)
        return self

    def __exit__(self, *_):
        self.handle.remove()

    @torch.no_grad()
    def hook(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != 1:
            raise ValueError("Residual shift requires batch size one")
        start = self.seen
        self.seen += hidden.shape[1]
        self.calls += 1
        count = max(0, min(3 - start, hidden.shape[1]))
        if not count:
            return None
        self.before.append(hidden[0, :count].detach().float().cpu())
        if self.mode == "observe":
            return None
        vector = self.vector[start:start + count].to(hidden.device)
        changed = hidden.clone()
        changed[0, :count] = (hidden[0, :count].float() + vector if self.mode == "add" else vector).to(hidden.dtype)
        if not torch.isfinite(changed).all():
            raise RuntimeError("Non-finite patched residual")
        self.after.append(changed[0, :count].detach().float().cpu())
        self.patched += count
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
