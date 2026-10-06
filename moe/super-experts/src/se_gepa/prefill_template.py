import torch


def materialize_table(table, length, prefix_length=44, suffix_length=34):
    if not isinstance(length, int) or not isinstance(prefix_length, int) or not isinstance(suffix_length, int):
        raise ValueError("Template lengths must be integers")
    if prefix_length < 3 or suffix_length < 0 or length - prefix_length - suffix_length < 1:
        raise ValueError("Template requires prefix >=3, suffix >=0 and at least one middle position")
    prefix_slots = prefix_length - 3
    if table.ndim != 2 or table.shape[0] != prefix_slots + 1 + suffix_length or table.shape[1] < 1:
        raise ValueError("Expected table shape [prefix_length - 3 + 1 + suffix_length, hidden size]")
    middle = table[prefix_slots:prefix_slots + 1].expand(length - prefix_length - suffix_length, -1)
    return torch.cat((table[:prefix_slots], middle, table[prefix_slots + 1:]), dim=0)


class PrefillTemplateShift:
    """Add fixed prefix, middle and end-aligned suffix vectors during prefill."""

    def __init__(self, model, layer, length, table, prefix_length=44, suffix_length=34):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.shifts = materialize_table(table, length, prefix_length, suffix_length)
        if self.shifts.shape[1] != base.config.hidden_size:
            raise ValueError("Template hidden size does not match model")
        self.layer = base.model.layers[layer]
        self.length = length
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
            raise ValueError("Template shift requires batch size one")
        self.calls += 1
        if self.calls > 1:
            return None
        if hidden.shape[1] < self.length:
            raise ValueError("First call must contain the original prefill")
        self.before = hidden[0, :self.length].detach().float().cpu().clone()
        self.patched_positions = self.length - 3
        if not torch.count_nonzero(self.shifts):
            self.after = self.before.clone()
            return None
        shifts = self.shifts.to(device=hidden.device, dtype=torch.float32)
        changed = hidden.clone()
        changed[0, 3:self.length] = (hidden[0, 3:self.length].float() + shifts).to(hidden.dtype)
        self.after = changed[0, :self.length].detach().float().cpu().clone()
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
