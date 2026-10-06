import torch
import torch.nn as nn


class TopKRouterAdapter:
    gate_class: str
    model_family: str

    def __init__(self, model):
        self._model = model
        self.gates = sorted(
            (int(name.split("layers.")[1].split(".")[0]), module)
            for name, module in model.named_modules()
            if type(module).__name__ == self.gate_class
        )
        if not self.gates:
            raise RuntimeError(
                f"No {self.gate_class} modules found; this code requires transformers==5.16.1"
            )
        gate = self.gates[0][1]
        self.num_experts = gate.num_experts
        self.top_k = gate.top_k
        self.n_group = self.topk_group = self.experts_per_group = None
        self._expert_norms = None

    @property
    def layer_ids(self):
        return [layer for layer, _ in self.gates]

    def register_hooks(self, collected):
        def make_hook(layer):
            def hook(_module, _inputs, output):
                logits, _weights, indices = output
                collected[layer] = (logits.detach().float().cpu(), indices.detach().long().cpu())
            return hook
        return [gate.register_forward_hook(make_hook(layer)) for layer, gate in self.gates]

    def selection_scores(self, logits):
        return torch.softmax(logits, dim=-1)

    def selected_groups(self, logits):
        return None

    def expert_output_norms(self):
        if self._expert_norms is None:
            parents = {id(child): parent for parent in self._model.modules() for child in parent.children()}
            rows = []
            for _, gate in self.gates:
                experts = parents[id(gate)].experts
                down = getattr(experts, "down_proj", None)
                if isinstance(down, nn.Parameter):
                    rows.append(down.detach().float().flatten(1).norm(dim=1))
                else:
                    rows.append(torch.stack([expert.down_proj.weight.detach().float().norm() for expert in experts]))
            self._expert_norms = torch.stack(rows).cpu()
        return self._expert_norms
