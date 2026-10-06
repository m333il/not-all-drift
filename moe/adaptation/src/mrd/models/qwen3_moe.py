from mrd.models.topk import TopKRouterAdapter


class Qwen3MoeAdapter(TopKRouterAdapter):
    model_family = "qwen3_moe"
    gate_class = "Qwen3MoeTopKRouter"
