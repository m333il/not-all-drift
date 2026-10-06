from mrd.models.topk import TopKRouterAdapter


class GptOssAdapter(TopKRouterAdapter):
    model_family = "gpt_oss"
    gate_class = "GptOssTopKRouter"
