"""Load a ``ModelSpec`` into a ``(model, tokenizer)`` pair.

Factored out of what used to be copy-pasted between ``scripts/measure_routing.py``
and ``scripts/smoke_test.py`` - same call, same defaults, one place to change
them (e.g. if a future architecture needs ``attn_implementation`` set explicitly).
"""
from __future__ import annotations

import logging

import torch

from mrd.models.base import ModelSpec

log = logging.getLogger(__name__)


def load_causal_lm(spec: ModelSpec, model_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    remote_code = spec.repo_id.startswith("inclusionAI/")
    source = model_path or spec.repo_id
    options = {}
    if spec.repo_id == "openai/gpt-oss-20b":
        options["quantization_config"] = Mxfp4Config(dequantize=True)
    if not remote_code:
        options.update(attn_implementation="eager", experts_implementation="grouped_mm")
    log.info("loading %s (stage=%s, revision=%s, bf16)",
              spec.repo_id, spec.stage, spec.revision)
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=spec.revision, trust_remote_code=remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=spec.revision,
        trust_remote_code=remote_code,
        dtype=torch.bfloat16,
        device_map="cuda",
        **options,
    )
    model.eval()
    log.info("loaded backends: %s", model_backends(model))
    return model, tokenizer


def model_backends(model):
    return {"attention": model.config._attn_implementation,
            "experts": getattr(model.config, "_experts_implementation", None)}
