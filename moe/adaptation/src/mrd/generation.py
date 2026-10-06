from contextlib import nullcontext
import time

import torch

from mrd.chat import decode_completion, encode_user_prompt
from mrd.route_interventions import RouteHooks


@torch.no_grad()
def generate_record(model, tokenizer, adapter, system, user, max_new_tokens, *, patches=None, capture=False, temperature=0.0):
    token_ids, user_positions = encode_user_prompt(tokenizer, system, user)
    ids = torch.tensor([token_ids], device=model.device)
    hooks = RouteHooks(adapter, patches) if capture or patches else nullcontext()
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    sampling = {"do_sample": temperature > 0}
    if temperature > 0:
        sampling["temperature"] = temperature
        sampling["top_p"] = 1.0
        sampling["top_k"] = 0
    model.eval()
    with hooks as trace:
        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   max_new_tokens=max_new_tokens, **sampling,
                                   pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id, use_cache=True,
                                   num_beams=1, repetition_penalty=1.0)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    generated_ids = generated[0, len(token_ids):].tolist()
    completion = decode_completion(tokenizer, generated_ids, adapter.model_family)
    offset = None
    if trace is not None:
        offsets = {len(state.indices) - len(token_ids) for state in trace.captured.values()}
        if len(offsets) != 1 or min(offsets) < 0:
            raise ValueError(f"Unexpected prefill geometry: {offsets}")
        offset = offsets.pop()
    record = {**vars(completion), "finish_reason": "stop" if completion.finished else "length",
              "input_ids": token_ids, "generated_ids": generated_ids,
              "user_positions": user_positions, "virtual_offset": offset,
              "routing_positions": [p + offset for p in user_positions] if offset is not None else None,
              "prefill_patch_coverage": trace.coverage if trace else {},
              "prompt_tokens": len(token_ids), "completion_tokens": len(generated_ids),
              "max_new_tokens": max_new_tokens, "temperature": temperature, "elapsed_seconds": elapsed}
    return record, trace.captured if trace else None
