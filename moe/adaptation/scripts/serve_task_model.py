#!/usr/bin/env python3
import argparse
from concurrent.futures import Future
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
import time
import traceback

import torch
import transformers

from mrd.chat import CHAT_DATE, REASONING_EFFORT, decode_completion, encode_user_prompt
from mrd.models.loading import load_causal_lm, model_backends
from mrd.models.registry import MODEL_SPECS, build_adapter


@torch.no_grad()
def generate_batch(model, tokenizer, adapter, requests):
    encoded = [encode_user_prompt(tokenizer, system, user) for system, user, _ in requests]
    lengths = [len(token_ids) for token_ids, _ in encoded]
    width = max(lengths)
    pad = tokenizer.pad_token_id
    ids = torch.full((len(requests), width), pad, dtype=torch.long, device=model.device)
    mask = torch.zeros_like(ids)
    for row, ((token_ids, _), length) in enumerate(zip(encoded, lengths, strict=True)):
        ids[row, width - length:] = torch.tensor(token_ids, device=model.device)
        mask[row, width - length:] = 1
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    model.eval()
    generated = model.generate(
        input_ids=ids,
        attention_mask=mask,
        max_new_tokens=requests[0][2],
        do_sample=False,
        pad_token_id=pad,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        repetition_penalty=1.0,
    )
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    records = []
    for row, ((token_ids, user_positions), length) in enumerate(zip(encoded, lengths, strict=True)):
        generated_ids = generated[row, width:].tolist()
        if tokenizer.eos_token_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(tokenizer.eos_token_id) + 1]
        completion = decode_completion(tokenizer, generated_ids, adapter.model_family)
        records.append({
            **vars(completion),
            "finish_reason": "stop" if completion.finished else "length",
            "input_ids": token_ids,
            "generated_ids": generated_ids,
            "user_positions": user_positions,
            "virtual_offset": None,
            "routing_positions": None,
            "prefill_patch_coverage": {},
            "prompt_tokens": length,
            "completion_tokens": len(generated_ids),
            "max_new_tokens": requests[0][2],
            "temperature": 0.0,
            "elapsed_seconds": elapsed,
        })
    return records


class BatchGenerator:
    def __init__(self, model, tokenizer, adapter, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.batch_size = batch_size
        self.queue = Queue()
        Thread(target=self.run, daemon=True).start()

    def submit(self, system, user, budget):
        future = Future()
        self.queue.put((future, system, user, budget))
        return future

    def run(self):
        while True:
            first = self.queue.get()
            items = [first]
            deadline = time.monotonic() + 0.025
            while len(items) < self.batch_size:
                try:
                    items.append(self.queue.get(timeout=max(0.0, deadline - time.monotonic())))
                except Empty:
                    break
            try:
                records = generate_batch(
                    self.model,
                    self.tokenizer,
                    self.adapter,
                    [(system, user, budget) for _, system, user, budget in items],
                )
                for (future, _, _, _), record in zip(items, records, strict=True):
                    future.set_result(record)
            except Exception as error:
                for future, _, _, _ in items:
                    future.set_exception(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-spec", choices=["qwen3-2507", "gpt-oss-20b"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    batch_size = int(os.environ.get("MRD_TASK_BATCH_SIZE", "1"))
    if batch_size < 1:
        raise ValueError("MRD_TASK_BATCH_SIZE must be positive")
    spec = MODEL_SPECS[args.model_spec]
    model, tokenizer = load_causal_lm(spec, args.model_dir)
    adapter = build_adapter(model)
    manifest = {"model": spec.repo_id, "revision": spec.revision,
                "backends": model_backends(model),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "date_in_template": CHAT_DATE, "reasoning_effort": REASONING_EFFORT,
                "device": torch.cuda.get_device_name(), "port": args.port,
                "batch_size": batch_size,
                "decoding": "native chat, dynamic microbatch, explicit final channel, BF16"}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cache = {}
    log = (args.out_dir / "requests.jsonl").open("w", buffering=1)
    lock = Lock()
    generator = BatchGenerator(model, tokenizer, adapter, batch_size)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, data):
            encoded = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self.respond(200 if self.path == "/health" else 404, manifest if self.path == "/health" else {"error": "Unknown endpoint"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.respond(404, {"error": "Unknown endpoint"})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                messages = request["messages"]
                roles = [message["role"] for message in messages]
                if roles not in [["user"], ["system", "user"]]:
                    raise ValueError("This experiment endpoint accepts one user turn with an optional system prompt")
                if request["model"] != spec.repo_id:
                    raise ValueError("Requested model differs from the loaded pinned model")
                budget = int(request.get("max_tokens", 2048))
                temperature = float(request.get("temperature", 0))
                if not 1 <= budget <= 8192 or temperature != 0:
                    raise ValueError("Invalid generation budget or temperature")
                seed = request.get("seed")
                if seed is not None:
                    raise ValueError("Batched deterministic endpoint does not accept a sampling seed")
            except (ValueError, KeyError, TypeError) as error:
                self.respond(400, {"error": str(error)})
                return
            identity = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            with lock:
                cached = identity in cache
                record = cache.get(identity)
            try:
                if not cached:
                    record = generator.submit(
                        messages[0]["content"] if roles[0] == "system" else "",
                        messages[-1]["content"],
                        budget,
                    ).result(timeout=900)
                    with lock:
                        cache[identity] = record
                with lock:
                    log.write(json.dumps({"time": time.time(), "request_id": identity, "request": request,
                                          "cached": cached, "record": record}, ensure_ascii=False) + "\n")
                self.respond(200, {"id": identity, "model": spec.repo_id,
                                   "choices": [{"index": 0, "message": {"role": "assistant", "content": record["final"],
                                                                         "reasoning": record["analysis"]},
                                                "finish_reason": record["finish_reason"]}],
                                   "usage": {"prompt_tokens": record["prompt_tokens"], "completion_tokens": record["completion_tokens"]},
                                   "revision_metadata": {"cached": cached, "revision": spec.revision}})
            except Exception as error:
                traceback.print_exc()
                log.write(json.dumps({"time": time.time(), "request_id": identity, "request": request,
                                      "error": type(error).__name__ + ": " + str(error)}) + "\n")
                self.respond(500, {"error": type(error).__name__ + ": " + str(error)})

    print(json.dumps({"status": "READY", **manifest}), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
