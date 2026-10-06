#!/usr/bin/env python3
"""Retrain only the router of the frozen MoE on top of one arm, then score it.

The question comes from the paper's MoE section. After adaptation the router still
routes with its pretrained weights. If the adapted states need a different routing,
retraining the gate alone on the arm's own training rows should raise the arm's
score; if adaptation already does what a retrained router would do, it should not.
On the base arm the same procedure measures what the router alone recovers.

Only ``mlp.gate.weight`` is trainable (48 x [128, 2048] on Qwen3-30B-A3B). The
adapter, the experts and every other weight stay frozen. The gate lives in bf16
inside the model; the optimizer keeps an fp32 master copy and writes it back after
every step, so epoch 0 is bit-identical to the frozen router and small updates are
not rounded away.

Rows, rendering and scoring are those of ``score_arms.py``: the v25 user-only
contract, the arm's own instruction (seed or GEPA), greedy free generation and the
empty-aware sample F1. The training target is the archived ``target_text`` of the
arm's training rows followed by the native end-of-turn token.

Protocol: epoch 0 is the frozen router. After every epoch the arm is scored on the
200 validation rows its checkpoint was selected on. The test rows are then scored
for epoch 0 and for the best trained epoch. Both selections are reported: "must
train" (best epoch >= 1) and "may decline" (best epoch >= 0), because a sweep that
cannot choose epoch 0 is forced to report the least bad trained epoch.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from se_gepa.arms import (  # noqa: E402
    SEED_KEY, build_base, check_contract, check_instructions, device_of, load_contract, render,
    resolve_instructions, wrap,
)
from se_gepa.router_training import (  # noqa: E402
    lr_factor, paired, select_epochs, token_weighted_windows,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--arm", required=True, help="the one arm this job retrains the router for")
    parser.add_argument("--train", type=Path, required=True, help="training rows (text, labels, key)")
    parser.add_argument("--train-targets", type=Path, required=True,
                        help="the arms' archived targets, keyed like --train")
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--contract-sample", type=Path, required=True)
    parser.add_argument("--contract-rows", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, help="preflight: train on the first N rows only")
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-train-minutes", type=float, default=240.0,
                        help="stop at an epoch boundary once the next epoch would pass this")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gate_modules(model):
    gates = [(name, module) for name, module in model.named_modules()
             if name.endswith("mlp.gate") and isinstance(getattr(module, "weight", None), torch.nn.Parameter)]
    if not gates:
        raise RuntimeError("No mlp.gate router found; this script knows the Qwen3-MoE layout only")
    return gates


@contextmanager
def checkpoint_attention(model):
    """Recompute eager attention in backward, after the cache update has happened.

    Same device as the PEFT campaign's ``checkpoint_native_attention``: it is safe
    with a prefix in ``past_key_values``, which checkpointing whole decoder layers
    is not, because a recomputed layer would append the prefix to the cache twice.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    module = importlib.import_module(type(base).__module__)
    original = module.eager_attention_forward

    def forward(attention, query, key, value, mask, **kwargs):
        if not torch.is_grad_enabled():
            return original(attention, query, key, value, mask, **kwargs)
        return checkpoint(original, attention, query, key, value, mask, use_reentrant=False, **kwargs)

    module.eager_attention_forward = forward
    try:
        yield
    finally:
        module.eager_attention_forward = original


def encode_training(tokenizer, contract, instruction, rows, targets, is_seed):
    by_key = {row["key"]: row for row in targets}
    eos = tokenizer.eos_token_id
    items = []
    for row in rows:
        target = by_key[row["key"]]
        prompt = render(tokenizer, instruction, row["text"], contract)
        if is_seed and target["input_text"] not in tokenizer.decode(prompt):
            raise RuntimeError(f"{row['key']}: the seed rendering does not contain the archived user turn")
        answer = tokenizer(target["target_text"], add_special_tokens=False)["input_ids"] + [eos]
        items.append((prompt, answer))
    return items


def comment_span(tokenizer, instruction, text, contract, ids):
    """Token positions of the classified comment inside one rendered prompt."""
    labels, _seeds, render_messages, apply_chat_template = contract
    rendered = apply_chat_template(tokenizer, render_messages(instruction, labels, text), non_thinking=True)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != list(ids):
        return None
    start = rendered.rfind(text.strip())
    if start < 0 or not text.strip():
        return None
    end = start + len(text.strip())
    positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start]
    return positions[0], positions[-1] + 1


def comment_routing(model, gates, sequences, spans):
    """Top-k selections of every layer at the comment positions, prefill only.

    Prompt tuning prepends its virtual tokens to what the router sees, so the
    selections are taken from the end of the routed sequence.
    """
    captured = {}
    hooks = [gate.register_forward_hook(lambda _m, _i, output, layer=layer: captured.__setitem__(layer, output[2]))
             for layer, (_name, gate) in enumerate(gates)]
    selections = []
    model.eval()
    try:
        with torch.no_grad():
            for ids, (lo, hi) in zip(sequences, spans):
                tensor = torch.tensor([ids], device=device_of(model))
                model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False)
                chosen = torch.stack([captured[layer][-len(ids):][lo:hi] for layer in range(len(gates))])
                selections.append(chosen.sort(dim=-1).values.to(torch.int16).cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return selections


def routing_change(before, after, n_experts):
    """Per-layer TV between the two comment-token load profiles, and the share of
    (token, layer) pairs whose selected expert set changed."""
    a = torch.cat(before, dim=1).long()
    b = torch.cat(after, dim=1).long()
    tv = []
    for layer in range(a.shape[0]):
        ca = torch.bincount(a[layer].flatten(), minlength=n_experts).double()
        cb = torch.bincount(b[layer].flatten(), minlength=n_experts).double()
        tv.append(0.5 * (ca / ca.sum() - cb / cb.sum()).abs().sum().item())
    changed = (a != b).any(dim=-1).double().mean().item()
    return {"tv_per_layer": tv, "tv_mean": sum(tv) / len(tv), "set_changed": changed}


def evaluate(model, tokenizer, sequences, rows, contract, max_new_tokens, stream, tag):
    from score_arms import score_one

    model.eval()
    # The paper's tables read an unreadable answer as the empty label set; the strict
    # score counts it as wrong. Both are kept, and the paper's convention selects.
    scores, strict, valid, truncated = [], [], [], []
    started = time.time()
    with torch.no_grad():
        for index, ids in enumerate(sequences):
            tensor = torch.tensor([ids], device=device_of(model))
            output = model.generate(
                input_ids=tensor, attention_mask=torch.ones_like(tensor), do_sample=False,
                max_new_tokens=max_new_tokens, use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            row = score_one(tokenizer, output, len(ids), rows[index]["labels"], contract)
            row.update(tag=tag, key=rows[index]["id"])
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            strict.append(row["score"])
            scores.append(row["score"] if row["valid"] else float(not rows[index]["labels"]))
            valid.append(row["valid"])
            truncated.append(row["completion_tokens"] >= max_new_tokens)
    summary = {"tag": tag, "n": len(scores), "score": sum(scores) / len(scores),
               "strict": sum(strict) / len(strict),
               "valid": sum(valid) / len(valid), "truncated": sum(truncated) / len(truncated),
               "seconds": round(time.time() - started, 1)}
    print(json.dumps(summary), flush=True)
    return summary, scores


def main():
    from transformers import AutoTokenizer

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    contract = load_contract(None)
    _labels, seeds, _render, _apply = contract
    arms = [arm for arm in json.loads(args.arms.read_text()) if arm["name"] == args.arm]
    if len(arms) != 1:
        raise SystemExit(f"--arm {args.arm} is not in {args.arms}")
    arm = arms[0]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    resolve_instructions([arm], tokenizer, seeds)
    check_instructions(tokenizer, [arm], contract)
    instruction = arm.get("instruction", seeds[SEED_KEY])
    checked = check_contract(tokenizer, contract, read_rows(args.contract_rows),
                             json.loads(args.contract_sample.read_text()))
    print(f"CONTRACT_TOKENS_VERIFIED={checked}", flush=True)

    model = build_base(args.model)
    if arm["kind"] == "peft":
        model = wrap(model, [arm])
        model.set_adapter(arm["name"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gates = gate_modules(model)
    for _name, gate in gates:
        gate.weight.requires_grad_(True)
    master = [gate.weight.detach().float().clone().requires_grad_(True) for _name, gate in gates]
    frozen = [tensor.detach().clone() for tensor in master]
    optimizer = torch.optim.AdamW(master, lr=args.lr, weight_decay=0.0)
    print(f"ROUTER_TRAINABLE layers={len(gates)} parameters={sum(t.numel() for t in master)}", flush=True)

    train_rows = read_rows(args.train)[: args.train_limit]
    items = encode_training(tokenizer, contract, instruction, train_rows, read_rows(args.train_targets),
                            is_seed=instruction == seeds[SEED_KEY])
    val_rows = read_rows(args.val)[: args.val_limit]
    test_rows = read_rows(args.test)[: args.test_limit]
    val_seq = [render(tokenizer, instruction, row["text"], contract) for row in val_rows]
    test_seq = [render(tokenizer, instruction, row["text"], contract) for row in test_rows]
    steps_per_epoch = math.ceil(len(items) / args.accumulation)
    total = steps_per_epoch * args.epochs
    warmup = math.ceil(total * args.warmup_ratio)
    print(f"ROUTER_PLAN arm={arm['name']} train={len(items)} steps/epoch={steps_per_epoch} "
          f"epochs={args.epochs} lr={args.lr} warmup={warmup} "
          f"max_prompt={max(len(p) for p, _ in items)}", flush=True)

    def load_gates(tensors):
        with torch.no_grad():
            for (_name, gate), tensor in zip(gates, tensors):
                gate.weight.copy_(tensor.to(gate.weight.device, gate.weight.dtype))

    rows_stream = (args.out / "rows.jsonl").open("w")
    steps_stream = (args.out / "steps.jsonl").open("w")
    snapshots = {0: frozen}
    val = {}
    val_scores = {}
    summary = {"arm": arm["name"], "kind": arm["kind"], "lr": args.lr, "accumulation": args.accumulation,
               "epochs_planned": args.epochs, "train_rows": len(items), "val_rows": len(val_rows),
               "test_rows": len(test_rows), "max_new_tokens": args.max_new_tokens,
               "trainable_parameters": sum(t.numel() for t in master)}

    def record():
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    spans = [comment_span(tokenizer, instruction, row["text"], contract, ids)
             for row, ids in zip(val_rows, val_seq)]
    routed = [(ids, span) for ids, span in zip(val_seq, spans) if span is not None]
    route_seq, val_spans = [ids for ids, _ in routed], [span for _, span in routed]
    summary["routing_rows"] = len(routed)
    frozen_routing = comment_routing(model, gates, route_seq, val_spans)
    n_experts = gates[0][1].weight.shape[0]
    val[0], val_scores[0] = evaluate(model, tokenizer, val_seq, val_rows, contract,
                                     args.max_new_tokens, rows_stream, "val-epoch0")
    summary["val"] = {"0": val[0]}
    record()

    rng = random.Random(args.seed)
    step = 0
    started = time.time()
    epoch_seconds = []
    with checkpoint_attention(model):
        for epoch in range(1, args.epochs + 1):
            elapsed = (time.time() - started) / 60
            if epoch_seconds and elapsed + epoch_seconds[-1] / 60 > args.max_train_minutes:
                print(f"ROUTER_STOP budget: {elapsed:.1f} min spent, next epoch would pass "
                      f"{args.max_train_minutes}", flush=True)
                break
            epoch_started = time.time()
            model.train()
            order = list(range(len(items)))
            rng.shuffle(order)
            for window in token_weighted_windows(order, [len(items[i][1]) for i in range(len(items))],
                                                 args.accumulation):
                loss_value = 0.0
                for tensor in master:
                    tensor.grad = torch.zeros_like(tensor)
                for index, weight in window:
                    prompt, answer = items[index]
                    ids = torch.tensor([prompt + answer], device=device_of(model))
                    labels = torch.tensor([[-100] * len(prompt) + answer], device=ids.device)
                    loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                                 use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    (loss * weight).backward()
                    loss_value += loss.item() * weight
                    # Accumulate in fp32: the bf16 buffer would round each addition.
                    for tensor, (_name, gate) in zip(master, gates):
                        if gate.weight.grad is None or not torch.isfinite(gate.weight.grad).all():
                            raise FloatingPointError("Missing or nonfinite router gradient")
                        tensor.grad.add_(gate.weight.grad.float())
                        gate.weight.grad = None
                norm = torch.nn.utils.clip_grad_norm_(master, 1.0).item()
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * lr_factor(step, warmup, total)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                load_gates(master)
                step += 1
                event = {"step": step, "epoch": epoch, "loss": loss_value, "gradient_norm": norm,
                         "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - started, 1)}
                steps_stream.write(json.dumps(event) + "\n")
                if step % 25 == 0 or step == 1:
                    steps_stream.flush()
                    print(json.dumps(event), flush=True)
            epoch_seconds.append(time.time() - epoch_started)
            snapshots[epoch] = [tensor.detach().clone() for tensor in master]
            moved = sum((a - b).norm().item() ** 2 for a, b in zip(master, frozen)) ** 0.5
            print(f"ROUTER_EPOCH {epoch} seconds={epoch_seconds[-1]:.0f} weight_change_norm={moved:.4f}",
                  flush=True)
            val[epoch], val_scores[epoch] = evaluate(model, tokenizer, val_seq, val_rows, contract,
                                                     args.max_new_tokens, rows_stream, f"val-epoch{epoch}")
            summary["val"][str(epoch)] = {**val[epoch], "train_seconds": round(epoch_seconds[-1], 1),
                                          "weight_change_norm": moved}
            record()

    must, may = select_epochs({epoch: result["score"] for epoch, result in val.items()})
    summary["selected"] = {"must_train": must, "may_decline": may}
    record()
    if must is not None:
        # How much the retrained router itself moves the comment tokens, in the
        # units of the paper's displacement figure.
        load_gates(snapshots[must])
        summary["routing_change_val_comment"] = routing_change(
            frozen_routing, comment_routing(model, gates, route_seq, val_spans), n_experts)
        print("ROUTER_ROUTING_CHANGE " + json.dumps(
            {k: v for k, v in summary["routing_change_val_comment"].items() if k != "tv_per_layer"}), flush=True)
        record()
    test = {}
    for epoch in sorted({0, must} - {None}):
        load_gates(snapshots[epoch])
        test[epoch] = evaluate(model, tokenizer, test_seq, test_rows, contract,
                               args.max_new_tokens, rows_stream, f"test-epoch{epoch}")
        summary.setdefault("test", {})[str(epoch)] = test[epoch][0]
        record()
    if must is not None:
        summary["test_must_train_minus_frozen"] = paired(test[must][1], test[0][1])
        summary["val_must_train_minus_frozen"] = paired(val_scores[must], val_scores[0])
        torch.save({"layers": [name for name, _gate in gates],
                    "weights": [tensor.to(torch.bfloat16).cpu() for tensor in snapshots[must]]},
                   args.out / f"gate_epoch{must}.pt")
    record()
    rows_stream.close()
    steps_stream.close()
    print("ROUTER_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
