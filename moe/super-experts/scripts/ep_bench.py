#!/usr/bin/env python3
"""Measured expert-parallel MoE step time under each arm's own expert load.

Run under ``torchrun --nproc_per_node N``: one rank per GPU, the experts of every
layer split contiguously over the ranks (expert e on rank e // (E / N)), as in the
default expert-parallel placement of Megatron and vLLM. Each rank holds a batch of
tokens; a layer is one real expert-parallel MoE forward:

    all-to-all dispatch -> grouped GEMM of the local experts (SwiGLU) -> all-to-all combine

and its time is the slowest rank's, since the step waits for it. Token routing is
drawn per layer from the arm's measured expert distribution (the routing maps,
every routed position, so the traffic the adapted model actually serves): each
token takes ``k`` distinct experts by Gumbel top-k on the log shares. Expert weights
are random with the real shapes; the time of a GEMM does not depend on the values.

This measures the MoE layers only -- no attention, no scheduler, no framework
overhead -- so it isolates exactly what load imbalance costs under expert
parallelism. It is not an end-to-end serving latency.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist

SHAPES = {  # experts, top-k, hidden, expert intermediate
    "qwen": (128, 8, 2048, 768),
    "gpt-oss": (32, 4, 2880, 2880),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loads", required=True, help="npz of '<model>|<arm>' -> [layers, experts] shares")
    parser.add_argument("--models", default="qwen,gpt-oss")
    parser.add_argument("--tokens", default="512,4096", help="tokens per rank per step, comma-separated")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu", action="store_true", help="logic smoke only: gloo, CPU, wall-clock timing")
    return parser.parse_args()


def grouped(x, weights, offsets):
    try:
        return torch._grouped_mm(x, weights, offs=offsets)
    except (AttributeError, RuntimeError):
        out, start = [], 0
        for group, end in enumerate(offsets.tolist()):
            out.append(x[start:end] @ weights[group])
            start = end
        return torch.cat(out) if out else x.new_zeros(0, weights.shape[-1])


def moe_layer(x, experts, weights, w1, w2, n_ranks, local, identity=False):
    """One expert-parallel MoE forward on this rank's tokens; returns [tokens, hidden]."""
    tokens, k = experts.shape
    flat = experts.reshape(-1)
    dest = flat // local
    order = torch.argsort(dest, stable=True)
    send_counts = torch.bincount(dest, minlength=n_ranks)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    send = x.repeat_interleave(k, dim=0)[order]
    send_expert = (flat % local)[order].to(torch.int32)
    splits_out, splits_in = send_counts.tolist(), recv_counts.tolist()
    recv = x.new_empty(sum(splits_in), x.shape[1])
    recv_expert = send_expert.new_empty(sum(splits_in))
    dist.all_to_all_single(recv, send, splits_in, splits_out)
    dist.all_to_all_single(recv_expert, send_expert, splits_in, splits_out)
    by_expert = torch.argsort(recv_expert, stable=True)
    rows = recv[by_expert]
    offsets = torch.cumsum(torch.bincount(recv_expert, minlength=local), 0).to(torch.int32)
    if not identity:
        hidden = grouped(rows, w1, offsets)
        gate, up = hidden.chunk(2, dim=-1)
        rows = grouped(torch.nn.functional.silu(gate) * up, w2, offsets)
    done = torch.empty_like(rows)
    done[by_expert] = rows
    back = x.new_empty(send.shape[0], x.shape[1])
    dist.all_to_all_single(back, done, splits_out, splits_in)
    combined = torch.empty_like(back)
    combined[order] = back
    return (combined.view(tokens, k, -1) * weights.unsqueeze(-1)).sum(1)


def sample_routing(shares, tokens, k, generator):
    """[layers, tokens, k] distinct experts per token, Gumbel top-k on log shares."""
    logits = torch.log(shares.clamp_min(1e-12)).unsqueeze(1)
    gumbel = -torch.log(-torch.log(torch.rand(shares.shape[0], tokens, shares.shape[1],
                                              generator=generator, device=shares.device).clamp_min(1e-20)))
    return torch.topk(logits + gumbel, k, dim=-1).indices


def main():
    args = parse_args()
    dist.init_process_group("gloo" if args.cpu else "nccl")
    rank, n_ranks = dist.get_rank(), dist.get_world_size()
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
        torch.cuda.set_device(device)
    loads = np.load(args.loads)
    results = {"ranks": n_ranks, "device": "cpu" if args.cpu else torch.cuda.get_device_name(device),
               "torch": torch.__version__,
               "grouped_mm": hasattr(torch, "_grouped_mm"), "cells": {}}
    for model in args.models.split(","):
        n_experts, k, hidden, inter = SHAPES[model]
        if n_experts % n_ranks:
            continue
        local = n_experts // n_ranks
        torch.manual_seed(args.seed + rank)
        w1 = (torch.randn(local, hidden, 2 * inter, device=device, dtype=torch.bfloat16) * 0.02).contiguous()
        w2 = (torch.randn(local, inter, hidden, device=device, dtype=torch.bfloat16) * 0.02).contiguous()
        arms = sorted(key.split("|", 1)[1] for key in loads.files if key.startswith(model + "|"))
        arms = ["uniform"] + arms
        for tokens in [int(t) for t in args.tokens.split(",")]:
            x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
            gate_weights = torch.full((tokens, k), 1.0 / k, device=device, dtype=torch.bfloat16)
            # Dispatch and combine must be exact inverses: with identity experts the
            # layer returns its input. Checked once per shape, before anything is timed.
            probe = sample_routing(torch.full((1, n_experts), 1.0 / n_experts, device=device),
                                   tokens * n_ranks, k, torch.Generator(device=device).manual_seed(1))
            echo = moe_layer(x, probe[0, rank * tokens:(rank + 1) * tokens], gate_weights, w1, w2,
                             n_ranks, local, identity=True)
            if not torch.allclose(echo.float(), x.float(), atol=2e-2, rtol=2e-2):
                raise RuntimeError("expert-parallel dispatch/combine does not round-trip")
            for arm in arms:
                if arm == "uniform":
                    n_layers = loads[f"{model}|base"].shape[0]
                    shares = torch.full((n_layers, n_experts), 1.0 / n_experts, device=device)
                else:
                    shares = torch.tensor(loads[f"{model}|{arm}"], device=device, dtype=torch.float32)
                    shares = shares / shares.sum(1, keepdim=True)
                generator = torch.Generator(device=device).manual_seed(args.seed)
                layer_ms = np.zeros(shares.shape[0])
                imbalance = np.zeros(shares.shape[0])
                for rep in range(args.warmup + args.reps):
                    # Every rank draws the same global batch and keeps its own slice.
                    routing = sample_routing(shares, tokens * n_ranks, k, generator)
                    mine = routing[:, rank * tokens:(rank + 1) * tokens]
                    for layer in range(shares.shape[0]):
                        dist.barrier()
                        if args.cpu:
                            began = time.perf_counter()
                            moe_layer(x, mine[layer], gate_weights, w1, w2, n_ranks, local)
                            elapsed = torch.tensor((time.perf_counter() - began) * 1e3)
                        else:
                            start = torch.cuda.Event(enable_timing=True)
                            stop = torch.cuda.Event(enable_timing=True)
                            start.record()
                            moe_layer(x, mine[layer], gate_weights, w1, w2, n_ranks, local)
                            stop.record()
                            torch.cuda.synchronize()
                            elapsed = torch.tensor(start.elapsed_time(stop), device=device)
                        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                        if rep >= args.warmup:
                            layer_ms[layer] += elapsed.item() / args.reps
                            per_rank = torch.bincount(routing[layer].reshape(-1) // local, minlength=n_ranks)
                            imbalance[layer] += (per_rank.max() / per_rank.float().mean()).item() / args.reps
                cell = {"model": model, "tokens_per_rank": tokens, "arm": arm,
                        "step_ms": float(layer_ms.sum()), "layer_ms": layer_ms.round(4).tolist(),
                        "max_over_mean_rank_load": float(imbalance.mean())}
                results["cells"][f"{model}|{tokens}|{arm}"] = cell
                if rank == 0:
                    print("EP_CELL " + json.dumps({k2: v for k2, v in cell.items() if k2 != "layer_ms"}), flush=True)
        del w1, w2
        if not args.cpu:
            torch.cuda.empty_cache()
    if rank == 0:
        with open(args.out, "w") as stream:
            json.dump(results, stream, indent=2)
        print("EP_RESULT " + json.dumps({key: round(cell["step_ms"], 3) for key, cell in results["cells"].items()}),
              flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
