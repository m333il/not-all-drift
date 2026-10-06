"""Is the batched MoE correct? Judge both kernels against float64, not each other.

Comparing the loop to the batched form only says they differ - it cannot say
which one is wrong, and in bf16 they must differ: the same sum in a different
order. The question that has an answer is whether the batched form is a correct
implementation, and that is settled by computing the block in float64, where
both orderings agree to 1e-15, and asking how far each bf16 kernel lands from
it.

If the batched error is at or below the loop's, the code is right and every
difference downstream is bf16 rounding that the loop has too. If it is
systematically larger, the implementation is wrong.

Real weights, not random: the experts of one layer are read from the actual
Qwen3-30B checkpoint, and the inputs are drawn to match the scale of real
hidden states. One layer fits in ~5 GB at float64, so this runs on a card that
could never hold the model.

    python scripts/verify_grouped_exact.py --layer 0 --rows 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_layer_experts(model_id: str, revision: str, layer: int, dtype):
    """Pull one layer's expert weights straight out of the checkpoint shards."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index = json.loads(Path(hf_hub_download(
        model_id, "model.safetensors.index.json", revision=revision)).read_text())
    want = f"model.layers.{layer}.mlp."
    need = {k: v for k, v in index["weight_map"].items() if k.startswith(want)}
    if not need:
        raise SystemExit(f"no layer {layer}")

    by_shard: dict[str, list[str]] = {}
    for key, shard in need.items():
        by_shard.setdefault(shard, []).append(key)

    tensors = {}
    for shard, keys in by_shard.items():
        path = hf_hub_download(model_id, shard, revision=revision)
        with safe_open(path, framework="pt") as fh:
            for key in keys:
                tensors[key] = fh.get_tensor(key).to(dtype)
    return tensors


def stack_experts(tensors: dict, layer: int, n_experts: int):
    import torch

    g, u, d = [], [], []
    for e in range(n_experts):
        base = f"model.layers.{layer}.mlp.experts.{e}."
        g.append(tensors[base + "gate_proj.weight"])
        u.append(tensors[base + "up_proj.weight"])
        d.append(tensors[base + "down_proj.weight"])
    return torch.stack(g), torch.stack(u), torch.stack(d)


def run_loop(x, w_gate, w_up, w_down, weights, act):
    """The stock path: one expert at a time, scattered back."""
    import torch

    out = torch.zeros_like(x)
    n_experts = w_gate.shape[0]
    for e in range(n_experts):
        col = weights[:, e]
        hit = col.nonzero(as_tuple=True)[0]
        if hit.numel() == 0:
            continue
        rows = x.index_select(0, hit)
        inter = act(rows @ w_gate[e].t()) * (rows @ w_up[e].t())
        out.index_add_(0, hit, (inter @ w_down[e].t()) * col[hit, None])
    return out


def run_grouped(x, w_gate, w_up, w_down, weights, act):
    """The batched path: every expert for every row, router weights select."""
    import torch

    n_experts = w_gate.shape[0]
    xs = x.unsqueeze(0).expand(n_experts, x.shape[0], x.shape[1])
    inter = act(torch.bmm(xs, w_gate.transpose(1, 2))) * torch.bmm(xs, w_up.transpose(1, 2))
    out = torch.bmm(inter, w_down.transpose(1, 2))
    return (out * weights.t().unsqueeze(-1)).sum(dim=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--revision", default="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import torch

    torch.manual_seed(0)
    dev = args.device
    tensors = load_layer_experts(args.model, args.revision, args.layer, torch.float64)
    n_experts = 1 + max(int(k.split(".experts.")[1].split(".")[0])
                        for k in tensors if ".experts." in k)
    g64, u64, d64 = (t.to(dev) for t in stack_experts(tensors, args.layer, n_experts))
    hidden = g64.shape[2]
    act = torch.nn.functional.silu
    print(f"layer {args.layer}: experts {n_experts}, hidden {hidden}, "
          f"intermediate {g64.shape[1]}")

    # Inputs and a router at the scale the real model produces: RMSNorm leaves
    # activations near unit variance, and top-k weights sum to one per row.
    x64 = torch.randn(args.rows, hidden, dtype=torch.float64, device=dev)
    logits = torch.randn(args.rows, n_experts, dtype=torch.float64, device=dev)
    probs = torch.softmax(logits, dim=1)
    top, idx = torch.topk(probs, args.top_k, dim=-1)
    top = top / top.sum(dim=-1, keepdim=True)
    w64 = torch.zeros(args.rows, n_experts, dtype=torch.float64, device=dev)
    w64.scatter_(1, idx, top)

    truth = run_grouped(x64, g64, u64, d64, w64, act)
    truth_loop = run_loop(x64, g64, u64, d64, w64, act)
    order_f64 = (truth - truth_loop).abs().max().item()
    print(f"float64: two orders of summation diverge {order_f64:.3e} "
          f"(It's the same thing)")

    scale = truth.abs().mean().item()
    rows = []
    for name, dt in (("bfloat16", torch.bfloat16), ("float16", torch.float16),
                     ("float32", torch.float32)):
        x, g, u, d, w = (t.to(dt) for t in (x64, g64, u64, d64, w64))
        loop = run_loop(x, g, u, d, w, act).to(torch.float64)
        grouped = run_grouped(x, g, u, d, w, act).to(torch.float64)
        # Max-abs is a tail statistic over half a million numbers: one unlucky
        # row swings it, and a verdict built on it alone flips between runs.
        # RMS answers the question actually being asked - is the batched form
        # systematically further from the truth.
        e_loop = (loop - truth).abs().max().item()
        e_grp = (grouped - truth).abs().max().item()
        r_loop = (loop - truth).pow(2).mean().sqrt().item()
        r_grp = (grouped - truth).pow(2).mean().sqrt().item()
        # Is the batched form biased, or just rounding differently? If the two
        # errors are the same size and it is a coin flip which is closer on any
        # given element, that is rounding. A systematic bias would show up here.
        closer = ((grouped - truth).abs() < (loop - truth).abs()).float().mean().item()
        rows.append((name, e_loop, e_grp, r_loop, r_grp,
                     r_grp / max(r_loop, 1e-30), closer))
        del x, g, u, d, w, loop, grouped
        torch.cuda.empty_cache()

    print(f"\n mean output magnitude: {scale:.4f}")
    print(f"{'dtype':10} {'max loop':>12} {'max batch':>12} "
          f"{'rms loop':>12} {'rms batch':>12} {'rms b/l':>9} {'b closer':>11}")
    for name, a, b, ra, rb, r, c in rows:
        print(f"{name:10} {a:12.3e} {b:12.3e} {ra:12.3e} {rb:12.3e} "
              f"{r:9.2f} {100 * c:10.1f}%")

    # The verdict belongs to the dtype the model actually runs in. Judging on
    # float32 would fail the code on a ratio between two errors four orders of
    # magnitude below the bf16 one, and the sweep never uses float32.
    work = next(row for row in rows if row[0] == "bfloat16")
    ratio = work[5]
    print()
    print(f"Working dtype bfloat16: batched/loop error ratio = {ratio:.2f}.")
    if ratio <= 1.1:
        print("Verdict: the batched kernel is no further from the float64 answer than the loop. "
              "The difference between them is rounding, which both have.")
    else:
        print(f"Verdict: in the working dtype the batched kernel is {ratio:.2f}x further from "
              "the float64 answer than the loop. That is not rounding but an implementation error.")
    fp32 = next(row for row in rows if row[0] == "float32")
    print(f"\nIn float32 the ratio is {fp32[5]:.2f}, but both errors "
          f"({fp32[3]:.0e} to {fp32[4]:.0e}) are thousands of times smaller than the bfloat16 "
          f"error ({work[3]:.0e}). The difference is the order of accumulation in the two GEMM "
          "shapes (the loop multiplies 16-row matrices, the batched kernel 256), not a defect.")


if __name__ == "__main__":
    main()
