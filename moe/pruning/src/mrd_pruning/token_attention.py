"""Per-token attention: how much each position is *read*, layer by layer.

The existing diagnostic asks where the read position looks. This asks the
mirror question - which positions the whole sequence looks at - because that is
the one that finds attention sinks, and sinks are how super experts announce
themselves.

The reported mechanism is a chain: a super expert emits a massive activation on
a particular token, that token's key becomes an outlier, and every query row
piles onto it. So a sink is visible as a single column of the attention matrix
holding a large share of the mass, and it should sit on the same token as the
hidden-state spike. Measuring both together is what makes the detection a claim
about mechanism rather than a coincidence of two curves.

**Why column sums and not the whole matrix.** The full tensor is
`layers x heads x q x k`; at 48 layers, 32 heads and a 512-token prompt that is
~25 GB for one example in float32. Summing over query rows inside the hook
leaves one vector per layer - `k` numbers, ~2 KB - and that vector is exactly
what a sink lives in. Everything else is dropped before it is ever assembled.

**Causal masking is already in the weights.** Row `i` can only attend to columns
`<= i`, so column `j` is summed over `n - j` rows. Dividing by that count gives
the mean attention a position receives from the rows allowed to see it, which is
comparable across positions; the raw sum is kept too, because a sink is
interesting precisely for holding mass that the normalisation would hide.

**Padding is excluded, not trusted to be zero.** Left-padded rows carry real
softmax mass in some implementations, and a pad column that collects 40% of the
attention would read as a spectacular sink. The mask says which positions are
real and both axes are restricted to them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = [
    "TokenAttention",
    "AttentionProfile",
    "sink_metrics",
    "massive_activations",
    "position_mode",
    "describe_edges",
    "causal_null_top1_share",
    "segment_mass",
    "sink_metrics_batch",
    "hidden_origin",
]


@dataclass
class AttentionProfile:
    """What one batch left behind, per layer.

    `received` is the attention each position collected averaged over heads,
    `received_top_head` the column vector of the single most concentrated head -
    a sink living in two heads out of thirty-two survives only in the latter.
    `hidden_norm` is the per-position hidden-state magnitude from the same
    forward, so a sink and a massive activation can be lined up by position.

    `received_per_head` and `edges` are the detailed layer, kept only when asked
    for. The first is every head's column vector rather than a collapse of them,
    `[batch, heads, positions]`; the second is the strongest `(query, key)`
    pairs per head - literally which token read which, with the weight - which
    is what turns "there is a sink on position 4" into "heads 7 and 19 of layer
    23 put 0.8 of their mass on position 4 from every row".
    """

    received: dict[int, "object"] = field(default_factory=dict)
    received_top_head: dict[int, "object"] = field(default_factory=dict)
    received_per_head: dict[int, "object"] = field(default_factory=dict)
    # Which head held the sink, per example. A layer index alone does not say
    # where to look; thirty-two heads later the trail is cold.
    top_head_index: dict[int, "object"] = field(default_factory=dict)
    edges: dict[int, "object"] = field(default_factory=dict)
    # The read position's own row and its per-head entropy: what the answer
    # position looks at, rather than what the sequence looks at.
    last_row: dict[int, "object"] = field(default_factory=dict)
    row_entropy: dict[int, "object"] = field(default_factory=dict)
    # The top-1 share a model with no sink at all would show on this layer's
    # actual mask - measured from the visibility pattern, not assumed causal.
    null_top1: dict[int, "object"] = field(default_factory=dict)
    received_mean: dict[int, "object"] = field(default_factory=dict)
    hidden_norm: dict[int, "object"] = field(default_factory=dict)
    # Index of the first real token per example: left padding is variable, so
    # absolute column indices are not comparable between examples without it.
    first_real: "object" = None
    n_positions: int = 0


class TokenAttention:
    """Capture per-token attention mass and hidden-state magnitude.

    Used as a context manager around a forward pass. Hooks come off on exit, so
    an exception mid-batch cannot leave the model instrumented.

    Requires eager attention: the fused kernels never build the weight matrix,
    so there is nothing to read and the hook silently sees `None`. The caller
    forces it, and :func:`assert_captured` turns a silent miss into an error.
    """

    def __init__(self, model, *, capture_hidden: bool = True,
                 keep_per_head: bool = False, top_edges: int = 0,
                 edge_min_context: int = 8) -> None:
        """`keep_per_head` and `top_edges` buy detail with memory.

        The collapsed vector is ~2 KB per layer; every head is that times the
        head count, still small (a megabyte per batch at 48 layers), and worth
        it because the head is where a sink lives. `top_edges` keeps the `n`
        strongest `(query, key)` pairs per head, which is the readable form:
        which token read which, and how hard. Both default off so the sweep
        over sixteen cells stays cheap, and the detailed dump is taken on a
        handful of examples.
        """
        self.model = model
        self.capture_hidden = capture_hidden
        self._n_virtual = 0
        self.keep_per_head = keep_per_head
        self.top_edges = top_edges
        self.edge_min_context = edge_min_context
        self.layers = _decoder_layers(model)
        self.profile = AttentionProfile()
        self._handles: list = []
        self._attn_mask = None

    def set_attention_mask(self, mask, n_virtual: int = 0) -> None:
        """Which positions are real. `[batch, positions]`, 1 for a real token.

        `n_virtual` is what makes this work on a PEFT arm at all. PEFT prepends
        the virtual tokens at position 0, *ahead* of the padding, so the routed
        sequence is `[virtual][pad…][text]` while the tokenizer's mask covers
        only the text. The key axis is therefore `n_virtual` wider than the mask
        and the two do not line up - multiplying them raises rather than
        mismeasuring, which is how this was found, but only once the shapes are
        built correctly does anything get measured.

        The virtual positions are real: they are attended to, and on a prefix
        arm they are most of what the read position looks at. They join the key
        mask as ones.
        """
        self._attn_mask = mask
        self._n_virtual = int(n_virtual)

    def _masks(self, full):
        """Key-axis and query-axis masks, sized to what the model actually did.

        The two axes are not the same width on a prefix arm: the virtual tokens
        arrive as past key/values, so they are keys without being queries and
        `q = T_padded` while `k = n_virtual + T_padded`. On a prompt arm they
        are ordinary positions and both axes carry them. Rather than trusting a
        flag, the widths are read off the tensor and checked, because a silent
        misalignment here shifts every position by the virtual-token count and
        the whole measurement is about positions.
        """
        import torch

        text = self._attn_mask.to(full.device).to(full.dtype)
        b, _, q, k = full.shape
        width = text.shape[1]
        expected = width + self._n_virtual
        if k != expected:
            raise RuntimeError(
                f"Attention has {k} keys, but the mask expects {expected} "
                f"({width} text + {self._n_virtual} virtual tokens). "
                "Check the attached arm and the virtual-token count in "
                "adapter_config.")
        ones = torch.ones((b, self._n_virtual), dtype=text.dtype, device=text.device)
        key_m = torch.cat([ones, text], dim=1) if self._n_virtual else text
        if q == k:
            return key_m, key_m
        if q == width:
            return key_m, text
        raise RuntimeError(
            f"Attention has {q} query rows, neither {k} (virtual tokens in the sequence) "
            f"nor {width} (virtual tokens in the cache): unfamiliar attention shape")

    def _attn_hook(self, idx: int):
        def hook(_module, _inputs, output):
            weights = None
            if isinstance(output, tuple):
                for item in output[1:]:
                    if hasattr(item, "ndim") and item.ndim == 4:
                        weights = item
                        break
            if weights is None:
                return output

            import torch

            full = weights.float()                        # [B, heads, q, k]
            if self._attn_mask is not None:
                key_m, query_m = self._masks(full)
                # Kill pad rows (they ask nothing) and pad columns (nothing may
                # legitimately land there). A pad column that keeps its mass is
                # the most convincing fake sink there is.
                full = full * query_m[:, None, :, None]
                full = full * key_m[:, None, None, :]
                m = key_m

            # Per head first, heads collapsed after - and collapsed two ways.
            #
            # Averaging over heads is the obvious move and it is how a sink gets
            # missed: sinks are reported to live in a minority of heads, so a
            # column holding 80% of the mass in two heads out of thirty-two
            # averages down to 5% and reads as ordinary. The mean is kept
            # because it is the sequence-level quantity; the most concentrated
            # head is kept beside it because that is where the sink actually is.
            per_head = full.sum(dim=2)                    # [B, heads, k]
            received = per_head.mean(dim=1)               # [B, k]

            # One whole head, not a per-column maximum. Taking the maximum
            # column by column mixes heads: each column comes from whichever
            # head happened to read it hardest, and the resulting vector is not
            # any head's distribution, so its "share" is a ratio between
            # unrelated numbers. The head that concentrates most is chosen by
            # its own top-1 share and then kept intact.
            head_total = per_head.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            head_top1 = (per_head / head_total).max(dim=-1).values   # [B, heads]
            best = head_top1.argmax(dim=-1)                          # [B]
            received_top_head = per_head[
                torch.arange(per_head.shape[0], device=per_head.device), best]

            # The null this layer should be judged against, measured rather
            # than assumed. `causal_null_top1_share` takes H_n/n, which is only
            # right for a square full-causal triangle. Two shapes in this
            # project are not that: gpt-oss alternates a 128-token sliding
            # window with full attention layer by layer, and a prefix arm's
            # virtual keys are visible to *every* query row. Both make the
            # formula too generous - by about sevenfold on a prefix arm - which
            # hides sinks rather than inventing them, but still misreports them.
            #
            # So: take the visibility pattern the model actually used, let every
            # row spread its mass evenly over what it can see, and ask what the
            # most-read column would then hold. That is the null for this exact
            # mask, whatever shape it has.
            visible = (full.sum(dim=1) > 0)               # [B, q, k]
            seen = visible.sum(dim=-1).clamp(min=1)       # [B, q]
            even = (visible.to(full.dtype) / seen[..., None].to(full.dtype))
            null_cols = even.sum(dim=1)                   # [B, k]
            n_rows = (seen > 1).sum(dim=-1).clamp(min=1).to(full.dtype)
            null_top1 = null_cols.max(dim=-1).values / n_rows

            # How many rows were allowed to see each column. Under a causal
            # mask that is n - j, but counting it from the mask covers padding
            # and any other shape the model imposes.
            # Taken from the same visibility pattern, so it is right for any
            # shape. The old no-mask branch built this from the *query* width,
            # which is not the key width whenever the two axes differ.
            rows = visible.sum(dim=1).clamp(min=1)        # [B, k]
            if self._attn_mask is not None:
                # Where the prompt begins, virtual block excluded. Left padding
                # makes column j mean a different position in every example, so
                # an index is only comparable once it is measured from here, and
                # the virtual block is a different width per arm on top of that.
                # Counting from the first prompt token puts every arm in the
                # same coordinates and leaves the virtual tokens at negative
                # offsets, which is what they are: before the prompt.
                text_m = self._attn_mask.to(full.device).to(full.dtype)
                first_real = text_m.argmax(dim=1) + self._n_virtual
            else:
                first_real = torch.zeros(full.shape[0], dtype=torch.long,
                                         device=full.device)
            mean = received / rows.to(received.dtype)

            self.profile.received[idx] = received.detach().to("cpu").numpy()
            self.profile.received_top_head[idx] = \
                received_top_head.detach().to("cpu").numpy()
            self.profile.received_mean[idx] = mean.detach().to("cpu").numpy()
            self.profile.null_top1[idx] = null_top1.detach().to("cpu").numpy()
            self.profile.first_real = first_real.detach().to("cpu").numpy()
            self.profile.n_positions = received.shape[-1]

            # The read position's own row, per head. This is the mirror of
            # everything above: the columns say which positions are read, the
            # row says what the position that produces the answer is looking at.
            # It is also the quantity the dense-model side of this project
            # measures, so keeping it here is what makes the two comparable.
            # Left padding puts the last real query at the end of the axis.
            last_row = full[:, :, -1, :]                  # [B, heads, k]
            self.profile.last_row[idx] = last_row.detach().to("cpu").numpy()
            p_row = last_row.clamp(min=1e-20)
            self.profile.row_entropy[idx] = (
                -(p_row * p_row.log()).sum(-1)).detach().to("cpu").numpy()

            self.profile.top_head_index[idx] = best.detach().to("cpu").numpy()
            if self.keep_per_head:
                self.profile.received_per_head[idx] = \
                    per_head.detach().to("cpu").numpy()
            if self.top_edges > 0:
                self.profile.edges[idx] = _top_edges(
                    full, self.top_edges, self.edge_min_context)
            return output

        return hook

    def _hidden_hook(self, idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not hasattr(hidden, "ndim") or hidden.ndim != 3:
                return output
            # Infinity norm, not L2: a massive activation is one coordinate
            # running away, and averaging it over the width hides exactly that.
            norm = hidden.float().abs().amax(dim=-1)      # [B, positions]
            self.profile.hidden_norm[idx] = norm.detach().to("cpu").numpy()
            return output

        return hook

    def __enter__(self) -> "TokenAttention":
        for i, layer in enumerate(self.layers):
            self._handles.append(
                layer.self_attn.register_forward_hook(self._attn_hook(i)))
            if self.capture_hidden:
                self._handles.append(
                    layer.register_forward_hook(self._hidden_hook(i)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def assert_captured(self) -> None:
        if not self.profile.received:
            raise RuntimeError(
                "Attention was not captured: the model is not running eager attention. "
                "Fused kernels do not materialize attention weights.")


def _decoder_layers(model):
    """Reach the decoder layers through PEFT and HF wrappers alike."""
    node = model
    for _ in range(6):
        layers = getattr(node, "layers", None)
        if layers is not None:
            return layers
        for attr in ("base_model", "model", "transformer"):
            child = getattr(node, attr, None)
            if child is not None:
                node = child
                break
        else:
            break
    raise ValueError("No decoder layers found")


def force_eager(model) -> bool:
    """Make the model build attention weights. True if it took."""
    ok = False
    try:
        model.config._attn_implementation = "eager"
        ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not switch the configuration of the model to eager: %s", exc)
    for layer in _decoder_layers(model):
        attn = getattr(layer, "self_attn", None)
        cfg = getattr(attn, "config", None)
        if cfg is not None:
            try:
                cfg._attn_implementation = "eager"
                ok = True
            except Exception:  # noqa: BLE001, S110
                pass
    return ok


def sink_metrics(received, first_real: int = 0, null: float | None = None) -> dict[str, float]:
    """How concentrated the attention is, for one layer of one example.

    `top1_share` is the sink itself: the fraction of all attention that landed
    on the single most-read position. A uniform layer over `n` positions gives
    about `1/n`; a sink gives a number that does not shrink as the prompt grows.

    `top4_share` catches the several-sinks case, and `entropy_ratio` is the
    distribution's entropy over the uniform one, so 1.0 is flat and values near
    zero mean everything is on a handful of tokens.

    `top1_over_null` is the share divided by what a model with no sink at all
    would produce - see :func:`causal_null_top1_share`. It is the number to read:
    a raw share of 0.024 at a 256-token window is not a weak sink, it is exactly
    nothing.

    `first_real` shifts the reported index. Batches are left-padded to a common
    width, so absolute column 12 is the fourth real token in one example and the
    twelfth in another; taking a mode over raw indices mixes those and returns a
    number that means nothing. `top1_offset` counts from the first real token
    and is the one to aggregate across examples. `top1_from_end` counts back
    from the last position, which is where a sink on the final delimiter sits
    regardless of length.
    """
    import numpy as np

    arr = np.asarray(received, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return {"top1_share": 0.0, "top1_over_null": 0.0, "top4_share": 0.0,
                "top1_index": None, "top1_offset": None, "top1_from_end": None,
                "entropy_ratio": 0.0, "n_positions": int(arr.size),
                "valid": False}
    p = arr / total
    order = np.argsort(p)[::-1]
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    n = int((arr > 0).sum()) or 1
    uniform = float(np.log(n)) if n > 1 else 1.0
    top1 = int(order[0])
    if null is None:
        null = causal_null_top1_share(n)
    return {
        "top1_share": float(p[order[0]]),
        # Against the causal null, not against uniform. See the helper: under a
        # causal mask the first position wins for free, and 1/n is the wrong
        # yardstick by a factor of six at a 256-token window.
        "top1_over_null": float(p[order[0]] / null) if null > 0 else 0.0,
        "top4_share": float(p[order[:4]].sum()),
        "top1_index": top1,
        "top1_offset": top1 - int(first_real),
        "top1_from_end": int(arr.size) - 1 - top1,
        "entropy_ratio": entropy / uniform if uniform > 0 else 0.0,
        "n_positions": int(arr.size),
        "valid": True,
    }


def massive_activations(hidden_norm, first_real: int = 0,
                        *, factor: float = 10.0) -> dict[str, float]:
    """Positions whose hidden state towers over the rest of the sequence.

    The threshold is relative to the median of the same row, because the scale
    of a hidden state varies by orders of magnitude between layers and a fixed
    cut would find everything in one layer and nothing in another.

    Padding positions are excluded before the median is taken: their hidden
    states are whatever the model makes of a pad token, and including them drags
    the median and manufactures a ratio.

    `argmax_offset` is measured from the first real token, matching
    :func:`sink_metrics`. Comparing a sink's position to an activation's is the
    whole point, and the two are only comparable in the same coordinates.
    """
    import numpy as np

    arr = np.asarray(hidden_norm, dtype=np.float64)
    first = int(first_real)
    if first >= arr.size:
        # The origin must be an index into *this* array. It is not on a prefix
        # arm, where hidden states are on the hidden axis (text width) while
        # `first_real` counts from the key axis (virtual + text): the origin
        # then runs past the end. The old code quietly fell back to the whole
        # row, which took the median over padding and reported an argmax in
        # padded-text coordinates - incommensurable between examples, and in a
        # different coordinate system from the sink it is meant to be paired
        # with. Use `hidden_origin` to convert before calling.
        raise ValueError(
            f"Origin {first_real} exceeds a row of length {arr.size}: "
            "hidden states and keys use different axes; convert the origin "
            "via hidden_origin()")
    real = arr[first:]
    if real.size == 0:
        return {"max": 0.0, "median": 0.0, "ratio": 0.0, "argmax": None,
                "argmax_offset": None, "n_massive": 0, "valid": False}
    med = float(np.median(real))
    mx = float(real.max())
    if mx <= 0:
        # Every position zero: there is no peak to speak of. Reporting this as a
        # measured activation at position `first` is how an all-zero layer ended
        # up with an offset in the table.
        return {"max": 0.0, "median": 0.0, "ratio": 0.0, "argmax": None,
                "argmax_offset": None, "n_massive": 0, "valid": False}
    ratio = mx / med if med > 0 else 0.0
    n_massive = int((real > factor * med).sum()) if med > 0 else 0
    argmax = int(np.argmax(real)) + first
    return {
        "max": mx,
        "median": med,
        "ratio": ratio,
        "argmax": argmax,
        "argmax_offset": argmax - first,
        "n_massive": n_massive,
        "valid": True,
    }


def position_mode(key: str, metrics) -> dict[str, "object"]:
    """The position that wins most often, and how often it wins.

    The share is the point. A sink is a *stable* position, not merely a peaked
    one, and a mode reported without it looks identical whether the same column
    won in every example or in three out of sixty-four. Distributions that never
    agree return a share near `1/n` and should be read as "no fixed position".

    Negative offsets are counted, not dropped. Offsets are measured from the
    first *prompt* token, so a sink sitting inside the virtual block is
    legitimately negative - and on a prefix arm that is where it usually sits.
    An earlier version treated `< 0` as "could not be computed" and reported
    mode `-1` with a share of zero for ten cells out of sixteen, which is to say
    it threw away the answer on exactly the arms the measurement is for.

    Failure is marked by `valid`, not by the sign: an all-zero layer has nothing
    to report and is skipped.
    """
    import numpy as np

    vals = [int(m[key]) for m in metrics
            if m.get(key) is not None and m.get("valid", True)]
    if not vals:
        return {f"{key}_mode": None, f"{key}_mode_frac": 0.0}
    lo = min(vals)
    counts = np.bincount([v - lo for v in vals])
    mode = int(counts.argmax())
    return {f"{key}_mode": mode + lo,
            f"{key}_mode_frac": float(counts[mode] / len(vals))}


def _top_edges(full, n: int, min_context: int = 8):
    """The `n` strongest `(query, key)` pairs of every head.

    Returns `[batch, heads, n, 3]` - query index, key index, weight - as a plain
    array, which is the readable form of an attention map: token *q* read token
    *k* this hard. The column sums say a position is popular; the edges say who
    made it popular, and whether it is every row or a few.

    The pairs come from the masked matrix, so a padded query row cannot appear
    as an edge and neither can a padded key.

    Query rows with fewer than `min_context` positions to choose from are
    dropped, and this is not a nicety. The first real row can attend to exactly
    one position, so its weight there is 1.0 by arithmetic; without the cut
    every head's strongest edge is one of the opening rows in every layer of
    every model, and the dump says nothing about where attention concentrates.
    A row is only evidence once it had somewhere else to go.
    """
    b, heads, q, k = full.shape
    if min_context > 1:
        context = (full.sum(dim=1) > 0).sum(dim=-1)        # [B, q]
        full = full * (context >= min_context)[:, None, :, None].to(full.dtype)
    flat = full.reshape(b, heads, q * k)
    n = min(n, flat.shape[-1])
    weight, idx = flat.topk(n, dim=-1)
    out = weight.new_empty((b, heads, n, 3))
    out[..., 0] = (idx // k).to(weight.dtype)
    out[..., 1] = (idx % k).to(weight.dtype)
    out[..., 2] = weight
    return out.detach().to("cpu").numpy()


def describe_edges(edges, tokens, *, layer: int, min_weight: float = 0.05,
                   first_real: int = 0):
    """Turn one example's edges into lines a person can read.

    `tokens` is the decoded sequence for the same example, aligned to the same
    columns as the attention matrix, padding included - the offsets printed are
    relative to `first_real` so they match the summary's positions.

    Weak edges are dropped: every head has a strongest pair, and in a head that
    is merely diffuse that pair carries a few percent and means nothing. The
    default floor is the point below which an edge is noise rather than a read.
    """
    lines = []
    n_heads = edges.shape[0]
    for head in range(n_heads):
        for q_i, k_i, w in edges[head]:
            if w < min_weight:
                continue
            q_i, k_i = int(q_i), int(k_i)
            q_tok = tokens[q_i] if q_i < len(tokens) else "?"
            k_tok = tokens[k_i] if k_i < len(tokens) else "?"
            lines.append(
                f"layer {layer:2d} head {head:2d}: "
                f"+{q_i - first_real} {q_tok!r} → +{k_i - first_real} {k_tok!r} "
                f"weight {float(w):.3f}")
    return lines


def causal_null_top1_share(n: int) -> float:
    """What the top-1 share is when there is no sink whatsoever.

    Under a causal mask row `i` may only attend to positions `<= i`, so the
    first position is summed over all `n` rows and the last over one. A model
    that spreads every row perfectly evenly therefore still puts

        H_n / n  =  (1 + 1/2 + ... + 1/n) / n

    of the total mass on position 0 - about 0.024 at `n = 256` and 0.074 at
    `n = 64`, against the `1/n` of 0.004 and 0.016 that an uncorrected reading
    would compare to. Six times too generous at the window this runs at, and in
    the direction that invents sinks.

    Two consequences the caller must not forget: shares are only comparable
    between layers at the *same* `n`, and a winning position of 0 is the null
    result rather than evidence - the first token wins for free.
    """
    if n <= 1:
        return 1.0
    import numpy as np

    harmonic = float(np.sum(1.0 / np.arange(1, n + 1)))
    return harmonic / n


def segment_mass(rows, segment_ids, n_segments: int):
    """Attention mass per segment: `[..., keys]` summed into `[..., segments]`.

    This is the quantity the dense-model measurements in this project report,
    and the reason to compute it here is that it answers the question a raw map
    cannot: of everything the answer position reads, how much is the soft
    prompt, how much the instruction, how much the comment itself. A prefix arm
    that puts most of its mass on the virtual block and a prompt arm that does
    not are two different mechanisms, and no per-position picture says so as
    plainly as four numbers.

    Positions labelled `-1` belong to no segment - padding, and generated
    tokens - and are dropped rather than folded into a neighbour.
    """
    import numpy as np

    arr = np.asarray(rows, dtype=np.float64)
    seg = np.asarray(segment_ids)
    if seg.shape[-1] != arr.shape[-1]:
        raise ValueError(
            f"Segment mask has {seg.shape[-1]} positions, but attention has {arr.shape[-1]}")
    return np.stack([arr[..., seg == s].sum(-1) for s in range(n_segments)],
                    axis=-1)


def sink_metrics_batch(received, first_real=None, null=None) -> dict:
    """:func:`sink_metrics` over a whole array at once, `[..., keys]`.

    The full table wants these per example *per head* - at 48 layers and 32
    heads that is fifteen hundred vectors per example, and a Python loop over
    them is the difference between a measurement that runs in a coffee break
    and one that does not. Every returned field is an array shaped like the
    input minus its last axis.

    Kept honest by a test that compares it element by element against the
    scalar version rather than by re-deriving the arithmetic twice.
    """
    import numpy as np

    arr = np.asarray(received, dtype=np.float64)
    k = arr.shape[-1]
    total = arr.sum(axis=-1)
    safe = np.where(total > 0, total, 1.0)
    p = arr / safe[..., None]

    order = np.argsort(p, axis=-1)
    top1_index = order[..., -1]
    top1_share = np.take_along_axis(p, top1_index[..., None], axis=-1)[..., 0]
    top4_share = np.sort(p, axis=-1)[..., -min(4, k):].sum(axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), 0.0)
    entropy = -(p * logs).sum(axis=-1)
    n = (arr > 0).sum(axis=-1)
    uniform = np.where(n > 1, np.log(np.maximum(n, 2)), 1.0)
    entropy_ratio = np.where(uniform > 0, entropy / uniform, 0.0)

    # H_n / n for every n at once: the prefix sums of the harmonic series.
    if null is None:
        harmonic = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, k + 1))])
        null = np.where(n > 1, harmonic[np.minimum(n, k)] / np.maximum(n, 1), 1.0)
    else:
        null = np.asarray(null, dtype=np.float64)
        null = null.reshape(null.shape + (1,) * (top1_index.ndim - null.ndim))

    if first_real is None:
        origin = np.zeros_like(top1_index)
    else:
        origin = np.asarray(first_real)
        if origin.shape != top1_index.shape[:1]:
            raise ValueError(
                f"Origins have shape {origin.shape}, but rows have shape {top1_index.shape[:1]}: "
                "the segment mask belongs to a different batch.")
        # Broadcast along the head axis, if there is one.
        origin = origin.reshape(origin.shape + (1,) * (top1_index.ndim - 1))
    dead = total <= 0
    return {
        "top1_share": np.where(dead, 0.0, top1_share),
        "top1_over_null": np.where(dead | (null <= 0), 0.0, top1_share / null),
        "top4_share": np.where(dead, 0.0, top4_share),
        "top1_index": np.where(dead, -1, top1_index),
        # Offsets keep their sign: negative means the virtual block, which is a
        # place, not a failure. `valid` is what marks a row with nothing in it.
        "top1_offset": top1_index - origin,
        "top1_from_end": k - 1 - top1_index,
        "entropy_ratio": np.where(dead, 0.0, entropy_ratio),
        "n_positions": np.full(top1_index.shape, k),
        "valid": ~dead,
    }


def hidden_origin(first_real: int, hidden_width: int, key_width: int,
                  n_virtual: int) -> int:
    """`first_real` translated from the key axis onto the hidden-state axis.

    The two axes coincide for a base model and for prompt tuning, whose virtual
    tokens are ordinary positions and therefore have hidden states. They do not
    coincide for prefix tuning: those virtual tokens arrive as past key/values,
    so they are keys that were never positions, and the hidden axis is narrower
    by exactly that block.

    Deciding from the widths rather than from the arm's name is deliberate - the
    arm's name is a label, the widths are what the model actually did.
    """
    if hidden_width == key_width:
        return int(first_real)
    if hidden_width == key_width - n_virtual:
        return max(0, int(first_real) - int(n_virtual))
    raise ValueError(
        f"Hidden axis {hidden_width} matches neither key axis {key_width} "
        f"nor key axis without {n_virtual} virtual tokens: unfamiliar shape")
