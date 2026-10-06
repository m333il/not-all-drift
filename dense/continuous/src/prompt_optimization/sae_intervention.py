"""Collective error-preserving interventions for paired SAE residual shifts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch

from prompt_optimization.gemma_scope import hidden_from_decoder_output, replace_decoder_hidden


class SAEEncoderDecoder(Protocol):
    W_dec: torch.Tensor

    def encode(self, inputs: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class CollectiveShift:
    """Dense method shift split into SAE-decoded and residual components."""

    dense: torch.Tensor
    sparse: torch.Tensor
    residual: torch.Tensor


PatchMode = Literal["add", "subtract", "replace"]


class OneShotAnchorPatch:
    """Patch one position per sample on the first decoder call of generation."""

    def __init__(
        self,
        vectors: torch.Tensor,
        positions: torch.Tensor,
        *,
        mode: PatchMode,
    ) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must have shape [batch, hidden_size]")
        if positions.ndim != 1 or len(positions) != len(vectors):
            raise ValueError("positions must have shape [batch]")
        if mode not in {"add", "subtract", "replace"}:
            raise ValueError(f"Unsupported patch mode: {mode}")
        self.vectors = vectors
        self.positions = positions.to(dtype=torch.long)
        self.mode = mode
        self.applied = False

    def __call__(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if self.applied:
            return output
        hidden = hidden_from_decoder_output(output)
        if hidden.ndim != 3:
            raise ValueError("decoder hidden states must have shape [batch, sequence, hidden]")
        if hidden.shape[0] != len(self.vectors) or hidden.shape[2] != self.vectors.shape[1]:
            raise ValueError("patch vectors do not match decoder hidden shape")
        positions = self.positions.to(device=hidden.device)
        if bool(((positions < 0) | (positions >= hidden.shape[1])).any()):
            raise ValueError("patch positions are outside decoder sequence length")
        vectors = self.vectors.to(device=hidden.device, dtype=hidden.dtype)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        modified = hidden.clone()
        if self.mode == "add":
            modified[batch, positions] += vectors
        elif self.mode == "subtract":
            modified[batch, positions] -= vectors
        else:
            modified[batch, positions] = vectors
        self.applied = True
        return replace_decoder_hidden(output, modified)


def final_text_window_positions(
    attention_mask: torch.Tensor,
    *,
    hidden_offset: int,
    window_size: int,
) -> torch.Tensor:
    """Return the final ``window_size`` valid text positions in model-state space."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if hidden_offset < 0:
        raise ValueError("hidden_offset must be non-negative")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    valid = attention_mask.to(dtype=torch.bool)
    counts = valid.sum(dim=1)
    if bool((counts < window_size).any()):
        raise ValueError(f"At least one prompt has fewer than {window_size} valid tokens")
    sequence_positions = torch.arange(
        valid.shape[1], device=valid.device, dtype=torch.long
    ).expand_as(valid)
    # Ranking valid positions makes this independent of left/right padding.
    ranks = valid.cumsum(dim=1)
    first_rank = counts[:, None] - window_size + 1
    selected = valid & (ranks >= first_rank)
    positions = sequence_positions[selected].reshape(len(valid), window_size)
    return positions + hidden_offset


class OneShotTokenWindowPatch:
    """Patch a fixed number of textual positions on the first decoder call."""

    def __init__(
        self,
        vectors: torch.Tensor,
        positions: torch.Tensor,
        *,
        mode: PatchMode,
    ) -> None:
        if vectors.ndim != 3:
            raise ValueError("vectors must have shape [batch, window, hidden_size]")
        if positions.shape != vectors.shape[:2]:
            raise ValueError("positions must have shape [batch, window]")
        if mode not in {"add", "subtract", "replace"}:
            raise ValueError(f"Unsupported patch mode: {mode}")
        if any(len(torch.unique(row)) != len(row) for row in positions):
            raise ValueError("positions must be unique within every sample")
        self.vectors = vectors
        self.positions = positions.to(dtype=torch.long)
        self.mode = mode
        self.applied = False

    def __call__(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> Any:
        if self.applied:
            return output
        hidden = hidden_from_decoder_output(output)
        if hidden.ndim != 3:
            raise ValueError("decoder hidden states must have shape [batch, sequence, hidden]")
        if hidden.shape[0] != len(self.vectors) or hidden.shape[2] != self.vectors.shape[2]:
            raise ValueError("patch vectors do not match decoder hidden shape")
        positions = self.positions.to(device=hidden.device)
        if bool(((positions < 0) | (positions >= hidden.shape[1])).any()):
            raise ValueError("patch positions are outside decoder sequence length")
        vectors = self.vectors.to(device=hidden.device, dtype=hidden.dtype)
        batch = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        modified = hidden.clone()
        if self.mode == "add":
            modified[batch, positions] += vectors
        elif self.mode == "subtract":
            modified[batch, positions] -= vectors
        else:
            modified[batch, positions] = vectors
        self.applied = True
        return replace_decoder_hidden(output, modified)


def decompose_collective_shift(
    manual: torch.Tensor,
    adapted: torch.Tensor,
    sae: SAEEncoderDecoder,
    *,
    chunk_size: int,
) -> CollectiveShift:
    """Encode paired states and decode their feature difference without bias."""
    if manual.shape != adapted.shape:
        raise ValueError("manual and adapted states must have identical shapes")
    if manual.ndim != 2:
        raise ValueError("states must have shape [samples, hidden_size]")
    if not len(manual):
        raise ValueError("states must contain at least one sample")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if manual.shape[1] != sae.W_dec.shape[1]:
        raise ValueError("state hidden size does not match SAE decoder")

    sparse_chunks: list[torch.Tensor] = []
    for start in range(0, len(manual), chunk_size):
        manual_chunk = manual[start : start + chunk_size]
        adapted_chunk = adapted[start : start + chunk_size]
        manual_features = sae.encode(manual_chunk)
        adapted_features = sae.encode(adapted_chunk)
        sparse_chunks.append((adapted_features - manual_features) @ sae.W_dec)
    sparse = torch.cat(sparse_chunks, dim=0).to(dtype=manual.dtype)
    dense = adapted - manual
    return CollectiveShift(dense=dense, sparse=sparse, residual=dense - sparse)


def norm_match(source: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Scale each source vector to the corresponding target-vector norm."""
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must have equal [samples, hidden_size] shapes")
    if eps <= 0:
        raise ValueError("eps must be positive")
    source_norm = source.float().norm(dim=-1, keepdim=True)
    target_norm = target.float().norm(dim=-1, keepdim=True)
    scale = torch.where(source_norm > eps, target_norm / source_norm, torch.zeros_like(source_norm))
    return (source.float() * scale).to(dtype=source.dtype)


def deterministic_derangement(size: int, *, seed: int, device: torch.device) -> torch.Tensor:
    """Return a deterministic permutation with no fixed sample indices."""
    if size < 2:
        raise ValueError("derangement requires at least two samples")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    identity = torch.arange(size)
    for _ in range(10_000):
        candidate = torch.randperm(size, generator=generator)
        if bool((candidate != identity).all()):
            return candidate.to(device=device)
    # This deterministic fallback is always a derangement for size >= 2.
    return torch.roll(identity, shifts=1).to(device=device)


def shuffled_norm_matched(shifts: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Use another sample's direction while preserving each target shift norm."""
    if shifts.ndim != 2:
        raise ValueError("shifts must have shape [samples, hidden_size]")
    permutation = deterministic_derangement(len(shifts), seed=seed, device=shifts.device)
    return norm_match(shifts[permutation], shifts)


def random_norm_matched(shifts: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Generate deterministic isotropic directions with matched per-sample norm."""
    if shifts.ndim != 2:
        raise ValueError("shifts must have shape [samples, hidden_size]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.randn(shifts.shape, generator=generator, dtype=torch.float32)
    random = random.to(device=shifts.device, dtype=shifts.dtype)
    return norm_match(random, shifts)


def shuffled_window_norm_matched(
    shifts: torch.Tensor, *, seed: int
) -> torch.Tensor:
    """Shuffle complete sample windows while matching every token-vector norm."""
    if shifts.ndim != 3:
        raise ValueError("shifts must have shape [samples, window, hidden_size]")
    permutation = deterministic_derangement(
        len(shifts), seed=seed, device=shifts.device
    )
    source = shifts[permutation]
    matched = norm_match(
        source.reshape(-1, source.shape[-1]),
        shifts.reshape(-1, shifts.shape[-1]),
    )
    return matched.reshape_as(shifts)


def random_window_norm_matched(shifts: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Generate deterministic isotropic token directions with matched norms."""
    if shifts.ndim != 3:
        raise ValueError("shifts must have shape [samples, window, hidden_size]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.randn(shifts.shape, generator=generator, dtype=torch.float32)
    random = random.to(device=shifts.device, dtype=shifts.dtype)
    matched = norm_match(
        random.reshape(-1, random.shape[-1]),
        shifts.reshape(-1, shifts.shape[-1]),
    )
    return matched.reshape_as(shifts)


def build_collective_interventions(
    manual: torch.Tensor,
    adapted: torch.Tensor,
    sparse: torch.Tensor,
    *,
    shuffle_seed: int,
    random_seed: int,
) -> dict[str, torch.Tensor]:
    """Build add/remove states and matched controls at one layer and anchor."""
    if not (manual.shape == adapted.shape == sparse.shape) or manual.ndim != 2:
        raise ValueError("manual, adapted, and sparse must share a 2D shape")
    dense = adapted - manual
    residual = dense - sparse
    shuffled = shuffled_norm_matched(sparse, seed=shuffle_seed)
    random = random_norm_matched(sparse, seed=random_seed)
    return {
        "manual": manual,
        "adapted": adapted,
        "zero_add": manual.clone(),
        "zero_remove": adapted.clone(),
        "dense_add": manual + dense,
        "dense_remove": adapted - dense,
        "sae_add": manual + sparse,
        "sae_remove": adapted - sparse,
        "residual_add": manual + residual,
        "residual_remove": adapted - residual,
        "shuffled_add": manual + shuffled,
        "shuffled_remove": adapted - shuffled,
        "random_add": manual + random,
        "random_remove": adapted - random,
    }


def kl_from_logits(target_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    """Per-sample KL(target || candidate) computed stably in float32."""
    if target_logits.shape != candidate_logits.shape or target_logits.ndim != 2:
        raise ValueError("logits must have equal [samples, vocabulary] shapes")
    target_log_probs = target_logits.float().log_softmax(dim=-1)
    candidate_log_probs = candidate_logits.float().log_softmax(dim=-1)
    target_probs = target_log_probs.exp()
    return (target_probs * (target_log_probs - candidate_log_probs)).sum(dim=-1)


def js_from_logits(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
    """Per-sample Jensen-Shannon divergence in nats."""
    if left_logits.shape != right_logits.shape or left_logits.ndim != 2:
        raise ValueError("logits must have equal [samples, vocabulary] shapes")
    left_log = left_logits.float().log_softmax(dim=-1)
    right_log = right_logits.float().log_softmax(dim=-1)
    mixture_log = torch.logaddexp(left_log, right_log) - torch.log(
        torch.tensor(2.0, device=left_log.device)
    )
    left = (left_log.exp() * (left_log - mixture_log)).sum(dim=-1)
    right = (right_log.exp() * (right_log - mixture_log)).sum(dim=-1)
    return 0.5 * (left + right)


def normalized_recovery(
    baseline_distance: torch.Tensor,
    candidate_distance: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return 1 - candidate/baseline, with NaN for unstable denominators."""
    if baseline_distance.shape != candidate_distance.shape:
        raise ValueError("distance tensors must have equal shapes")
    if eps <= 0:
        raise ValueError("eps must be positive")
    recovery = 1.0 - candidate_distance / baseline_distance.clamp_min(eps)
    nan = torch.full_like(recovery, float("nan"))
    return torch.where(baseline_distance > eps, recovery, nan)

def distribution_metrics(
    target: torch.Tensor,
    candidate: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, list[float | int]]:
    """Compute full-vocabulary KL, JS, and top-1 agreement in bounded batches."""
    if target.shape != candidate.shape:
        raise ValueError("Distribution logits must have equal shapes")
    result: dict[str, list[float | int]] = {"kl": [], "js": [], "top1_agreement": []}
    for start in range(0, len(target), batch_size):
        left = target[start : start + batch_size].to(device=device, dtype=torch.float32)
        right = candidate[start : start + batch_size].to(device=device, dtype=torch.float32)
        result["kl"].extend(float(value) for value in kl_from_logits(left, right).cpu())
        result["js"].extend(float(value) for value in js_from_logits(left, right).cpu())
        result["top1_agreement"].extend(
            int(value) for value in (left.argmax(-1) == right.argmax(-1)).cpu()
        )
    return result
