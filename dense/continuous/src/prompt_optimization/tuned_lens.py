"""Training and evaluation primitives for vanilla and tuned logit lenses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional


class ResidualTranslator(nn.Module):
    """Affine residual translator initialized as the identity map.

    The learned map is ``T(h) = h + Delta(h)``. Zero initialization of
    ``Delta`` makes the untrained module exactly equivalent to the vanilla
    logit lens.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.delta = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != self.hidden_size:
            raise ValueError("state hidden size does not match translator")
        return states + self.delta(states)


@dataclass(frozen=True)
class LensMetrics:
    """Distribution-level agreement with the model's final prediction."""

    kl_to_final: float
    top1_agreement: float
    teacher_top1_probability: float
    entropy: float
    samples: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "kl_to_final": self.kl_to_final,
            "top1_agreement": self.top1_agreement,
            "teacher_top1_probability": self.teacher_top1_probability,
            "entropy": self.entropy,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class TunedLensFit:
    """Best validation state and optimization history for one depth point."""

    translator: ResidualTranslator
    best_step: int
    best_validation_kl: float
    direct_validation: LensMetrics
    tuned_validation: LensMetrics
    history: tuple[dict[str, float | int], ...]


def _validate_state_pair(layer_states: torch.Tensor, final_states: torch.Tensor) -> None:
    if layer_states.ndim != 2 or final_states.ndim != 2:
        raise ValueError("states must have shape [samples, hidden_size]")
    if layer_states.shape != final_states.shape:
        raise ValueError("layer and final states must have identical shapes")
    if len(layer_states) == 0:
        raise ValueError("states must not be empty")


def decoder_dtype(lm_head: nn.Module) -> torch.dtype:
    """Return the floating dtype expected by a frozen unembedding module."""
    parameter = next(lm_head.parameters(), None)
    return parameter.dtype if parameter is not None else torch.float32


def project_final_states(final_states: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
    """Project the final, already-normalized Hugging Face hidden state.

    Gemma 2 appends its final hidden state after the final RMSNorm. Applying the
    norm again here would not reproduce the model's actual next-token logits.
    """
    return lm_head(final_states.to(dtype=decoder_dtype(lm_head)))


def project_intermediate_states(
    layer_states: torch.Tensor,
    *,
    final_norm: nn.Module,
    lm_head: nn.Module,
    translator: ResidualTranslator | None = None,
) -> torch.Tensor:
    """Apply an optional translator, the final norm, and frozen unembedding."""
    translated = translator(layer_states) if translator is not None else layer_states
    translated = translated.to(dtype=decoder_dtype(lm_head))
    return lm_head(final_norm(translated))


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Compute ``KL(final || lens)`` with standard distillation scaling."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_log_probs = functional.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = functional.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    return (
        functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        * temperature**2
    )


def _batch_metrics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    student_log_probs = functional.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = functional.softmax(teacher_logits.float() / temperature, dim=-1)
    kl = (teacher_probs * (teacher_probs.clamp_min(1e-30).log() - student_log_probs)).sum(
        dim=-1
    )
    teacher_top1 = teacher_logits.argmax(dim=-1)
    top1_agreement = (student_logits.argmax(dim=-1) == teacher_top1).to(torch.float32)
    teacher_top1_probability = student_log_probs.gather(
        dim=-1, index=teacher_top1[:, None]
    ).exp()[:, 0]
    entropy = -(student_log_probs.exp() * student_log_probs).sum(dim=-1)
    return kl, top1_agreement, teacher_top1_probability, entropy


def evaluate_lens_layer(
    layer_states: torch.Tensor,
    final_states: torch.Tensor,
    *,
    final_norm: nn.Module,
    lm_head: nn.Module,
    device: torch.device,
    translator: ResidualTranslator | None,
    batch_size: int,
    temperature: float = 1.0,
) -> LensMetrics:
    """Evaluate one direct or tuned lens against the final model distribution."""
    _validate_state_pair(layer_states, final_states)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    totals = torch.zeros(4, dtype=torch.float64)
    if translator is not None:
        translator.eval()
    with torch.inference_mode():
        for start in range(0, len(layer_states), batch_size):
            stop = min(start + batch_size, len(layer_states))
            layer_batch = layer_states[start:stop].to(device=device, dtype=torch.float32)
            final_batch = final_states[start:stop].to(
                device=device, dtype=decoder_dtype(lm_head)
            )
            teacher_logits = project_final_states(final_batch, lm_head)
            student_logits = project_intermediate_states(
                layer_batch,
                final_norm=final_norm,
                lm_head=lm_head,
                translator=translator,
            )
            values = _batch_metrics(
                student_logits,
                teacher_logits,
                temperature=temperature,
            )
            totals += torch.tensor([item.sum().item() for item in values], dtype=torch.float64)

    denominator = float(len(layer_states))
    return LensMetrics(
        kl_to_final=float(totals[0] / denominator),
        top1_agreement=float(totals[1] / denominator),
        teacher_top1_probability=float(totals[2] / denominator),
        entropy=float(totals[3] / denominator),
        samples=len(layer_states),
    )


def fit_tuned_lens_layer(
    train_layer_states: torch.Tensor,
    train_final_states: torch.Tensor,
    validation_layer_states: torch.Tensor,
    validation_final_states: torch.Tensor,
    *,
    final_norm: nn.Module,
    lm_head: nn.Module,
    device: torch.device,
    steps: int,
    batch_size: int,
    evaluation_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    evaluation_interval: int,
    patience: int,
    seed: int,
) -> TunedLensFit:
    """Fit one full-rank affine translator by distilling final next-token logits."""
    _validate_state_pair(train_layer_states, train_final_states)
    _validate_state_pair(validation_layer_states, validation_final_states)
    if train_layer_states.shape[1] != validation_layer_states.shape[1]:
        raise ValueError("train and validation hidden sizes must agree")
    if steps <= 0 or batch_size <= 0 or evaluation_batch_size <= 0:
        raise ValueError("steps and batch sizes must be positive")
    if learning_rate <= 0 or weight_decay < 0 or temperature <= 0:
        raise ValueError("invalid optimizer or temperature configuration")
    if evaluation_interval <= 0 or patience <= 0:
        raise ValueError("evaluation_interval and patience must be positive")

    for module in (final_norm, lm_head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    translator = ResidualTranslator(train_layer_states.shape[1]).to(
        device=device, dtype=torch.float32
    )
    optimizer = torch.optim.AdamW(
        translator.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direct_validation = evaluate_lens_layer(
        validation_layer_states,
        validation_final_states,
        final_norm=final_norm,
        lm_head=lm_head,
        device=device,
        translator=None,
        batch_size=evaluation_batch_size,
        temperature=temperature,
    )

    best_validation_kl = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    evaluations_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for step in range(1, steps + 1):
        translator.train()
        indices = torch.randint(
            len(train_layer_states),
            (min(batch_size, len(train_layer_states)),),
            generator=generator,
        )
        layer_batch = train_layer_states[indices].to(device=device, dtype=torch.float32)
        final_batch = train_final_states[indices].to(
            device=device, dtype=decoder_dtype(lm_head)
        )
        with torch.no_grad():
            teacher_logits = project_final_states(final_batch, lm_head)
        student_logits = project_intermediate_states(
            layer_batch,
            final_norm=final_norm,
            lm_head=lm_head,
            translator=translator,
        )
        loss = distillation_kl(
            student_logits,
            teacher_logits,
            temperature=temperature,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_evaluate = step == 1 or step % evaluation_interval == 0 or step == steps
        if not should_evaluate:
            continue
        tuned_validation = evaluate_lens_layer(
            validation_layer_states,
            validation_final_states,
            final_norm=final_norm,
            lm_head=lm_head,
            device=device,
            translator=translator,
            batch_size=evaluation_batch_size,
            temperature=temperature,
        )
        history.append(
            {
                "step": step,
                "train_kl": float(loss.detach().cpu()),
                "validation_kl": tuned_validation.kl_to_final,
                "validation_top1_agreement": tuned_validation.top1_agreement,
            }
        )
        if tuned_validation.kl_to_final < best_validation_kl:
            best_validation_kl = tuned_validation.kl_to_final
            best_step = step
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in translator.state_dict().items()
            }
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
            if evaluations_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("tuned lens optimization produced no checkpoint")
    translator.load_state_dict(best_state)
    translator.to(device)
    tuned_validation = evaluate_lens_layer(
        validation_layer_states,
        validation_final_states,
        final_norm=final_norm,
        lm_head=lm_head,
        device=device,
        translator=translator,
        batch_size=evaluation_batch_size,
        temperature=temperature,
    )
    return TunedLensFit(
        translator=translator,
        best_step=best_step,
        best_validation_kl=best_validation_kl,
        direct_validation=direct_validation,
        tuned_validation=tuned_validation,
        history=tuple(history),
    )


def save_translator(
    path: Path,
    translator: ResidualTranslator,
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    """Save one translator without pickled Python objects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = translator.state_dict()
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in state.items()},
        path,
        metadata=metadata,
    )


def load_translator(
    path: Path,
    *,
    hidden_size: int,
    device: torch.device,
) -> ResidualTranslator:
    """Load a translator and validate its hidden-size contract."""
    tensors = load_file(path, device="cpu")
    expected = {
        "delta.weight": (hidden_size, hidden_size),
        "delta.bias": (hidden_size,),
    }
    if set(tensors) != set(expected):
        raise ValueError(f"unexpected translator keys: {sorted(tensors)}")
    for key, shape in expected.items():
        if tuple(tensors[key].shape) != shape:
            raise ValueError(f"unexpected shape for {key}: {tuple(tensors[key].shape)}")
    translator = ResidualTranslator(hidden_size)
    translator.load_state_dict(tensors)
    return translator.to(device=device, dtype=torch.float32)


def evaluate_lens_bank(
    states: torch.Tensor,
    *,
    translators: Sequence[ResidualTranslator | None],
    final_norm: nn.Module,
    lm_head: nn.Module,
    device: torch.device,
    batch_size: int,
    temperature: float,
) -> list[dict[str, Any]]:
    """Evaluate direct and tuned trajectories for all stored depth points."""
    if states.ndim != 3:
        raise ValueError("states must have shape [samples, depth_points, hidden_size]")
    if len(translators) != states.shape[1] - 1:
        raise ValueError("provide one translator entry per non-final depth point")
    final_states = states[:, -1]
    rows: list[dict[str, Any]] = []
    for layer, translator in enumerate(translators):
        direct = evaluate_lens_layer(
            states[:, layer],
            final_states,
            final_norm=final_norm,
            lm_head=lm_head,
            device=device,
            translator=None,
            batch_size=batch_size,
            temperature=temperature,
        )
        tuned = (
            evaluate_lens_layer(
                states[:, layer],
                final_states,
                final_norm=final_norm,
                lm_head=lm_head,
                device=device,
                translator=translator,
                batch_size=batch_size,
                temperature=temperature,
            )
            if translator is not None
            else None
        )
        rows.append(
            {
                "depth_point": layer,
                "direct": direct.as_dict(),
                "tuned": tuned.as_dict() if tuned is not None else None,
            }
        )
    rows.append(
        {
            "depth_point": states.shape[1] - 1,
            "direct": {
                "kl_to_final": 0.0,
                "top1_agreement": 1.0,
                "teacher_top1_probability": None,
                "entropy": None,
                "samples": len(states),
            },
            "tuned": None,
        }
    )
    return rows
