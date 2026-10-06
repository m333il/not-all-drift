"""SAE interventions on token spans belonging to editable instructions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from prompt_optimization.gemma_scope import (
    GemmaScopeJumpReLU,
    ReconstructionAccumulator,
    ReconstructionMetrics,
    hidden_from_decoder_output,
    replace_decoder_hidden,
)

InstructionWindow = int | Literal["all"]
InstructionIntervention = Literal[
    "sae_reconstruction",
    "zero_feature",
    "shuffled_reconstruction",
]


def select_instruction_window(
    positions: list[int],
    *,
    window: InstructionWindow,
) -> list[int]:
    """Return the complete or right-aligned tail of an instruction span."""
    if not positions:
        raise ValueError("Instruction positions must not be empty")
    normalized = [int(position) for position in positions]
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError("Instruction positions must be strictly increasing")
    if window == "all":
        return normalized
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("Instruction window must be a positive integer or 'all'")
    return normalized[-window:]


def build_instruction_window_mask(
    position_rows: list[list[int]],
    *,
    sequence_length: int,
    window: InstructionWindow,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a boolean mask selecting instruction tokens in each batch row."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not position_rows:
        raise ValueError("At least one instruction-position row is required")
    mask = torch.zeros(
        (len(position_rows), sequence_length),
        dtype=torch.bool,
        device=device,
    )
    for row_index, positions in enumerate(position_rows):
        selected = select_instruction_window(positions, window=window)
        if selected[0] < 0 or selected[-1] >= sequence_length:
            raise ValueError("Instruction position is outside the sequence")
        mask[row_index, torch.tensor(selected, device=mask.device)] = True
    return mask


@dataclass(frozen=True)
class InstructionInterventionResult:
    """Diagnostics captured by one instruction-token intervention."""

    selected_states: int
    reconstruction: ReconstructionMetrics


class OneShotInstructionSAEHook:
    """Replace selected resid-post states during the first decoder call."""

    def __init__(
        self,
        sae: GemmaScopeJumpReLU,
        selection_mask: torch.Tensor,
        *,
        mode: InstructionIntervention,
        chunk_size: int,
        shuffle_seed: int = 0,
        accumulator: ReconstructionAccumulator | None = None,
    ) -> None:
        if selection_mask.ndim != 2:
            raise ValueError("selection_mask must have shape [batch, sequence]")
        if mode not in {
            "sae_reconstruction",
            "zero_feature",
            "shuffled_reconstruction",
        }:
            raise ValueError(f"Unknown instruction intervention: {mode}")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.sae = sae
        self.selection_mask = selection_mask.to(dtype=torch.bool)
        self.mode = mode
        self.chunk_size = chunk_size
        self.shuffle_seed = int(shuffle_seed)
        self.accumulator = accumulator or ReconstructionAccumulator()
        self.applied = False
        self.result: InstructionInterventionResult | None = None

    def _replacement(
        self,
        selected: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "sae_reconstruction":
            return reconstruction
        if self.mode == "zero_feature":
            return self.sae.b_dec.expand_as(selected)
        if len(reconstruction) < 2:
            raise ValueError("Shuffled reconstruction requires at least two selected states")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.shuffle_seed)
        permutation = torch.randperm(len(reconstruction), generator=generator)
        if bool((permutation == torch.arange(len(permutation))).all()):
            permutation = torch.roll(permutation, shifts=1)
        return reconstruction[permutation.to(device=reconstruction.device)]

    def __call__(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if self.applied:
            return output
        hidden = hidden_from_decoder_output(output)
        if hidden.shape[:2] != self.selection_mask.shape:
            raise ValueError(
                "Decoder output and instruction mask disagree: "
                f"{tuple(hidden.shape[:2])} vs {tuple(self.selection_mask.shape)}"
            )
        mask = self.selection_mask.to(device=hidden.device)
        selected = hidden[mask]
        reconstruction, metrics = self.sae.reconstruct_chunked(
            selected,
            chunk_size=self.chunk_size,
            accumulator=self.accumulator,
        )
        replacement = self._replacement(selected, reconstruction)
        modified = hidden.clone()
        modified[mask] = replacement.to(device=hidden.device, dtype=hidden.dtype)
        self.applied = True
        self.result = InstructionInterventionResult(
            selected_states=len(selected),
            reconstruction=metrics,
        )
        return replace_decoder_hidden(output, modified)


class OneShotPositionReplacementHook:
    """Replace selected resid-post states with precomputed aligned states."""

    def __init__(
        self,
        selection_mask: torch.Tensor,
        replacement: torch.Tensor,
    ) -> None:
        if selection_mask.ndim != 2:
            raise ValueError("selection_mask must have shape [batch, sequence]")
        if replacement.ndim != 2:
            raise ValueError("replacement must have shape [selected states, d_model]")
        if int(selection_mask.sum()) != len(replacement):
            raise ValueError("selection mask and replacement state counts differ")
        self.selection_mask = selection_mask.to(dtype=torch.bool)
        self.replacement = replacement
        self.applied = False

    def __call__(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if self.applied:
            return output
        hidden = hidden_from_decoder_output(output)
        if hidden.shape[:2] != self.selection_mask.shape:
            raise ValueError(
                "Decoder output and selection mask disagree: "
                f"{tuple(hidden.shape[:2])} vs {tuple(self.selection_mask.shape)}"
            )
        mask = self.selection_mask.to(device=hidden.device)
        modified = hidden.clone()
        modified[mask] = self.replacement.to(device=hidden.device, dtype=hidden.dtype)
        self.applied = True
        return replace_decoder_hidden(output, modified)


class OneShotPositionCaptureHook:
    """Capture selected resid-post states during the first decoder call."""

    def __init__(self, selection_mask: torch.Tensor) -> None:
        if selection_mask.ndim != 2:
            raise ValueError("selection_mask must have shape [batch, sequence]")
        self.selection_mask = selection_mask.to(dtype=torch.bool)
        self.applied = False
        self.states: torch.Tensor | None = None

    def __call__(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if self.applied:
            return output
        hidden = hidden_from_decoder_output(output)
        if hidden.shape[:2] != self.selection_mask.shape:
            raise ValueError(
                "Decoder output and selection mask disagree: "
                f"{tuple(hidden.shape[:2])} vs {tuple(self.selection_mask.shape)}"
            )
        mask = self.selection_mask.to(device=hidden.device)
        self.states = hidden[mask].detach().to(device="cpu", dtype=torch.bfloat16)
        self.applied = True
        return output


class OneShotPositionDeltaHook:
    """Add or subtract an aligned delta from selected resid-post states."""

    def __init__(
        self,
        selection_mask: torch.Tensor,
        delta: torch.Tensor,
        *,
        scale: float = 1.0,
    ) -> None:
        if selection_mask.ndim != 2:
            raise ValueError("selection_mask must have shape [batch, sequence]")
        if delta.ndim != 2:
            raise ValueError("delta must have shape [selected states, d_model]")
        if int(selection_mask.sum()) != len(delta):
            raise ValueError("selection mask and delta state counts differ")
        self.selection_mask = selection_mask.to(dtype=torch.bool)
        self.delta = delta
        self.scale = float(scale)
        self.applied = False

    def __call__(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if self.applied:
            return output
        hidden = hidden_from_decoder_output(output)
        if hidden.shape[:2] != self.selection_mask.shape:
            raise ValueError(
                "Decoder output and selection mask disagree: "
                f"{tuple(hidden.shape[:2])} vs {tuple(self.selection_mask.shape)}"
            )
        mask = self.selection_mask.to(device=hidden.device)
        modified = hidden.clone()
        update = self.delta.to(device=hidden.device, dtype=hidden.dtype)
        modified[mask] = modified[mask] + self.scale * update
        self.applied = True
        return replace_decoder_hidden(output, modified)
