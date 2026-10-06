"""Explicit one-shot and recurrent residual interventions during generation."""

from __future__ import annotations

from typing import Any, Literal

import torch


InterventionMode = Literal[
    "prefill_once",
    "fixed_recurrent",
    "repredict_recurrent",
    "recurrent",
]


class AutoregressiveDeltaHook:
    """Add a predicted delta at the prompt anchor and optionally every decode step.

    The first layer invocation is the prompt prefill and uses ``prompt_positions``.
    With greedy, single-beam cached generation, later invocations contain the
    current decode token and the intervention is applied at its final sequence
    position. ``prefill_once`` reproduces the historical one-shot protocol.
    ``fixed_recurrent`` reuses the delta predicted at prompt prefill on every
    decode step. ``repredict_recurrent`` recomputes the delta from the current
    decode-token state. ``recurrent`` remains a backwards-compatible alias for
    ``repredict_recurrent``.
    """

    def __init__(
        self,
        prompt_positions: torch.Tensor,
        *,
        predictor: torch.nn.Module,
        mode: InterventionMode,
    ) -> None:
        if mode not in (
            "prefill_once",
            "fixed_recurrent",
            "repredict_recurrent",
            "recurrent",
        ):
            raise ValueError(f"Unknown intervention mode: {mode}")
        if prompt_positions.ndim != 1 or len(prompt_positions) < 1:
            raise ValueError("prompt_positions must be a non-empty vector")
        self.prompt_positions = prompt_positions.detach().clone()
        self.predictor = predictor
        self.mode = mode
        self.calls = 0
        self.patched_calls = 0
        self.prefill_delta_norm_sum = 0.0
        self.prefill_vectors = 0
        self.decode_delta_norm_sum = 0.0
        self.decode_vectors = 0
        self.prefill_hidden_norm_sum = 0.0
        self.decode_hidden_norm_sum = 0.0
        self._initial_delta: torch.Tensor | None = None

    def __call__(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != len(self.prompt_positions):
            raise ValueError(
                "Generation hook batch/shape changed: "
                f"hidden={tuple(hidden.shape)}, expected batch={len(self.prompt_positions)}"
            )
        is_prefill = self.calls == 0
        self.calls += 1
        if not is_prefill and self.mode == "prefill_once":
            return output

        if is_prefill:
            positions = self.prompt_positions.to(hidden.device)
            if (positions < 0).any() or (positions >= hidden.shape[1]).any():
                raise ValueError("Prompt intervention position is outside the prefill sequence")
        else:
            positions = torch.full(
                (hidden.shape[0],), hidden.shape[1] - 1, device=hidden.device, dtype=torch.long
            )
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        original = hidden[rows, positions]
        if is_prefill or self.mode != "fixed_recurrent":
            delta = self.predictor(original.float())
        else:
            if self._initial_delta is None:
                raise RuntimeError("Fixed recurrent intervention has no prefill delta")
            delta = self._initial_delta.to(original.device)
        if delta.shape != original.shape or not torch.isfinite(delta).all():
            raise ValueError("Predictor returned an invalid autoregressive delta")
        if is_prefill:
            self._initial_delta = delta.detach().clone()
        norm_sum = float(delta.float().norm(dim=-1).sum())
        hidden_norm_sum = float(original.float().norm(dim=-1).sum())
        if is_prefill:
            self.prefill_delta_norm_sum += norm_sum
            self.prefill_hidden_norm_sum += hidden_norm_sum
            self.prefill_vectors += len(delta)
        else:
            self.decode_delta_norm_sum += norm_sum
            self.decode_hidden_norm_sum += hidden_norm_sum
            self.decode_vectors += len(delta)
        modified = hidden.clone()
        modified[rows, positions] = (original.float() + delta).to(original.dtype)
        self.patched_calls += 1
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    def diagnostics(self) -> dict[str, float | int | None]:
        return {
            "calls": self.calls,
            "patched_calls": self.patched_calls,
            "prefill_vectors": self.prefill_vectors,
            "decode_vectors": self.decode_vectors,
            "mean_prefill_delta_norm": (
                self.prefill_delta_norm_sum / self.prefill_vectors if self.prefill_vectors else None
            ),
            "mean_decode_delta_norm": (
                self.decode_delta_norm_sum / self.decode_vectors if self.decode_vectors else None
            ),
            "mean_prefill_hidden_norm": (
                self.prefill_hidden_norm_sum / self.prefill_vectors if self.prefill_vectors else None
            ),
            "mean_decode_hidden_norm": (
                self.decode_hidden_norm_sum / self.decode_vectors if self.decode_vectors else None
            ),
            "mean_prefill_delta_to_hidden_norm_ratio": (
                self.prefill_delta_norm_sum / self.prefill_hidden_norm_sum
                if self.prefill_hidden_norm_sum
                else None
            ),
            "mean_decode_delta_to_hidden_norm_ratio": (
                self.decode_delta_norm_sum / self.decode_hidden_norm_sum
                if self.decode_hidden_norm_sum
                else None
            ),
        }
