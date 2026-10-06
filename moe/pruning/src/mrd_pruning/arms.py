"""Arm definitions and loading, with checkpoint provenance recorded up front.

Two asymmetries in the earlier runs are made explicit here instead of being
inherited by accident:

* the base arm carried a hand-written system prompt while the PEFT arms ran
  with no system turn at all, so "base vs arm" also compared "with instruction
  vs without". The policy is a required field, not a default;
* the top-level ``adapter/`` directory of both PEFT arms is byte-identical to
  ``epoch_011``, not the val-selected checkpoint. Loading it silently measures
  a different model than the one the reports name, so an unpinned path has to
  be opted into.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .task import SEED_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

ArmKind = Literal["base", "gepa", "prompt_tuning", "prefix_tuning"]
SystemPolicy = Literal["as_trained", "seed_all", "none_all"]


@dataclass(frozen=True)
class ArmSpec:
    """One arm of the experiment: what to load and what prompt it runs with."""

    name: str
    kind: ArmKind
    adapter_path: Path | None = None
    system_prompt_text: str | None = None
    # The v25 ("user-only") contract has no system message at all: a GEPA arm's
    # optimised instruction leads the *user* turn instead. The text is then held
    # by the prompt renderer rather than by the arm, so demanding it here would
    # reject a correctly configured arm.
    prompt_in_user_turn: bool = False
    allow_unpinned_checkpoint: bool = False
    metadata: dict[str, str] = field(default_factory=dict)

    def system_prompt(self, policy: SystemPolicy) -> str | None:
        """Resolve the system turn under the sweep's policy.

        ``as_trained`` reproduces each arm's training-time condition, which is
        what a faithful replication needs and what makes cross-arm F1 gaps
        partly a prompt artefact. ``seed_all`` / ``none_all`` remove that
        asymmetry in the two available directions.
        """
        if policy == "as_trained":
            return self.system_prompt_text
        if policy == "seed_all":
            return SEED_SYSTEM_PROMPT
        if policy == "none_all":
            return None
        raise ValueError(f"unknown system prompt policy {policy!r}")

    def validate(self) -> None:
        if self.kind in ("prompt_tuning", "prefix_tuning"):
            if self.adapter_path is None:
                raise ValueError(f"arm {self.name!r}: PEFT arm needs adapter_path")
            path = Path(self.adapter_path)
            if not path.exists():
                raise FileNotFoundError(f"arm {self.name!r}: {path} does not exist")
            if _is_unpinned_adapter(path) and not self.allow_unpinned_checkpoint:
                raise ValueError(
                    f"arm {self.name!r}: {path} is the arm's top-level adapter directory, "
                    "which is a copy of the last epoch rather than the val-selected "
                    "checkpoint. Point at checkpoints/epoch_XXX/adapter, or set "
                    "allow_unpinned_checkpoint=True on purpose."
                )
        if self.kind == "gepa" and not self.system_prompt_text \
                and not self.prompt_in_user_turn:
            raise ValueError(
                f"arm {self.name!r}: GEPA arm needs its optimised prompt text, "
                "either as a system turn or, under the v25 contract, in the user "
                "turn (set prompt_in_user_turn)"
            )


def _is_unpinned_adapter(path: Path) -> bool:
    """True for ``arms/<arm>/adapter``, false for ``checkpoints/epoch_002/adapter``.

    Both directories are named ``adapter`` in the published checkpoints repo, and
    only the first one is the ambiguous copy of the last epoch. The epoch is in
    the parent directory's name, so that is what decides.
    """
    if path.name != "adapter":
        return False
    return not path.parent.name.startswith("epoch_")


def sha256_of_dir(path: Path, patterns: tuple[str, ...] = ("*.safetensors", "*.bin", "*.json")) -> dict[str, str]:
    """Hash the files that define an adapter, so a summary pins what ran.

    Names are unreliable here: two epochs can share a path and differ, or share
    content and differ in name. The digest is the identity that matters.
    """
    digests: dict[str, str] = {}
    for pattern in patterns:
        for file in sorted(Path(path).glob(pattern)):
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            digests[file.name] = digest
    if not digests:
        raise FileNotFoundError(f"no adapter files matched {patterns} in {path}")
    return digests


def provenance(spec: ArmSpec) -> dict[str, object]:
    """Everything needed to say later which artefact produced a number."""
    record: dict[str, object] = {
        "name": spec.name,
        "kind": spec.kind,
        "system_prompt_text": spec.system_prompt_text,
        "metadata": dict(spec.metadata),
    }
    if spec.adapter_path is not None:
        path = Path(spec.adapter_path)
        record["adapter_path"] = str(path)
        record["adapter_sha256"] = sha256_of_dir(path)
    return record


def require_on_gpu(model, spec: "ArmSpec") -> None:
    """Refuse a model that `device_map="auto"` has quietly spilled to the host.

    When the card cannot hold the weights, accelerate does not fail - it places
    the overflow on CPU (or disk) and generation still runs, at a speed that
    looks exactly like a hang: `gpt-oss/base` once spent more than four hours
    without finishing its first sixty-four rows, with all weights in host RAM.

    The card's free memory at load time decides this, and on a shared GPU that
    depends on other jobs. Failing here costs one retry; not failing costs hours
    per level and is invisible in the log.
    """
    placement = getattr(model, "hf_device_map", None)
    if not placement:
        return
    offloaded = sorted({str(d) for d in placement.values() if str(d) in ("cpu", "disk")})
    if offloaded:
        where = {mod: str(dev) for mod, dev in placement.items() if str(dev) in ("cpu", "disk")}
        raise RuntimeError(
            f"{spec.name}: model is not only on the GPU: {len(where)} modules "
            f"on {', '.join(offloaded)} (for example {next(iter(where))}). "
            "The weights do not fit on the GPU; CPU execution is much slower "
            "and can appear to stall."
        )



def claim_memory(reserve_mb: int, device: int = 0) -> int:
    """Take the card's memory now, before a five-minute load loses the race.

    The queue picks a card by its free memory and only then starts loading.
    Qwen's sixteen shards take about five minutes, and on a box with seven
    other tenants that is long enough for the memory to be gone by the time the
    last shard lands: measured twice within eleven minutes on 22-09-2026 -
    `qwen/prefix-m200` and `qwen/prompt-m200` both died in `load_shard_file`
    with 41 GB left of the 74 the queue had seen.

    One allocation closes the window. The block is freed immediately, but
    PyTorch's caching allocator keeps the segment, so the driver still counts it
    as ours and the loader draws from it instead of from the card.

    Two things this must not do. It must not be used with
    ``device_map="auto"``: accelerate plans placement from `mem_get_info`, which
    does not see our cache, so it would read the card as full and spill the
    model to the host. And it must not call `empty_cache`, which would hand the
    reservation straight back.

    Returns the megabytes actually held, 0 when nothing was reserved.
    """
    import torch

    if reserve_mb <= 0 or not torch.cuda.is_available():
        return 0
    try:
        block = torch.empty(int(reserve_mb) * 1024 * 1024,
                            dtype=torch.uint8, device=f"cuda:{device}")
    except torch.OutOfMemoryError:
        # Losing here costs one retry and says plainly that the card is gone;
        # losing five minutes later costs the same retry and looks like a bug.
        free = torch.cuda.mem_get_info(device)[0] / 2**20
        raise RuntimeError(
            f"GPU cannot reserve {reserve_mb} MB for weights: only {free:.0f} MB free. "
            "A neighbour used the memory between GPU selection and model loading."
        ) from None
    del block
    logger.info("reserved %d MB on the %d card before loading the weights", reserve_mb, device)
    return int(reserve_mb)


def load_arm(
    spec: ArmSpec,
    *,
    model_id: str,
    revision: str | None = None,
    dtype: str = "bfloat16",
    device_map: str | dict[str, int] = "auto",
    reserve_mb: int = 0,
):
    """Load the backbone and, for PEFT arms, attach the adapter.

    Imports live inside the function so the pure-logic modules of this package
    stay importable (and testable) without transformers or peft installed.
    """
    # Before any import or download: a reservation under `device_map="auto"` is
    # a configuration error, and finding that out after five minutes of loading
    # is the failure this whole mechanism exists to avoid.
    if reserve_mb > 0 and device_map == "auto":
        raise ValueError(
            "reserve_mb with device_map='auto' forbidden: accelerate plans "
            "placement on mem_get_info, which the reserve does not see, and decompose "
            "Specify the map explicitly, device_map={'': 0}."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec.validate()
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # generation with a batch requires left padding

    claim_memory(reserve_mb)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch_dtype, device_map=device_map
    )
    require_on_gpu(model, spec)
    if spec.kind in ("prompt_tuning", "prefix_tuning"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(spec.adapter_path), torch_dtype=torch_dtype)
        logger.info("arm %s: attached %s adapter from %s", spec.name, spec.kind, spec.adapter_path)
    model.eval()
    return model, tokenizer


def default_arms(
    *,
    gepa_prompt_text: str | None = None,
    prompt_tuning_ckpt: Path | None = None,
    prefix_tuning_ckpt: Path | None = None,
) -> list[ArmSpec]:
    """The four arms of the 2x2, with their training-time system prompts."""
    arms = [
        ArmSpec(name="base", kind="base", system_prompt_text=SEED_SYSTEM_PROMPT),
    ]
    if gepa_prompt_text is not None:
        arms.append(ArmSpec(name="gepa", kind="gepa", system_prompt_text=gepa_prompt_text))
    if prompt_tuning_ckpt is not None:
        arms.append(ArmSpec(
            name="prompt_tuning", kind="prompt_tuning",
            adapter_path=Path(prompt_tuning_ckpt), system_prompt_text=None,
        ))
    if prefix_tuning_ckpt is not None:
        arms.append(ArmSpec(
            name="prefix_tuning", kind="prefix_tuning",
            adapter_path=Path(prefix_tuning_ckpt), system_prompt_text=None,
        ))
    return arms
