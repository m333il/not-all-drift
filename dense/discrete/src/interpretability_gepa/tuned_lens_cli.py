"""Train and apply per-layer tuned-lens translators on a leased GPU."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from .activations import forward_with_residuals
from .artifacts import RunDirectory
from .config import load_config
from .datasets import labels_for_dataset, load_jsonl_split
from .modeling import (
    load_hf_model,
    model_family_factory,
    readout_logits,
    transformer_layers,
    unwrap_hf_model,
)
from .prompts import SEED_INSTRUCTIONS, apply_chat_template, render_condition_messages
from .tuned_lens import (
    TunedLensConfig,
    build_translators,
    load_translators,
    sample_positions,
    save_translators,
    translator_kl,
)

app = typer.Typer(no_args_is_help=True)


def _instruction_pool(bundle: Path | None, dataset: str) -> list[tuple[str, str]]:
    """(condition, instruction) pairs to draw training prompts from, over all frozen conditions."""
    if bundle is None:
        return [("C_seed", SEED_INSTRUCTIONS[dataset])]
    prompts = sorted((bundle / "prompts").iterdir())
    if not prompts:
        raise typer.BadParameter(f"no prompts under {bundle}/prompts")
    return [(path.name, (path / "instruction.txt").read_text(encoding="utf-8")) for path in prompts]


@app.command("train")
def train(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    split_file: Annotated[Path, typer.Option("--split-file", exists=True, dir_okay=False)],
    model_key: str,
    bundle: Annotated[Path | None, typer.Option("--bundle", exists=True, file_okay=False)] = None,
    max_steps: int = 500,
    batch_size: int = 4,
    positions_per_sequence: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    warmup_steps: int = 20,
    max_length: int = 1024,
    seed: int = 0,
) -> None:
    """Fit affine translators so each layer reads out in the final layer's basis."""
    import torch

    cfg = load_config(config)
    model_cfg = cfg.model(model_key)
    model, tokenizer = load_hf_model(model_cfg)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    adapter = model_family_factory(model_cfg.family)
    final_norm = adapter.final_norm(model)
    device = next(model.parameters()).device
    layers = len(transformer_layers(model)) + 1
    hidden_size = int(model.get_input_embeddings().weight.shape[1])

    labels = labels_for_dataset(cfg.dataset.id)
    examples = load_jsonl_split(split_file)
    pool = _instruction_pool(bundle, cfg.dataset.id)
    rng = np.random.default_rng(seed)

    lens_config = TunedLensConfig(
        hidden_size=hidden_size,
        layers=layers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        positions_per_sequence=positions_per_sequence,
        max_steps=max_steps,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        seed=seed,
    )
    translators = build_translators(lens_config, device=device, dtype=torch.float32)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _batch() -> tuple[Any, Any, Any]:
        picked = rng.integers(0, len(examples), size=batch_size)
        texts = []
        for index in picked:
            condition, instruction = pool[int(rng.integers(0, len(pool)))]
            rendered = render_condition_messages(
                condition, instruction, labels, examples[int(index)].text
            )
            texts.append(
                apply_chat_template(tokenizer, rendered, non_thinking=model_cfg.non_thinking)
            )
        # The chat template already emits <bos>; match the scoring tokenisation.
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output, states = forward_with_residuals(model, **encoded, use_cache=False)
            # Translators are fitted on the base model only.
            if int(states[0].shape[1]) != int(encoded["attention_mask"].shape[1]):
                raise RuntimeError("virtual tokens shifted the residual sequence axis")
            positions = sample_positions(encoded["attention_mask"], positions_per_sequence)
            target = torch.gather(
                output.logits, 1, positions[..., None].expand(-1, -1, output.logits.shape[-1])
            )
            return states, positions, torch.log_softmax(target.float(), dim=-1)

    # SGD with a per-layer learning rate scaled by the initial KL of each layer:
    # Adam keeps moving already converged upper layers.
    probe_states, probe_positions, probe_target = _batch()
    with torch.no_grad():
        initial_kl = [
            float(
                translator_kl(
                    model=model,
                    translator=translators[index],
                    final_norm=final_norm,
                    hidden=hidden.detach().float(),
                    target_log_probs=probe_target,
                    positions=probe_positions,
                )
            )
            for index, hidden in enumerate(probe_states)
        ]
    ceiling = max(initial_kl) or 1.0
    layer_rates = [learning_rate * value / ceiling for value in initial_kl]
    optimizer = torch.optim.SGD(
        [
            {"params": translators[index].parameters(), "lr": layer_rates[index]}
            for index in range(len(translators))
        ],
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    # Linear warmup, then cosine decay.
    warmup = min(warmup_steps, max(max_steps - 1, 0))

    def lr_scale(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(max_steps - warmup, 1)
        return float(0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    history: list[dict[str, Any]] = []
    for step in range(max_steps):
        hidden_states, positions, target_log_probs = _batch()
        optimizer.zero_grad(set_to_none=True)
        per_layer: list[float] = []
        # Free each layer's graph right away.
        for index, hidden in enumerate(hidden_states):
            loss = translator_kl(
                model=model,
                translator=translators[index],
                final_norm=final_norm,
                hidden=hidden.detach().float(),
                target_log_probs=target_log_probs,
                positions=positions,
            )
            loss.backward()
            per_layer.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(translators.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        history.append({"step": step, "mean_kl": float(np.mean(per_layer)), "per_layer": per_layer})
        if step % 25 == 0 or step == max_steps - 1:
            typer.echo(f"TUNED_LENS step={step} mean_kl={np.mean(per_layer):.4f}")

    with RunDirectory(cfg, "tuned_lens.train") as run:
        save_translators(run.path / "translators.pt", translators, lens_config)
        run.write_json("history.json", history)
        run.write_json(
            "summary.json",
            {
                "model_key": model_key,
                "model_id": model_cfg.id,
                "revision": model_cfg.revision,
                "layers": layers,
                "hidden_size": hidden_size,
                "conditions": [name for name, _ in pool],
                "split": split_file.name,
                "initial_per_layer_kl": initial_kl,
                "layer_learning_rates": layer_rates,
                "final_mean_kl": history[-1]["mean_kl"],
                "final_per_layer_kl": history[-1]["per_layer"],
            },
        )
        typer.echo(run.path)


@app.command("score")
def score(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    split_file: Annotated[Path, typer.Option("--split-file", exists=True, dir_okay=False)],
    translators_path: Annotated[Path, typer.Option("--translators", exists=True, dir_okay=False)],
    model_key: str,
    condition: str = "C_seed",
    instruction_file: Annotated[
        Path | None, typer.Option("--instruction-file", exists=True, dir_okay=False)
    ] = None,
    adapter: Annotated[Path | None, typer.Option("--adapter", exists=True, file_okay=False)] = None,
) -> None:
    """Score candidate labels through the trained translators.

    Adapters do not change model weights, so the same translators are used for them.
    """
    import torch

    from .prompts import format_labels

    cfg = load_config(config)
    model_cfg = cfg.model(model_key)
    model, tokenizer = load_hf_model(model_cfg)
    model.eval()
    if adapter is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise typer.BadParameter("install the models extra for adapter conditions") from exc
        model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
        model.eval()
    family_adapter = model_family_factory(model_cfg.family)
    final_norm = family_adapter.final_norm(model)
    device = next(model.parameters()).device
    translators, _ = load_translators(translators_path, device=device, dtype=torch.float32)
    readout_dtype = unwrap_hf_model(model).lm_head.weight.dtype

    labels = labels_for_dataset(cfg.dataset.id)
    examples = load_jsonl_split(split_file)
    instruction = (
        instruction_file.read_text(encoding="utf-8")
        if instruction_file
        else SEED_INSTRUCTIONS[cfg.dataset.id]
    )
    candidates = [format_labels(labels, [label]) for label in labels]

    rows: list[np.ndarray] = []
    counts: list[int] = []
    for example in examples:
        rendered = render_condition_messages(condition, instruction, labels, example.text)
        prompt = apply_chat_template(tokenizer, rendered, non_thinking=model_cfg.non_thinking)
        candidate_rows: list[np.ndarray] = []
        for candidate in candidates:
            complete = prompt + candidate
            encoded = tokenizer(complete, add_special_tokens=False, return_offsets_mapping=True)
            ids = list(encoded.input_ids)
            targets = [
                index
                for index, (_start, end) in enumerate(encoded.offset_mapping)
                if end > len(prompt)
            ]
            if not targets or targets[0] == 0:
                raise typer.BadParameter("candidate has no causally scoreable tokens")
            input_ids = torch.as_tensor([ids], device=device)
            with torch.no_grad():
                _, hidden_states = forward_with_residuals(
                    model, input_ids=input_ids, use_cache=False
                )
                # Prompt tuning shifts hidden states by its virtual tokens.
                offset = int(hidden_states[0].shape[1]) - int(input_ids.shape[1])
                if offset < 0:
                    raise ValueError("residual stream is shorter than the input sequence")
                state_positions = torch.as_tensor([p - 1 + offset for p in targets], device=device)
                gold = torch.as_tensor([ids[p] for p in targets], device=device)
                layer_scores = []
                for index, hidden in enumerate(hidden_states):
                    states = hidden[:, state_positions].float()
                    translated = translators[index](states).to(readout_dtype)
                    normed = final_norm(translated) if final_norm is not None else translated
                    log_probs = torch.log_softmax(readout_logits(model, normed)[0].float(), dim=-1)
                    total = log_probs.gather(-1, gold[:, None]).sum()
                    layer_scores.append(float(total / len(targets)))
            candidate_rows.append(np.asarray(layer_scores, dtype=np.float32))
            counts.append(len(targets))
        rows.append(np.stack(candidate_rows))
    scores = np.stack(rows)

    with RunDirectory(cfg, "tuned_lens.score") as run:
        np.savez_compressed(
            run.path / "tuned_lens.npz",
            scores=scores,
            example_ids=np.asarray([example.id for example in examples]),
            labels=np.asarray(labels),
            candidate_token_counts=np.asarray(counts[: len(candidates)], dtype=np.int32),
        )
        run.write_json(
            "summary.json",
            {
                "shape": list(scores.shape),
                "model_key": model_key,
                "condition": condition,
                "split": split_file.name,
                "translators": str(translators_path),
                "adapter": None if adapter is None else str(adapter.resolve()),
            },
        )
        typer.echo(run.path)


__all__ = ["app"]
