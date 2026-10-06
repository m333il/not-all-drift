from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from .artifacts import RunDirectory
from .config import load_config
from .datasets import labels_for_dataset, load_jsonl_split
from .metrics import backend_parity
from .modeling import load_hf_model, model_family_factory
from .prompts import SEED_INSTRUCTIONS, apply_chat_template, render_messages

app = typer.Typer(no_args_is_help=True)


@app.command("architecture")
def architecture(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    model_key: str,
    adapter: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
) -> None:
    """Validate residual hooks, final logits, PEFT positions, and Qwen non-thinking."""
    import torch

    from .activations import forward_with_residuals, verify_final_logits
    from .datasets import labels_for_dataset

    cfg = load_config(config)
    model_cfg = cfg.model(model_key)
    model, tokenizer = load_hf_model(model_cfg)
    if adapter is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise typer.BadParameter("install the models extra for PEFT validation") from exc
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    labels = labels_for_dataset(cfg.dataset.id)
    prompt = apply_chat_template(
        tokenizer,
        render_messages(SEED_INSTRUCTIONS[cfg.dataset.id], labels, "Architecture smoke test."),
        non_thinking=model_cfg.non_thinking,
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    family = model_family_factory(model_cfg.family)
    layers = family.transformer_layers(model)
    with torch.no_grad():
        output, residuals = forward_with_residuals(
            model, **encoded, return_dict=True, use_cache=False
        )
    verify_final_logits(model, residuals[-1], output.logits)
    if residuals[-1].shape[1] != encoded["input_ids"].shape[1]:
        raise RuntimeError("PEFT virtual tokens shifted the residual sequence axis")
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=False,
            temperature=None,
            max_new_tokens=8,
            pad_token_id=tokenizer.pad_token_id,
        )
    continuation = generated[:, encoded["input_ids"].shape[1] :]
    decoded = tokenizer.batch_decode(continuation, skip_special_tokens=False)[0]
    non_thinking_ok = family.output_is_non_thinking(decoded)
    configured_layers_ok = model_cfg.layers is None or len(layers) == model_cfg.layers
    report = {
        "model_key": model_key,
        "adapter": None if adapter is None else str(adapter),
        "layer_count": len(layers),
        "configured_layers_ok": configured_layers_ok,
        "residual_sequence_length": int(residuals[-1].shape[1]),
        "input_sequence_length": int(encoded["input_ids"].shape[1]),
        "final_logits_ok": True,
        "non_thinking_ok": non_thinking_ok,
        "passed": configured_layers_ok and non_thinking_ok,
    }
    with RunDirectory(cfg, "phase0.architecture") as run:
        run.write_json("architecture.json", report)
        typer.echo(run.path)
    if not report["passed"]:
        raise typer.Exit(3)


def _label_rows(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    if "labels" not in frame:
        raise typer.BadParameter("prediction parquet must contain labels")
    return [tuple(str(label) for label in labels) for labels in frame["labels"]]


@app.command("parity")
def parity(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    split_file: Annotated[Path, typer.Option("--split-file", exists=True, dir_okay=False)],
    hf_predictions: Annotated[
        Path, typer.Option("--hf-predictions", exists=True, dir_okay=False)
    ],
    vllm_predictions: Annotated[
        Path, typer.Option("--vllm-predictions", exists=True, dir_okay=False)
    ],
) -> None:
    """Compare already generated HF/vLLM predictions on identical IDs."""
    cfg = load_config(config)
    examples = load_jsonl_split(split_file)
    hf = pd.read_parquet(hf_predictions)
    vllm = pd.read_parquet(vllm_predictions)
    expected_ids = [example.id for example in examples]
    for name, frame in (("HF", hf), ("vLLM", vllm)):
        if frame["example_id"].tolist() != expected_ids:
            raise typer.BadParameter(f"{name} prediction example ordering differs from split")
    labels = labels_for_dataset(cfg.dataset.id)
    passed, difference = backend_parity(
        [example.labels for example in examples],
        _label_rows(hf),
        _label_rows(vllm),
        labels,
    )
    with RunDirectory(cfg, "phase0.parity") as run:
        run.write_json(
            "parity.json",
            {"passed": passed, "absolute_f1_difference": difference, "tolerance": 0.02},
        )
        typer.echo(run.path)
    if not passed:
        raise typer.Exit(3)


__all__ = ["app"]
