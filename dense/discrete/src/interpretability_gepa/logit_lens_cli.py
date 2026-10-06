from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from .artifacts import RunDirectory
from .config import load_config
from .datasets import labels_for_dataset, load_jsonl_split
from .logit_lens import layerwise_teacher_forced_logprobs
from .modeling import load_hf_model
from .prompts import (
    SEED_INSTRUCTIONS,
    apply_chat_template,
    format_labels,
    render_condition_messages,
)

app = typer.Typer(no_args_is_help=True)


@app.command("score")
def score(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    split_file: Annotated[Path, typer.Option("--split-file", exists=True, dir_okay=False)],
    model_key: str,
    condition: str = "C_seed",
    instruction_file: Annotated[
        Path | None, typer.Option("--instruction-file", exists=True, dir_okay=False)
    ] = None,
    adapter: Annotated[Path | None, typer.Option("--adapter", exists=True, file_okay=False)] = None,
) -> None:
    """Run full-serialization teacher-forced logit lens on a leased remote GPU."""
    cfg = load_config(config)
    model_cfg = cfg.model(model_key)
    model, tokenizer = load_hf_model(model_cfg)
    if adapter is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise typer.BadParameter("install the models extra for adapter conditions") from exc
        model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
        model.eval()
    examples = load_jsonl_split(split_file)
    labels = labels_for_dataset(cfg.dataset.id)
    instruction = (
        instruction_file.read_text(encoding="utf-8")
        if instruction_file
        else SEED_INSTRUCTIONS[cfg.dataset.id]
    )
    prompts = [
        apply_chat_template(
            tokenizer,
            render_condition_messages(condition, instruction, labels, example.text),
            non_thinking=model_cfg.non_thinking,
        )
        for example in examples
    ]
    # Teacher-force the same serialization the parser accepts.
    candidates = [format_labels(labels, [label]) for label in labels]
    scores, candidate_token_counts = layerwise_teacher_forced_logprobs(
        model=model,
        tokenizer=tokenizer,
        family=model_cfg.family,
        prompts=prompts,
        candidate_strings=candidates,
    )
    with RunDirectory(cfg, "logit_lens.score") as run:
        np.savez_compressed(
            run.path / "logit_lens.npz",
            scores=scores,
            example_ids=np.asarray([example.id for example in examples]),
            labels=np.asarray(labels),
            candidate_token_counts=candidate_token_counts,
        )
        run.write_json(
            "summary.json",
            {
                "shape": list(scores.shape),
                "model_key": model_key,
                "condition": condition,
                "split": split_file.name,
                "adapter": None if adapter is None else str(adapter.resolve()),
            },
        )
        typer.echo(run.path)


__all__ = ["app"]
