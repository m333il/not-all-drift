from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .artifacts import ArtifactStore
from .config import load_config
from .orchestration import Stage, expand_jobs

app = typer.Typer(no_args_is_help=True)


@app.command("plan")
def plan(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    stage: Stage = Stage.GEPA,
    dataset: Annotated[list[str] | None, typer.Option("--dataset")] = None,
    model: Annotated[list[str] | None, typer.Option("--model")] = None,
) -> None:
    """Expand a deterministic matrix without running or downloading anything."""
    cfg = load_config(config)
    jobs = expand_jobs(
        cfg,
        stage,
        dataset_ids=None if dataset is None else tuple(dataset),
        model_keys=None if model is None else tuple(model),
    )
    payload = {
        "stage": stage.value,
        "config_hash": cfg.content_hash(),
        "job_count": len(jobs),
        "jobs": [
            {
                "id": job.id,
                "model": job.model_key,
                "dataset": job.dataset_id,
                "seed": job.seed,
                "resource_pool": job.resource_pool,
            }
            for job in jobs
        ],
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("status")
def status(
    root: Annotated[Path, typer.Option("--root", exists=True, file_okay=False)],
) -> None:
    """Summarize hash-addressed job states without mutating artifacts."""
    counts: dict[str, int] = {}
    for path in root.glob("jobs/*/status.json"):
        state = str(json.loads(path.read_text(encoding="utf-8")).get("state", "unknown"))
        counts[state] = counts.get(state, 0) + 1
    typer.echo(json.dumps({"root": str(root), "states": counts}, indent=2, sort_keys=True))


@app.command("claim")
def claim(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    root: Annotated[Path, typer.Option("--root")],
    study_id: str,
    stage: Stage,
    dataset: Annotated[list[str] | None, typer.Option("--dataset")] = None,
    model: Annotated[list[str] | None, typer.Option("--model")] = None,
) -> None:
    """Atomically claim the next incomplete hash-addressed job for one worker."""
    cfg = load_config(config)
    jobs = expand_jobs(
        cfg,
        stage,
        dataset_ids=None if dataset is None else tuple(dataset),
        model_keys=None if model is None else tuple(model),
    )
    store = ArtifactStore(root, study_id=study_id)
    for job in jobs:
        if store.try_claim(job):
            typer.echo(
                json.dumps(
                    {"job_id": job.id, "path": str(store.job_path(job)), "payload": job.payload},
                    indent=2,
                    sort_keys=True,
                )
            )
            return
    raise typer.Exit(4)


__all__ = ["app"]
