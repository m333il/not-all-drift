from contextlib import contextmanager
import hashlib
import importlib
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from mrd.peft_training import _length_bucketed_batches, _pad_batch


@contextmanager
def checkpoint_native_attention(model):
    module = importlib.import_module(type(model.get_base_model()).__module__)
    original = module.eager_attention_forward

    def forward(attention, query, key, value, mask, **kwargs):
        if not torch.is_grad_enabled() or not attention.training:
            return original(attention, query, key, value, mask, **kwargs)
        # Cache updates already happened before this pure attention computation.
        return checkpoint(original, attention, query, key, value, mask,
                          use_reentrant=False, **kwargs)

    module.eager_attention_forward = forward
    try:
        yield
    finally:
        module.eager_attention_forward = original


def save_checkpoint(model, optimizer, scheduler, state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=path.name + ".partial-", dir=path.parent))
    model.save_pretrained(staging / "adapter")
    payload = {"trainable": {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad},
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               "state": state}
    torch.save(payload, staging / "training.pt")
    (staging / "COMPLETE").write_text("complete\n")
    staging.rename(path)


def save_adapter_snapshot(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=path.name + ".partial-", dir=path.parent))
    model.save_pretrained(staging / "adapter")
    (staging / "ADAPTER_ONLY").write_text("adapter only\n")
    if path.exists():
        names = ("adapter_config.json", "adapter_model.safetensors")
        same = (path / "ADAPTER_ONLY").is_file()
        for name in names:
            old = path / "adapter" / name
            new = staging / "adapter" / name
            if not old.is_file() or old.stat().st_size != new.stat().st_size:
                same = False
                break
            with old.open("rb") as left, new.open("rb") as right:
                if hashlib.file_digest(left, "sha256").digest() != hashlib.file_digest(right, "sha256").digest():
                    same = False
                    break
        shutil.rmtree(staging)
        if not same:
            raise ValueError(f"Existing adapter snapshot differs: {path}")
    else:
        staging.rename(path)


def restore_checkpoint(model, optimizer, scheduler, path):
    if not (path / "COMPLETE").is_file():
        raise ValueError("Checkpoint is incomplete")
    payload = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != payload["trainable"].keys():
        raise ValueError("Checkpoint trainable parameter set differs")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(payload["trainable"][name])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["state"]


def train_encoded(model, encoded, keys, optimizer, scheduler, out_dir, *, pad_id,
                  epochs, batch_size, accumulation, seed, contract, resume=None, stop_after=None,
                  checkpoint_every=1, extend_from=None, epoch_adapter_only=False):
    if resume and extend_from:
        raise ValueError("Choose resume or a new extension, not both")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    state = {"contract": contract, "epoch": 0, "next_batch": 0, "batches": None,
             "rng": rng.getstate(), "step": 0, "elapsed_seconds": 0.0}
    if resume or extend_from:
        state = restore_checkpoint(model, optimizer, scheduler, resume or extend_from)
        if extend_from:
            parent = state["contract"]
            changed_fields = {"epochs", "total_steps", "schedule", "extension"}
            if ({k: v for k, v in parent.items() if k not in changed_fields}
                    != {k: v for k, v in contract.items() if k not in changed_fields}):
                raise ValueError("Extension may not change data, model or optimizer settings")
            extension = contract["extension"]
            if ("extension" in parent or state["epoch"] != parent["epochs"]
                    or state["step"] != parent["total_steps"] or state["batches"] is not None
                    or state["next_batch"] != 0 or epochs <= state["epoch"]
                    or extension["start_step"] != state["step"]
                    or extension["completed_epochs"] != state["epoch"]):
                raise ValueError("Extension requires the completed parent training boundary")
            # Keep Adam moments and both data/model RNG streams; explicitly restart only LR.
            for group, lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = lr
            scheduler.load_state_dict({**scheduler.state_dict(), "_last_lr": scheduler.base_lrs[:]})
            state["contract"] = contract
        elif state["contract"] != contract:
            raise ValueError("Checkpoint data/model/training contract differs")
        rng.setstate(state["rng"])
        log_path = (Path(extend_from).parent if extend_from else out_dir) / "steps.jsonl"
        events = []
        if log_path.exists():
            with log_path.open() as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        if line.endswith("\n") or stream.read():
                            raise
                        break
                    if event["step"] <= state["step"]:
                        events.append(event)
        if state.get("last_event") and (not events or events[-1]["step"] < state["step"]):
            events.append(state["last_event"])
        if [event["step"] for event in events] != list(range(1, state["step"] + 1)):
            raise ValueError("Committed step log is incomplete")
        with tempfile.NamedTemporaryFile(mode="w", dir=out_dir, prefix="steps.partial-", delete=False) as stream:
            stream.write("".join(json.dumps(event) + "\n" for event in events))
            restored_log = Path(stream.name)
        restored_log.replace(out_dir / "steps.jsonl")
        if extend_from:
            save_checkpoint(model, optimizer, scheduler, state, out_dir / f"step_{state['step']:06d}")
    else:
        save_checkpoint(model, optimizer, scheduler, state, out_dir / "step_000000")
    parameters = [p for p in model.parameters() if p.requires_grad]
    started = time.monotonic()
    elapsed_before = state["elapsed_seconds"]
    with (out_dir / "steps.jsonl").open("a", buffering=1) as log:
        while state["epoch"] < epochs:
            if state["batches"] is None:
                state["batches"] = _length_bucketed_batches(encoded, batch_size, rng)
                state["rng"] = rng.getstate()
            batches = state["batches"]
            model.train()
            while state["next_batch"] < len(batches):
                start = state["next_batch"]
                window = batches[start:start + accumulation]
                n_tokens = sum(sum(label != -100 for label in encoded[i][1][1:]) for batch in window for i in batch)
                optimizer.zero_grad(set_to_none=True)
                weighted_loss = 0.0
                for batch in window:
                    ids, mask, labels = _pad_batch([encoded[i] for i in batch], pad_id, model.device)
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    count = int((labels[:, 1:] != -100).sum())
                    (loss * (count / n_tokens)).backward()
                    weighted_loss += loss.detach().item() * count / n_tokens
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError("Missing or nonfinite adapter gradient")
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0).item()
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state["step"] += 1
                state["next_batch"] += len(window)
                epoch = state["epoch"]
                if state["next_batch"] == len(batches):
                    state["epoch"] += 1
                    state["next_batch"] = 0
                    state["batches"] = None
                state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
                checkpoint = out_dir / f"step_{state['step']:06d}"
                should_save = (state["step"] % checkpoint_every == 0 or state["batches"] is None
                               or (stop_after is not None and state["step"] >= stop_after))
                event = {"step": state["step"], "epoch": epoch, "keys": [keys[i] for batch in window for i in batch],
                         "loss": weighted_loss, "tokens": n_tokens, "lr": lr, "gradient_norm": gradient_norm,
                         "elapsed_seconds": state["elapsed_seconds"], "checkpoint": str(checkpoint) if should_save else None}
                state["last_event"] = event
                if should_save:
                    if (epoch_adapter_only and state["batches"] is None
                            and state["step"] % checkpoint_every != 0
                            and not (stop_after and state["step"] >= stop_after)):
                        save_adapter_snapshot(model, checkpoint)
                    else:
                        save_checkpoint(model, optimizer, scheduler, state, checkpoint)
                log.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
                if stop_after and state["step"] >= stop_after:
                    return state
                if state["batches"] is None:
                    break
    return state
