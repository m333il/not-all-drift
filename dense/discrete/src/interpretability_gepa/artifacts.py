from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from . import __version__
from .config import ExperimentConfig
from .errors import ArtifactError
from .orchestration import JobSpec


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    for item in paths:
        if path.is_dir():
            digest.update(str(item.relative_to(path)).encode())
            digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def collect_provenance() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("torch", "transformers", "peft", "gepa", "numpy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "package_version": __version__,
        "git": _git_metadata(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": packages,
    }


class RunDirectory(AbstractContextManager["RunDirectory"]):
    def __init__(self, config: ExperimentConfig, command: str, *, now: datetime | None = None):
        stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
        suffix = config.content_hash()[:8]
        command_slug = command.replace(".", "-")
        name = f"{stamp}_{command_slug}_{config.name}_{suffix}_{os.getpid()}"
        self.path = config.output.root / name
        self.config = config
        self.command = command

    def __enter__(self) -> RunDirectory:
        try:
            self.path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ArtifactError(f"run directory already exists: {self.path}") from exc
        (self.path / "logs").mkdir()
        (self.path / "tensorboard").mkdir()
        resolved = yaml.safe_dump(self.config.public_dict(), sort_keys=True, allow_unicode=True)
        (self.path / "config.resolved.yaml").write_text(resolved, encoding="utf-8")
        self.write_json(
            "meta.json",
            {
                **collect_provenance(),
                "command": self.command,
                "config_hash": self.config.content_hash(),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        self._write_status("running")
        return self

    def write_json(self, relative: str | Path, value: Any) -> Path:
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(target)
        return target

    def _write_status(self, state: str, error: str | None = None) -> None:
        payload = {"state": state, "updated_at": datetime.now(UTC).isoformat()}
        if error:
            payload["error"] = error
        self.write_json("status.json", payload)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Literal[False]:
        self._write_status("failed" if exc else "completed", None if exc is None else str(exc))
        return False


class ArtifactStore:
    """Deterministic, resume-safe storage keyed by immutable job IDs."""

    def __init__(self, root: Path, *, study_id: str):
        self.root = root / study_id

    def job_path(self, job: JobSpec) -> Path:
        return self.root / "jobs" / job.id

    @staticmethod
    def _job_dict(job: JobSpec) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "job_id": job.id,
            "stage": job.stage.value,
            "model_key": job.model_key,
            "dataset_id": job.dataset_id,
            "seed": job.seed,
            "resource_pool": job.resource_pool,
            "payload": job.payload,
            "dependencies": list(job.dependencies),
        }

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(path)

    def prepare(self, job: JobSpec) -> Path:
        path = self.job_path(job)
        path.mkdir(parents=True, exist_ok=True)
        expected = self._job_dict(job)
        spec_path = path / "spec.json"
        if spec_path.exists():
            existing = json.loads(spec_path.read_text(encoding="utf-8"))
            if existing != expected:
                raise ArtifactError(f"job ID collision or changed specification: {job.id}")
        else:
            self._atomic_json(spec_path, expected)
        self._atomic_json(
            path / "status.json",
            {"state": "running", "updated_at": datetime.now(UTC).isoformat()},
        )
        return path

    @staticmethod
    def _claim_is_active(claim: Path, stale_after_seconds: float) -> bool:
        try:
            payload = json.loads(claim.read_text(encoding="utf-8"))
            if payload.get("hostname") == platform.node():
                pid = int(payload["pid"])
                try:
                    os.kill(pid, 0)
                    return True
                except (ProcessLookupError, OverflowError):
                    return False
                except PermissionError:
                    return True
            return time.time() - claim.stat().st_mtime <= stale_after_seconds
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def try_claim(self, job: JobSpec, *, stale_after_seconds: float = 86_400) -> bool:
        """Atomically claim an incomplete job across concurrent workers."""
        if stale_after_seconds <= 0:
            raise ValueError("claim TTL must be positive")
        if self.is_complete(job):
            return False
        path = self.job_path(job)
        path.mkdir(parents=True, exist_ok=True)
        claim = path / ".claim"
        guard_descriptor = os.open(path / ".claim.guard", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            try:
                fcntl.flock(guard_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            if self.is_complete(job):
                return False
            if claim.exists() and self._claim_is_active(claim, stale_after_seconds):
                return False
            claim.unlink(missing_ok=True)
            descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "hostname": platform.node(),
                            "pid": os.getpid(),
                            "claimed_at": datetime.now(UTC).isoformat(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            try:
                self.prepare(job)
            except Exception:
                claim.unlink(missing_ok=True)
                raise
        finally:
            fcntl.flock(guard_descriptor, fcntl.LOCK_UN)
            os.close(guard_descriptor)
        return True

    def commit(self, job: JobSpec, outputs: dict[str, Any]) -> Path:
        path = self.job_path(job)
        if not (path / "spec.json").exists():
            raise ArtifactError(f"job was not prepared: {job.id}")
        self._atomic_json(path / "outputs.json", outputs)
        self._atomic_json(
            path / "status.json",
            {"state": "completed", "updated_at": datetime.now(UTC).isoformat()},
        )
        (path / "_SUCCESS").write_text(job.id + "\n", encoding="utf-8")
        (path / ".claim").unlink(missing_ok=True)
        return path

    def fail(self, job: JobSpec, error: str) -> Path:
        path = self.job_path(job)
        self._atomic_json(
            path / "status.json",
            {
                "state": "failed",
                "error": error,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        (path / ".claim").unlink(missing_ok=True)
        return path

    def is_complete(self, job: JobSpec) -> bool:
        path = self.job_path(job)
        marker = path / "_SUCCESS"
        if not marker.exists() or marker.read_text(encoding="utf-8").strip() != job.id:
            return False
        try:
            return json.loads((path / "spec.json").read_text(encoding="utf-8")) == self._job_dict(
                job
            )
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def outputs(self, job: JobSpec) -> dict[str, Any]:
        if not self.is_complete(job):
            raise ArtifactError(f"job is incomplete: {job.id}")
        result = json.loads((self.job_path(job) / "outputs.json").read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ArtifactError(f"job outputs are not a mapping: {job.id}")
        return result
