"""Minimal OpenAI-compatible chat client (vLLM speaks this protocol).

Trimmed port of ``src/auto_labeling/services/llm_client.py`` from the
auto-labeling harness - only the pieces GEPA needs, plus a ``system`` role so
the optimized instruction block lives where the routing measurement expects it.
"""
from __future__ import annotations

import logging
import json
import os
from pathlib import Path
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    model_name: str = ""
    api_key: str = ""
    api_base_url: str = ""
    max_tokens: int = 1024
    temperature: float | None = 0.0
    timeout: float = 300.0
    max_retries: int = 2
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Optional proxy for this client only, e.g. "socks5h://127.0.0.1:1080" for a
    # host without direct access to the reflection API. The task client talks to
    # a local server and is never proxied.
    proxy: str | None = None


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict[str, int | None] = field(default_factory=dict)
    error: str | None = None
    finish_reason: str | None = None
    reasoning: str = ""
    request_id: str | None = None
    request_attempts: int = 1
    retry_events: list[dict] = field(default_factory=list)
    raw_usage: dict = field(default_factory=dict)


class OpenAICompatibleClient:
    """Calls any server implementing ``/v1/chat/completions``."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        base = config.api_base_url.rstrip("/") if config.api_base_url else "https://api.openai.com/v1"
        self.url = f"{base}/chat/completions"
        self.headers: dict[str, str] = {"Content-Type": "application/json"}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        requested_temperature = temperature if temperature is not None else self.config.temperature
        if requested_temperature is not None:
            payload["temperature"] = requested_temperature
        if seed is not None:
            payload["seed"] = seed
        payload.update(self.config.extra_body)

        retry_events = []
        with httpx.Client(timeout=self.config.timeout, proxy=self.config.proxy) as client:
            for attempt in range(self.config.max_retries + 1):
                try:
                    resp = client.post(self.url, headers=self.headers, json=payload)
                except httpx.TransportError as error:
                    if attempt == self.config.max_retries:
                        raise
                    retry_events.append({"attempt": attempt + 1, "error_type": type(error).__name__,
                                         "usage": "unknown; request may have reached provider"})
                    logger.warning("Retrying %s after %s", self.config.model_name, type(error).__name__)
                    time.sleep(2 ** attempt)
                    continue
                if (resp.status_code not in {429, 502, 503, 504}
                        or attempt == self.config.max_retries):
                    break
                retry_events.append({"attempt": attempt + 1, "http_status": resp.status_code})
                time.sleep(2 ** attempt)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code} from {self.url}: {resp.text[:500]}")
            data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]
        raw = msg.get("content") or ""
        text = re.sub(r"<think>.*?(?:</think>|$)", "", raw, flags=re.DOTALL).strip()

        if choice.get("finish_reason") == "length":
            logger.warning("Truncated (finish_reason=length); raise max_tokens for %s",
                           self.config.model_name)

        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", self.config.model_name),
            usage={
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            },
            finish_reason=choice.get("finish_reason"),
            reasoning=msg.get("reasoning") or "",
            request_id=data.get("id"),
            request_attempts=attempt + 1,
            retry_events=retry_events,
            raw_usage=usage,
        )


class FileQueueLLMClient:
    def __init__(self, model_name: str, queue_dir: Path, timeout: float = 900.0) -> None:
        self.model_name = model_name
        self.queue_dir = queue_dir
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> LLMResponse:
        if system is not None or temperature is not None or seed is not None:
            raise ValueError("Reflection queue accepts only the frozen reflection request")
        deadline = time.monotonic() + self.timeout
        while not (self.queue_dir / "ready.json").is_file():
            if (self.queue_dir / "stopped.json").is_file():
                raise RuntimeError("Reflection queue worker stopped before the request")
            if time.monotonic() >= deadline:
                raise TimeoutError("Reflection queue worker did not become ready")
            time.sleep(1)

        request_id = uuid.uuid4().hex
        request_path = self.queue_dir / f"request-{request_id}.json"
        response_path = self.queue_dir / f"response-{request_id}.json"
        temporary = self.queue_dir / f".{request_path.name}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps({
            "id": request_id,
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }))
        os.replace(temporary, request_path)
        while not response_path.is_file():
            if (self.queue_dir / "stopped.json").is_file():
                raise RuntimeError("Reflection queue worker stopped during the request")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Reflection queue timed out for request {request_id}")
            time.sleep(1)
        payload = json.loads(response_path.read_text())
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        return LLMResponse(**payload["response"])
