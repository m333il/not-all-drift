from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from .config import ProviderConfig
from .errors import ConfigurationError, ReflectorUnavailable


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    truncated: bool = False


class TextProvider(Protocol):
    def complete(
        self, messages: Sequence[dict[str, str]], *, max_tokens: int = 128
    ) -> Completion: ...


class FakeProvider:
    def __init__(self, responses: Sequence[str]):
        self._responses = iter(responses)
        self.calls = 0

    def complete(self, messages: Sequence[dict[str, str]], *, max_tokens: int = 128) -> Completion:
        del messages, max_tokens
        self.calls += 1
        return Completion(next(self._responses))


class UsageTrackingProvider:
    def __init__(self, provider: TextProvider, *, failure_limit: int | None = None):
        self.provider = provider
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.truncated_calls = 0
        self.failed_calls = 0
        self.wall_seconds = 0.0
        self._failure_limit = failure_limit
        self._consecutive_failures = 0

    def complete(self, messages: Sequence[dict[str, str]], *, max_tokens: int = 128) -> Completion:
        started = time.monotonic()
        try:
            result = self.provider.complete(messages, max_tokens=max_tokens)
        except Exception:
            self.wall_seconds += time.monotonic() - started
            self.failed_calls += 1
            self._consecutive_failures += 1
            # Stop after too many consecutive failures.
            limit = self._failure_limit
            if limit is not None and self._consecutive_failures >= limit:
                raise ReflectorUnavailable(
                    f"reflection endpoint failed {self._consecutive_failures} calls in a row"
                ) from None
            raise
        self.wall_seconds += time.monotonic() - started
        self._consecutive_failures = 0
        self.calls += 1
        self.prompt_tokens += result.prompt_tokens or 0
        self.completion_tokens += result.completion_tokens or 0
        self.truncated_calls += int(result.truncated)
        return result

    def summary(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "truncated_calls": self.truncated_calls,
            "failed_calls": self.failed_calls,
            "provider_wall_seconds": self.wall_seconds,
        }


class OpenAICompatibleProvider:
    def __init__(self, config: ProviderConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigurationError("install the 'gepa' extra for provider clients") from exc
        key = os.environ.get(config.api_key_env or "", "not-needed")
        if config.api_key_env and not os.environ.get(config.api_key_env):
            raise ConfigurationError(
                f"missing credential environment variable {config.api_key_env}"
            )
        self._client = OpenAI(
            api_key=key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        self._model = config.model
        self._temperature = config.temperature
        self._extra_body: dict[str, Any] = dict(config.extra_body)
        if config.chat_template_kwargs:
            self._extra_body["chat_template_kwargs"] = dict(config.chat_template_kwargs)

    def complete(self, messages: Sequence[dict[str, str]], *, max_tokens: int = 128) -> Completion:
        typed_messages = cast(Any, list(messages))
        optional = {"extra_body": self._extra_body} if self._extra_body else {}
        result = self._client.chat.completions.create(
            model=self._model,
            messages=typed_messages,
            temperature=self._temperature,
            max_tokens=max_tokens,
            **cast(Any, optional),
        )
        usage = result.usage
        return Completion(
            result.choices[0].message.content or "",
            None if usage is None else usage.prompt_tokens,
            None if usage is None else usage.completion_tokens,
            result.choices[0].finish_reason == "length",
        )


class AnthropicProvider:
    def __init__(self, config: ProviderConfig):
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ConfigurationError("install the 'gepa' extra for Anthropic") from exc
        if not config.api_key_env or not os.environ.get(config.api_key_env):
            raise ConfigurationError("Anthropic provider requires a populated api_key_env")
        self._client = Anthropic(
            api_key=os.environ[config.api_key_env],
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        self._model = config.model
        self._thinking = config.thinking

    def complete(self, messages: Sequence[dict[str, str]], *, max_tokens: int = 128) -> Completion:
        system = "\n".join(x["content"] for x in messages if x["role"] == "system")
        body = cast(Any, [x for x in messages if x["role"] != "system"])
        # Anthropic models reject sampling params with thinking; thinking tokens share max_tokens.
        result = self._client.messages.create(
            model=self._model,
            system=system,
            messages=body,
            max_tokens=max_tokens,
            thinking=cast(Any, {"type": self._thinking}),
        )
        text = "".join(block.text for block in result.content if block.type == "text")
        return Completion(
            text,
            result.usage.input_tokens,
            result.usage.output_tokens,
            result.stop_reason == "max_tokens",
        )


def build_provider(config: ProviderConfig) -> TextProvider:
    if config.kind in {"vllm", "openai"}:
        return OpenAICompatibleProvider(config)
    if config.kind == "anthropic":
        return AnthropicProvider(config)
    if config.kind == "fake":
        return FakeProvider([])
    raise ConfigurationError(f"unsupported provider kind: {config.kind}")
