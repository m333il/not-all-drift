from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from interpretability_gepa.config import ProviderConfig
from interpretability_gepa.errors import ReflectorUnavailable
from interpretability_gepa.evaluation import evaluate_examples
from interpretability_gepa.providers import (
    AnthropicProvider,
    Completion,
    FakeProvider,
    OpenAICompatibleProvider,
    UsageTrackingProvider,
)
from interpretability_gepa.schemas import Example


def test_fake_provider_and_evaluation_capture_parse_errors() -> None:
    raw_provider = FakeProvider(['["a"]', "garbage"])
    provider = UsageTrackingProvider(raw_provider)
    examples = [Example("1", "x", "one", ("a",), "1"), Example("2", "x", "two", (), "2")]
    predictions = evaluate_examples(
        provider, examples, condition="seed", instruction="classify", labels=("a",)
    )
    assert predictions[0].parse_ok
    assert not predictions[1].parse_ok
    assert provider.summary()["calls"] == 2


@pytest.mark.parametrize(
    ("chat_template_kwargs", "expected_extra_body"),
    [
        ({}, None),
        (
            {"enable_thinking": False},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
    ],
)
def test_openai_compatible_provider_forwards_configured_temperature(
    monkeypatch: pytest.MonkeyPatch,
    chat_template_kwargs: dict[str, bool],
    expected_extra_body: dict[str, dict[str, bool]] | None,
) -> None:
    requests: list[dict[str, Any]] = []

    class StubCompletions:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="response"), finish_reason="stop"
                    )
                ],
                usage=None,
            )

    class StubOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.chat = SimpleNamespace(completions=StubCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=StubOpenAI))
    config = ProviderConfig(
        kind="vllm",
        model="local-reflector",
        temperature=0.7,
        chat_template_kwargs=chat_template_kwargs,
    )

    provider = OpenAICompatibleProvider(config)
    provider.complete([{"role": "user", "content": "Reflect"}])

    assert requests[0]["temperature"] == 0.7
    if expected_extra_body is None:
        assert "extra_body" not in requests[0]
    else:
        assert requests[0]["extra_body"] == expected_extra_body


def test_provider_temperature_defaults_to_greedy() -> None:
    config = ProviderConfig(kind="fake", model="test")

    assert config.temperature == 0.0


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_provider_temperature_accepts_closed_interval_boundaries(temperature: float) -> None:
    config = ProviderConfig(kind="fake", model="test", temperature=temperature)

    assert config.temperature == temperature


@pytest.mark.parametrize("temperature", [-0.01, 1.01])
def test_provider_temperature_rejects_values_outside_closed_interval(
    temperature: float,
) -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(kind="fake", model="test", temperature=temperature)


def _stub_anthropic(
    monkeypatch: pytest.MonkeyPatch, requests: list[dict[str, Any]], stop_reason: str
) -> None:
    class StubMessages:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            requests.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="response")],
                usage=SimpleNamespace(input_tokens=5, output_tokens=2),
                stop_reason=stop_reason,
            )

    class StubAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.messages = StubMessages()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=StubAnthropic))
    monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "test-key")


def test_anthropic_provider_omits_sampling_parameters_and_sends_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    _stub_anthropic(monkeypatch, requests, "end_turn")
    config = ProviderConfig(
        kind="anthropic",
        model="reflector",
        api_key_env="TEST_ANTHROPIC_API_KEY",
        temperature=0.7,
    )

    provider = AnthropicProvider(config)
    completion = provider.complete(
        [
            {"role": "system", "content": "Improve the instruction."},
            {"role": "user", "content": "Reflect"},
        ]
    )

    # Current Claude models reject temperature/top_p/top_k outright.
    assert "temperature" not in requests[0]
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert not completion.truncated


def test_anthropic_provider_reports_ceiling_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    _stub_anthropic(monkeypatch, requests, "max_tokens")
    config = ProviderConfig(
        kind="anthropic", model="reflector", api_key_env="TEST_ANTHROPIC_API_KEY"
    )

    provider = UsageTrackingProvider(AnthropicProvider(config))
    assert provider.complete([{"role": "user", "content": "Reflect"}]).truncated
    assert provider.summary()["truncated_calls"] == 1


class _DeadProvider:
    """Stands in for a reflection endpoint behind a dropped tunnel."""

    def __init__(self) -> None:
        self.attempts = 0

    def complete(self, messages: Any, *, max_tokens: int = 128) -> Any:
        del messages, max_tokens
        self.attempts += 1
        raise ConnectionError("Connection error.")


def test_a_flaky_call_is_reraised_and_counted_without_aborting() -> None:
    raw_provider = _DeadProvider()
    provider = UsageTrackingProvider(raw_provider, failure_limit=3)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            provider.complete([{"role": "user", "content": "Reflect"}])
    assert provider.summary()["failed_calls"] == 2
    assert provider.summary()["calls"] == 0


def test_a_dead_reflector_aborts_the_run_instead_of_being_swallowed() -> None:
    # GEPA catches Exception around every reflection, so the guard has to raise
    # something outside that hierarchy or the run burns its whole budget and
    # still reports completed.
    raw_provider = _DeadProvider()
    provider = UsageTrackingProvider(raw_provider, failure_limit=3)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            provider.complete([{"role": "user", "content": "Reflect"}])
    with pytest.raises(ReflectorUnavailable):
        provider.complete([{"role": "user", "content": "Reflect"}])
    assert raw_provider.attempts == 3
    assert not isinstance(ReflectorUnavailable("x"), Exception)


def test_a_success_clears_the_consecutive_failure_count() -> None:
    class _FlakyThenFine:
        def __init__(self) -> None:
            self.attempts = 0

        def complete(self, messages: Any, *, max_tokens: int = 128) -> Any:
            del messages, max_tokens
            self.attempts += 1
            if self.attempts != 3:
                raise ConnectionError("Connection error.")
            return Completion("ok", 1, 1)

    provider = UsageTrackingProvider(_FlakyThenFine(), failure_limit=3)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            provider.complete([{"role": "user", "content": "Reflect"}])
    assert provider.complete([{"role": "user", "content": "Reflect"}]).text == "ok"
    # Two more failures must not trip a limit of three: the counter restarted.
    for _ in range(2):
        with pytest.raises(ConnectionError):
            provider.complete([{"role": "user", "content": "Reflect"}])
    assert provider.summary()["failed_calls"] == 4
    assert provider.summary()["calls"] == 1


