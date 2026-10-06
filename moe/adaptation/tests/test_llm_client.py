import json
from pathlib import Path
import threading
import time

import httpx
import pytest

from mrd.llm_client import FileQueueLLMClient, LLMConfig, OpenAICompatibleClient


@pytest.mark.parametrize("content,reasoning,expected", [
    (None, "unfinished reasoning", ""),
    ("<think>unfinished reasoning", None, ""),
    ("<think>private intermediate text</think>final", None, "final"),
    ("final", "intermediate", "final"),
])
def test_reasoning_is_never_used_as_final(monkeypatch, content, reasoning, expected):
    sent = {}

    def post(_self, url, *, headers, json):
        sent.update(json)
        return httpx.Response(200, json={
            "model": "openai/gpt-5.6-luna-pro",
            "choices": [{"message": {"content": content, "reasoning": reasoning},
                         "finish_reason": "length"}],
            "usage": {"completion_tokens": 100,
                      "completion_tokens_details": {"reasoning_tokens": 90}},
        })

    monkeypatch.setattr(httpx.Client, "post", post)
    result = OpenAICompatibleClient(LLMConfig(
        model_name="openai/gpt-5.6-luna-pro", temperature=None,
    )).generate("test")
    assert "temperature" not in sent
    assert result.text == expected
    assert result.finish_reason == "length"
    assert result.usage["reasoning_tokens"] == 90


def test_local_task_temperature_is_preserved(monkeypatch):
    sent = {}

    def post(_self, url, *, headers, json):
        sent.update(json)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "yes"}, "finish_reason": "stop"}],
        })

    monkeypatch.setattr(httpx.Client, "post", post)
    OpenAICompatibleClient(LLMConfig()).generate("test")
    assert sent["temperature"] == 0.0


def test_extra_body_is_passed_at_top_level(monkeypatch):
    sent = {}

    def post(_self, url, *, headers, json):
        sent.update(json)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
        })

    monkeypatch.setattr(httpx.Client, "post", post)
    OpenAICompatibleClient(LLMConfig(
        extra_body={"reasoning": {"effort": "medium"}},
    )).generate("test")
    assert sent["reasoning"] == {"effort": "medium"}


@pytest.mark.parametrize("usage", [None, {"completion_tokens": 7, "cost": 0.001,
                                       "prompt_tokens_details": {"cached_tokens": 3}}])
def test_usage_preserves_provider_fields_and_missing_counts(monkeypatch, usage):
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **kw: httpx.Response(200, json={
        "choices": [{"message": {"content": "final"}, "finish_reason": "stop"}], "usage": usage,
    }))
    result = OpenAICompatibleClient(LLMConfig()).generate("test")
    assert result.usage["prompt_tokens"] is None
    assert result.usage["completion_tokens"] == (7 if usage else None)
    assert result.raw_usage == (usage or {})


def test_transient_http_errors_have_bounded_retry(monkeypatch):
    codes = iter([429, 503, 200])
    waits = []

    def post(*args, **kwargs):
        return httpx.Response(next(codes), json={
            "choices": [{"message": {"content": "final"}, "finish_reason": "stop"}],
        })

    monkeypatch.setattr(httpx.Client, "post", post)
    monkeypatch.setattr("mrd.llm_client.time.sleep", waits.append)
    assert OpenAICompatibleClient(LLMConfig()).generate("test").text == "final"
    assert waits == [1, 2]


def test_exhausted_retry_raises(monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return httpx.Response(503, text="unavailable")

    monkeypatch.setattr(httpx.Client, "post", post)
    monkeypatch.setattr("mrd.llm_client.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        OpenAICompatibleClient(LLMConfig()).generate("test")
    assert len(calls) == 3


def test_transport_retry_records_unknown_usage(monkeypatch):
    attempts = []

    def post(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.RemoteProtocolError("incomplete response body")
        return httpx.Response(200, json={"choices": [{"message": {"content": "final"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(httpx.Client, "post", post)
    monkeypatch.setattr("mrd.llm_client.time.sleep", lambda _: None)
    result = OpenAICompatibleClient(LLMConfig()).generate("test")
    assert result.text == "final"
    assert result.request_attempts == 2
    assert result.retry_events == [{"attempt": 1, "error_type": "RemoteProtocolError",
                                    "usage": "unknown; request may have reached provider"}]


def test_file_queue_client(tmp_path: Path):
    (tmp_path / "ready.json").write_text("{}")

    def worker():
        while not (requests := list(tmp_path.glob("request-*.json"))):
            time.sleep(0.01)
        request = json.loads(requests[0].read_text())
        response = tmp_path / requests[0].name.replace("request-", "response-", 1)
        response.write_text(json.dumps({"response": {
            "text": "answer", "model": request["model"], "finish_reason": "stop",
        }}))

    thread = threading.Thread(target=worker)
    thread.start()
    response = FileQueueLLMClient("reflector", tmp_path, timeout=2).generate("prompt")
    thread.join()
    assert response.text == "answer"
    assert not list(tmp_path.glob("request-*.json"))
    assert not list(tmp_path.glob("response-*.json"))
