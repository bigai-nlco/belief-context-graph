"""Verify construction thinking controls on the serialized wire, without a model."""
from __future__ import annotations

import json

import httpx
import pytest
from openai import OpenAI

from bcg.construct.unified import llm
from bcg.construct.unified.llm import call_model
from tools.case_study_studio.control import Control


@pytest.mark.parametrize("profile", ["gpt6-luna", "qwen35-9b"])
def test_studio_graph_thinking_request(profile: str) -> None:
    spec = Control.graph_profiles()[profile]
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline-test", "object": "chat.completion", "created": 0, "model": spec["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": '{"beliefs": []}', "reasoning_content": "Offline reasoning.",
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    with OpenAI(api_key="offline-test", base_url=spec["base_url"],
                http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        assert call_model(client, spec["model"], "Extract beliefs.", reasoning_effort=spec["reasoning_effort"],
                          max_tokens=spec["max_tokens"], retries=1) == '{"beliefs": []}'
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == spec["model"]
    assert request["max_tokens"] == (16384 if profile == "qwen35-9b" else 8192)
    if profile == "qwen35-9b":
        assert request["chat_template_kwargs"] == {"enable_thinking": True}
        assert "reasoning_effort" not in request
    else:
        assert request["reasoning_effort"] == "none"
        assert "chat_template_kwargs" not in request


def test_qwen_graph_can_still_disable_thinking() -> None:
    from types import SimpleNamespace

    calls = []

    def create(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert call_model(client, "Qwen3.5-9B", "Extract.", reasoning_effort="none", retries=1) == "{}"
    assert calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "reasoning_effort" not in calls[0]


@pytest.mark.parametrize("model, configured, expected", [
    ("Qwen3.5-9B", 1.0, 1.0),
    ("Qwen3.5-9B", 0.7, 0.7),
    ("Qwen3.5-9B", 0.0, 0.0),
    ("Qwen3.5-9B", None, 0.0),
    ("gpt-6-luna", 1.0, 0.0),
    ("gpt-5.6-luna", 1.0, 1.0),
])
def test_configured_qwen_temperature_reaches_every_graph_request(
    monkeypatch: pytest.MonkeyPatch, model: str, configured: float | None, expected: float,
) -> None:
    requests, prompts = [], []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "temperature-test", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": '{"beliefs": []}',
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    def configured_client(**kwargs: object) -> OpenAI:
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(llm, "OpenAI", configured_client)
    monkeypatch.setattr(llm, "_log_prompt", prompts.append)
    with llm.make_client({"model": model, "api_key": "offline-test", "base_url": "http://local.test/v1",
                          "temperature": configured}) as client:
        for label in ('extract:user', 'extract:assistant', 'extract:tool', 'relations', 'merge'):
            # Existing stage callers still use zero; model configuration must
            # win for Qwen consistently, not just for the initial User turn.
            call_model(client, model, 'Return JSON.', temperature=0, max_tokens=16384,
                       reasoning_effort='low' if 'Qwen' in model else 'none',
                       retries=1, usage_label=label)
    assert len(requests) == len(prompts) == 5
    assert all(request['temperature'] == expected for request in requests)
    assert all(prompt['temperature'] == expected for prompt in prompts)
    for request in requests:
        assert 'thinking_token_budget' not in request
        assert 'thinking_token_budget' not in request.get('chat_template_kwargs', {})
        assert 'top_p' not in request and 'presence_penalty' not in request
        if 'Qwen' in model:
            assert request['chat_template_kwargs'] == {'enable_thinking': True}


def test_configured_temperatures_are_isolated_between_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    calls = []

    def client(**kwargs: object) -> SimpleNamespace:
        def create(**body: object) -> SimpleNamespace:
            calls.append(body)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))], usage=None)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(llm, 'OpenAI', client)
    first = llm.make_client({'api_key': 'unused', 'base_url': 'http://local.test/v1', 'temperature': 1.0})
    second = llm.make_client({'api_key': 'unused', 'base_url': 'http://local.test/v1', 'temperature': 0.7})
    for configured in (first, second, first):
        call_model(configured, 'Qwen3.5-9B', 'Return JSON.', temperature=0, retries=1)
    assert [request['temperature'] for request in calls] == [1.0, 0.7, 1.0]


@pytest.mark.parametrize("json_limit", [768, 1024, 1536, 4096])
@pytest.mark.parametrize("model, effort, budget, expected", [
    ("Qwen3.5-9B", "low", 16384, 16384),
    ("Qwen3.5-9B", "low", 8192, 8192),
    ("Qwen3.5-9B", "none", 16384, None),
    ("gpt-6-luna", "none", 8192, None),
])
def test_compact_extraction_keeps_qwen_thinking_headroom(
    json_limit: int, model: str, effort: str, budget: int, expected: int | None,
) -> None:
    from bcg.construct.unified.extract import _extraction_token_limit

    assert _extraction_token_limit(model, effort, budget, json_limit) == (expected or json_limit)


@pytest.mark.parametrize("stage, json_limit", [("single", 768), ("batch", 768), ("combined", 1536)])
@pytest.mark.parametrize("model, effort, budget", [("Qwen3.5-9B", "low", 16384), ("gpt-6-luna", "none", 8192)])
def test_tool_extraction_paths_send_the_selected_output_limit(
    monkeypatch: pytest.MonkeyPatch, stage: str, json_limit: int, model: str, effort: str, budget: int,
) -> None:
    from bcg.construct.unified import extract

    calls = []

    def respond(*args: object, **kwargs: object) -> str:
        calls.append(kwargs)
        return json.dumps({"beliefs": [], "assistant": {"beliefs": [], "decisions": []}, "items": []})

    monkeypatch.setattr(extract.llm, "call_model", respond)
    content = '<tool_result>' + json.dumps({
        "name": "web_search", "tool_call_id": "call-1",
        "content": "## Search Results\n[1] **Museum**\nThe museum opened in 2024.\nURL: https://example.org/museum\n",
    }) + '</tool_result>'
    common = {"client": None, "model": model, "mode": "excerpt", "max_tokens": budget, "reasoning_effort": effort}
    if stage == "single":
        extract.extract_compact_tool_result_nodes(**common, role="tool", content=content, query="museum opening")
    elif stage == "batch":
        extract.extract_compact_tool_result_nodes_batch(**common, items=[{"content": content, "query": "museum opening"}])
    else:
        extract.extract_assistant_tool_result_nodes_batch(**common, assistant_content="Check the opening date.",
            assistant_sentences=["Check the opening date."], graph_nodes_str="[]",
            items=[{"content": content, "query": "museum opening"}])
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == (budget if "Qwen" in model else json_limit)
