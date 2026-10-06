"""Offline coverage for live-test safeguards and the read-only search adapter."""

from __future__ import annotations

import io
import json

import httpx
import pytest

from bcg.integrations.codex_proxy import search_mcp
from bcg.integrations.codex_proxy.config import ProxyConfig
from bcg.integrations.codex_proxy.context import ContextEngine, Session
from bcg.integrations.codex_proxy.protocol import partition
from bcg.integrations.codex_proxy.server import ModelIoLog, StreamUsage


def test_all_initial_user_messages_are_seeded_before_first_assistant():
    items = [
        {
            "role": "user",
            "content": "<environment_context>Isolated directory</environment_context>",
        },
        {"role": "user", "content": "The actual research question"},
    ]
    plan = partition(items, 2)
    assert [turn["content"] for turn in plan.graph_turns] == [
        item["content"] for item in items
    ]
    assert plan.retained == items


@pytest.mark.parametrize("prior", [False, True])
def test_optional_reasoning_context_follows_replay(prior):
    config = ProxyConfig(
        upstream_base_url="https://model.invalid/v1",
        mode="default",
        replay_reasoning_context=True,
    )
    items = [{"role": "user", "content": "question"}]
    if prior:
        items.append({"role": "assistant", "content": "answer"})
    engine = ContextEngine(config, None)
    out, _ = engine.prepare(
        {"model": "test", "reasoning": {"effort": "low"}, "input": items},
        Session("test"),
    )
    assert out["reasoning"]["context"] == ("all_turns" if prior else "current_turn")
    assert out["reasoning"]["summary"] == "auto"
    assert out["input"] == items


def test_model_io_is_explicit_private_and_bounded(tmp_path):
    logger = ModelIoLog(str(tmp_path / "disabled"), False)
    logger.write("request", {"input": "sensitive"})
    assert not (tmp_path / "disabled").exists()
    logger = ModelIoLog(str(tmp_path / "enabled"), True)
    logger.write("request", {"event": "request", "incoming": {"input": "data"}})
    path = tmp_path / "enabled/model_io/request.jsonl"
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["incoming"]["input"] == "data"


def test_stream_capture_is_opt_in_and_preserves_response():
    response = {
        "output": [{"type": "reasoning", "encrypted_content": "opaque"}],
        "usage": {"input_tokens": 7},
    }
    chunk = (
        "data: "
        + json.dumps({"type": "response.completed", "response": response})
        + "\n\n"
    ).encode()
    normal = StreamUsage(4096)
    normal.feed(chunk)
    assert normal.response is None
    recorder = StreamUsage(4096, capture_response=True)
    recorder.feed(chunk)
    assert recorder.response == response


def test_serper_request_and_results_without_exposing_key(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "fake-secret")
    monkeypatch.setenv("SERPER_ENDPOINT", "http://search.invalid/search")
    seen = []

    def backend(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "title": "Source",
                        "link": "https://source.invalid/page",
                        "snippet": "A supported fact.",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(backend)) as client:
        text = search_mcp.search({"query": "test query", "top_k": 5}, client)
    assert "A supported fact." in text
    assert "fake-secret" not in text
    assert seen[0].headers["x-api-key"] == "fake-secret"
    assert json.loads(seen[0].content) == {
        "q": "test query",
        "num": 5,
        "gl": "us",
        "hl": "en",
    }


def test_mcp_stdio_handshake_list_and_notifications(monkeypatch, capsys):
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
    ]
    monkeypatch.setattr(
        "sys.stdin", io.StringIO("\n".join(json.dumps(m) for m in messages))
    )
    search_mcp.serve()
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [reply["id"] for reply in replies] == [1, 2, 3]
    assert replies[0]["result"]["capabilities"] == {"tools": {}}
    assert [tool["name"] for tool in replies[1]["result"]["tools"]] == ["web_search"]


def test_mcp_budget_blocks_without_http(monkeypatch, capsys):
    monkeypatch.setenv("BCG_SEARCH_MAX_CALLS", "0")
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "web_search", "arguments": {"query": "test"}},
                }
            )
            + "\n"
        ),
    )
    search_mcp.serve()
    reply = json.loads(capsys.readouterr().out)
    assert reply["result"]["isError"] is True
    assert "budget exhausted" in reply["result"]["content"][0]["text"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"record_model_io": True, "audit_dir": None},
        {"max_session_requests": 0},
        {"max_session_requests": True},
    ],
)
def test_invalid_live_debug_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        ProxyConfig(upstream_base_url="http://model.invalid/v1", **kwargs)
