from __future__ import annotations

import json
from pathlib import Path

from bcg.agent.config import AgentRolloutConfig
from bcg.agent.runner import _resolve_tools
from bcg.agent.tools.serper_search import (
    SerperSearchTool,
    resolve_serper_api_key,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._payload


def test_serper_search_posts_and_returns_structured_evidence() -> None:
    captured: dict = {}

    def fake_open(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "answerBox": {
                    "title": "Official answer",
                    "answer": "BrowseComp contains 1,266 questions.",
                    "link": "https://example.test/official",
                },
                "organic": [
                    {
                        "title": "BrowseComp paper",
                        "link": "https://example.test/paper",
                        "snippet": "A difficult benchmark for browsing agents.",
                    },
                    {
                        "title": "Second source",
                        "link": "https://example.test/second",
                        "snippet": "An independent summary.",
                    },
                ],
            }
        )

    tool = SerperSearchTool(
        api_key="test-secret",
        endpoint="https://serper.test/search",
        max_results=3,
        timeout=7,
        http_open=fake_open,
    )
    result = tool.forward("BrowseComp benchmark", top_k=3)

    assert result.error is None
    assert captured["url"] == "https://serper.test/search"
    assert captured["headers"]["x-api-key"] == "test-secret"
    assert captured["payload"] == {
        "q": "BrowseComp benchmark",
        "num": 3,
        "gl": "us",
        "hl": "en",
    }
    assert captured["timeout"] == 7
    assert result.metadata["num_results"] == 3
    assert result.metadata["evidences"][0]["source_type"] == "answer_box"
    assert result.metadata["evidences"][1]["url"] == "https://example.test/paper"
    assert "URL: https://example.test/paper" in result.metadata["evidences"][1]["text"]
    assert "test-secret" not in str(result.output)
    assert "test-secret" not in json.dumps(result.metadata)


def test_serper_key_environment_resolution_and_missing_key_error(monkeypatch) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "from-root-env")
    assert resolve_serper_api_key() == "from-root-env"
    assert SerperSearchTool().configured

    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    missing = SerperSearchTool()
    result = missing.forward("query")
    assert result.error is not None
    assert "SERPER_API_KEY is not configured" in result.error


def test_runner_resolves_serper_with_optional_file_reader(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    cfg = AgentRolloutConfig(
        model="test-model",
        tools=["serper_search", "read_file"],
        file_tool_root=str(tmp_path / "workspace"),
        retrieval_max_results=4,
    )

    resolved = _resolve_tools(cfg)
    assert set(resolved["tool_map"]) == {"serper_search", "read_file"}
    search = resolved["tool_map"]["serper_search"]()
    assert search.configured
    assert search.max_results == 4
    assert "serper_search" in resolved["tool_prompt"]
    assert "exact answer" in resolved["tool_prompt"]
