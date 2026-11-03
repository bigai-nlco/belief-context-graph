"""Offline tests for the default query-PCST path and embedding HTTP contract."""

from __future__ import annotations

import json

import httpx
import pytest

from workflows.stateful_react_agent.bcg import embeddings, pcst
from workflows.stateful_react_agent.bcg.client import GraphError
from workflows.stateful_react_agent.bcg.runtime import BCGMemory, BCGSettings


@pytest.fixture
def mock_service(monkeypatch):
    original = httpx.Client
    pcst._remote_encoder.cache_clear()

    def install(handler):
        def client(**kwargs):
            assert kwargs["trust_env"] is False
            return original(**kwargs, transport=httpx.MockTransport(handler))
        monkeypatch.setattr(embeddings.httpx, "Client", client)
    yield install
    pcst._remote_encoder.cache_clear()


def response_for(request, vector=None):
    payload = json.loads(request.content)
    return httpx.Response(200, json={
        "model": payload["model"],
        "data": [
            {"index": i, "embedding": vector(text) if vector else [3.0, 4.0]}
            for i, text in reversed(list(enumerate(payload["input"])))
        ],
    })


@pytest.mark.parametrize("limit", [4, 12, 240])
def test_utf8_chunks_preserve_every_character_and_respect_limit(limit):
    text = "😀" + "中文与 English words are mixed.\n" * 50 + "结尾证据"
    chunks = embeddings._chunks(text, limit)
    assert "".join(chunks) == text
    assert all(0 < len(part.encode("utf-8")) <= limit for part in chunks)
    assert embeddings._chunks("", limit) == [""]


def test_remote_batch_order_normalization_and_cache(mock_service):
    seen = []
    def handle(request):
        payload = json.loads(request.content)
        seen.append(payload)
        assert str(request.url) == BCGSettings.embedding_url
        assert request.headers["Authorization"] == "Bearer test-only"
        assert payload["model"] == BCGSettings.embedding_model
        assert payload["encoding_format"] == "float"
        return response_for(request, lambda text: [3, 0] if text == "query" else [0, 7])
    mock_service(handle)
    encoder = embeddings.RemoteRelevance(
        BCGSettings.embedding_url, BCGSettings.embedding_model, "test-only", 5, 240,
    )
    texts = ["query", *[f"node-{i}" for i in range(40)]]
    scores, audit = encoder.score("query", "", texts)
    assert scores == [1.0, *[0.0] * 40]
    assert [len(p["input"]) for p in seen] == [32, 9]
    assert audit["new_chunks_encoded"] == 41 and audit["dimensions"] == 2
    assert "test-only" not in json.dumps(audit)
    assert encoder.score("query", "", texts)[1]["new_chunks_encoded"] == 0
    assert len(seen) == 2
    encoder.score("query", "", [*texts, "new-node"])
    assert seen[-1]["input"] == ["new-node"]


def test_remote_long_node_tail_and_focus_both_affect_relevance(mock_service):
    mock_service(lambda r: response_for(r, lambda text: [4, 0] if "z" in text else [0, 9]))
    encoder = embeddings.RemoteRelevance("http://test/embeddings", "test", "", 5, 8)
    scores, audit = encoder.score("z", "", ["a" * 40 + "z"])
    assert scores == [1.0] and audit["node_chunks"] == 6
    scores, _ = encoder.score("z", "a", ["z"])
    assert scores == [0.5]


@pytest.mark.parametrize("data", [
    [],
    [{"index": 0, "embedding": [1, 0]}],
    [{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [0, 1]}],
    [{"index": 0, "embedding": [1, 0]}, {"index": 2, "embedding": [0, 1]}],
    [{"index": 0, "embedding": []}, {"index": 1, "embedding": []}],
    [{"index": 0, "embedding": [0, 0]}, {"index": 1, "embedding": [0, 1]}],
    [{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [1, 0, 0]}],
    [{"index": 0, "embedding": [1, float("nan")]}, {"index": 1, "embedding": [0, 1]}],
    [{"index": 0, "embedding": [True, 0]}, {"index": 1, "embedding": [0, 1]}],
])
def test_invalid_vectors_are_rejected_without_populating_cache(mock_service, data):
    mock_service(lambda r: httpx.Response(200, text=json.dumps({"data": data})))
    encoder = embeddings.RemoteRelevance("http://test/embeddings", "test", "", 5, 240)
    with pytest.raises(GraphError, match="pcst_embedding_"):
        encoder.score("q", "", ["node"])
    assert not encoder.cache


def test_dimension_change_cannot_mix_with_cached_vectors(mock_service):
    mock_service(lambda r: response_for(r))
    encoder = embeddings.RemoteRelevance("http://test/embeddings", "test", "", 5, 240)
    encoder.score("q", "", ["node"])
    mock_service(lambda r: response_for(r, lambda t: [1, 0, 0]))
    with pytest.raises(GraphError, match="dimensions_changed"):
        encoder.score("q", "", ["new-node"])
    assert "new-node" not in encoder.cache


@pytest.mark.asyncio
@pytest.mark.parametrize("team_mode", [False, True])
async def test_default_memory_uses_remote_query_pcst(tmp_path, mock_service, team_mode):
    from tests.test_bcg_pcst import assert_connected
    from tests.test_react_bcg import FakeGraph, history
    from workflows.agent_team.bcg import TeamMemory

    calls = []
    def handle(request):
        calls.append(request)
        assert str(request.url) == BCGSettings.embedding_url
        assert "Authorization" not in request.headers
        return response_for(request)
    mock_service(handle)
    backend = FakeGraph()
    if team_mode:
        team = TeamMemory(BCGSettings(), tmp_path, "default-team", backend_factory=lambda: backend)
        memory = team.member("coordinator", agent="coordinator", question="Find the answer")
    else:
        memory = BCGMemory(BCGSettings(), tmp_path, "default-react", backend=backend)
    messages = history(5)
    projected = await memory.prepare(messages)
    selection = memory.last_event["selector_result"]
    assert calls and not backend.selections
    assert memory.last_event["fallback"] is None
    assert selection["strategy"] == "query_pcst"
    assert selection["embedding"]["provider"] == "openai"
    assert selection["embedding"]["model"] == BCGSettings.embedding_model
    saved = [json.loads(line) for line in (memory.directory / "requests.jsonl").read_text().splitlines()]
    assert selection["node_scores"]
    scored_events = [event for event in saved if "selector_result" in event]
    assert scored_events[-1]["selector_result"]["node_scores"] == selection["node_scores"]
    assert selection["rendered_chars"] <= (4600 if team_mode else 6600)
    assert len(projected) < len(messages)
    assert_connected(memory.snapshot, selection)
    assert await memory.prepare(messages) == projected
    assert len(calls) == 1
    await memory.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "timeout", "unavailable"])
async def test_embedding_failure_preserves_full_history(tmp_path, mock_service, failure):
    from tests.test_react_bcg import FakeGraph, history

    def handle(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("PRIVATE", request=request)
        if failure == "unavailable":
            raise httpx.ConnectError("PRIVATE", request=request)
        return httpx.Response(401, text="PRIVATE RESPONSE BODY")
    mock_service(handle)
    memory = BCGMemory(BCGSettings(), tmp_path, "embedding-failure", backend=FakeGraph())
    messages = history(5)
    assert await memory.prepare(messages) == messages
    assert memory.last_event["fallback"].startswith("pcst_embedding_")
    assert "PRIVATE" not in json.dumps(memory.last_event)
    await memory.close()


def test_default_settings_and_all_builtin_profiles(monkeypatch, tmp_path):
    from workflows.agent_team import profile as team_profile
    from workflows.stateful_react_agent import profile as react_profile

    # Ignore personal .env overrides when verifying shipped defaults.
    monkeypatch.setattr(react_profile, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(team_profile, "_PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("BCG_RETRIEVAL", raising=False)
    monkeypatch.delenv("BCG_RETRIEVAL_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("BCG_RETRIEVAL_EMBEDDING_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_MODEL", "test")
    assert BCGSettings.from_agent({}).retrieval == "query_pcst"
    for loader in (react_profile.load_react_profile, team_profile.load_swarm_profile):
        for name in ("simple", "benchmark", "tui"):
            settings = BCGSettings.from_agent(loader(name)["agent"])
            assert settings.retrieval == "query_pcst"
            assert settings.embedding_provider == "openai"
            assert settings.embedding_url == "http://10.1.100.76:8016/v1/embeddings"


def test_embedding_config_environment_and_profile_precedence(monkeypatch):
    monkeypatch.setenv("BCG_RETRIEVAL_EMBEDDING_URL", "http://test/v1/embeddings")
    monkeypatch.setenv("BCG_RETRIEVAL_EMBEDDING_API_KEY", "test-only")
    settings = BCGSettings.from_agent({"bcg": {"embedding_model": "custom"}})
    assert settings.embedding_url == "http://test/v1/embeddings"
    assert settings.embedding_api_key == "test-only" and settings.embedding_model == "custom"
    assert "test-only" not in repr(settings)
    local = BCGSettings.from_agent({"bcg": {"embedding_provider": "local", "embedding_store": "memory"}})
    assert isinstance(pcst._encoder(local), pcst.LocalRelevance)


@pytest.mark.parametrize("config", [
    {"embedding_url": "ftp://test"}, {"embedding_url": "http://key@test/embeddings"},
    {"embedding_url": "http://test/embeddings?key=secret"}, {"embedding_provider": "typo"},
    {"embedding_timeout": 0}, {"embedding_timeout": float("nan")},
    {"embedding_chunk_bytes": 3},
])
def test_invalid_embedding_settings(config):
    with pytest.raises(ValueError):
        BCGSettings.from_agent({"bcg": config})
