"""TongGraph HTTP contract, restart reuse, invalidation and interrupted writes."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx
import pytest

from workflows.stateful_react_agent.bcg import pcst
from workflows.stateful_react_agent.bcg.client import GraphError
from workflows.stateful_react_agent.bcg.runtime import BCGMemory, BCGSettings
from workflows.stateful_react_agent.bcg.vector_store import TongGraphVectorStore


class Server:
    def __init__(self):
        self.nodes = {}
        self.external = {}
        self.vectors = {}
        self.indexes = {}
        self.namespaces = set()
        self.encoded = []
        self.fail_vectors = False
        self.fail_storage = False
        self.lock = threading.Lock()

    def handle(self, request):
        with self.lock:
            return self._handle(request)

    def _handle(self, request):
        payload = json.loads(request.content) if request.content and request.content != b"null" else {}
        path = request.url.path
        method = request.method
        if request.url.host == "embedding.test":
            self.encoded.extend(payload["input"])
            return httpx.Response(200, json={"model": payload["model"], "data": [
                {"index": i, "embedding": [3, 0] if "z" in text else [0, 7]}
                for i, text in enumerate(payload["input"])
            ]})
        assert request.url.host == "tonggraph.test"
        assert request.headers["Authorization"] == "Bearer test-only"
        if self.fail_storage:
            return httpx.Response(503, text="PRIVATE SERVER BODY")
        path = path.removeprefix("/graphs/agent_workspace")
        ns = request.url.params.get("logical_graph_id") or payload.get("logical_graph_id")
        if path == "/logical-graphs":
            if ns in self.namespaces:
                return httpx.Response(409, json={"error": {"code": "conflict"}})
            self.namespaces.add(ns)
            return httpx.Response(200, json={"created": True})
        if path == "/vector/indexes":
            if method == "GET":
                return httpx.Response(200, json={"indexes": list(self.indexes.values())})
            self.indexes[payload["name"]] = payload
            return httpx.Response(200, json={"created": True})
        assert ns in self.namespaces
        if path.startswith("/nodes/by-external-id/"):
            external = path.removeprefix("/nodes/by-external-id/")
            return httpx.Response(200, json={"id": self.external.get((ns, external))})
        if path == "/nodes/batch":
            if any((ns, r["external_id"]) in self.external for r in payload["records"]):
                return httpx.Response(409, json={"error": {"code": "conflict"}})
            ids = []
            for record in payload["records"]:
                node_id = len(self.nodes)
                self.nodes[node_id] = {"id": node_id, **record}
                self.external[ns, record["external_id"]] = node_id
                ids.append(node_id)
            return httpx.Response(200, json={"ids": ids})
        if path.startswith("/nodes/"):
            node_id = int(path.rsplit("/", 1)[1])
            if method == "PATCH":
                self.nodes[node_id]["properties"].update(payload["set_properties"])
            return httpx.Response(200, json={"node": self.nodes[node_id]})
        if path.startswith("/vector/"):
            _, _, index, entity = path.split("/")
            if method == "PUT":
                if self.fail_vectors:
                    raise httpx.ReadTimeout("PRIVATE", request=request)
                self.vectors.update({(ns, index, int(k)): v for k, v in payload["vectors"].items()})
                return httpx.Response(200, json={"upserted": len(payload["vectors"])})
            return httpx.Response(200, json={"vector": self.vectors.get((ns, index, int(entity)))})
        raise AssertionError((method, path))


@pytest.fixture
def server(monkeypatch):
    instance = Server()
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(instance.handle),
    ))
    pcst._remote_encoder.cache_clear()
    yield instance
    pcst._remote_encoder.cache_clear()


def settings(**kwargs):
    return replace(BCGSettings(
        embedding_url="http://embedding.test/v1/embeddings", embedding_chunk_bytes=8,
        embedding_store="tonggraph", tonggraph_url="http://tonggraph.test",
        tonggraph_token="test-only",
    ), **kwargs)


def new_encoder(cfg=None):
    pcst._remote_encoder.cache_clear()  # Simulate losing all process-local vectors.
    return pcst._encoder(cfg or settings())


def test_restart_reuses_all_chunk_vectors_without_reencoding_nodes(server):
    text = "a" * 40 + "z"
    first = new_encoder()
    scores, audit = first.score("z", "", [text, "other"])
    assert scores == [1, 0]
    assert audit["persistent_cache"]["written_texts"] == 2
    assert audit["persistent_cache"]["errors"] == []
    assert len(server.vectors) == 7  # Six chunks plus one short node.
    assert {n["properties"]["text"] for n in server.nodes.values() if "BCGRetrievalText" in n["labels"]} == {text, "other"}
    server.encoded.clear()
    second = new_encoder()
    restored, audit = second.score("z", "", [text, "other"])
    assert restored == scores
    assert audit["persistent_cache"]["loaded_texts"] == 2
    assert audit["persistent_cache"]["written_texts"] == 0
    assert server.encoded == ["z"]  # Only the transient query must be encoded again.
    assert len(second.cache[text]) == 6


def test_text_edits_invalidate_only_changed_node_and_confidence_is_not_in_key(server):
    first = new_encoder()
    first.score("q", "", ["old", "stable"])
    server.encoded.clear()
    second = new_encoder()
    _, audit = second.score("q", "", ["new", "stable"])
    assert server.encoded == ["q", "new"]
    assert audit["persistent_cache"]["loaded_texts"] == 1
    assert audit["persistent_cache"]["written_texts"] == 1


@pytest.mark.parametrize("change", [
    {"embedding_revision": "2"}, {"embedding_model": "another-model"},
    {"embedding_chunk_bytes": 16}, {"embedding_url": "http://embedding.test/new"},
])
def test_model_revision_endpoint_and_chunk_config_isolate_indexes(server, change):
    original = new_encoder()
    original.score("q", "", ["node"])
    server.encoded.clear()
    updated = new_encoder(settings(**change))
    _, audit = updated.score("q", "", ["node"])
    assert audit["persistent_cache"]["loaded_texts"] == 0
    assert original.store.index != updated.store.index
    assert server.encoded == ["q", "node"]
    assert len(server.indexes) == 2


def test_partial_write_is_not_published_and_retry_is_idempotent(server):
    encoder = new_encoder()
    server.fail_vectors = True
    _, audit = encoder.score("z", "", ["longbody" * 5])
    assert audit["persistent_cache"]["errors"] == ["tonggraph_unavailable"]
    parent = next(n for n in server.nodes.values() if "BCGRetrievalText" in n["labels"])
    assert "chunk_ids" not in parent["properties"]
    assert not encoder.persisted
    initial_count = len(server.nodes)
    server.fail_vectors = False
    server.encoded.clear()
    _, audit = encoder.score("z", "", ["longbody" * 5])
    assert audit["persistent_cache"]["errors"] == []
    assert len(server.nodes) == initial_count and not server.encoded
    assert audit["persistent_cache"]["written_texts"] == 1
    assert "chunk_ids" in parent["properties"]
    assert len(new_encoder().score("z", "", ["longbody" * 5])[0]) == 1


@pytest.mark.parametrize("corruption", ["missing", "nonfinite", "zero", "manifest"])
def test_missing_or_corrupt_vectors_are_recomputed(server, corruption):
    encoder = new_encoder()
    encoder.score("z", "", ["node"])
    key = next(iter(server.vectors))
    if corruption == "missing":
        del server.vectors[key]
    elif corruption == "nonfinite":
        # JSON cannot carry NaN in this mock response; wrong shape is invalid too.
        server.vectors[key] = ["NaN", 1]
    elif corruption == "zero":
        server.vectors[key] = [0, 0]
    else:
        next(n for n in server.nodes.values() if "BCGRetrievalText" in n["labels"])["properties"]["chunk_ids"] = "broken"
    server.encoded.clear()
    _, audit = new_encoder().score("z", "", ["node"])
    assert server.encoded == ["z", "node"]
    assert audit["persistent_cache"]["written_texts"] == 1


def test_storage_failure_is_audited_and_retried_without_losing_retrieval(server):
    server.fail_storage = True
    encoder = new_encoder()
    scores, audit = encoder.score("z", "", ["z node"])
    assert scores == [1]
    assert audit["persistent_cache"]["errors"] == ["tonggraph_http_503"]
    assert "PRIVATE" not in json.dumps(audit) and "test-only" not in json.dumps(audit)
    server.fail_storage = False
    _, audit = encoder.score("z", "", ["z node"])
    assert audit["persistent_cache"]["written_texts"] == 1
    assert not audit["persistent_cache"]["errors"]


def test_concurrent_writers_reuse_stable_external_ids(server):
    store = new_encoder().store
    other = TongGraphVectorStore(store.url, store.token, store.graph, store.timeout, store.descriptor)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda s: s.save({"same text": [[1, 0], [0, 1]]}), [store, other]))
    assert len(server.nodes) == 3 and len(server.vectors) == 2
    assert store.load(["same text"]) == {"same text": [[1, 0], [0, 1]]}


def test_incompatible_index_is_never_overwritten(server):
    encoder = new_encoder()
    encoder.score("q", "", ["node"])
    server.indexes[encoder.store.index]["dimensions"] = 1024
    with pytest.raises(GraphError, match="index_schema_mismatch"):
        encoder.store.save({"new": [[1, 0]]})
    assert server.indexes[encoder.store.index]["dimensions"] == 1024


@pytest.mark.asyncio
async def test_final_flush_persists_recent_nodes_even_without_online_pcst(server, tmp_path, monkeypatch):
    from tests.test_react_bcg import FakeGraph, history
    monkeypatch.setattr(pcst, "select_prized_tree", lambda *a, **k: pytest.fail("final export must not run PCST"))
    memory = BCGMemory(settings(), tmp_path, "final", backend=FakeGraph())
    messages = history(1)
    await memory.prepare(messages)
    assert not server.encoded
    await memory.flush_final(messages)
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    final = rows[-1]
    assert final["status"] == "ok"
    assert final["embedding"]["persistent_cache"]["written_texts"] == len(memory.snapshot["nodes"])
    assert final["embedding"]["persistent_cache"]["errors"] == []
    await memory.close()


@pytest.mark.parametrize("config", [
    {"embedding_store": "wrong"}, {"embedding_revision": ""},
    {"embedding_store": "tonggraph", "tonggraph_url": ""},
    {"embedding_store": "tonggraph", "tonggraph_url": "http://key@host"},
    {"embedding_store": "tonggraph", "tonggraph_url": "http://host", "embedding_provider": "local"},
    {"embedding_store": "tonggraph", "tonggraph_url": "http://host", "tonggraph_timeout": 0},
])
def test_invalid_store_configuration(config):
    with pytest.raises(ValueError):
        BCGSettings.from_agent({"bcg": config})
