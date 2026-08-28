"""HTTP service and concurrency fault-injection tests."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import pytest

from bcg.apps.online_server import serve
from bcg.construct._shared.session import (
    StreamingTrajectorySession,
    TrajectoryClosedError,
)
from bcg.construct._shared.writers import ArtifactWriter


class FakeManager:
    def __init__(self) -> None:
        self.graphs: dict[str, dict[str, Any]] = {}
        self.input_calls: list[dict[str, Any]] = []

    def active_problem_ids(self) -> list[str]:
        return sorted(self.graphs)

    def all_problem_ids(self) -> list[str]:
        return sorted(self.graphs)

    def get_graph(self, problem_id: str) -> dict[str, Any] | None:
        return self.graphs.get(problem_id)

    def push(self, turn: dict[str, Any]) -> dict[str, Any]:
        content = turn.get("content")
        if content == "closed":
            raise TrajectoryClosedError("trajectory is already finalized")
        if content == "missing":
            raise KeyError("missing trajectory")
        if content == "invalid":
            raise ValueError("invalid turn")
        if content == "boom":
            raise RuntimeError("backend unavailable")
        problem_id = turn.get("problem_id")
        if not problem_id:
            raise ValueError("missing problem_id")
        graph = {
            "problem_id": str(problem_id),
            "content": content,
            "finalized": bool(turn.get("is_trajectory_end")),
        }
        self.graphs[str(problem_id)] = graph
        return graph

    def push_many(self, turns: list[dict[str, Any]]) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        finalized: list[str] = []
        for turn in turns:
            graph = self.push(turn)
            problem_id = graph["problem_id"]
            latest[problem_id] = graph
            if graph["finalized"]:
                finalized.append(problem_id)
        return {"pushed": len(turns), "finalized": finalized, "latest": latest}

    def push_input(
        self,
        data: Any,
        *,
        keep_order: bool,
        item_selector: str | None,
        finalize: bool,
    ) -> dict[str, Any]:
        call = {
            "data": data,
            "keep_order": keep_order,
            "item_selector": item_selector,
            "finalize": finalize,
        }
        self.input_calls.append(call)
        return {"items": 1, **call}

    def finalize(self, problem_id: str) -> dict[str, Any]:
        if problem_id not in self.graphs:
            raise KeyError(problem_id)
        graph = {**self.graphs[problem_id], "finalized": True}
        self.graphs[problem_id] = graph
        return graph

    def release(self, problem_id: str) -> dict[str, Any]:
        return {
            "problem_id": problem_id,
            "released": self.graphs.pop(problem_id, None) is not None,
        }

    def select_context(
        self,
        problem_id: str,
        query: str,
        *,
        node_char_budget: int,
        max_depth: int,
    ) -> dict[str, Any]:
        if problem_id not in self.graphs:
            raise KeyError(problem_id)
        return {
            "problem_id": problem_id,
            "strategy": "connected",
            "retrieval": "embedding",
            "node_ids": [2, 1],
            "relation_ids": [3],
            "node_chars": node_char_budget,
            "max_depth": max_depth,
            "query": query,
        }


@pytest.fixture
def http_service() -> Any:
    manager = FakeManager()
    server = serve(manager, TrajectoryClosedError, "127.0.0.1", 0, quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield manager, server.server_address
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def request(
    address: tuple[str, int],
    method: str,
    path: str,
    *,
    payload: Any | None = None,
    raw: str | None = None,
) -> tuple[int, dict[str, Any]]:
    body = (
        raw
        if raw is not None
        else (json.dumps(payload) if payload is not None else None)
    )
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection = HTTPConnection(*address, timeout=3)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        response_body = response.read().decode("utf-8")
        return response.status, json.loads(response_body)
    finally:
        connection.close()


def test_http_happy_path_and_query_mapping(http_service: Any) -> None:
    manager, address = http_service

    status, health = request(address, "GET", "/health")
    assert status == 200
    assert health == {"status": "ok", "active": [], "all": [], "schema_version": 2}

    status, graph = request(
        address,
        "POST",
        "/turn",
        payload={"problem_id": "p1", "role": "user", "content": "hello"},
    )
    assert status == 200
    assert graph["problem_id"] == "p1"

    status, fetched = request(address, "GET", "/graph?problem_id=p1")
    assert status == 200
    assert fetched == graph

    status, selection = request(
        address,
        "POST",
        "/context-selection",
        payload={
            "problem_id": "p1",
            "query": "which evidence supports the answer?",
            "node_char_budget": 2048,
            "max_depth": 3,
        },
    )
    assert status == 200
    assert selection["node_ids"] == [2, 1]
    assert selection["node_chars"] == 2048
    assert selection["max_depth"] == 3

    status, result = request(
        address,
        "POST",
        "/input?keep_order=yes&finalize=0&item=item-2",
        payload={"trajectory": []},
    )
    assert status == 200
    assert result["keep_order"] is True
    assert result["finalize"] is False
    assert result["item_selector"] == "item-2"
    assert manager.input_calls[-1]["data"] == {"trajectory": []}

    status, finalized = request(
        address, "POST", "/finalize", payload={"problem_id": "p1"}
    )
    assert status == 200
    assert finalized["finalized"] is True

    status, released = request(
        address, "POST", "/release", payload={"problem_id": "p1"}
    )
    assert status == 200
    assert released == {"problem_id": "p1", "released": True}


def test_http_turns_accepts_json_array_and_ndjson(http_service: Any) -> None:
    _manager, address = http_service
    turns = [
        {"problem_id": "array", "content": "one"},
        {"problem_id": "array", "content": "two", "is_trajectory_end": True},
    ]

    status, result = request(address, "POST", "/turns", payload=turns)
    assert status == 200
    assert result["pushed"] == 2
    assert result["finalized"] == ["array"]

    ndjson = "\n".join(
        [
            json.dumps({"problem_id": "ndjson", "content": "one"}),
            json.dumps({"problem_id": "ndjson", "content": "two"}),
        ]
    )
    status, result = request(address, "POST", "/turns", raw=ndjson)
    assert status == 200
    assert result["pushed"] == 2
    assert result["latest"]["ndjson"]["content"] == "two"


@pytest.mark.parametrize(
    ("path", "payload", "raw", "expected_status", "message"),
    [
        ("/turn", None, "{broken", 400, "invalid JSON"),
        ("/turn", ["not", "an", "object"], None, 400, "single JSON object"),
        ("/turn", {"problem_id": "p", "content": "closed"}, None, 409, "finalized"),
        ("/turn", {"problem_id": "p", "content": "missing"}, None, 404, "missing"),
        ("/turn", {"problem_id": "p", "content": "invalid"}, None, 400, "invalid"),
        ("/finalize", {}, None, 400, "problem_id"),
        ("/release", {}, None, 400, "problem_id"),
        ("/unknown", {}, None, 404, "unknown path"),
    ],
)
def test_http_maps_client_and_state_errors(
    http_service: Any,
    path: str,
    payload: Any,
    raw: str | None,
    expected_status: int,
    message: str,
) -> None:
    _manager, address = http_service

    status, body = request(address, "POST", path, payload=payload, raw=raw)

    assert status == expected_status
    assert message in body["error"]


def test_http_backend_error_returns_500_and_server_survives(
    http_service: Any,
) -> None:
    _manager, address = http_service

    status, body = request(
        address,
        "POST",
        "/turn",
        payload={"problem_id": "p", "content": "boom"},
    )
    assert status == 500
    assert body == {"error": "RuntimeError: backend unavailable"}

    status, health = request(address, "GET", "/health")
    assert status == 200
    assert health["status"] == "ok"


def test_threading_http_server_processes_different_ids_concurrently() -> None:
    class ConcurrentManager(FakeManager):
        def __init__(self) -> None:
            super().__init__()
            self.barrier = threading.Barrier(2)
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def push(self, turn: dict[str, Any]) -> dict[str, Any]:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                self.barrier.wait(timeout=2)
                return super().push(turn)
            finally:
                with self.lock:
                    self.active -= 1

    manager = ConcurrentManager()
    server = serve(manager, TrajectoryClosedError, "127.0.0.1", 0, quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:

        def push(problem_id: str) -> tuple[int, dict[str, Any]]:
            return request(
                server.server_address,
                "POST",
                "/turn",
                payload={"problem_id": problem_id, "content": "hello"},
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(push, ["p1", "p2"]))

        assert [status for status, _body in responses] == [200, 200]
        assert manager.max_active == 2
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


@pytest.mark.parametrize("backend", ["unified", "hybrid"])
def test_session_manager_creates_one_session_under_same_id_race(
    backend: str,
    tmp_path: Path,
) -> None:
    if backend == "unified":
        from bcg.construct.unified.online import SessionManager

        manager = SessionManager(
            client=None,
            model="fake",
            embedder=None,
            output_root=tmp_path / backend,
        )
    else:
        from bcg.construct.hybrid.online import SessionManager
        from bcg.construct.hybrid.stream import StreamOptions

        manager = SessionManager(
            client=None,
            model="fake",
            embedder=None,
            options=StreamOptions(),
            output_root=tmp_path / backend,
        )

    barrier = threading.Barrier(16)

    def get_session(_index: int) -> Any:
        barrier.wait(timeout=3)
        return manager.get_session("shared")

    with ThreadPoolExecutor(max_workers=16) as executor:
        sessions = list(executor.map(get_session, range(16)))

    assert len({id(session) for session in sessions}) == 1
    assert manager.all_problem_ids() == ["shared"]


@pytest.mark.parametrize("backend", ["unified", "hybrid"])
def test_push_many_parallelizes_ids_and_preserves_per_id_order(
    backend: str,
) -> None:
    if backend == "unified":
        from bcg.construct.unified.online import SessionManager
    else:
        from bcg.construct.hybrid.online import SessionManager

    barrier = threading.Barrier(2)

    class FakeSession:
        def __init__(self, problem_id: str) -> None:
            self.problem_id = problem_id
            self.received: list[int] = []
            self.lock = threading.RLock()

        @contextmanager
        def exclusive(self) -> Any:
            with self.lock:
                yield self

        def push(self, turn: dict[str, Any]) -> dict[str, Any]:
            if not self.received:
                barrier.wait(timeout=2)
            self.received.append(turn["sequence"])
            return {
                "problem_id": self.problem_id,
                "finalized": bool(turn.get("is_trajectory_end")),
            }

    class FakeMultiplexer:
        def __init__(self) -> None:
            self.sessions: dict[str, FakeSession] = {}

        def get_session(self, problem_id: str) -> FakeSession:
            return self.sessions.setdefault(problem_id, FakeSession(problem_id))

    manager = FakeMultiplexer()
    turns = [
        {"problem_id": "a", "sequence": 0},
        {"problem_id": "b", "sequence": 0},
        {"problem_id": "a", "sequence": 1, "is_trajectory_end": True},
        {"problem_id": "b", "sequence": 1},
    ]

    result = SessionManager.push_many(manager, turns)

    assert result["pushed"] == 4
    assert result["finalized"] == ["a"]
    assert manager.sessions["a"].received == [0, 1]
    assert manager.sessions["b"].received == [0, 1]


def test_streaming_session_batches_only_consecutive_complete_tool_results(
    tmp_path: Path,
) -> None:
    class DummyGraph:
        def snapshot(self, *, extra: dict[str, Any]) -> dict[str, Any]:
            return {**extra, "nodes": [], "relations": [], "merges": []}

    class DummyBuilder:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.graph = DummyGraph()
            self.ingested: list[tuple[str, str]] = []
            self.prepared: list[list[str]] = []

        def ingest_turn(
            self,
            role: str,
            content: str,
            **kwargs: Any,
        ) -> None:
            del kwargs
            self.ingested.append((role, content))

        def prepare_tool_result_batch(self, contents: list[str]) -> int:
            self.prepared.append(list(contents))
            return len(contents)

    session = StreamingTrajectorySession(
        "batch-case",
        client=object(),
        model="unused",
        output_root=tmp_path,
        builder_cls=DummyBuilder,
    )
    session.push_many(
        [
            {"problem_id": "batch-case", "role": "assistant", "content": "calls"},
            {"problem_id": "batch-case", "role": "tool", "content": "result-1"},
            {"problem_id": "batch-case", "role": "tool", "content": "result-2"},
        ]
    )

    builder = session._builder
    assert isinstance(builder, DummyBuilder)
    assert builder.prepared == [["result-1", "result-2"]]
    assert builder.ingested == [
        ("assistant", "calls"),
        ("tool", "result-1"),
        ("tool", "result-2"),
    ]


def test_streaming_session_batches_assistant_with_following_tool_results(
    tmp_path: Path,
) -> None:
    class DummyGraph:
        def snapshot(self, *, extra: dict[str, Any]) -> dict[str, Any]:
            return {**extra, "nodes": [], "relations": [], "merges": []}

    class DummyBuilder:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.graph = DummyGraph()
            self.ingested: list[tuple[str, str]] = []
            self.prepared: list[tuple[str, list[str]]] = []

        def ingest_turn(
            self,
            role: str,
            content: str,
            **kwargs: Any,
        ) -> None:
            del kwargs
            self.ingested.append((role, content))

        def prepare_assistant_tool_result_batch(
            self,
            assistant_content: str,
            tool_contents: list[str],
        ) -> int:
            self.prepared.append((assistant_content, list(tool_contents)))
            return 1 + len(tool_contents)

    session = StreamingTrajectorySession(
        "assistant-tool-batch",
        client=object(),
        model="unused",
        output_root=tmp_path,
        builder_cls=DummyBuilder,
    )
    session.push_many(
        [
            {
                "problem_id": "assistant-tool-batch",
                "role": "assistant",
                "content": "thinking and calls",
            },
            {
                "problem_id": "assistant-tool-batch",
                "role": "tool",
                "content": "result-1",
            },
            {
                "problem_id": "assistant-tool-batch",
                "role": "tool",
                "content": "result-2",
            },
        ]
    )

    builder = session._builder
    assert isinstance(builder, DummyBuilder)
    assert builder.prepared == [("thinking and calls", ["result-1", "result-2"])]
    assert builder.ingested == [
        ("assistant", "thinking and calls"),
        ("tool", "result-1"),
        ("tool", "result-2"),
    ]


def test_artifact_writer_preserves_target_and_cleans_temp_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = ArtifactWriter(tmp_path)
    target = writer.write_json("result.json", {"version": "old"})

    def fail_replace(_source: str, _target: Path) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr("bcg.construct._shared.writers.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated disk failure"):
        writer.write_json("result.json", {"version": "new"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"version": "old"}
    assert not list(tmp_path.glob(".result.json.*.tmp"))
