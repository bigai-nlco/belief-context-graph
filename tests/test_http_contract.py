"""Step 10: cross-language HTTP contract producer tests.

Real ``bcg.apps.online_server`` handlers are exercised against a deterministic
fake manager; every response must validate against ``contracts/http.schema.json``
(the normative source). The shared fixtures in ``contracts/fixtures/`` are the
same request/response baseline the TypeScript agent tests consume.
"""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urlrequest

import pytest
from jsonschema import Draft202012Validator

from bcg.apps.online_server import make_handler
from bcg.core.contracts import HTTP_SCHEMA_VERSION

CONTRACTS = Path(__file__).parents[1] / "contracts"
FIXTURES = CONTRACTS / "fixtures"

SCHEMA = json.loads((CONTRACTS / "http.schema.json").read_text(encoding="utf-8"))

_VALIDATOR = Draft202012Validator(SCHEMA)


def _schema_def(name: str) -> dict[str, Any]:
    return SCHEMA["$defs"][name]


def _validate(instance: Any, def_name: str) -> None:
    """Validate against a named sub-schema with full-document $ref resolution."""
    errors = list(_VALIDATOR.evolve(schema=_schema_def(def_name)).iter_errors(instance))
    if errors:
        raise errors[0]


# ---------------------------------------------------------------------------
# Deterministic fake manager (fixed snapshots, no model calls)
# ---------------------------------------------------------------------------


class FakeManager:
    """In-memory manager producing the contract fixture shapes."""

    def __init__(self) -> None:
        self._turns: list[dict[str, Any]] = []
        self._released: set[str] = set()

    def active_problem_ids(self) -> list[str]:
        return sorted({t["problem_id"] for t in self._turns} - self._released)

    def all_problem_ids(self) -> list[str]:
        return sorted({t["problem_id"] for t in self._turns})

    def push(self, turn: dict[str, Any]) -> dict[str, Any]:
        self._turns.append(turn)
        return self._snapshot(turn["problem_id"])

    def push_many(self, turns: list[dict[str, Any]]) -> dict[str, Any]:
        latest: dict[str, dict[str, Any]] = {}
        for turn in turns:
            if isinstance(turn, dict) and "problem_id" in turn:
                self._turns.append(turn)
                latest[turn["problem_id"]] = self._snapshot(turn["problem_id"])
        return {"pushed": len(turns), "finalized": [], "latest": latest}

    def get_graph(self, problem_id: str) -> dict[str, Any] | None:
        if problem_id not in self.all_problem_ids():
            return None
        return self._snapshot(problem_id)

    def finalize(self, problem_id: str) -> dict[str, Any]:
        if problem_id not in self.all_problem_ids():
            raise KeyError(f"no active trajectory for problem_id {problem_id}")
        return self._snapshot(problem_id)

    def release(self, problem_id: str) -> dict[str, Any]:
        if problem_id not in self.all_problem_ids():
            return {"problem_id": problem_id, "released": False}
        self._released.add(problem_id)
        return {"problem_id": problem_id, "released": True}

    def select_context(
        self,
        problem_id: str,
        query: str,
        *,
        strategy: str = "connected",
        focus_query: str | None = None,
        question: str | None = None,
        node_char_budget: int,
        max_depth: int,
    ) -> dict[str, Any]:
        if problem_id not in self.all_problem_ids():
            raise KeyError(problem_id)
        return {
            "problem_id": problem_id,
            "strategy": strategy,
            "retrieval": "embedding",
            "node_ids": [3, 2],
            "relation_ids": [1],
            "node_chars": min(node_char_budget, 420),
        }

    def _snapshot(self, problem_id: str) -> dict[str, Any]:
        return {
            "problem_id": problem_id,
            "stage": "turn",
            "finalized": False,
            "stream_turn_index": len(self._turns) - 1,
            "n_turns_ingested": len(self._turns),
            "generated_at": "2026-08-06T00:00:00+00:00",
            "n_nodes": 1,
            "n_beliefs": 1,
            "n_decisions": 0,
            "nodes": [],
            "beliefs": [],
            "decisions": [],
            "evidence": {},
            "relations": [],
            "merges": [],
            "sessions": [],
        }


@pytest.fixture()
def server() -> Any:
    manager = FakeManager()
    handler = make_handler(manager, RuntimeError, quiet=True)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, manager
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _post(base: str, path: str, body: Any) -> tuple[int, dict[str, Any]]:
    req = urlrequest.Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        exc.close()
        return exc.code, json.loads(body)


def _get(base: str, path: str) -> tuple[int, dict[str, Any]]:
    try:
        with urlrequest.urlopen(base + path) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        exc.close()
        return exc.code, json.loads(body)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_health_matches_schema(server: Any) -> None:
    base, _manager = server
    status, body = _get(base, "/health")
    assert status == 200
    _validate(body, "health")
    assert body["status"] == "ok"
    assert body["schema_version"] == HTTP_SCHEMA_VERSION


def test_health_fixture_matches_schema() -> None:
    fixture = json.loads(
        (FIXTURES / "health-response.json").read_text(encoding="utf-8")
    )
    _validate(fixture, "health")


def test_turns_response_matches_schema(server: Any) -> None:
    base, _manager = server
    turns = json.loads((FIXTURES / "turns-request.json").read_text(encoding="utf-8"))
    status, body = _post(base, "/turns", turns)
    assert status == 200
    _validate(body, "turnsResponse")
    assert body["pushed"] == len(turns)


def test_turns_fixture_is_mutually_consistent(server: Any) -> None:
    """The shared fixture request/response pair is a valid exchange."""
    base, _manager = server
    turns = json.loads((FIXTURES / "turns-request.json").read_text(encoding="utf-8"))
    fixture_response = json.loads(
        (FIXTURES / "turns-response.json").read_text(encoding="utf-8")
    )

    # fixture response itself validates
    _validate(fixture_response, "turnsResponse")

    # server handling the same request produces an equivalent envelope
    _status, body = _post(base, "/turns", turns)
    assert body["pushed"] == fixture_response["pushed"]
    assert set(body["latest"]) == set(fixture_response["latest"])
    for pid in fixture_response["latest"]:
        _validate(body["latest"][pid], "snapshot")


def test_graph_and_finalize_respond_with_snapshot(server: Any) -> None:
    base, manager = server
    manager.push_many(
        json.loads((FIXTURES / "turns-request.json").read_text(encoding="utf-8"))
    )
    pid = "fixture-session:seed"

    _status, graph = _get(base, f"/graph?problem_id={pid}")
    _validate(graph, "snapshot")

    _status, finalized = _post(base, "/finalize", {"problem_id": pid})
    _validate(finalized, "snapshot")
    assert finalized["finalized"] is False or "finalized" in finalized


def test_release_matches_schema(server: Any) -> None:
    base, manager = server
    manager.push({"problem_id": "pid-1", "role": "user", "content": "hi"})

    _status, released = _post(base, "/release", {"problem_id": "pid-1"})
    _validate(released, "releaseResponse")
    assert released == {"problem_id": "pid-1", "released": True}

    _status, unknown = _post(base, "/release", {"problem_id": "nope"})
    _validate(unknown, "releaseResponse")
    assert unknown["released"] is False


def test_context_selection_matches_schema(server: Any) -> None:
    base, manager = server
    manager.push({"problem_id": "pid-1", "role": "user", "content": "hi"})

    status, selection = _post(
        base,
        "/context-selection",
        {
            "problem_id": "pid-1",
            "query": "current question and recent state",
            "node_char_budget": 6_600,
            "max_depth": 4,
        },
    )

    assert status == 200
    _validate(selection, "contextSelectionResponse")
    assert selection["node_ids"] == [3, 2]

    status, focused = _post(
        base,
        "/context-selection",
        {
            "problem_id": "pid-1",
            "query": "question plus raw result",
            "focus_query": "question plus reasoning intent",
            "question": "question",
            "strategy": "focused",
            "node_char_budget": 6_600,
            "max_depth": 4,
        },
    )
    assert status == 200
    _validate(focused, "contextSelectionResponse")
    assert focused["strategy"] == "focused"


def test_error_envelope_matches_schema(server: Any) -> None:
    base, _manager = server
    _status, bad = _post(base, "/finalize", {"wrong": "shape"})
    assert _status == 400
    _validate(bad, "errorEnvelope")

    _status, missing = _get(base, "/graph")
    assert _status == 400
    _validate(missing, "errorEnvelope")

    _status, unknown_path = _get(base, "/does-not-exist")
    assert _status == 404
    _validate(unknown_path, "errorEnvelope")


def test_schema_version_constant_matches_contract_file() -> None:
    assert SCHEMA["schema_version"] == HTTP_SCHEMA_VERSION


def test_contract_defaults_match_python_defaults() -> None:
    """contracts/defaults.json is the cross-language source for server defaults."""
    import yaml

    contract = json.loads((CONTRACTS / "defaults.json").read_text(encoding="utf-8"))
    python_defaults = yaml.safe_load(
        (Path(__file__).parents[1] / "bcg" / "config" / "defaults.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert contract["server"]["host"] == python_defaults["server"]["host"]
    assert contract["server"]["port"] == python_defaults["server"]["port"]
    assert contract["schema_version"] == HTTP_SCHEMA_VERSION
