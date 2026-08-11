"""Step 10: artifact contract tests for memory-document and stream schemas.

Uses a referencing Registry so stream/memory schemas can share node/relation
definitions with the HTTP contract instead of duplicating them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

CONTRACTS = Path(__file__).parents[1] / "contracts"


def _registry() -> Registry:
    http = json.loads((CONTRACTS / "http.schema.json").read_text(encoding="utf-8"))
    registry = Registry().with_resource(
        "http.schema.json", Resource.from_contents(http)
    )
    return registry


def _load(name: str) -> dict[str, Any]:
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


def _validate(document: dict[str, Any], schema_name: str) -> None:
    schema = _load(schema_name)
    validator = Draft202012Validator(schema, registry=_registry())
    errors = list(validator.iter_errors(document))
    if errors:
        raise errors[0]


MINIMAL_NODE = {
    "id": 1,
    "node_type": "belief",
    "belief": "a belief",
    "confidence": 0.8,
    "stance": "asserted",
    "role": "user",
    "layer": "io",
}

MINIMAL_RELATION = {
    "id": 1,
    "from_id": 1,
    "to_id": 2,
    "type": "depends_on",
}


def test_minimal_memory_document_validates() -> None:
    document = {
        "schema": "bcg.memory.v2",
        "engine": "unified",
        "run_id": "run-1",
        "mode": "stream",
        "trajectory": [{"role": "user", "content": "hi"}],
        "nodes": [MINIMAL_NODE],
        "beliefs": [MINIMAL_NODE],
        "relations": [MINIMAL_RELATION],
    }
    _validate(document, "memory-document.schema.json")


def test_memory_document_rejects_wrong_schema_marker() -> None:
    document = {
        "schema": "bcg.memory.v1",
        "engine": "unified",
        "run_id": "run-1",
        "mode": "stream",
        "trajectory": [],
        "nodes": [],
        "beliefs": [],
        "relations": [],
    }
    with pytest.raises(Exception, match="bcg.memory.v1"):
        _validate(document, "memory-document.schema.json")


def test_minimal_stream_result_validates() -> None:
    document = {
        "prompt_name": "construct_beliefs",
        "model": "gpt-5.5",
        "item_id": "item-1",
        "generated_at": "2026-08-06T00:00:00+00:00",
        "mode": "stream",
        "trajectory": [{"role": "user", "content": "hi"}],
        "all_nodes": [MINIMAL_NODE],
        "relations": [MINIMAL_RELATION],
        "final": {"n_nodes": 1, "n_beliefs": 1},
    }
    _validate(document, "stream.schema.json")


def test_stream_snapshot_line_reuses_http_snapshot() -> None:
    line = {
        "problem_id": "pid",
        "stage": "turn",
        "finalized": False,
        "generated_at": "2026-08-06T00:00:00+00:00",
        "n_nodes": 1,
        "n_beliefs": 1,
        "n_decisions": 0,
        "nodes": [MINIMAL_NODE],
        "beliefs": [MINIMAL_NODE],
        "decisions": [],
        "relations": [],
    }
    # the shared snapshot def comes from the HTTP contract via cross-file $ref
    _validate(line, "stream.schema.json")
    http_snapshot = _load("http.schema.json")["$defs"]["snapshot"]
    errors = list(
        Draft202012Validator(
            {"$defs": http_snapshot}, registry=_registry()
        ).iter_errors(line)
    )
    assert not errors
