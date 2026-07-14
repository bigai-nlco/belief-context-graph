from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from bcg.agent.tonggraph_sync import (
    build_edge_records,
    build_node_record,
    infer_logical_graph_id,
    iter_agent_nodes,
    load_graph_payload,
    sync_graph_payload,
    TongGraphSyncError,
)


def test_final_graph_mapping_matches_tonggraph_contract() -> None:
    payload = {
        "generated_at": "2026-06-25T12:52:28+00:00",
        "item_id": "q450",
        "nodes": [
            {
                "id": 0,
                "node_type": "belief",
                "belief": "Places of worship are essential.",
                "stance": "asserted",
                "entities": ["Trump", "places of worship"],
                "source": {"role": "assistant", "turn_index": 2},
                "evidence": [{"text": "evidence sentence", "start": 0, "end": 17}],
                "supporting_excerpts": ["evidence sentence"],
                "confidence": 0.91,
            },
            {"id": 1, "node_type": "decision", "decision": "Search official guidance."},
        ],
        "relations": [{"from_id": 1, "to_id": 0, "type": "depends on", "note": "needs evidence"}],
    }

    logical_graph_id = infer_logical_graph_id(None, payload)
    nodes = iter_agent_nodes(payload)
    records = [
        build_node_record(node, item_id="q450", generated_at=payload["generated_at"], logical_graph_id=logical_graph_id)
        for node in nodes
    ]
    edges = build_edge_records(payload, id_map={"0": 10, "1": 11}, item_id="q450", logical_graph_id=logical_graph_id)

    assert logical_graph_id == "q450"
    assert records[0]["external_id"] == "agent:q450:node:0"
    assert records[0]["labels"] == ["AgentNode", "Belief"]
    assert records[0]["properties"]["text"] == "Places of worship are essential."
    assert json.loads(records[0]["properties"]["entities"]) == ["Trump", "places of worship"]
    assert json.loads(records[0]["properties"]["source"]) == {"role": "assistant", "turn_index": 2}
    assert json.loads(records[0]["properties"]["evidence"]) == [
        {"text": "evidence sentence", "start": 0, "end": 17}
    ]
    assert json.loads(records[0]["properties"]["payload_json"])["stance"] == "asserted"
    assert records[1]["labels"] == ["AgentNode", "Decision"]
    assert edges == [
        {
            "source": 11,
            "target": 10,
            "edge_type": "DEPENDS_ON",
            "properties": {
                "agent_edge_id": "agent:q450:edge:1:DEPENDS_ON:0:0",
                "type": "depends on",
                "note": "needs evidence",
                "item_id": "q450",
                "logical_graph_id": "q450",
                "from_agent_id": 1,
                "to_agent_id": 0,
                "payload_json": json.dumps(
                    {"from_id": 1, "to_id": 0, "type": "depends on", "note": "needs evidence"},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        }
    ]


def test_snapshot_list_input_uses_last_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "u1_54a24790.json"
    path.write_text(
        '[{"beliefs":[{"id":"n1","belief":"old"}]}, {"beliefs":[{"id":"n2","belief":"new"}]}]',
        encoding="utf-8",
    )

    payload = load_graph_payload(path)
    nodes = iter_agent_nodes(payload)

    assert infer_logical_graph_id(path, payload) == "u1_54a24790"
    assert len(nodes) == 1
    assert nodes[0]["id"] == "n2"
    assert nodes[0]["node_type"] == "belief"


def test_logical_graph_id_is_sanitized_for_server_names() -> None:
    assert infer_logical_graph_id(None, {}, "gpqa_smoke_0:0:5631e825") == "gpqa_smoke_0_0_5631e825"


class _FakeTongGraphClient:
    nodes: dict[int, dict]
    node_by_external: dict[str, int]
    edges: dict[int, dict]
    edge_by_agent_id: dict[str, int]
    next_node_id: int
    next_edge_id: int

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    @classmethod
    def reset(cls) -> None:
        cls.nodes = {}
        cls.node_by_external = {}
        cls.edges = {}
        cls.edge_by_agent_id = {}
        cls.next_node_id = 10
        cls.next_edge_id = 100

    def create_logical_graph(self, *_args, **_kwargs) -> None:
        return None

    def get_node_id(self, _graph: str, external_id: str, _logical_graph_id: str) -> int | None:
        return self.node_by_external.get(external_id)

    def add_nodes(self, _graph: str, _logical_graph_id: str, records: list[dict]) -> list[int]:
        ids: list[int] = []
        for record in records:
            node_id = self.next_node_id
            type(self).next_node_id += 1
            self.node_by_external[record["external_id"]] = node_id
            self.nodes[node_id] = {
                "id": node_id,
                "external_id": record["external_id"],
                "labels": list(record.get("labels") or []),
                "properties": dict(record.get("properties") or {}),
            }
            ids.append(node_id)
        return ids

    def get_node(self, _graph: str, node_id: int, _logical_graph_id: str) -> dict:
        return {
            **self.nodes[node_id],
            "labels": list(self.nodes[node_id]["labels"]),
            "properties": dict(self.nodes[node_id]["properties"]),
        }

    def node_ids(self, _graph: str, _logical_graph_id: str) -> list[int]:
        return sorted(self.nodes)

    def update_node(
        self,
        _graph: str,
        node_id: int,
        _logical_graph_id: str,
        *,
        add_labels: list[str] | None = None,
        remove_labels: list[str] | None = None,
        set_properties: dict | None = None,
        remove_properties: list[str] | None = None,
    ) -> dict:
        node = self.nodes[node_id]
        labels = set(node["labels"])
        labels.update(add_labels or [])
        labels.difference_update(remove_labels or [])
        node["labels"] = sorted(labels)
        for key in remove_properties or []:
            node["properties"].pop(key, None)
        node["properties"].update(set_properties or {})
        return self.get_node(_graph, node_id, _logical_graph_id)

    def delete_node(self, _graph: str, node_id: int, _logical_graph_id: str) -> bool:
        if any(edge["source"] == node_id or edge["target"] == node_id for edge in self.edges.values()):
            raise AssertionError(f"node {node_id} still has edges")
        node = self.nodes.pop(node_id)
        self.node_by_external.pop(node["external_id"], None)
        return True

    def edge_ids_with_property(
        self,
        _graph: str,
        _logical_graph_id: str,
        *,
        key: str,
        value,
    ) -> list[int]:
        if key != "agent_edge_id":
            return []
        edge_id = self.edge_by_agent_id.get(str(value))
        return [] if edge_id is None else [edge_id]

    def add_edges(self, _graph: str, _logical_graph_id: str, records: list[dict]) -> list[int]:
        ids: list[int] = []
        for record in records:
            edge_id = self.next_edge_id
            type(self).next_edge_id += 1
            props = dict(record.get("properties") or {})
            self.edge_by_agent_id[str(props.get("agent_edge_id"))] = edge_id
            self.edges[edge_id] = {
                "id": edge_id,
                "source": record["source"],
                "target": record["target"],
                "edge_type": record["edge_type"],
                "properties": props,
            }
            ids.append(edge_id)
        return ids

    def get_edge(self, _graph: str, edge_id: int, _logical_graph_id: str) -> dict:
        return {
            **self.edges[edge_id],
            "properties": dict(self.edges[edge_id]["properties"]),
        }

    def edge_ids(self, _graph: str, _logical_graph_id: str) -> list[int]:
        return sorted(self.edges)

    def update_edge(
        self,
        _graph: str,
        edge_id: int,
        _logical_graph_id: str,
        *,
        set_properties: dict | None = None,
        remove_properties: list[str] | None = None,
    ) -> dict:
        edge = self.edges[edge_id]
        for key in remove_properties or []:
            edge["properties"].pop(key, None)
        edge["properties"].update(set_properties or {})
        return self.get_edge(_graph, edge_id, _logical_graph_id)

    def delete_edge(self, _graph: str, edge_id: int, _logical_graph_id: str) -> bool:
        edge = self.edges.pop(edge_id)
        agent_edge_id = str(edge["properties"].get("agent_edge_id") or "")
        if self.edge_by_agent_id.get(agent_edge_id) == edge_id:
            self.edge_by_agent_id.pop(agent_edge_id, None)
        return True

    def create_fulltext_index(self, *_args, **_kwargs) -> None:
        return None


def test_sync_updates_existing_records_and_verifies_readback(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeTongGraphClient.reset()
    monkeypatch.setattr("bcg.agent.tonggraph_sync.TongGraphHTTPClient", _FakeTongGraphClient)

    first = {
        "item_id": "q1",
        "beliefs": [
            {
                "id": "n1",
                "belief": "old belief",
                "stance": "asserted",
                "source": {"turn_index": 1},
                "evidence": [{"text": "old evidence"}],
            }
        ],
    }
    second = {
        "item_id": "q1",
        "beliefs": [
            {
                "id": "n1",
                "belief": "new belief",
                "evidence": [{"text": "new evidence"}],
            }
        ],
    }

    created = sync_graph_payload(
        first,
        base_url="http://tonggraph",
        token="token",
        logical_graph_id="q1",
        text_index=None,
        verify_readback=True,
    )
    updated = sync_graph_payload(
        second,
        base_url="http://tonggraph",
        token="token",
        logical_graph_id="q1",
        text_index=None,
        verify_readback=True,
    )

    assert created.nodes_created == 1
    assert created.nodes_deleted == 0
    assert created.edges_deleted == 0
    assert created.readback_verified is True
    assert updated.nodes_created == 0
    assert updated.nodes_reused == 1
    assert updated.nodes_deleted == 0
    assert updated.edges_deleted == 0
    assert updated.readback_verified is True

    node = _FakeTongGraphClient.nodes[10]
    assert node["properties"]["belief"] == "new belief"
    assert json.loads(node["properties"]["evidence"]) == [{"text": "new evidence"}]
    assert json.loads(node["properties"]["source"]) == {}
    assert "stance" not in node["properties"]
    assert json.loads(node["properties"]["payload_json"]) == {
        "belief": "new belief",
        "evidence": [{"text": "new evidence"}],
        "id": "n1",
        "node_type": "belief",
    }


def test_sync_replaces_latest_state_and_deletes_stale_nodes_and_edges(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _FakeTongGraphClient.reset()
    monkeypatch.setattr("bcg.agent.tonggraph_sync.TongGraphHTTPClient", _FakeTongGraphClient)
    caplog.set_level(logging.INFO, logger="bcg.agent.tonggraph_sync")

    first = {
        "item_id": "real-question-1",
        "nodes": [
            {"id": "claim", "node_type": "belief", "belief": "The claim is true."},
            {"id": "search", "node_type": "decision", "decision": "Search official sources."},
        ],
        "relations": [
            {"from_id": "search", "to_id": "claim", "type": "supports"},
        ],
    }
    second = {
        "item_id": "real-question-1",
        "nodes": [
            {"id": "claim", "node_type": "decision", "decision": "The claim needs revision."},
        ],
        "relations": [],
    }

    sync_graph_payload(
        first,
        base_url="http://tonggraph",
        token="token",
        logical_graph_id="real-question-1",
        text_index=None,
        verify_readback=True,
    )
    replaced = sync_graph_payload(
        second,
        base_url="http://tonggraph",
        token="token",
        logical_graph_id="real-question-1",
        text_index=None,
        verify_readback=True,
    )

    assert replaced.nodes_created == 0
    assert replaced.nodes_reused == 1
    assert replaced.nodes_deleted == 1
    assert replaced.edges_deleted == 1
    assert replaced.readback_verified is True
    assert replaced.readback_mismatches == []
    assert list(_FakeTongGraphClient.nodes) == [10]
    assert _FakeTongGraphClient.edges == {}
    assert _FakeTongGraphClient.nodes[10]["labels"] == ["AgentNode", "Decision"]
    assert "belief" not in _FakeTongGraphClient.nodes[10]["properties"]
    assert _FakeTongGraphClient.nodes[10]["properties"]["decision"] == "The claim needs revision."
    assert "Removing stale records" in caplog.text
    assert "Full-graph readback verified" in caplog.text


def test_full_graph_readback_rejects_unexpected_records(monkeypatch: pytest.MonkeyPatch) -> None:
    class RefusesDeletionClient(_FakeTongGraphClient):
        def delete_node(self, _graph: str, node_id: int, _logical_graph_id: str) -> bool:
            return True

    RefusesDeletionClient.reset()
    monkeypatch.setattr("bcg.agent.tonggraph_sync.TongGraphHTTPClient", RefusesDeletionClient)

    sync_graph_payload(
        {"item_id": "q1", "beliefs": [{"id": "old", "belief": "stale"}]},
        base_url="http://tonggraph",
        token="token",
        logical_graph_id="q1",
        text_index=None,
        verify_readback=True,
    )

    with pytest.raises(TongGraphSyncError, match="unexpected node ids"):
        sync_graph_payload(
            {"item_id": "q1", "beliefs": []},
            base_url="http://tonggraph",
            token="token",
            logical_graph_id="q1",
            text_index=None,
            verify_readback=True,
        )


def test_sync_readback_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class CorruptReadbackClient(_FakeTongGraphClient):
        def get_node(self, graph: str, node_id: int, logical_graph_id: str) -> dict:
            node = super().get_node(graph, node_id, logical_graph_id)
            node["properties"]["belief"] = "corrupted"
            return node

    CorruptReadbackClient.reset()
    monkeypatch.setattr("bcg.agent.tonggraph_sync.TongGraphHTTPClient", CorruptReadbackClient)

    with pytest.raises(TongGraphSyncError):
        sync_graph_payload(
            {"item_id": "q1", "beliefs": [{"id": "n1", "belief": "expected"}]},
            base_url="http://tonggraph",
            token="token",
            logical_graph_id="q1",
            text_index=None,
            verify_readback=True,
        )
