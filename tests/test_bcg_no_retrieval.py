"""Full-graph ablation contracts; no model, embedding or network calls."""

from __future__ import annotations

import copy
import json

import pytest

from tests.test_react_bcg import FakeGraph, history, step
from workflows.stateful_react_agent.bcg import pcst
from workflows.stateful_react_agent.bcg.client import GraphError
from workflows.stateful_react_agent.bcg.rendering import full_graph_selection, render_graph
from workflows.stateful_react_agent.bcg.runtime import BCGMemory, BCGSettings


def test_named_mode_overrides_retrieval_but_preserves_other_settings(monkeypatch):
    monkeypatch.setenv("BCG_RETRIEVAL", "query_pcst")
    agent = {"bcg": {"retrieval": "focused", "recent_turns": 2, "graph_chars": 6600}}
    baseline = BCGSettings.from_agent(agent)
    full = BCGSettings.from_agent({**agent, "context_mode": "bcg_no_retrieval"})
    assert baseline.retrieval == "focused" and baseline.graph_budget == 6600
    assert full.retrieval == "all" and full.graph_budget is None
    assert {**vars(baseline), "retrieval": "all"} == vars(full)
    # Stale retrieval connection settings cannot block a mode that never uses them.
    full = BCGSettings.from_agent({"context_mode": "bcg_no_retrieval", "bcg": {
        "embedding_url": "not-used", "embedding_store": "tonggraph", "tonggraph_url": "",
    }})
    assert full.graph_budget is None


class FullGraph(FakeGraph):
    def snapshot(self, pid):
        value = super().snapshot(pid)
        for node in value["nodes"]:
            node["belief"] += " Preserve this complete fact, including its qualifiers." * 150
        value["nodes"] += [
            {"id": 900, "decision": "A disconnected final decision", "confidence": 0.0},
            {"id": 901, "belief": "The web_search tool returned no results.", "confidence": 0.1},
        ]
        value["relations"] += [
            {"id": 900, "from_id": 900, "to_id": 901, "type": "contradicts", "note": "recorded contradiction"},
            {"id": 901, "from_id": 901, "to_id": 900, "type": "supplements", "note": "recorded supplement"},
        ]
        return value

    async def select(self, *args, **kwargs):
        raise AssertionError("Full-graph mode must never call the server's selector")


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["v0", "split", "split_confidence"])
async def test_full_graph_bypasses_retrieval_and_budget_without_changing_ingestion(tmp_path, monkeypatch, style):
    def no_selector(*args, **kwargs):
        raise AssertionError("Full-graph mode must never instantiate PCST or embeddings")

    monkeypatch.setattr(pcst, "PCSTSelector", no_selector)
    backend = FullGraph()
    memory = BCGMemory(BCGSettings(retrieval="all"), tmp_path / "all", "all", backend=backend, render_style=style)
    ordinary_backend = FakeGraph()
    ordinary = BCGMemory(BCGSettings(retrieval="focused"), tmp_path / "baseline", "base", backend=ordinary_backend)
    messages = [*history(3), {"role": "user", "content": "Keep this additional requirement"}, *step(3, parallel=3)]
    original = copy.deepcopy(messages)
    projected = await memory.prepare(messages)
    await ordinary.prepare(messages)
    assert [turns for _, turns in backend.pushes] == [turns for _, turns in ordinary_backend.pushes]
    assert memory.acknowledged == ordinary.acknowledged
    assert messages == original
    assert projected[0]["role"] == "system" and projected[0]["content"].startswith("original instructions")
    assert projected[1:] == memory.project(messages, 2).retained[1:]
    assert all(m["role"] != "system" for m in projected[1:])
    assert [m["content"] for m in projected if m["role"] == "user"] == ["Find the answer", "Keep this additional requirement"]
    event = memory.last_event
    assert event["fallback"] is None and event["evicted_steps"] == 2
    assert event["retrieval_strategy"] == "all" and event["graph_chars_budget"] is None
    assert event["graph_chars"] > 6600
    assert event["selected_node_ids"] == sorted(n["id"] for n in memory.snapshot["nodes"])
    assert event["selected_relation_ids"] == sorted(r["id"] for r in memory.snapshot["relations"])
    assert "[B900]" in event["system_memory"] and "[B901]" in event["system_memory"]
    assert "recorded contradiction" in event["system_memory"] and "recorded supplement" in event["system_memory"]
    expected, _, _ = render_graph(memory.snapshot, full_graph_selection(memory.snapshot), 1_000_000, style=style)
    assert event["system_memory"] == memory.context_guide + "\n\n" + expected
    assert await memory.prepare(messages) == projected  # Cached retry does not rewrite the graph.
    assert len(backend.pushes) == 1 and memory._pcst_selector is None
    await memory.prepare([*messages, *step(4)])
    assert set(event["selected_node_ids"]) < set(memory.last_event["selected_node_ids"])
    await memory.flush_final([*messages, *step(4), {"role": "assistant", "content": "Final report"}])
    assert memory.acknowledged[-1] == {"role": "assistant", "content": "Final report"}
    await memory.close()
    await ordinary.close()
    audit = [json.loads(line) for line in (tmp_path / "all/requests.jsonl").read_text().splitlines()]
    assert audit[0]["selected_relation_ids"] == event["selected_relation_ids"]
    assert backend.closed


@pytest.mark.asyncio
async def test_full_graph_preserves_initial_timing_and_graph_failure_fallback(tmp_path):
    graph = FakeGraph()
    memory = BCGMemory(BCGSettings(retrieval="all"), tmp_path, "session", backend=graph)
    assert await memory.prepare(history(2)) == history(2)
    assert [t["role"] for t in memory.acknowledged] == ["system", "user"]
    graph.fail_after_commit = True
    assert await memory.prepare(history(3)) == history(3)
    assert memory.last_event["fallback"] == "lost_ack"
    assert not graph.selections
    await memory.close()


def test_all_nodes_includes_beliefs_and_decisions_and_checks_edge_integrity():
    snapshot = {"beliefs": [{"id": 2, "belief": "Fact"}], "decisions": [{"id": 3, "decision": "Answer"}],
                "relations": [{"id": 7, "from_id": 3, "to_id": 2, "type": "depends_on"}]}
    selection = full_graph_selection(snapshot)
    _, ids, edges = render_graph(snapshot, selection, None, strict=True)
    assert ids == [2, 3] and edges == [7]
    snapshot["relations"][0]["to_id"] = 99
    with pytest.raises(GraphError, match="invalid_selected_relation"):
        render_graph(snapshot, selection, None, strict=True)
    empty, ids, edges = render_graph({"nodes": [], "relations": []}, {"node_ids": [], "relation_ids": []}, None)
    assert "<belief_graph>" in empty and ids == edges == []
