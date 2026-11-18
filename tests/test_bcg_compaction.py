"""BCG Compaction request contracts. Offline; no live model calls."""

from __future__ import annotations

import copy
import json

import pytest

from tests.test_bcg_recent_steps import TaskGraph
from tests.test_react_bcg import FakeGraph, history, step
from workflows.stateful_react_agent.bcg.compaction import compact_history
from workflows.stateful_react_agent.bcg.messages import partition
from workflows.stateful_react_agent.bcg.rendering import relation_line, render_graph
from workflows.stateful_react_agent.bcg.runtime import BCGMemory, BCGSettings


def settings(**overrides):
    return BCGSettings.from_react_agent({"context_mode": "bcg_compaction", "bcg": {
        "url": "http://unused", **overrides,
    }})


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["v0", "split", "split_confidence"])
async def test_roles_window_internal_edges_global_state_and_unchanged_format(tmp_path, style):
    backend = TaskGraph()
    memory = BCGMemory(settings(compaction_turns=3), tmp_path, "compact", backend=backend, render_style=style)
    original = history(10)
    before = copy.deepcopy(original)
    result = await memory.prepare(original)
    event = memory.last_event
    assert not event["fallback"]
    assert [m["role"] for m in result] == ["system", *["user"] * 5]
    assert result[1] == original[1] and original == before
    assert "<belief_graph>\n" not in result[0]["content"]
    assert result[0]["content"].startswith("original instructions\n\n")
    assert "No completed raw Assistant/Tool interactions are retained" in result[0]["content"]
    assert "last 3 interactions" in result[0]["content"]
    assert "user-role <belief_graph> blocks supplied by the framework" in result[0]["content"]
    assert event["system_memory"] == memory.context_guide
    assert result[2] == event["task_state_message"]
    assert "[T11]" in result[2]["content"]  # Initial goal outside recent window.
    assert event["task_state"]["scope"] == "full_graph"
    assert event["retained_indices"] == [0, 1]
    assert len(memory.acknowledged) == 22  # All ten steps + initial user + marker.
    assert event["selected_node_ids"] == list(range(26, 32))
    assert event["selected_relation_ids"] == list(range(14, 19))
    assert event["selector_result"]["agent_step_numbers"] == [8, 9, 10]
    assert not backend.selections and memory._pcst_selector is None
    assert all(set(m) == {"role", "content"} for m in result[3:])
    blocks = event["compacted_messages"]
    edges = {e["id"]: e for e in memory.snapshot["relations"]}
    for block, sent in zip(blocks, result[3:], strict=True):
        assert block["role"] == sent["role"] == "user"
        assert sent["content"] == block["content"]
        expected, *_ = render_graph(memory.snapshot, {"node_ids": block["node_ids"], "relation_ids": []},
                                     None, strict=True, style=style)
        expected = expected.removesuffix("</belief_graph>") + "".join(
            relation_line(edges[rid]) + "\n" for rid in block["relation_ids"]) + "</belief_graph>"
        assert block["content"] == expected
        assert all(edges[rid]["from_id"] in block["node_ids"] for rid in block["relation_ids"])
    text = "\n".join(m["content"] for m in result[3:])
    for rid in event["selected_relation_ids"]:
        edge = edges[rid]
        assert edge["from_id"] in event["selected_node_ids"] and edge["to_id"] in event["selected_node_ids"]
        assert text.count(relation_line(edge)) == 1
    assert "[B27] depends_on [B28]" in result[3]["content"]  # Cross-step edge retained.
    assert "[B25]" not in text  # Old endpoint excluded.
    assert await memory.prepare(original) == result
    assert len(backend.pushes) == 1
    await memory.prepare([*original, *step(10)])
    assert len(backend.pushes[-1][1]) == 2
    assert memory.last_event["selected_node_ids"] == list(range(28, 34))
    assert all(not t["content"].startswith(("<belief_graph>", "<task_state>")) for t in memory.acknowledged)
    await memory.flush_final([*original, *step(10), {"role": "assistant", "content": "Final report"}])
    assert memory.acknowledged[-1]["content"] == "Final report"
    await memory.close()
    audit = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert audit[0]["compacted_messages"] == blocks


@pytest.mark.asyncio
async def test_first_call_has_global_state_no_synthetic_assistant_steps(tmp_path):
    memory = BCGMemory(settings(), tmp_path, "first", backend=TaskGraph())
    result = await memory.prepare(history(0))
    assert [m["role"] for m in result] == ["system", "user", "user"]
    assert memory.last_event["compacted_messages"] == []
    assert "[T11]" in result[2]["content"]
    assert [t["role"] for t in memory.acknowledged] == ["system", "user"]
    await memory.close()


@pytest.mark.asyncio
async def test_parallel_batch_is_one_step_and_all_user_inputs_stay_in_place(tmp_path):
    followup = {"role": "user", "content": "Also verify the date."}
    messages = [*history(1), followup, *step(1, parallel=4, native=True),
                {"role": "assistant", "content": "A tool-free conclusion."}]
    memory = BCGMemory(settings(compaction_turns=2), tmp_path, "batch", backend=TaskGraph())
    result = await memory.prepare(messages)
    assert not memory.last_event["fallback"]
    assert [m["role"] for m in result] == ["system", *["user"] * 5]
    assert result[1] == messages[1] and result[3] == followup
    blocks = memory.last_event["compacted_messages"]
    assert [x["agent_step"] for x in blocks] == [2, 3]
    assert [len(x["graph_turn_ids"]) for x in blocks] == [2, 1]
    assert memory.acknowledged[blocks[0]["graph_turn_ids"][1]]["content"].count("<tool_result>") == 4
    text = json.dumps(result)
    assert "CIPHERTEXT" not in text and "SIGNATURE" not in text and "REDACTED" not in text
    await memory.close()


def test_empty_steps_count_no_backfill_and_merged_nodes_only_once():
    plan = partition(history(3), 0)
    snapshot = {"nodes": [
        {"id": 11, "belief": "Old goal", "source": {"turn_id": 1}},
        {"id": 15, "belief": "Merged evidence", "source": {"turn_id": 4}, "merged_from": [16]},
        {"id": 17, "belief": "Latest evidence", "source": {"turn_id": 7}},
    ], "merges": [{"absorbed_snapshots": [{"id": 16, "source": {"turn_id": 6}}]}],
        "relations": [{"id": 0, "from_id": 17, "to_id": 15, "type": "depends_on"},
                      {"id": 1, "from_id": 15, "to_id": 11, "type": "supplements"}],
        "task_state": {"version": 2, "nodes": [], "edges": []}}
    before = copy.deepcopy(snapshot)
    _, audit = compact_history(history(3), plan, snapshot, 2, "v0", 1800)
    assert [b["node_ids"] for b in audit["compacted_messages"]] == [[], [15, 17]]
    assert audit["selected_relation_ids"] == [0]
    assert snapshot == before
    # Empty latest step must not borrow nodes from an older step.
    snapshot["nodes"] = snapshot["nodes"][:1]
    snapshot["relations"] = []
    _, audit = compact_history(history(3), plan, snapshot, 2, "v0", 1800)
    assert audit["selected_node_ids"] == [] and len(audit["compacted_messages"]) == 2


@pytest.mark.asyncio
async def test_failure_keeps_original_transcript_not_partial_compressed_history(tmp_path):
    memory = BCGMemory(settings(), tmp_path, "old-server", backend=FakeGraph())
    messages = history(3)
    assert await memory.prepare(messages) == messages
    assert memory.last_event["fallback"] == "graph_server_task_state_required"
    await memory.close()


@pytest.mark.asyncio
async def test_incomplete_parallel_batch_is_not_dropped_or_orphaned(tmp_path):
    pending = step(1, parallel=2)[:-1]
    messages = [*history(1), *pending]
    memory = BCGMemory(settings(), tmp_path, "pending", backend=TaskGraph())
    projected = await memory.prepare(messages)
    assert not memory.last_event["fallback"]
    assert projected[-2:] == pending
    assert len(memory.last_event["compacted_messages"]) == 1
    assert len(memory.acknowledged) == 4
    await memory.close()


@pytest.mark.asyncio
async def test_new_human_followup_is_graphed_immediately_but_remains_pinned(tmp_path):
    memory = BCGMemory(settings(), tmp_path, "followup", backend=TaskGraph())
    messages = history(2)
    await memory.prepare(messages)
    followup = {"role": "user", "content": "The final report must cover only primary sources."}
    result = await memory.prepare([*messages, followup])
    assert not memory.last_event["fallback"]
    assert memory.acknowledged[-1] == followup and result[-1] == followup
    assert memory.last_event["graph_turns_written"] == 1
    await memory.prepare([*messages, followup, *step(2)])
    assert not memory.last_event["fallback"]
    assert memory.last_event["graph_turns_written"] == 2
    assert memory.last_event["compacted_messages"][-1]["graph_turn_ids"] == [7, 8]
    await memory.close()


@pytest.mark.asyncio
async def test_selected_window_is_not_truncated_to_legacy_budget(tmp_path):
    memory = BCGMemory(settings(graph_chars=512), tmp_path, "large", backend=TaskGraph())
    messages = history(1)
    messages[-1]["content"] = "Long external observation. " * 500
    result = await memory.prepare(messages)
    assert not memory.last_event["fallback"]
    assert len(result[-1]["content"]) > 6600
    assert memory.last_event["selected_node_ids"] == [12, 13]
    await memory.close()


def test_named_settings_override_old_window_and_selector(monkeypatch):
    monkeypatch.setenv("BCG_RECENT_TURNS", "2")
    monkeypatch.setenv("BCG_RETRIEVAL", "query_pcst")
    value = settings(recent_turns=2, retrieval="focused", task_state_enabled=False, embedding_url="unused")
    assert value.recent_turns == 0 and value.retrieval == "compaction"
    assert value.compaction_turns == 5 and value.task_state_enabled
    assert value.graph_budget is None
    monkeypatch.setenv("BCG_COMPACTION_TURNS", "3")
    assert settings().compaction_turns == 3
    assert settings(compaction_turns=1).compaction_turns == 1
    with pytest.raises(ValueError, match="BCG_COMPACTION_TURNS"):
        settings(compaction_turns=0)
    assert BCGSettings.from_react_agent({}).retrieval == "recent_steps"
