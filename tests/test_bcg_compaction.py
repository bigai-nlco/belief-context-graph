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


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("retention", [5, "all"])
def test_empty_task_state_is_not_injected_but_keeps_user_and_audits(retention, version):
    original = history(0)
    graph = {"nodes": [], "relations": [], "task_state": {"version": version, "nodes": [], "relations": []}}
    projected, audit = compact_history(original, partition(original, 0), graph, retention, "v1", 1800)
    assert projected == original and audit["task_state_message"] is None
    assert audit["task_state"]["available"] and not audit["task_state"]["injected"]


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
async def test_retention_changes_only_the_visible_step_window(tmp_path):
    memories = [BCGMemory(settings(compaction_turns=n), tmp_path / str(n), str(n),
                          backend=TaskGraph(), render_style="v1") for n in (5, "all")]
    messages = history(0)
    for done in range(8):
        projected = [await m.prepare(messages) for m in memories]
        recent, all_steps = (m.last_event for m in memories)
        assert not recent["fallback"] and not all_steps["fallback"]
        assert "All completed interactions appear in chronological order" in projected[1][0]["content"]
        assert recent["compaction_turns"] == 5 and all_steps["compaction_turns"] == "all"
        assert memories[0].acknowledged == memories[1].acknowledged  # identical graph input
        assert recent["task_state_message"] == all_steps["task_state_message"]
        for event, sent, first in ((recent, projected[0], max(1, done - 4)), (all_steps, projected[1], 1)):
            steps = list(range(first, done + 1))
            blocks = event["compacted_messages"]
            assert [b["agent_step"] for b in blocks] == event["selector_result"]["agent_step_numbers"] == steps
            assert sent[1:3] == [messages[1], event["task_state_message"]]
            assert sent[3:] == [{"role": "user", "content": b["content"]} for b in blocks]
            assert len({nid for b in blocks for nid in b["node_ids"]}) == sum(len(b["node_ids"]) for b in blocks)
            visible = set(event["selected_node_ids"])
            internal = {e["id"] for e in memories[0].snapshot["relations"]
                        if e["from_id"] in visible and e["to_id"] in visible}
            edge_ids = [rid for b in blocks for rid in b["relation_ids"]]
            assert set(edge_ids) == internal and len(edge_ids) == len(internal)
        if done <= 5:
            assert projected[0][1:] == projected[1][1:]
        if done == 6:
            assert len(all_steps["compacted_messages"][-1]["graph_turn_ids"]) == 2
            assert all_steps["compacted_messages"][0]["node_ids"][0] not in recent["selected_node_ids"]
        messages += step(done, parallel=4 if done == 5 else 1)
    for m in memories:
        await m.close()


def test_all_window_is_not_a_large_fixed_count():
    messages = history(307)
    plan = partition(messages, 0)
    snapshot = {"nodes": [], "relations": [], "task_state": {"version": 2, "nodes": [], "edges": []}}
    _, event = compact_history(messages, plan, snapshot, "all", "v1", 1800)
    assert [b["agent_step"] for b in event["compacted_messages"]] == list(range(1, 308))


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
    monkeypatch.setenv("BCG_COMPACTION_TURNS", "all")
    assert settings().compaction_turns == "all"
    assert settings(compaction_turns=5).compaction_turns == 5
    for invalid in (0, -1, False, None, "unlimited", 1.5):
        with pytest.raises(ValueError, match="BCG_COMPACTION_TURNS"):
            settings(compaction_turns=invalid)


def test_task_state_filter_parses_flags_and_updates_guide_and_scope(monkeypatch):
    base = settings()
    assert base.compaction_task_state_filter is False and base.compaction_task_state_hops == 2
    monkeypatch.setenv("BCG_COMPACTION_TASK_STATE_FILTER", "true")
    monkeypatch.setenv("BCG_COMPACTION_TASK_STATE_HOPS", "3")
    enabled = settings()
    assert enabled.compaction_task_state_filter is True and enabled.compaction_task_state_hops == 3
    for bad_bool in ("maybe", 1, None):
        with pytest.raises(ValueError, match="BCG_COMPACTION_TASK_STATE_FILTER"):
            settings(compaction_task_state_filter=bad_bool)
    for bad_hops in (-1, 11, True, "wide"):
        with pytest.raises(ValueError, match="BCG_COMPACTION_TASK_STATE_HOPS"):
            settings(compaction_task_state_hops=bad_hops)


@pytest.mark.parametrize("window", [5, "all"])
def test_task_state_filter_orthogonal_khop_pairing_blocking_latest_step_and_skipping(tmp_path, window):
    from bcg.construct.unified.task_state import build_task_state

    messages = history(7)
    plan = partition(messages, 0)

    def n(nid, turn_id, text, *, role="assistant", task_role="evidence", **extra):
        return {"id": nid, "node_type": "belief", "belief": text, "role": role,
                "stance": "asserted", "confidence": 0.8, "task_role": task_role,
                "supporting_excerpts": [text], "source": {"turn_id": turn_id}, **extra}

    nodes = [
        # Step 1 (turns 2,3 - outside 5-turn window): active T10, completed T11, closed gap pair B13->B14.
        n(10, 2, "Verify active program X.", task_role="task"),
        n(11, 2, "Verify completed program Y.", task_role="task"),
        n(12, 3, "Completed program Y.", task_role="progress", task_ids=[11], task_status="reported_complete"),
        n(13, 3, "Original missing date gap.", task_scope="root", gap_quote="missing date gap"),
        n(14, 3, "Resolved the date gap.", task_scope="root",
          closes_gaps=[{"id": 13, "quote": "Resolved the date gap"}]),
        # Step 2 (turns 4,5): only completed-task / closed-gap neighbors -> skipped!
        n(20, 4, "Completed task noise.", role="tool"),
        n(21, 5, "Closed gap noise.", role="tool"),
        # Step 3 (turns 6,7): paired tool_call B30 (via tool_call_id) + hop-1 finding B31.
        n(30, 6, 'The assistant is using web_search to search for "active query".',
          task_role="tool_action", extraction_method="rule_tool_call", tool_call_id="c30", tool_name="web_search"),
        n(31, 7, "Active hop-1 finding.", role="tool", tool_call_id="c30", tool_name="web_search"),
        # Step 4 (turns 8,9): hop-1 action B40 + suppressed blocked result B41 -> renders B40 -> BLOCKED.
        n(40, 8, 'The assistant is using web_fetch to call "https://blocked.example".',
          task_role="tool_action", extraction_method="rule_tool_call", tool_call_id="c40", tool_name="web_fetch"),
        n(41, 9, "[BLOCKED] source is unavailable for this task",
          role="tool", extraction_method="rule_tool_result", tool_call_id="c40", tool_name="web_fetch"),
        # Step 5 (turns 10,11): hop-2 conclusion B50, unrelated B51, unassigned gap B52 (active seed).
        n(50, 10, "Hop-2 derived claim."),
        n(51, 11, "Unrelated step-5 noise.", role="tool"),
        n(52, 11, "Nepal registry records still missing.", gap_quote="registry records still missing"),
        # Step 6 (turns 12,13): hop-3 finding B60 -> excluded at hops=2 (step 6 skipped), kept at hops=3.
        n(60, 12, "Hop-3 distant finding.", role="tool"),
        # Step 7 (turns 14,15, latest step): disconnected B70 -> unconditionally kept.
        n(70, 14, "Latest disconnected observation.", role="tool"),
    ]
    relations = [
        {"id": 0, "from_id": 20, "to_id": 11, "type": "supplements"},
        {"id": 1, "from_id": 21, "to_id": 14, "type": "supplements"},
        {"id": 2, "from_id": 31, "to_id": 10, "type": "supplements"},
        {"id": 3, "from_id": 40, "to_id": 10, "type": "supplements"},
        {"id": 4, "from_id": 50, "to_id": 31, "type": "depends_on"},
        {"id": 5, "from_id": 60, "to_id": 50, "type": "depends_on"},
        {"id": 6, "from_id": 51, "to_id": 12, "type": "supplements"},
    ]
    snapshot = {"nodes": nodes, "relations": relations}
    snapshot["task_state"] = build_task_state(snapshot)
    before = copy.deepcopy(snapshot)

    projected, audit = compact_history(
        messages, plan, snapshot, window, "v1", 1800,
        task_state_filter=True, task_state_hops=2,
    )
    assert snapshot == before
    assert audit["compaction_task_state_filter"] is True
    assert audit["compaction_task_state_hops"] == 2
    assert audit["selector_result"]["strategy"] == "compaction_task_state"
    # Active seeds must exclude completed T11 / B12 and resolved closure B13/B14, keeping T10 & unassigned B52.
    assert set(audit["selector_result"]["task_state_seed_ids"]) == {10, 52}

    expected_steps = [1, 3, 4, 5, 7] if window == "all" else [3, 4, 5, 7]
    assert projected[1] == messages[1]
    assert projected[2] == audit["task_state_message"]
    assert projected[3:] == [{"role": "user", "content": b["content"]} for b in audit["compacted_messages"]]
    expected_nodes = {30, 31, 40, 50, 52, 70} | ({10} if window == "all" else set())
    assert [b["agent_step"] for b in audit["compacted_messages"]] == expected_steps
    assert audit["selector_result"]["agent_step_numbers"] == expected_steps
    assert set(audit["selected_node_ids"]) == expected_nodes
    assert 41 not in audit["selected_node_ids"] and 60 not in audit["selected_node_ids"]
    # Step 2 (both modes) and Step 6 (hops=2) skipped; true Step t-n distance preserved!
    blocks_by_step = {b["agent_step"]: b for b in audit["compacted_messages"]}
    assert "Step t-5" in blocks_by_step[3]["content"]
    assert "Step t-4" in blocks_by_step[4]["content"] and "→ BLOCKED" in blocks_by_step[4]["content"]
    assert "[B41]" not in blocks_by_step[4]["content"]
    assert "Step t-3" in blocks_by_step[5]["content"] and "[B50] depends_on [B31]" in blocks_by_step[5]["content"]
    assert "Step t-1" in blocks_by_step[7]["content"]
    assert blocks_by_step[3]["node_ids"] == [30, 31]

    # hops=3 pulls in B60 (step 6).
    _, audit_h3 = compact_history(
        messages, plan, snapshot, window, "v1", 1800,
        task_state_filter=True, task_state_hops=3,
    )
    assert 60 in audit_h3["selected_node_ids"]
    assert 6 in [b["agent_step"] for b in audit_h3["compacted_messages"]]

    # Even with zero seeds, the latest completed step (Step 7) stays unconditionally retained.
    empty_state = {**snapshot, "task_state": {"version": 4, "nodes": [], "relations": [], "root": {}}}
    _, audit_zero = compact_history(
        messages, plan, empty_state, window, "v1", 1800,
        task_state_filter=True, task_state_hops=2,
    )
    assert [b["agent_step"] for b in audit_zero["compacted_messages"]] == [7]
    assert audit_zero["selected_node_ids"] == [70]

    # BCGMemory separates scope & context_guide cleanly when filter is enabled vs disabled.
    m_off = BCGMemory(settings(compaction_turns=window, compaction_task_state_filter=False),
                      tmp_path / f"off-{window}", "p", backend=TaskGraph(), render_style="v1")
    m_on = BCGMemory(settings(compaction_turns=window, compaction_task_state_filter=True, compaction_task_state_hops=2),
                     tmp_path / f"on-{window}", "p", backend=TaskGraph(), render_style="v1")
    assert m_off.scope != m_on.scope
    assert "2 relation hops of active <task_state>" in m_on.context_guide
    assert "2 relation hops" not in m_off.context_guide
    assert BCGSettings.from_react_agent({}).retrieval == "recent_steps"


@pytest.mark.asyncio
async def test_runtime_nudges_stay_visible_for_immediate_retry_then_age_out_without_graphing(tmp_path):
    """Loop/observer control nudges (bcg_origin='runtime_nudge' or untagged user
    messages once explicit human provenance is active) must be visible on the
    immediate retry call, age out after the next completed step, and never be
    ingested into the belief graph as human follow-up tasks."""
    from frontier_agent.core.loop_types import BaseObserver, Intervention
    from frontier_agent.core.runtime.loop.agent_loop import run_agent_loop

    memory = BCGMemory(settings(compaction_turns=2), tmp_path, "nudge", backend=TaskGraph())
    initial = history(1)
    initial[1]["bcg_origin"] = "user"
    # Untagged observer/loop injection after an explicit-provenance initial user turn.
    nudge = {"role": "user", "content": "Invoke the tool through the proper tool_calls interface now."}
    with_trailing_nudge = [*initial, nudge]

    # 1. Immediate retry call: trailing nudge remains visible at the tail, NEVER pushed to graph!
    projected_retry = await memory.prepare(with_trailing_nudge)
    assert projected_retry[-1] == nudge
    assert all(t["content"] != nudge["content"] for t in memory.acknowledged)

    # 2. Once the retry step completes, the old nudge ages out and is still absent from the graph;
    #    meanwhile a genuine human steer (bcg_origin='user') stays pinned and IS graphed.
    human_steer = {"role": "user", "content": "Also compare rural vs urban rates.", "bcg_origin": "user"}
    after_step = [*with_trailing_nudge, *step(1), human_steer]
    projected_next = await memory.prepare(after_step)
    assert all(m.get("content") != nudge["content"] for m in projected_next)
    assert projected_next[-1] == human_steer
    assert any(t["content"] == human_steer["content"] for t in memory.acknowledged)
    assert all(t["content"] != nudge["content"] for t in memory.acknowledged)
    await memory.close()

    # 3. End-to-end through run_agent_loop + BCGObserver: initial user stays clean and pinned,
    #    observer inject_messages becomes 'runtime_nudge' (never graphed), and human steer stays 'user'.
    from frontier_agent.core.llm import LLMResponse
    from frontier_agent.core.loop_types import LoopConfig, LoopPolicy
    from workflows.stateful_react_agent.bcg.llm import BCGClient
    from workflows.stateful_react_agent.bcg.runtime import BCGObserver

    loop_mem = BCGMemory(settings(compaction_turns=2), tmp_path / "loop", "loop", backend=TaskGraph())

    class _FakeLLM:
        model = "fake"

        def __init__(self):
            self.calls = 0

        async def chat(self, msgs, **kw):
            self.calls += 1
            return LLMResponse(content=f"reply {self.calls}")

    class _NudgeAndSteerObserver(BaseObserver):
        critical = True

        def __init__(self):
            self.fired = False

        async def on_llm_response(self, ctx):
            if not self.fired:
                self.fired = True
                ctx.metadata.setdefault("_bcg_human_steers", []).append("human steer line")
                return Intervention(
                    continue_to_next_turn=True,
                    inject_messages=["observer format nudge", "human steer line"],
                )
            return None

    res = await run_agent_loop(
        system_prompt="sys",
        user_message="original task",
        llm=BCGClient(_FakeLLM(), loop_mem),
        tools=[],
        config=LoopConfig(loop_policy=LoopPolicy(no_tool_behavior="stop")),
        observers=[_NudgeAndSteerObserver(), BCGObserver(loop_mem)],
    )
    user_msgs = [(m["content"], m.get("bcg_origin")) for m in res.messages if m.get("role") == "user"]
    assert user_msgs == [
        ("original task", None),
        ("observer format nudge", "runtime_nudge"),
        ("human steer line", None),
    ]
    assert any(t["content"] == "human steer line" for t in loop_mem.acknowledged)
    assert all(t["content"] != "observer format nudge" for t in loop_mem.acknowledged)
    await loop_mem.close()
