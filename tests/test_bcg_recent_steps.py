"""Single-agent window contracts. No model calls or live services."""

from __future__ import annotations

import copy
import re

import pytest

from bcg.construct.unified.task_state import build_task_state, render_task_state
from tests.test_react_bcg import FakeGraph, history, step
from workflows.stateful_react_agent.bcg.messages import partition
from workflows.stateful_react_agent.bcg.recent import recent_step_selection
from workflows.stateful_react_agent.bcg.rendering import render_graph
from workflows.stateful_react_agent.bcg.runtime import BCGMemory, BCGSettings


class TaskGraph(FakeGraph):
    def snapshot(self, pid):
        value = super().snapshot(pid)
        for node in value["nodes"]:
            index = node["source"]["turn_id"]
            node["role"] = self.turns[pid][index]["role"]
            node["task_role"] = "other"
            node["task_ids"] = []
            if index in {1, 4, 12}:
                node.update(task_role="task", belief=f"Task from graph turn {index}")
            if index == 16:
                node.update(task_role="progress", task_ids=[14], task_status="blocked",
                            belief="The task from step 2 still lacks a primary source.")
        value["task_state"] = build_task_state(value)
        return value


def test_window_is_three_complete_steps_before_two_raw_steps():
    messages = history(10)
    before = copy.deepcopy(messages)
    plan = partition(messages, 2)
    assert plan.retained == messages[:2] + messages[-4:]
    assert plan.evicted_steps == 8
    assert len(plan.evicted_step_turn_indices) == 8
    assert plan.evicted_step_turn_indices[-3:] == [[11, 12], [13, 14], [15, 16]]
    graph = FakeGraph()
    graph.turns["p"] = [{"role": "system", "content": "marker"}, *plan.turns]
    selection = recent_step_selection(graph.snapshot("p"), plan, 3)
    assert selection["agent_step_numbers"] == [6, 7, 8]
    assert selection["graph_turn_ids"] == list(range(12, 18))
    assert selection["node_ids"] == list(range(22, 28))
    assert selection["relation_ids"] == list(range(10, 15))
    assert messages == before


def test_parallel_results_one_step_and_all_user_inputs_are_pinned():
    messages = history(2)
    followup = {"role": "user", "content": "Also verify the date."}
    messages += [followup, *step(2, parallel=4),
                 {"role": "assistant", "content": "A tool-free observation."},
                 *step(3, parallel=3, native=True), *step(4, native=True)]
    plan = partition(messages, 2)
    assert plan.evicted_steps == 4
    # Last three evicted steps: ordinary tools, four parallel tools, no tools.
    assert [len(group) for group in plan.evicted_step_turn_indices[-3:]] == [2, 2, 1]
    assert plan.retained == [*messages[:2], followup, *step(3, parallel=3, native=True), *step(4, native=True)]
    parallel = plan.evicted_step_turn_indices[-2]
    assert plan.turns[parallel[1]]["content"].count("<tool_result>") == 4
    selected_sources = {i for group in plan.evicted_step_turn_indices[-3:] for i in group}
    assert all(plan.turns[i]["role"] != "user" for i in selected_sources)


def test_empty_extraction_does_not_backfill_old_steps_and_relations_stay_internal():
    plan = partition(history(10), 2)
    graph = FakeGraph()
    graph.turns["p"] = [{"role": "system", "content": "marker"}, *plan.turns]
    snapshot = graph.snapshot("p")
    snapshot["nodes"] = [n for n in snapshot["nodes"] if n["source"]["turn_id"] not in {12, 13, 14, 15}]
    snapshot["relations"] = [
        {"id": 1, "from_id": 27, "to_id": 26, "type": "depends_on", "note": "real direction"},
        {"id": 2, "from_id": 27, "to_id": 11, "type": "supplements"},
    ]
    before = copy.deepcopy(snapshot)
    selection = recent_step_selection(snapshot, plan, 3)
    assert selection["node_ids"] == [26, 27]
    assert selection["agent_step_numbers"] == [6, 7, 8]
    assert selection["relation_ids"] == [1]
    block, *_ = render_graph(snapshot, selection, None, strict=True, style="split")
    assert "[B27] depends_on [B26]: real direction" in block
    assert snapshot == before
    snapshot["nodes"] = []
    assert recent_step_selection(snapshot, plan, 3)["node_ids"] == []


def test_recent_merged_members_use_active_canonical_node():
    plan = partition(history(10), 2)
    snapshot = {"nodes": [
        {"id": 1, "belief": "Merged fact", "source": {"turn_id": 2}, "merged_from": [30, 40]},
        {"id": 2, "belief": "Old fact", "source": {"turn_id": 4}, "merged_from": [20]},
        {"id": 50, "belief": "Recent fact", "source": {"turn_id": 17}},
    ], "merges": [
        {"absorbed_snapshots": [{"id": 30, "source": {"turn_id": 12}}]},
        {"absorbed_snapshots": [{"id": 40, "source": {"turn_id": 15}, "merged_from": [30]},
                                {"id": 20, "source": {"turn_id": 5}}]},
    ], "relations": [{"id": 10, "from_id": 50, "to_id": 1, "type": "depends_on"}]}
    before = copy.deepcopy(snapshot)
    assert recent_step_selection(snapshot, plan, 3)["node_ids"] == [1, 50]
    assert recent_step_selection(snapshot, plan, 3)["relation_ids"] == [10]
    assert snapshot == before


@pytest.mark.parametrize("style", ["v0", "split", "split_confidence"])
@pytest.mark.asyncio
async def test_global_state_uses_whole_graph_and_injection_format_is_unchanged(tmp_path, style):
    settings = BCGSettings.from_react_agent({"bcg": {"url": "http://unused"}})
    backend = TaskGraph()
    memory = BCGMemory(settings, tmp_path, "window", backend=backend, render_style=style)
    messages = history(10)
    result = await memory.prepare(messages)
    event = memory.last_event
    assert not event["fallback"]
    assert event["retained_indices"] == [0, 1, 18, 19, 20, 21]
    assert result[1:] == [messages[1], *messages[-4:]]
    assert result[0]["content"].startswith("original instructions\n\n")
    assert len([m for m in result if m["role"] == "system"]) == 1
    assert len(memory.acknowledged) == 18  # initial user + marker + eight steps
    assert event["selector_result"]["node_ids"] == list(range(22, 28))
    assert event["task_state"]["scope"] == "full_graph"
    # Includes the initial user task, an older step-2 task, and a recent task.
    assert set(event["task_state"]["task_ids"]) == {"T11", "T14", "T22"}
    task = next(t for t in memory.snapshot["task_state"]["nodes"] if t["id"] == "T14")
    assert task["status"] == "blocked" and task["status_source_id"] == 26
    assert not backend.selections and memory._pcst_selector is None
    # Byte-identical graph formatting from the existing renderers. No new block,
    # fake dialogue, changed node syntax, or new relation types are introduced.
    expected_graph, *_ = render_graph(memory.snapshot, event["selector_result"], None, strict=True, style=style)
    task_text, *_ = render_task_state(memory.snapshot, settings.task_state_chars)
    expected_graph = expected_graph.replace("<belief_graph>\n", "<belief_graph>\n" + task_text + "\n", 1)
    actual = re.search(r"<belief_graph>\n.*?</belief_graph>", result[0]["content"], re.S).group()
    assert actual == expected_graph
    await memory.prepare(messages)
    assert len(backend.pushes) == 1
    await memory.flush_final([*messages, {"role": "assistant", "content": "Final answer"}])
    assert len(memory.acknowledged) == 23
    assert memory.acknowledged[-1]["content"] == "Final answer"
    await memory.close()


@pytest.mark.asyncio
async def test_window_does_not_silently_prune_nodes_to_old_character_budget(tmp_path):
    messages = history(10)
    messages[13]["content"] = "A long source claim. " * 500
    settings = BCGSettings.from_react_agent({"bcg": {"url": "http://unused", "graph_chars": 512}})
    backend = TaskGraph()
    memory = BCGMemory(settings, tmp_path, "large-window", backend=backend)
    await memory.prepare(messages)
    assert not memory.last_event["fallback"]
    assert memory.last_event["graph_chars"] > 6600
    assert memory.last_event["selector_result"]["node_ids"] == list(range(22, 28))
    await memory.close()


@pytest.mark.asyncio
async def test_missing_task_state_preserves_original_context(tmp_path):
    memory = BCGMemory(BCGSettings.from_react_agent({}), tmp_path, "old-server", backend=FakeGraph())
    messages = history(3)
    assert await memory.prepare(messages) == messages
    assert memory.last_event["fallback"] == "graph_server_task_state_required"
    await memory.close()


@pytest.mark.parametrize("count", [3, 4, 5])
def test_single_agent_settings_override_stale_retrieval_not_team_or_ablation(monkeypatch, count):
    monkeypatch.setenv("BCG_RETRIEVAL", "query_pcst")
    agent = {"bcg": {"retrieval": "focused", "task_state_enabled": False,
                     "graph_recent_turns": count, "embedding_url": "unused"}}
    settings = BCGSettings.from_react_agent(agent)
    assert settings.retrieval == "recent_steps" and settings.task_state_enabled
    assert settings.recent_turns == 2 and settings.graph_recent_turns == count
    assert settings.graph_budget is None
    plan = partition(history(10), 2)
    assert len(recent_step_selection({"nodes": [], "relations": []}, plan, count)["agent_step_numbers"]) == count
    settings = BCGSettings.from_react_agent({**agent, "context_mode": "bcg_no_retrieval"})
    assert settings.retrieval == "all" and not settings.task_state_enabled
    assert BCGSettings.from_agent({}).retrieval == "query_pcst"  # Team unchanged
    with pytest.raises(ValueError, match="BCG_GRAPH_RECENT_TURNS"):
        BCGSettings.from_react_agent({"bcg": {"graph_recent_turns": 0}})


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_startup_uses_available_steps_without_inventing_history(count):
    plan = partition(history(count), 2)
    selection = recent_step_selection({"nodes": [], "relations": []}, plan, 3)
    assert selection["agent_step_numbers"] == list(range(1, max(0, count - 2) + 1))


@pytest.mark.parametrize("style", ["v0", "split", "split_confidence"])
@pytest.mark.parametrize("raw_steps,graph_steps", [(2, 3), (1, 5), (0, 4)])
def test_guide_matches_scope_and_grounded_next_action_policy(tmp_path, style, raw_steps, graph_steps):
    settings = BCGSettings.from_react_agent({"bcg": {
        "recent_turns": raw_steps, "graph_recent_turns": graph_steps,
    }})
    backend = TaskGraph()
    memory = BCGMemory(settings, tmp_path, "guide", backend=backend, render_style=style)
    guide = " ".join(memory.context_guide.split())
    tag = "context_blocks_guide" if style == "v0" else "belief_graph_guide"
    assert guide.startswith(f"<{tag}>") and guide.endswith(f"</{tag}>")
    assert guide.count(f"<{tag}>") == 1
    assert f"last {raw_steps} complete interactions remain verbatim" in guide
    assert "One interaction is one Assistant output plus all its Tool Results." in guide
    assert "entire acknowledged historical graph" in guide
    assert "only task records fitting its display budget are shown" in guide
    assert f"up to {graph_steps} complete interactions immediately before the retained raw interactions" in guide
    assert "relations between those nodes" in guide
    assert "additional tasks, nodes and relations" in guide
    assert "Reconcile Task state, when shown, with recent raw interactions" in guide
    assert "unresolved task, blocker or missing premise" in guide
    assert "Recorded plans and completion claims are not verified outcomes" in guide
    assert "source and stance priors" in guide and "evidence and relation propagation" in guide
    assert "extractor's confidence" not in guide and "if B falls" not in guide
    assert "confidence alone does not establish truth" in guide
    assert "Repeat queries only for a specific gap" in guide
    assert "recoverable failure" in guide and "Empty results do not disprove a claim" in guide
    assert "when needed detail is missing, not for every figure" in guide
    assert "recorded message or tool-result batch, not fresh verification" in guide
    if style != "v0":
        assert '"Completed actions" records attempted tools' in guide
        assert "not verified success" in guide and "allowing necessary verification" in guide
    # Constructing guidance neither runs a model nor alters the graph.
    assert not backend.pushes and not backend.selections and not memory.snapshot


def test_support_label_guide_does_not_claim_a_source_count_audit(tmp_path):
    memory = BCGMemory(BCGSettings.from_react_agent({}), tmp_path, "labels",
                       backend=TaskGraph(), render_style="split_confidence")
    guide = " ".join(memory.context_guide.split())
    assert "not verified truth or an audited count of independent sources" in guide
    assert "verify source independence before claiming corroboration" in guide
    assert "Strongest to weakest" not in guide
