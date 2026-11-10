"""Grounded task-state contracts: fake models, no credentials or paid calls."""
from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from bcg.config.loader import defaults_dict
from bcg.config.schema import BCGSettings as GraphSettings
from bcg.construct.unified import extract, stream
from bcg.construct.unified.graph import BeliefGraph
from bcg.construct.unified.prompts import build_node_extraction_prompt
from bcg.construct.unified.task_state import (
    EXTRACTION_GUIDANCE,
    build_task_state,
    canonical_task_role,
    classify_legacy_snapshot,
    classify_node,
    clean_annotation,
    render_task_state,
    task_history,
)
from bcg.core.runner import _bcg_from_construct, _memory_document
from tests.test_react_bcg import FakeGraph, history
from workflows.stateful_react_agent.bcg.runtime import BCGMemory, BCGSettings


def node(i, kind="other", text="A source-grounded claim.", **extra):
    return {"id": i, "node_type": "belief", "belief": text, "role": "assistant",
            "stance": "asserted", "entities": [], "supporting_excerpts": [text],
            "confidence": .8, "task_role": kind, "task_ids": [], **extra}


def snapshot():
    value = {"nodes": [
        node(0, "task", "Compare two licensing schemes.", role="user"),
        node(1, "task", "Verify the operating license.", task_ids=[0]),
        node(2, "evidence", "The source describes a pilot, not an operating license.", role="tool"),
        node(3, "intermediate_conclusion", "Pilot status does not establish licensed operation.", task_ids=[1]),
        node(4, "plan", "Fetch the licensing authority's original notice.", task_ids=[1]),
        node(5, "tool_action", "The assistant is using web_search to find the license."),
        node(6, "progress", "License verification is blocked until the notice is available.",
             task_ids=[1], task_status="blocked"),
    ], "relations": [
        {"id": 0, "from_id": 3, "to_id": 2, "type": "depends_on", "note": "distinguish pilot from operation"},
        {"id": 1, "from_id": 4, "to_id": 3, "type": "depends_on", "note": "verification is still needed"},
        {"id": 2, "from_id": 2, "to_id": 5, "type": "depends_on", "note": "search provenance"},
    ]}
    value["task_state"] = build_task_state(value)
    return value


def test_annotations_reject_invalid_types_and_non_anchor_references():
    assert clean_annotation({"task_role": ["goal"]}) == {}
    raw = {"task_role": "progress", "task_ids": [0, 0, 1, True, "0", -1], "task_status": "reported_complete"}
    assert clean_annotation(raw)["task_ids"] == [0, 1]
    value = node(8)
    classify_node(value, raw, [node(0, "task"), node(1, "evidence")])
    assert value["task_ids"] == [0] and value["task_rejected_ids"] == [1]
    assert value["confidence"] == .8 and value["task_annotation_source"] == "graph_model"
    assert "task_status" not in clean_annotation({"task_role": "plan", "task_status": "reported_complete"})


@pytest.mark.parametrize("role,method,expected", [
    ("tool", "", "evidence"), ("function", "", "evidence"),
    ("assistant", "rule_tool_call", "tool_action"), ("system", "", "other"),
])
def test_model_cannot_reclassify_tool_results_or_calls_as_progress(role, method, expected):
    value = node(9, role=role, extraction_method=method)
    classify_node(value, {"task_role": "progress", "task_ids": [0], "task_status": "reported_complete"}, [node(0, "task")])
    assert value["task_role"] == expected and "task_status" not in value


def test_missing_source_and_unassigned_progress_never_complete_a_task():
    value = node(8, supporting_excerpts=[])
    classify_node(value, {"task_role": "task"}, [])
    assert value["task_role"] == "other"
    original = snapshot()
    original["nodes"] += [node(10, "progress", task_status="reported_complete"),
                          node(11, "decision", node_type="decision")]
    state = build_task_state(original)
    assert state["unassigned_progress_ids"] == [10]
    assert {n["id"]: n["status"] for n in state["nodes"]} == {"T0": "open", "T1": "blocked"}


def test_rebuild_keeps_real_chain_provenance_and_confidence_unchanged():
    original = snapshot()
    before = copy.deepcopy(original)
    state = build_task_state(original)
    assert original == before
    task = next(t for t in state["nodes"] if t["id"] == "T1")
    assert task["parent_ids"] == ["T0"]
    assert task["kind"] == "task" and task["plan_ids"] == [4]
    assert state["version"] == 2
    assert task["status_source_id"] == 6 and task["status_provenance"] == "assistant_report"
    assert task["evidence_ids"] == [2] and task["intermediate_conclusion_ids"] == [3]
    assert task["relation_paths"]["2"] == [0] and task["path_roots"]["2"] == 3
    assert not any("confidence" in t for t in state["nodes"])
    original["nodes"].append(node(7, "progress", task_ids=[1], task_status="reported_complete"))
    original["nodes"].append(node(8, "evidence", role="tool"))
    original["relations"].append({"id": 3, "from_id": 8, "to_id": 3, "type": "contradicts"})
    task = build_task_state(original)["nodes"][1]
    assert task["status"] == "reported_complete" and task["update_ids"] == [6, 7]
    assert task["needs_review"] and task["contradiction_ids"] == [3]
    original["nodes"].append(node(9, "progress", task_ids=[1], task_status="open"))
    assert build_task_state(original)["nodes"][1]["status"] == "open"


def test_connected_components_are_not_automatically_tasks_and_traversal_is_bounded():
    value = snapshot()
    value["nodes"] += [node(i, "evidence", role="tool") for i in range(10, 30)]
    value["relations"] += [{"id": i, "from_id": i, "to_id": i + 1, "type": "supplements"} for i in range(10, 29)]
    value["relations"].append({"id": 30, "from_id": 3, "to_id": 10, "type": "supplements"})
    value["relations"].append({"id": 31, "from_id": 11, "to_id": 3, "type": "supplements"})
    state = build_task_state(value, max_hops=2)
    assert len(state["nodes"]) == 2
    assert all(len(p) <= 2 for t in state["nodes"] for p in t["relation_paths"].values())
    assert 29 not in state["nodes"][1]["evidence_ids"]


@pytest.mark.parametrize("budget", [0, 200, 1800, 6600, None])
def test_render_is_bounded_whole_bundles_and_cites_only_real_sources(budget):
    value = snapshot()
    text, ids, edges, tasks = render_task_state(value, budget)
    assert budget is None or len(text) <= budget
    assert all(f"[B{i}]" in text for i in ids)
    assert set(ids) <= {n["id"] for n in value["nodes"]}
    assert set(edges) <= {r["id"] for r in value["relations"]}
    if budget is None:
        assert set(tasks) == {"T0", "T1"}
        assert "[B4] plan:" in text and "[T4]" not in text
        assert "[B3] depends_on [B2]" in text
        assert "explicit task reference, not a factual edge" in text
    assert render_task_state({}, budget) == ("", [], [], [])


@pytest.mark.parametrize("status,phrase", [
    ("open", "reported open."),
    ("in_progress", "reported in progress."),
    ("blocked", "reported blocked."),
    ("reported_complete", "reported complete (not verified)."),
    ("cancelled", "reported cancelled."),
])
def test_render_explains_reported_status_and_keeps_original_sources(status, phrase):
    value = snapshot()
    value["nodes"][-1]["task_status"] = status
    value["task_state"] = build_task_state(value)
    before = copy.deepcopy(value)
    text, ids, edges, tasks = render_task_state(value, None)
    assert f"- [T1] source: [B1] | status: {phrase}" in text
    assert "- [T0] source: [B0] | status: unknown (no progress reported)." in text
    assert text.splitlines()[:2] == [
        "## Task state",
        "[Tn] tracks the task recorded in [Bn]. Status reflects reported progress, "
        "not verified completion. Unknown means no progress was recorded.",
    ]
    for repeated_guidance in ("Plans are intent", "These are older records",
                              "avoid repeating completed work", "Paths do not prove completion"):
        assert repeated_guidance not in text
    assert "task |" not in text
    assert value == before, "rendering must not rewrite status, beliefs, edges or confidence"
    assert {0, 1, 2, 3, 4, 6} <= set(ids) and set(tasks) == {"T0", "T1"}
    assert edges == [0]
    for i in ids:
        original = next(n for n in value["nodes"] if n["id"] == i)
        assert original["belief"] in text


def test_render_reports_conflicts_without_overriding_claimed_completion():
    value = snapshot()
    value["nodes"][-1]["task_status"] = "reported_complete"
    value["relations"].append({"id": 9, "from_id": 2, "to_id": 3, "type": "contradicts"})
    value["task_state"] = build_task_state(value)
    text, *_ = render_task_state(value, None)
    assert "[T1] source: [B1] | status: reported complete (not verified). Conflicting evidence: review." in text


def test_natural_state_budget_keeps_whole_task_and_status_sources():
    value = snapshot()
    # One task, no optional path or plan, makes the exact-fit boundary deterministic.
    value["nodes"] = [value["nodes"][1], value["nodes"][6]]
    value["relations"] = []
    value["task_state"] = build_task_state(value)
    text, ids, edges, tasks = render_task_state(value, None)
    # The renderer budgets a trailing newline before stripping it for the prompt.
    assert render_task_state(value, len(text) + 1) == (text, ids, edges, tasks)
    assert render_task_state(value, len(text)) == ("", [], [], [])
    assert ids == [1, 6] and tasks == ["T1"] and not edges


def test_task_history_keeps_old_anchors_inside_cap_without_edges():
    value = snapshot()
    history_text = task_history(value["nodes"][-2:], value["nodes"], 1200)
    rows = json.loads(history_text)
    assert len(history_text) <= 1200 and [n["id"] for n in rows] == sorted(n["id"] for n in rows)
    assert any(n["id"] == 1 and n["task_role"] == "task" for n in rows)
    assert all("confidence" not in n and "relations" not in n for n in rows)
    assert task_history([], [], 0) == ""


def test_extraction_reuses_one_model_call_and_rejects_unshown_task_ids(monkeypatch):
    calls = []
    def respond(client, model, prompt, **kwargs):
        calls.append(prompt)
        return json.dumps({"beliefs": [{"belief": "Verification is blocked.", "stance": "asserted",
            "entities": [], "supporting_sentence_indices": [0], "task_role": "progress",
            "task_ids": [0, 1234], "task_status": "blocked"}], "decisions": []})
    monkeypatch.setattr(extract.llm, "call_model", respond)
    displayed = task_history([], [node(0, "task"), node(1, "plan", task_ids=[0])], 12000)
    result = extract.extract_nodes(None, "fixture", role="assistant", mode="sentence",
        sentences=["Verification is blocked."], graph_nodes_str=displayed, task_state_enabled=True)
    assert len(calls) == 1 and "## Task-state annotations" in calls[0]
    assert result["nodes"][0]["task_ids"] == [0]
    assert result["nodes"][0]["task_rejected_ids"] == [1234]
    base = build_node_extraction_prompt("assistant", sentences_block="[0] Source.")
    assert base == build_node_extraction_prompt("assistant", sentences_block="[0] Source.", task_state_enabled=False)
    assert "Task-state annotations" not in base
    rule = extract.extract_nodes(None, "fixture", role="assistant", content='<tool_call>{"name":"web_search","arguments":{"query":"license"}}</tool_call>', task_state_enabled=True)
    assert rule["nodes"][0]["extraction_method"] == "rule_tool_call"
    assert len(calls) == 1, "actual tool calls remain deterministic"


def test_stream_and_sdk_keep_task_state_in_snapshots(tmp_path, monkeypatch):
    calls = []
    def respond(client, model, prompt, **kwargs):
        calls.append(prompt)
        text = "Compare A and B." if len(calls) == 1 else "The comparison is blocked."
        return json.dumps({"beliefs": [{"belief": text, "stance": "asserted", "entities": [],
            "supporting_sentence_indices": [0], "task_role": "task" if len(calls) == 1 else "progress",
            "task_ids": [] if len(calls) == 1 else [0], "task_status": "blocked"}]})
    monkeypatch.setattr(extract.llm, "call_model", respond)
    monkeypatch.setattr(stream, "extract_layered_relations", lambda *a, **kw: {"relations": []})
    monkeypatch.setattr(stream, "extract_relations", lambda *a, **kw: {"relations": []})
    builder = stream.StreamingBeliefBuilder(client=None, model="fixture", item_id="task",
        out_dir=tmp_path, options=stream.StreamOptions(task_state_enabled=True, incremental_merge=False))
    builder.ingest_turn("user", "Compare A and B.")
    builder.ingest_turn("assistant", "The comparison is blocked.")
    assert len(calls) == 2, "no independent task-state generation calls"
    native = builder.graph.snapshot()
    assert native["task_state"]["nodes"][0]["status"] == "blocked"
    graph = _bcg_from_construct(native, sessions=[], metadata={})
    result = _memory_document(graph, native=native, run_id="task", trajectory=[], options_dict={}, metadata={}, engine="unified")
    assert result["task_state"] == native["task_state"]
    assert result["nodes"][0]["task_role"] == "task"
    builder.finalize()
    assert json.loads((tmp_path / "final_graph.json").read_text())["task_state"] == native["task_state"]
    assert "task_state" not in BeliefGraph().snapshot()


def test_schema_and_agent_flags_are_strict_and_opt_in():
    defaults = defaults_dict()
    assert not BCGSettings.from_agent({}).task_state_enabled
    defaults.setdefault("pipeline", {})["task_state"] = {"enabled": True}
    assert GraphSettings.model_validate(defaults).pipeline.task_state.enabled
    opts = stream.StreamOptions()
    opts.apply_belief_graph_config(defaults["pipeline"])
    assert opts.task_state_enabled and opts.to_dict()["task_state"]["enabled"]
    defaults["backend"] = "hybrid"
    with pytest.raises(ValidationError, match="unified"):
        GraphSettings.model_validate(defaults)
    with pytest.raises(ValueError, match="boolean"):
        BCGSettings.from_agent({"bcg": {"task_state_enabled": "false"}})


class TaskGraph(FakeGraph):
    def snapshot(self, pid):
        value = super().snapshot(pid)
        labelled = snapshot()
        value["nodes"], value["relations"], value["task_state"] = labelled["nodes"], labelled["relations"], labelled["task_state"]
        return value


@pytest.mark.asyncio
@pytest.mark.parametrize("retrieval", ["focused", "all"])
async def test_agent_injection_uses_same_retention_and_graph_budget(tmp_path, retrieval):
    graph = TaskGraph()
    memory = BCGMemory(BCGSettings(retrieval=retrieval, task_state_enabled=True), tmp_path / "on", "task", backend=graph)
    baseline = BCGMemory(BCGSettings(retrieval=retrieval), tmp_path / "off", "off", backend=TaskGraph())
    assert await memory.prepare(history(2)) == history(2), "no early context eviction"
    messages = history(3)
    before = copy.deepcopy(messages)
    projected = await memory.prepare(messages)
    original = await baseline.prepare(messages)
    assert projected[1:] == original[1:] and messages == before
    assert memory.last_event["fallback"] is None
    assert "## Task state" in projected[0]["content"] and "## Task state" not in original[0]["content"]
    assert memory.last_event["task_state"]["task_ids"]
    assert memory.last_event["graph_chars"] <= 6600
    assert all(i in memory.last_event["selected_node_ids"] for i in memory.last_event["task_state"]["source_node_ids"])
    if retrieval == "focused":
        assert graph.selections[-1][-1] < 6600
    else:
        assert not graph.selections
    await memory.flush_final([*messages, {"role": "assistant", "content": "Final report."}])
    rows = [json.loads(line) for line in (tmp_path / "on" / "graph_snapshots.jsonl").read_text().splitlines()]
    assert rows[-1]["snapshot"]["task_state"]
    await memory.close()
    await baseline.close()


def test_legacy_replay_is_explicit_labelled_and_does_not_mutate_source():
    old = snapshot()
    for n in old["nodes"]:
        for key in ("task_role", "task_ids", "task_status"):
            n.pop(key, None)
    old.pop("task_state")
    before = copy.deepcopy(old)
    replay = classify_legacy_snapshot(old, {0: {"task_role": "task"}})
    assert old == before and len(replay["task_state"]["nodes"]) == 1
    assert replay["nodes"][0]["task_annotation_source"] == "offline_explicit_label"


def test_excerpt_without_source_match_cannot_create_task(tmp_path):
    builder = stream.StreamingBeliefBuilder(client=None, model="fixture", item_id="excerpt",
        out_dir=tmp_path, options=stream.StreamOptions(task_state_enabled=True, evidence_mode="excerpt"))
    cleaned = {"belief": "Do something absent from the source.", "stance": "asserted",
               "task_role": "task", "supporting_excerpts": ["This text was not said."]}
    evidence = builder._evidence_for(cleaned, "Unrelated source.", {}, "excerpt", "assistant")
    result = builder._make_node(cleaned, {}, evidence, "assistant")
    assert result["task_role"] == "other"
    assert not builder.graph.snapshot()["task_state"]["nodes"]


@pytest.mark.parametrize("task_role,source", [
    ("task", "Verify the operating license."),
    ("plan", "Search the licensing authority's registry."),
    ("progress", "License verification is blocked."),
])
def test_task_sources_are_not_lost_to_incremental_merge(tmp_path, monkeypatch, task_role, source):
    from types import SimpleNamespace

    excluded = []
    def fake_merge(**kwargs):
        excluded.append(kwargs["exclude_node_ids"])
        return {"skipped": True}

    monkeypatch.setattr(stream, "run_merge_pass", fake_merge)
    monkeypatch.setattr(stream, "extract_nodes", lambda *a, **kw: {"nodes": [
        {"tmp_id": "n0", "belief": source, "stance": "asserted",
         "supporting_sentence_indices": [0], "task_role": task_role}]})
    builder = stream.StreamingBeliefBuilder(client=None, model="fixture", item_id="merge",
        out_dir=tmp_path, options=stream.StreamOptions(task_state_enabled=True),
        embedder=SimpleNamespace(set_log_path=lambda _: None))
    builder.ingest_turn("user", source)
    assert excluded == [{0}]


def test_demoted_decision_remains_an_intermediate_conclusion(tmp_path):
    builder = stream.StreamingBeliefBuilder(client=None, model="fixture", item_id="decision",
        out_dir=tmp_path, options=stream.StreamOptions(task_state_enabled=True))
    for i in (0, 1):
        builder.graph.add_belief(node(i, "decision", node_type="decision"))
    builder._keep_only_latest_decision()
    assert builder.graph.beliefs[0]["task_role"] == "intermediate_conclusion"
    assert builder.graph.beliefs[1]["task_role"] == "decision"
    assert not builder.graph.snapshot()["task_state"]["nodes"]


def test_replay_command_artifacts_and_non_overwrite(tmp_path):
    from tools.task_state_replay import replay

    source = tmp_path / "state.json"
    labels = tmp_path / "labels.json"
    source.write_text(json.dumps({"snapshot": snapshot()}))
    labels.write_text(json.dumps({0: {"task_role": "task"}}))
    original = source.read_bytes()
    summary = replay(source, labels, tmp_path / "output")
    assert summary["model_calls"] == 0 and summary["automatic_labelling_evaluated"] is False
    assert source.read_bytes() == original
    assert (tmp_path / "output/task-context.md").is_file()
    with pytest.raises(FileExistsError):
        replay(source, labels, tmp_path / "output")


@pytest.mark.parametrize("legacy,current", [("goal", "task"), ("subgoal", "task"), ("next_action", "plan")])
def test_legacy_role_aliases_are_read_as_current_roles(legacy, current):
    assert canonical_task_role(legacy) == current
    assert clean_annotation({"task_role": legacy})["task_role"] == current
    value = node(1)
    classify_node(value, {"task_role": legacy, "task_ids": [0]}, [node(0, "goal")])
    assert value["task_role"] == current and value["task_ids"] == [0]


def test_only_tasks_become_records_and_unassigned_plans_stay_beliefs():
    value = snapshot()
    value["nodes"].append(node(7, "plan", "Try a different search query."))
    value["nodes"].append(node(8, "next_action", "Fetch a different page."))
    before = copy.deepcopy(value)
    state = build_task_state(value)
    assert {n["id"] for n in state["nodes"]} == {"T0", "T1"}
    assert state["unassigned_plan_ids"] == [7, 8]
    assert value == before
    assert all(link["from_id"] not in {"T4", "T7", "T8"} for link in state["relations"])


def test_latest_explicit_plan_is_rendered_under_task_with_budget():
    value = snapshot()
    value["nodes"].append(node(7, "plan", "Use the regulator's registry instead.", task_ids=[1]))
    value["task_state"] = build_task_state(value)
    text, ids, _, tasks = render_task_state(value, None)
    assert value["task_state"]["nodes"][1]["plan_ids"] == [4, 7]
    assert "[B7] plan: Use the regulator's registry instead." in text
    assert 7 in ids and "T7" not in tasks and "[B4] plan:" not in text
    value["nodes"][-1]["belief"] = "Oversized plan. " * 1000
    text, ids, _, tasks = render_task_state(value, 1800)
    assert len(text) <= 1800 and "T1" in tasks and 7 not in ids
    assert "[B6] progress:" in text


def test_plan_only_history_does_not_create_tasks_or_injected_task_state():
    value = {"nodes": [node(0, "plan"), node(1, "next_action")], "relations": []}
    value["task_state"] = build_task_state(value)
    assert value["task_state"]["nodes"] == []
    assert value["task_state"]["unassigned_plan_ids"] == [0, 1]
    assert render_task_state(value, None) == ("", [], [], [])


def test_legacy_projection_keeps_hierarchy_without_promoting_plan_completion():
    value = {"nodes": [
        node(0, "goal"), node(1, "subgoal", task_ids=[0]),
        node(2, "plan", task_ids=[1]), node(3, "next_action", task_ids=[2]),
        node(4, "progress", task_ids=[3], task_status="reported_complete"),
        node(5, "plan", task_ids=[6]), node(6, "next_action", task_ids=[5]),
    ], "relations": [], "task_state": {"version": 1, "nodes": [{"id": "T3", "kind": "next_action"}]}}
    before = copy.deepcopy(value)
    state = build_task_state(value)
    task = state["nodes"][1]
    assert task["parent_ids"] == ["T0"] and task["plan_ids"] == [2, 3]
    assert task["status"] == "open" and task["status_source_id"] is None
    assert state["unassigned_progress_ids"] == [4]
    assert state["unassigned_plan_ids"] == [5, 6]
    text, _, _, tasks = render_task_state(value, None)
    assert set(tasks) == {"T0", "T1"} and "[B3] plan:" in text
    assert "next_action" not in text and "subgoal" not in text
    assert value == before


def test_history_prioritizes_tasks_not_recent_plans_and_normalizes_labels():
    values = [node(0, "goal"), node(1, "subgoal", task_ids=[0])]
    values += [node(i, "next_action", task_ids=[1]) for i in range(2, 42)]
    before = copy.deepcopy(values)
    history = json.loads(task_history(values[-2:], values, 12000))
    assert [row["id"] for row in history] == [0, 1, 40, 41]
    assert [row["task_role"] for row in history] == ["task", "task", "plan", "plan"]
    assert values == before


def test_new_extraction_can_reference_shown_tasks_but_never_plans(monkeypatch):
    calls = []
    def respond(client, model, prompt, **kwargs):
        calls.append(prompt)
        return json.dumps({"beliefs": [{"belief": "The work is blocked.", "stance": "asserted",
            "entities": [], "supporting_sentence_indices": [0], "task_role": "progress",
            "task_ids": [0, 1, 2], "task_status": "blocked"}]})
    monkeypatch.setattr(extract.llm, "call_model", respond)
    shown = json.dumps([node(0, "goal"), node(1, "plan", task_ids=[0]), node(2, "next_action", task_ids=[0])])
    out = extract.extract_nodes(None, "fixture", role="assistant", mode="sentence",
        sentences=["The work is blocked."], graph_nodes_str=shown, task_state_enabled=True)
    assert len(calls) == 1
    assert out["nodes"][0]["task_ids"] == [0]
    assert out["nodes"][0]["task_rejected_ids"] == [1, 2]
    assert "Only task creates a task record." in EXTRACTION_GUIDANCE
    assert "Never reference\na plan as a task." in EXTRACTION_GUIDANCE
    assert "goal/subgoal/plan/next_action" not in EXTRACTION_GUIDANCE
