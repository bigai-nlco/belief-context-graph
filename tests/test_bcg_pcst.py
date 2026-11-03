"""PCST invariants and model-boundary integration; no downloaded model needed."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from workflows.stateful_react_agent.bcg import pcst
from workflows.stateful_react_agent.bcg.client import GraphError
from workflows.stateful_react_agent.bcg.rendering import GRAPH_OVERHEAD, node_line, render_graph
from workflows.stateful_react_agent.bcg.runtime import BCGMemory, BCGSettings


def graph():
    return {
        "problem_id": "test",
        "nodes": [
            {"id": i, "belief": text, "confidence": confidence, "source": {"turn_id": 2}}
            for i, text, confidence in [
                (10, "小林", 1.0), (20, "王教授", 0.0),
                (30, "海川大学", 1.0), (40, "篮球", 0.0),
            ]
        ],
        "relations": [
            {"id": 1, "from_id": 10, "to_id": 20, "type": "depends_on", "note": "导师"},
            {"id": 2, "from_id": 30, "to_id": 20, "type": "supplements", "note": "工作单位"},
            {"id": 3, "from_id": 10, "to_id": 40, "type": "supplements", "note": "爱好"},
        ],
    }


@pytest.fixture
def solver():
    pytest.importorskip("pcst_fast")
    return pcst.select_prized_tree


def assert_connected(snapshot, selection):
    ids = set(selection["node_ids"])
    edges = [e for e in snapshot["relations"] if e["id"] in selection["relation_ids"]]
    assert len(edges) == len(ids) - 1
    seen, todo = set(), [min(ids)]
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        for e in edges:
            if i in (e["from_id"], e["to_id"]):
                todo.extend([e["from_id"], e["to_id"]])
    assert seen == ids


def test_zero_prize_bridge_survives_and_unrewarded_leaf_does_not(solver):
    snapshot = graph()
    original = copy.deepcopy(snapshot)
    selected = solver(snapshot, {10: 2, 30: 1}, 1000)
    assert selected["node_ids"] == [10, 20, 30]
    assert selected["relation_ids"] == [1, 2]
    assert selected["objective"] == pytest.approx(2.6)
    text, _, _ = render_graph(snapshot, selected, 1000, strict=True)
    assert "[B30] supplements [B20]" in text
    assert snapshot == original


def test_budget_reduction_preserves_connectivity_and_exact_render_size(solver):
    snapshot = graph()
    for budget in [90, 140, 210, 290, 1000]:
        selected = solver(snapshot, {10: 2, 30: 1}, budget)
        assert_connected(snapshot, selected)
        text, ids, edges = render_graph(snapshot, selected, budget, strict=True)
        assert len(text) == selected["rendered_chars"] <= budget
        assert ids == selected["node_ids"] and edges == selected["relation_ids"]


def test_disconnected_input_selects_one_component_without_inventing_edges(solver):
    snapshot = graph()
    snapshot["nodes"].append({"id": 50, "belief": "孤立证据", "confidence": 1.0})
    selected = solver(snapshot, {10: 2, 30: 1, 50: 2.5}, 1000)
    assert selected["node_ids"] == [10, 20, 30]
    assert_connected(snapshot, selected)
    snapshot["relations"] = []
    selected = solver(snapshot, {10: 2, 50: 2.5}, 1000)
    assert selected["node_ids"] == [50] and selected["relation_ids"] == []


def test_oversized_bridge_is_infeasible_but_never_cut_or_skipped(solver):
    snapshot = graph()
    snapshot["nodes"][1]["belief"] = "long bridge " * 1000
    selected = solver(snapshot, {10: 2, 30: 1}, 300)
    assert selected["node_ids"] == [10]
    assert selected["oversized_nodes"] == 1
    assert_connected(snapshot, selected)


def test_escaping_notes_and_confidence_count_towards_budget(solver):
    snapshot = graph()
    snapshot["nodes"][0]["belief"] = "<belief>&" * 4
    snapshot["relations"][0]["note"] = "<relation>&" * 10
    for budget in [230, 420, 900]:
        selection = solver(snapshot, {10: 2, 30: 1}, budget)
        rendered, _, _ = render_graph(snapshot, selection, budget, strict=True)
        assert len(rendered) <= budget
        assert_connected(snapshot, selection)


def test_strict_renderer_refuses_to_destroy_selected_tree():
    snapshot = graph()
    selection = {"node_ids": [10, 20, 30], "relation_ids": [1, 2]}
    with pytest.raises(GraphError, match="selection_exceeds_render_budget"):
        render_graph(snapshot, selection, 100, strict=True)
    one = {"node_ids": [10], "relation_ids": []}
    text, _, _ = render_graph(snapshot, one, 1000, strict=True)
    assert len(text) == GRAPH_OVERHEAD + len(node_line(snapshot["nodes"][0])) + 1


def test_no_prizes_or_no_feasible_node_returns_empty(solver):
    assert solver(graph(), {}, 1000)["node_ids"] == []
    assert solver(graph(), {10: 1}, 5)["node_ids"] == []


def test_query_prizes_apply_confidence_floor_and_keep_bridge(solver, monkeypatch):
    encoder = SimpleNamespace(score=lambda *args: ([1.0, 0.0, 1.0, 0.0], {"model": "fake"}))
    monkeypatch.setattr(pcst, "_encoder", lambda *args: encoder)
    snapshot = graph()
    snapshot["nodes"][0]["confidence"] = 0.0
    settings = BCGSettings(retrieval="query_pcst", pcst_edge_cost=0.05)
    result = pcst.PCSTSelector(settings).select(snapshot, "question", "focus", 1000)
    assert result["node_ids"] == [10, 20, 30]
    assert result["selected_prizes"]["10"] == pytest.approx(0.2)
    assert result["selected_prizes"]["20"] == 0.0
    assert result["selected_prizes"]["30"] == 1.0
    assert result["retrieval"] == "embedding"


def test_confidence_mode_does_not_load_encoder(solver, monkeypatch):
    def unexpected(*args):
        pytest.fail("confidence retrieval must not use an encoder")
    monkeypatch.setattr(pcst, "_encoder", unexpected)
    result = pcst.PCSTSelector(BCGSettings(retrieval="confidence_pcst")).select(graph(), "q", "", 1000)
    assert result["node_ids"] == [10, 20, 30]


def test_focus_uses_latest_intent_and_query_without_tool_result_or_report():
    messages = [
        {"role": "user", "content": "initial task"},
        {"role": "assistant", "content": "old intent"},
        {"role": "assistant", "reasoning_content": "verify affiliation", "content": "",
         "tool_calls": [{"id": "a", "function": {"name": "web_search", "arguments": {"q": ["小林 海川大学"]}}}]},
        {"role": "tool", "tool_call_id": "a", "content": "huge page body" * 1000},
        {"role": "user", "bcg_origin": "agent_report", "content": "internal report body"},
    ]
    focus = pcst.retrieval_focus(messages)
    assert "小林 海川大学" in focus and "verify affiliation" in focus
    assert "body" not in focus and "old intent" not in focus
    assert pcst.retrieval_focus([*messages, {"role": "user", "content": "new task"}]) == ""


def test_encoder_chunks_both_question_and_nodes_without_silent_truncation(monkeypatch):
    np = pytest.importorskip("numpy")
    class Tokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2
        def encode(self, text, add_special_tokens=True, truncation=False, verbose=False):
            ids = [ord(c) for c in text]
            return [0, *ids, 0] if add_special_tokens else ids
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids)
    class Model:
        max_seq_length = 10
        tokenizer = Tokenizer()
        calls = 0
        def encode(self, texts, **kwargs):
            self.calls += 1
            assert all(len(self.tokenizer.encode(t)) <= self.max_seq_length for t in texts)
            return np.asarray([[1.0, 0.0] if "z" in t else [0.0, 1.0] for t in texts])
    encoder = pcst.LocalRelevance("fake", "cpu")
    encoder.model = Model()
    scores, audit = encoder.score("z" * 20, "z", ["a" * 16 + "z" * 8])
    assert scores == [1.0]  # Evidence at the end of a long node is visible.
    assert audit["question_chunks"] == 3 and audit["node_chunks"] == 3
    encoder.score("z" * 20, "z", ["a" * 16 + "z" * 8])
    assert encoder.model.calls == 1


@pytest.mark.parametrize("cfg", [
    {"retrieval": "pcst_typo"}, {"pcst_edge_cost": 0}, {"pcst_edge_cost": float("nan")},
    {"pcst_confidence_floor": -1}, {"pcst_confidence_floor": 1.1}, {"embedding_model": ""},
])
def test_bad_pcst_settings_are_rejected(cfg):
    with pytest.raises(ValueError):
        BCGSettings.from_agent({"bcg": cfg})


def test_pcst_environment_and_profile_precedence(monkeypatch):
    monkeypatch.setenv("BCG_RETRIEVAL", "confidence_pcst")
    monkeypatch.setenv("BCG_PCST_EDGE_COST", "0.3")
    settings = BCGSettings.from_agent({})
    assert settings.retrieval == "confidence_pcst" and settings.pcst_edge_cost == 0.3
    assert BCGSettings.from_agent({"bcg": {"retrieval": "query_pcst"}}).retrieval == "query_pcst"


@pytest.mark.asyncio
async def test_model_boundary_uses_local_pcst_and_records_exact_selection(tmp_path, solver):
    from tests.test_react_bcg import FakeGraph, history
    backend = FakeGraph()
    settings = BCGSettings(retrieval="confidence_pcst", graph_chars=1000)
    memory = BCGMemory(settings, tmp_path, "pcst", backend=backend)
    messages = history(5)
    projected = await memory.prepare(messages)
    assert not backend.selections
    assert memory.last_event["fallback"] is None
    assert memory.last_event["selected_node_ids"] == memory.last_event["selector_result"]["node_ids"]
    assert memory.last_event["selected_relation_ids"] == memory.last_event["selector_result"]["relation_ids"]
    assert memory.last_event["graph_chars"] <= 1000
    assert len(projected) < len(messages)
    assert_connected(memory.snapshot, memory.last_event["selector_result"])
    assert await memory.prepare(messages) == projected  # Cached prewarm projection.
    await memory.close()


@pytest.mark.asyncio
async def test_missing_solver_dependency_preserves_full_history(tmp_path, monkeypatch):
    from tests.test_react_bcg import FakeGraph, history
    original = pcst.importlib.import_module
    def missing(name):
        if name == "pcst_fast":
            raise ImportError("not installed")
        return original(name)
    monkeypatch.setattr(pcst.importlib, "import_module", missing)
    memory = BCGMemory(BCGSettings(retrieval="confidence_pcst"), tmp_path, "missing", backend=FakeGraph())
    messages = history(5)
    assert await memory.prepare(messages) == messages
    assert "pcst_dependency_missing" in memory.last_event["fallback"]
    await memory.close()


@pytest.mark.asyncio
async def test_zero_recent_retention_still_supplies_latest_focus(tmp_path, monkeypatch, solver):
    from tests.test_react_bcg import FakeGraph, history
    observed = []
    def score(question, focus, texts):
        observed.append(focus)
        return [1.0] * len(texts), {}
    monkeypatch.setattr(pcst, "_encoder", lambda *args: SimpleNamespace(score=score))
    memory = BCGMemory(
        BCGSettings(retrieval="query_pcst", recent_turns=0), tmp_path, "zero", backend=FakeGraph(),
    )
    await memory.prepare(history(5))
    assert memory.last_event["fallback"] is None
    assert "query 4/0" in observed[0] and "reason 4" in observed[0]
    await memory.close()


@pytest.mark.asyncio
async def test_team_reuses_pcst_with_its_local_share_of_the_budget(tmp_path, solver):
    from tests.test_team_bcg import Graph, history
    from workflows.agent_team.bcg import TeamMemory
    team = TeamMemory(BCGSettings(retrieval="confidence_pcst"), tmp_path, "team-pcst", backend_factory=Graph)
    memory = team.member("coordinator", agent="coordinator", question="Find the date")
    await memory.prepare(history(5))
    assert memory.last_event["fallback"] is None
    assert memory.last_event["selector_result"]["strategy"] == "confidence_pcst"
    assert memory.last_event["graph_chars"] <= 4600
    await memory.close()
