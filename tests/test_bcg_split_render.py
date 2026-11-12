"""Split graph layout (completed actions / findings / relations) for the single agent."""

from __future__ import annotations

import asyncio
import copy

import pytest

from tests.test_react_bcg import FakeGraph, history
from workflows.stateful_react_agent.bcg import pcst
from workflows.stateful_react_agent.bcg.client import GraphError
from workflows.stateful_react_agent.bcg.rendering import (
    CONFIDENCE_LABELS,
    GRAPH_OVERHEAD,
    SPLIT_ACTIONS_HEADER,
    SPLIT_GRAPH_OVERHEAD,
    classify_node,
    confidence_label,
    node_line,
    render_graph,
    styled_node_line,
)
from workflows.stateful_react_agent.bcg.runtime import (
    CONTEXT_GUIDE,
    CONTEXT_GUIDE_SPLIT,
    CONTEXT_GUIDE_SPLIT_CONFIDENCE,
    BCGMemory,
    BCGSettings,
)

SEARCH = (
    'The assistant is using web_search to search for '
    '\\"\\"terminal wealth standard deviation\\" portfolio diversification stocks\\".'
)


def snapshot():
    return {
        "problem_id": "split",
        "nodes": [
            {"id": 65, "belief": SEARCH, "confidence": 0.78, "source": {"turn_id": 9}},
            {
                "id": 74,
                "belief": "The web_search tool returned [BLOCKED] content: source is unavailable.",
                "confidence": 0.88,
                "source": {"turn_id": 10},
            },
            {
                "id": 68,
                "belief": "The web_search tool returned a page that describes TWSD as dispersion.",
                "confidence": 0.88,
                "source": {"turn_id": 10},
            },
            {
                "id": 21,
                "belief": "Statman (1987) states at least 30 stocks for a borrowing investor.",
                "confidence": 0.88,
                "source": {"turn_id": 4},
            },
            {
                "id": 40,
                "belief": "The assistant is calling the web_fetch tool on the Statman paper.",
                "confidence": 0.78,
                "source": {"turn_id": 3},
            },
        ],
        "relations": [
            {"id": 1, "from_id": 68, "to_id": 65, "type": "depends_on", "note": "result of call"},
            {"id": 2, "from_id": 21, "to_id": 68, "type": "supplements", "note": "same topic"},
        ],
    }


def test_classification_keeps_tool_results_in_findings():
    nodes = {n["id"]: n for n in snapshot()["nodes"]}
    assert classify_node(nodes[65])["is_action"] and classify_node(nodes[65])["tool"] == "web_search"
    assert classify_node(nodes[74])["blocked"] and classify_node(nodes[74])["tool"] == "web_search"
    assert not classify_node(nodes[68])["is_action"]
    assert not classify_node(nodes[21])["is_action"]
    assert classify_node(nodes[40])["is_action"] and classify_node(nodes[40])["query"] is None


def test_split_sections_order_and_compact_action_lines():
    snap = snapshot()
    selection = {"node_ids": [21, 40, 65, 68, 74], "relation_ids": [1, 2]}
    text, ids, edges = render_graph(snap, selection, 6600, style="split")
    lines = text.splitlines()
    actions = lines[lines.index(SPLIT_ACTIONS_HEADER) + 1 : lines.index("## Findings") - 1]
    findings = lines[lines.index("## Findings") + 1 : lines.index("## Relations") - 1]
    # Chronological by turn, no confidence, template prose removed.
    assert actions == [
        "- [B40] The assistant is calling the web_fetch tool on the Statman paper.",
        '- [B65] web_search "\\"terminal wealth standard deviation\\" portfolio diversification stocks"',
        "- [B74] web_search returned a blocked source",
    ]
    assert all("confidence" not in line for line in actions)
    assert [line.split("]")[0] for line in findings] == ["- [B21", "- [B68"]
    assert all("(confidence 0.88)" in line for line in findings)
    assert "- [B68] depends_on [B65]: result of call" in text
    assert ids == [21, 40, 65, 68, 74] and edges == [1, 2]


def test_split_findings_follow_pcst_prizes_and_selector_order():
    snap = snapshot()
    prized = {
        "node_ids": [21, 65, 68],
        "relation_ids": [1, 2],
        "selected_prizes": {"21": 0.1, "65": 0.9, "68": 0.7},
    }
    text, _, _ = render_graph(snap, prized, 6600, style="split")
    assert text.index("[B68] The web_search") < text.index("[B21] Statman")
    # Without prizes (focused selector) the selector's own order is preserved.
    text, _, _ = render_graph(snap, {"node_ids": [21, 68], "relation_ids": [2]}, 6600, style="split")
    assert text.index("[B21] Statman") < text.index("[B68] The web_search")


def test_split_without_actions_omits_the_section():
    snap = snapshot()
    text, _, _ = render_graph(snap, {"node_ids": [21, 68], "relation_ids": [2]}, 6600, style="split")
    assert SPLIT_ACTIONS_HEADER not in text
    assert text.startswith("<belief_graph>\n## Findings\n")


def test_split_trim_drops_least_relevant_not_last_displayed():
    snap = snapshot()
    selection = {"node_ids": [65, 21, 68], "relation_ids": [1, 2]}
    full, _, _ = render_graph(snap, selection, 6600, style="split")
    _, ids, edges = render_graph(snap, selection, len(full) - 1, style="split")
    assert 68 not in ids and set(ids) == {21, 65} and edges == []
    with pytest.raises(GraphError):
        render_graph(snap, selection, len(full) - 1, style="split", strict=True)


def test_v0_layout_is_unchanged_by_default():
    snap = snapshot()
    selection = {"node_ids": [21, 65], "relation_ids": []}
    text, _, _ = render_graph(snap, selection, 6600)
    nodes = {n["id"]: n for n in snap["nodes"]}
    assert text == "\n".join(
        ["<belief_graph>", "## Beliefs", node_line(nodes[21]), node_line(nodes[65]),
         "", "## Relations", "</belief_graph>"]
    )
    with pytest.raises(GraphError):
        render_graph(snap, selection, 6600, style="tiered")


def test_split_cost_model_is_exact_with_actions_and_conservative_without():
    snap = snapshot()
    nodes = {n["id"]: n for n in snap["nodes"]}
    ids = [21, 65, 68]
    text, _, _ = render_graph(snap, {"node_ids": ids, "relation_ids": []}, 6600, style="split")
    assert len(text) == SPLIT_GRAPH_OVERHEAD + sum(len(styled_node_line(nodes[i], "split")) + 1 for i in ids)
    text, _, _ = render_graph(snap, {"node_ids": [21], "relation_ids": []}, 6600, style="split")
    assert len(text) < SPLIT_GRAPH_OVERHEAD + len(styled_node_line(nodes[21], "split")) + 1
    assert SPLIT_GRAPH_OVERHEAD > GRAPH_OVERHEAD


def test_pcst_split_budget_never_fails_strict_render():
    pytest.importorskip("pcst_fast")
    snap = snapshot()
    prizes = {65: 1.0, 68: 0.8, 21: 0.6, 74: 0.2, 40: 0.1}
    full = pcst.select_prized_tree(snap, prizes, 6600, style="split")
    assert full["node_ids"]
    for budget in range(SPLIT_GRAPH_OVERHEAD, full["rendered_chars"] + 40):
        selection = pcst.select_prized_tree(snap, prizes, budget, style="split")
        if selection["node_ids"]:
            text, _, _ = render_graph(snap, selection, budget, strict=True, style="split")
            assert len(text) <= budget


def test_settings_default_and_validation(monkeypatch):
    monkeypatch.delenv("BCG_RENDER_STYLE", raising=False)
    assert BCGSettings.from_agent({}).render_style == "split"
    monkeypatch.setenv("BCG_RENDER_STYLE", "v0")
    assert BCGSettings.from_agent({}).render_style == "v0"
    assert BCGSettings.from_agent({"bcg": {"render_style": "split"}}).render_style == "split"
    with pytest.raises(ValueError):
        BCGSettings.from_agent({"bcg": {"render_style": "tiered"}})


@pytest.mark.parametrize("style", ["split", "v0"])
def test_memory_injects_guide_matching_style(tmp_path, style):
    async def run():
        memory = BCGMemory(
            BCGSettings(retrieval="focused"), tmp_path, "session",
            backend=FakeGraph(), render_style=style,
        )
        messages = history(3)
        original = copy.deepcopy(messages)
        projected = await memory.prepare(messages)
        injected = memory.last_event["system_memory"]
        assert memory.last_event["render_style"] == style
        assert projected[0]["content"].startswith("original instructions\n\n")
        if style == "split":
            assert injected.startswith(CONTEXT_GUIDE_SPLIT + "\n\n<belief_graph>\n")
            assert "## Findings" in injected and "## Beliefs" not in injected
        else:
            assert injected.startswith(CONTEXT_GUIDE + "\n\n<belief_graph>\n## Beliefs")
        assert messages == original
        await memory.close()

    asyncio.run(run())


def test_memory_defaults_to_legacy_layout_for_subclasses(tmp_path):
    memory = BCGMemory(BCGSettings(), tmp_path, "session", backend=FakeGraph())
    assert memory.render_style == "v0" and memory.context_guide == CONTEXT_GUIDE


# ── split_confidence: high / medium / low confidence tiers ─────────────────


def node(confidence, **extra):
    return {"id": 1, "belief": "claim", "confidence": confidence, **extra}


@pytest.mark.parametrize(
    ("confidence", "label"),
    [
        (0.999, "high confidence"),
        (0.893, "high confidence"),   # user, confirmed
        (0.733, "high confidence"),   # scholarly publisher, confirmed
        (0.70, "high confidence"),    # inclusive boundary
        (0.699, "medium confidence"),
        (0.677, "medium confidence"),  # assistant, confirmed
        (0.545, "medium confidence"),  # general web, confirmed
        (0.50, "medium confidence"),   # inclusive boundary
        (0.499, "low confidence"),
        (0.429, "low confidence"),     # general web, probable
        (0.3, "low confidence"),
        (0.001, "low confidence"),
    ],
)
def test_tier_follows_final_confidence(confidence, label):
    assert confidence_label(node(confidence)) == label


def test_tier_ignores_role_stance_and_prior():
    # Only the propagated value counts; provenance no longer picks the label.
    assert confidence_label(node(0.4, role="user", stance="confirmed")) == "low confidence"
    assert confidence_label(node(0.8, role="assistant", stance="possible",
                                 initial_confidence=0.43)) == "high confidence"


def test_missing_or_invalid_confidence_has_no_tier():
    bare = {"id": 1, "belief": "claim"}
    assert confidence_label(bare) is None
    for bad in (None, True, "0.8", float("nan"), float("inf")):
        assert confidence_label({**bare, "confidence": bad}) is None
    assert CONFIDENCE_LABELS == ("high confidence", "medium confidence", "low confidence")


def test_thresholds_can_be_overridden(monkeypatch):
    monkeypatch.setenv("BCG_CONFIDENCE_HIGH", "0.8")
    monkeypatch.setenv("BCG_CONFIDENCE_LOW", "0.6")
    assert confidence_label(node(0.75)) == "medium confidence"
    assert confidence_label(node(0.55)) == "low confidence"
    monkeypatch.setenv("BCG_CONFIDENCE_LOW", "0.9")
    with pytest.raises(ValueError):
        confidence_label(node(0.75))
    monkeypatch.setenv("BCG_RENDER_STYLE", "split_confidence")
    with pytest.raises(ValueError):
        BCGSettings.from_agent({})


def labelled_snapshot():
    snap = snapshot()
    extra = {
        21: {"role": "tool", "stance": "confirmed", "initial_confidence": 0.733, "confidence": 0.863},
        68: {"role": "tool", "stance": "probable", "initial_confidence": 0.429, "confidence": 0.429},
        40: {"role": "assistant", "stance": "confirmed", "initial_confidence": 0.677, "confidence": 0.677},
    }
    for n in snap["nodes"]:
        n.update(extra.get(n["id"], {}))
    snap["nodes"].append({
        "id": 90, "belief": "Statman's 30-stock figure is probably still valid.",
        "role": "assistant", "stance": "probable", "initial_confidence": 0.533, "confidence": 0.533,
        "source": {"turn_id": 12},
    })
    return snap


def test_split_confidence_replaces_numbers_and_keeps_split_layout():
    snap = labelled_snapshot()
    selection = {"node_ids": [21, 40, 65, 68, 74, 90], "relation_ids": [1, 2]}
    text, ids, _ = render_graph(snap, selection, 6600, style="split_confidence")
    assert "confidence 0." not in text
    assert "- [B21] Statman (1987) states at least 30 stocks for a borrowing investor. (high confidence)" in text
    assert "(medium confidence)" in text and "(low confidence)" in text
    actions = text.split(SPLIT_ACTIONS_HEADER)[1].split("## Findings")[0]
    assert "(" not in actions.replace("(blocked source)", "")  # actions never carry a label
    split_text, _, _ = render_graph(snap, selection, 6600, style="split")
    assert split_text.split("## Findings")[0] == text.split("## Findings")[0]
    assert ids == [21, 40, 65, 68, 74, 90]


def test_pcst_split_confidence_budget_is_exact():
    pytest.importorskip("pcst_fast")
    snap = labelled_snapshot()
    prizes = {65: 1.0, 68: 0.8, 21: 0.6, 74: 0.2, 40: 0.1, 90: 0.5}
    full = pcst.select_prized_tree(snap, prizes, 6600, style="split_confidence")
    assert full["node_ids"]
    for budget in range(SPLIT_GRAPH_OVERHEAD, full["rendered_chars"] + 40):
        selection = pcst.select_prized_tree(snap, prizes, budget, style="split_confidence")
        if selection["node_ids"]:
            text, _, _ = render_graph(snap, selection, budget, strict=True, style="split_confidence")
            assert len(text) <= budget


def test_split_confidence_setting_and_guide(tmp_path, monkeypatch):
    monkeypatch.setenv("BCG_RENDER_STYLE", "split_confidence")
    assert BCGSettings.from_agent({}).render_style == "split_confidence"
    memory = BCGMemory(BCGSettings(), tmp_path, "s", backend=FakeGraph(), render_style="split_confidence")
    assert memory.context_guide == CONTEXT_GUIDE_SPLIT_CONFIDENCE
    for label in CONFIDENCE_LABELS:
        assert label in CONTEXT_GUIDE_SPLIT_CONFIDENCE
