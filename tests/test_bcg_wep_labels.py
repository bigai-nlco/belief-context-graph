"""WEP labels for the planned split_wep styles, on nodes shaped like real snapshots."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from workflows.stateful_react_agent.bcg import rendering as r

ROOT = Path(__file__).resolve().parents[1]


def _hist(value, **extra):
    return [{"step": "initial", "value": value, "evidence_confidence": 0.0, "factor_confidence": 0.0}, *extra.get("more", [])]


# Shapes copied from a version1117_actions graph_snapshots.jsonl line.
USER = {"id": 1, "node_type": "belief", "role": "user", "stance": "confirmed",
        "belief": "The user requests explanations of VIX, GVZ, and OVX.", "confidence": 0.893,
        "evidence_confidence": 0, "factor_confidence": 0.0, "confidence_history": _hist(0.893)}
PLAN = {"id": 9, "node_type": "belief", "role": "assistant", "stance": "unknown",
        "belief": "The assistant plans to search for information about VIX, GVZ, and OVX.",
        "confidence": 0.558, "evidence_confidence": 0, "factor_confidence": 0.2325,
        "confidence_history": _hist(0.5, more=[{
            "step": "relation_propagation_after_layer_bundle", "value": 0.558, "delta": 0.058,
            "factor_confidence": 0.2325, "iteration": 1, "relation_contributions": [{
                "relation_id": 0, "relation_type": "depends_on", "input_id": 1, "output_id": 9,
                "weight": 0.5, "input_confidence": 0.893, "activation_strength": 0.465,
                "contribution": 0.2325}]}])}
ACTION = {"id": 10, "node_type": "belief", "role": "assistant", "stance": "confirmed",
          "belief": 'The assistant is using web_search to search for "VIX index CBOE volatility index".',
          "confidence": 0.677, "tool_name": "web_search", "extraction_method": "rule_tool_call",
          "confidence_history": _hist(0.677)}
CBOE = {"id": 15, "node_type": "belief", "role": "tool", "stance": "confirmed",
        "belief": "The Cboe Gold ETF Volatility Index (GVZ) estimates the expected 30-day volatility of GLD.",
        "source_category": "primary_official", "source_coefficient": 0.88, "confidence": 0.827,
        "evidence_confidence": 0, "factor_confidence": 0.0,
        "confidence_history": _hist(0.827, more=[{
            "step": "merge_evidence", "value": 0.827, "delta": 0.0, "evidence_confidence": 0,
            "evidence_delta": 0.0, "scored_evidence_ids": [], "added_evidence_ids": [],
            "from_belief_id": 16, "absorbed_belief_ids": [16]}])}


@pytest.mark.parametrize(("value", "label"), [
    (0.893, "confirmed"),        # confirmed x user
    (0.827, "confirmed"),        # confirmed x official
    (0.733, "almost certain"),   # confirmed x scholarly
    (0.677, "probable"),         # confirmed x assistant
    (0.545, "possible"),         # confirmed x general web
    (0.50, "possible"),          # unknown stance, any role
    (0.432, "possible"),         # possible x assistant: not "unlikely"
    (0.395, "possible"),         # confirmed x community: weak source, not doubt
    (0.30, "possible"),          # possible x unattributed (result digests): off the boundary
    (0.273, "unlikely"),         # unlikely x scholarly
    (0.19, "doubtful"),
])
def test_score_grid_maps_to_wep(value, label, monkeypatch):
    monkeypatch.delenv("BCG_WEP_THRESHOLDS", raising=False)
    assert r.wep_label_for(value) == label


def test_no_unknown_label_exists():
    assert "unknown" not in r.WEP_LABELS
    assert r.wep_label(PLAN) == "possible"


def test_real_shaped_nodes_get_source_and_why(monkeypatch):
    monkeypatch.delenv("BCG_WEP_THRESHOLDS", raising=False)
    assert r.wep_suffix(USER) == " (confirmed · from the task)"
    # merge with no new evidence adds nothing; label equals the score's.
    assert r.wep_suffix(CBOE) == " (confirmed · official source)"
    assert r.wep_suffix(PLAN) == " (possible · agent's reasoning · supported by [B1])"
    assert r.wep_suffix(CBOE, annotate=False) == " (confirmed)"


def test_contradiction_and_label_change():
    node = dict(USER, id=33, role="assistant", stance="probable", confidence=0.25,
                evidence_confidence=0, confidence_history=_hist(0.533, more=[{
                    "step": "relation_propagation", "value": 0.25, "relation_contributions": [
                        {"relation_id": 3, "relation_type": "contradicts", "input_id": 12,
                         "output_id": 33, "contribution": -0.5}]}]))
    assert r.wep_suffix(node) == " (unlikely · agent's reasoning · contradicted by [B12] · was possible)"


def test_corroboration_counts_only_when_evidence_moved_the_score():
    node = dict(CBOE, evidence_confidence=0.12, confidence_history=CBOE["confidence_history"] + [{
        "step": "evidence_recompute", "value": 0.84, "scored_evidence_ids": [40, 41]}])
    assert "corroborated by 2 other sources" in r.wep_suffix(node)


def test_line_depends_on_node_only():
    # Budget exactness: the same node renders the same line in any selection.
    assert r.wep_suffix(PLAN) == r.wep_suffix(json.loads(json.dumps(PLAN)))


@pytest.mark.parametrize("raw", ["0.8,0.7", "0.7,0.8,0.6,0.4,0.3", "a,b,c,d,e", "1.2,0.7,0.6,0.4,0.3"])
def test_threshold_env_is_validated(raw, monkeypatch):
    monkeypatch.setenv("BCG_WEP_THRESHOLDS", raw)
    with pytest.raises(ValueError):
        r.wep_thresholds()


def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv("BCG_WEP_THRESHOLDS", "0.85,0.75,0.65,0.45,0.30")
    assert r.wep_label_for(0.827) == "almost certain"


def test_render_preview_on_a_trial(tmp_path):
    bcg = tmp_path / "task9" / "trials" / "9" / "agent" / "bcg"
    bcg.mkdir(parents=True)
    snap = {"nodes": [USER, PLAN, ACTION, CBOE], "relations": []}
    (bcg / "graph_snapshots.jsonl").write_text(
        json.dumps({"call": 1, "snapshot": {"nodes": [USER]}}) + "\n" + json.dumps({"call": 4, "snapshot": snap}) + "\n")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "render_preview.py"), str(tmp_path)],
                         capture_output=True, text=True, check=True).stdout
    assert "findings 3   action nodes skipped 1" in out
    assert "medium confidence  -> possible 1" in out
    assert "supported" in out and "LINE COST" in out


# ── registered styles ────────────────────────────────────────────────────────

DIGEST = {"id": 145, "node_type": "belief", "role": "tool", "stance": "possible",
          "extraction_method": "rule_tool_result", "tool_name": "web_search", "tool_result_count": 10,
          "source_category": "mixed_or_unattributed", "confidence": 0.30,
          "belief": "The web_search tool returned 10 result(s). The bounded graph summary contains:\n"
                    + "\n".join(f"{i}. **Emerging Equity Markets {i}** — " + "x" * 280 for i in range(1, 11)),
          "tool_result_items": [{"rank": i, "title": f"**Emerging Equity Markets {i}**", "snippet": "x" * 280}
                                for i in range(1, 11)],
          "confidence_history": _hist(0.30)}


def test_wep_styles_render_lines(monkeypatch):
    monkeypatch.delenv("BCG_DIGEST_CHARS", raising=False)
    snap = {"nodes": [USER, PLAN, ACTION, CBOE, DIGEST], "relations": [
        {"id": 0, "from_id": 9, "to_id": 1, "type": "depends_on"}]}
    sel = {"node_ids": [1, 9, 10, 15, 145], "relation_ids": [0]}
    text, _, _ = r.render_graph(snap, sel, None, style="split_wep")
    assert "- [B15] The Cboe Gold" in text and "(confirmed · official source)" in text
    assert "(possible · agent's reasoning · supported by [B1])" in text
    assert "## Completed actions" in text and "confidence" not in text.split("## Findings")[1]
    assert "(search result list, not yet checked)" in text and "possible ·" not in text.split("[B145]")[1].split("\n")[0]
    label_only, _, _ = r.render_graph(snap, sel, None, style="split_wep_label")
    assert "[B9] The assistant plans to search for information about VIX, GVZ, and OVX. (possible)" in label_only


def test_digest_compaction(monkeypatch):
    monkeypatch.setenv("BCG_DIGEST_CHARS", "300")
    line = r.styled_node_line(DIGEST, "split_wep")
    assert len(line) <= 300 and line.startswith("- [B145] web_search returned 10 result(s): 1. Emerging")
    assert line.endswith("(search result list, not yet checked)") and "**" not in line
    # same compaction under the baseline style, so the two can be compared fairly
    assert r.styled_node_line(DIGEST, "split_confidence") == line
    monkeypatch.setenv("BCG_DIGEST_CHARS", "0")
    assert len(r.styled_node_line(DIGEST, "split_confidence")) > 2900


@pytest.mark.parametrize("raw", ["-1", "x"])
def test_digest_env_validated(raw, monkeypatch):
    monkeypatch.setenv("BCG_DIGEST_CHARS", raw)
    with pytest.raises(ValueError):
        r.digest_chars()


@pytest.mark.parametrize("digest", ["0", "300"])
@pytest.mark.parametrize("style", ["split_wep", "split_wep_label"])
def test_pcst_budget_exact_for_wep(style, digest, monkeypatch):
    pytest.importorskip("pcst_fast")
    from workflows.stateful_react_agent.bcg import pcst

    monkeypatch.setenv("BCG_DIGEST_CHARS", digest)
    snap = {"nodes": [USER, PLAN, ACTION, CBOE, DIGEST], "relations": [
        {"id": 0, "from_id": 9, "to_id": 1, "type": "depends_on"}]}
    prizes = {1: 0.9, 9: 0.5, 15: 0.8, 145: 0.6, 10: 0.2}
    full = pcst.select_prized_tree(snap, prizes, 20000, style=style)
    for budget in range(r.graph_overhead(style), full["rendered_chars"] + 60, 7):
        selection = pcst.select_prized_tree(snap, prizes, budget, style=style)
        if selection["node_ids"]:
            text, _, _ = r.render_graph(snap, selection, budget, strict=True, style=style)
            assert len(text) <= budget


def test_settings_and_guides(tmp_path, monkeypatch):
    from tests.test_react_bcg import FakeGraph
    from workflows.stateful_react_agent.bcg.runtime import (
        CONTEXT_GUIDE_SPLIT_WEP,
        CONTEXT_GUIDE_SPLIT_WEP_LABEL,
        BCGMemory,
        BCGSettings,
    )

    for style, guide in (("split_wep", CONTEXT_GUIDE_SPLIT_WEP), ("split_wep_label", CONTEXT_GUIDE_SPLIT_WEP_LABEL)):
        monkeypatch.setenv("BCG_RENDER_STYLE", style)
        assert BCGSettings.from_agent({}).render_style == style
        memory = BCGMemory(BCGSettings(), tmp_path, "s", backend=FakeGraph(), render_style=style)
        assert memory.context_guide == guide
        for word in r.WEP_LABELS:
            assert f"  {word}" in guide
        assert "unknown" not in guide.split("most to least reliable")[1].split("Rely on")[0]
    assert "corroborated by 2 other sources" in CONTEXT_GUIDE_SPLIT_WEP
    assert "corroborated by" not in CONTEXT_GUIDE_SPLIT_WEP_LABEL
    monkeypatch.setenv("BCG_RENDER_STYLE", "split_wep")
    monkeypatch.setenv("BCG_WEP_THRESHOLDS", "0.7,0.8,0.6,0.3,0.2")
    with pytest.raises(ValueError):
        BCGSettings.from_agent({})
