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
    (0.395, "unlikely"),         # confirmed x community
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
    node = dict(USER, id=33, role="assistant", stance="probable", confidence=0.38,
                evidence_confidence=0, confidence_history=_hist(0.533, more=[{
                    "step": "relation_propagation", "value": 0.38, "relation_contributions": [
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
