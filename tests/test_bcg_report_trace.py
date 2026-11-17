"""bcg_report: --dump-step-with-memory and the confidence trace on a synthetic trial."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from bcg.construct.unified.confidence import propagate_relation_confidences
from workflows.stateful_react_agent.bcg.rendering import render_graph

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bcg_report

STYLE = "split_confidence_noactions"


def _node(i, text, conf, turn, role="tool"):
    return {"id": i, "belief": text, "confidence": conf, "initial_confidence": conf,
            "evidence_confidence": 0.0, "factor_confidence": 0.0, "role": role,
            "source": {"turn_id": turn},
            "confidence_history": [{"step": "initial", "value": conf}]}


def _call(cid, query):
    return json.dumps({"id": cid, "name": "web_search", "arguments": {"query": query}})


def build_trial(root: Path) -> Path:
    trial = root / "trials" / "7"
    bcg = trial / "agent" / "bcg"
    bcg.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({"status": "ok"}))

    queries = ["statman 1987 diversification", "evans archer 1968", "emerging markets stocks", "final"]
    thinking = [
        "Start with Statman.",
        "Look for Evans and Archer.",
        "[B11] says 30 stocks but it is only medium confidence; I should verify with an independent source.",
        "B11 was contradicted ([B12]), its tier is now low confidence, so I will not rely on [B11]; use [B10].",
    ]
    acknowledged = [{"role": "system", "content": "preamble"}, {"role": "user", "content": "task"}]
    for i, q in enumerate(queries):
        acknowledged.append({"role": "assistant",
                             "content": f"<thinking>\n{thinking[i]}\n</thinking>\n<tool_call>\n{_call(f'c{i}', q)}\n</tool_call>"})
        acknowledged.append({"role": "tool", "content": "<tool_result>\n"
                             + json.dumps({"tool_call_id": f"c{i}", "content": "ok"}) + "\n</tool_result>"})

    action = _node(13, 'The assistant is using web_search to search for \\"statman 1987 diversification\\".', 0.7, 2,
                   role="assistant")
    nodes = {
        10: _node(10, "Statman (1987) finds 30-40 stocks are needed for a borrowing investor.", 0.85, 3),
        11: _node(11, "Thirty stocks are therefore enough in all markets.", 0.40, 4, role="assistant"),
        13: action,
    }
    rels = [{"id": 1, "from_id": 11, "to_id": 10, "type": "depends_on", "note": "generalises Statman"},
            {"id": 2, "from_id": 10, "to_id": 13, "type": "depends_on", "note": "result of call"}]
    propagate_relation_confidences(nodes, rels)  # B11 raised through R1
    snap1 = {"problem_id": "p", "nodes": copy.deepcopy(list(nodes.values())), "relations": copy.deepcopy(rels)}

    nodes[12] = _node(12, "Emerging-market studies recommend far more than 30 stocks.", 0.9, 7)
    rels.append({"id": 3, "from_id": 12, "to_id": 11, "type": "contradicts", "note": "EM evidence"})
    propagate_relation_confidences(nodes, rels, seed_output_node_ids=[11])  # B11 lowered through R3
    snap2 = {"problem_id": "p", "nodes": copy.deepcopy(list(nodes.values())), "relations": copy.deepcopy(rels)}

    (bcg / "graph_snapshots.jsonl").write_text(
        json.dumps({"call": 3, "snapshot": snap1}) + "\n" + json.dumps({"call": 4, "snapshot": snap2}) + "\n")
    (bcg / "state.json").write_text(json.dumps(
        {"version": 1, "problem_id": "p", "acknowledged": acknowledged, "snapshot": snap2}))

    def prepare(call, before, snap, evicted):
        event = {"call": call, "messages_before": before, "render_style": STYLE,
                 "retrieval_strategy": "all", "evicted_steps": evicted, "duration_ms": 5}
        if evicted:
            ids = [n["id"] for n in snap["nodes"]]
            graph, shown, rel_ids = render_graph(
                snap, {"node_ids": ids, "relation_ids": [r["id"] for r in snap["relations"]]}, None, style=STYLE)
            event.update(system_memory="<belief_graph_guide>g</belief_graph_guide>\n\n" + graph,
                         selected_node_ids=shown, selected_relation_ids=rel_ids)
        return event

    events = [prepare(1, 2, snap1, 0), prepare(2, 4, snap1, 0), prepare(3, 6, snap1, 1), prepare(4, 8, snap2, 2)]
    (bcg / "requests.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return trial


@pytest.fixture()
def run_dir(tmp_path):
    build_trial(tmp_path)
    return tmp_path


def test_trace_names_the_edges_that_moved_confidence(run_dir):
    trace = bcg_report.confidence_trace(run_dir / "trials" / "7")
    assert trace["history_available"] and trace["snapshot_timing_available"]
    by_id = {t["node"]: t for t in trace["nodes"]}
    b11 = by_id[11]
    raise_, lower = b11["changes"][0], b11["changes"][-1]
    assert raise_["delta"] > 0 and raise_["call"] == 3
    assert [e["relation_id"] for e in raise_["edges"]] == [1] and raise_["edges"][0]["effect"] == "raise"
    assert {e["relation_id"]: e["status"] for e in lower["edges"]} == {3: "new", 1: "unchanged"}
    assert lower["call"] == 4 and any(e["relation_id"] == 3 and e["effect"] == "lower" for e in lower["edges"])
    assert any("[B12] contradicts [B11]" in e["edge"] for e in lower["edges"])
    # The model saw B11 (no actions style) and cited it afterwards.
    assert b11["times_shown_after_change"] >= 1 and b11["cited_in_steps_after_change"]
    assert trace["changes_by_cause"]["relation_propagation"] >= 2
    assert 13 not in by_id or by_id[13]["kind"] == "action"


def test_dump_confidence_and_node_filter(run_dir):
    text = bcg_report.dump_confidence(run_dir, None, 11)
    assert "B11" in text and "R3: [B12] contradicts [B11]" in text and "R1: [B11] depends_on [B10]" in text
    assert "B10 " not in text.split("\n\n", 1)[1].split("B11", 1)[0]  # filtered to one node


def test_dump_step_with_memory_marks_cited_lines(run_dir):
    text = bcg_report.dump_step_with_memory(run_dir, 4, None)
    assert "prepare call 4" in text and STYLE in text
    assert "## Completed actions" not in text
    marked = [line for line in text.splitlines() if line.startswith(">> ")]
    assert any("[B11]" in line for line in marked) and any("[B12]" in line for line in marked)
    assert "MODEL THINKING:" in text and "low confidence" in text
    assert "cited & shown" in text and "B11 (" in text
    assert "support language" in text


def test_full_report_has_confidence_section(run_dir):
    run = bcg_report.load_run(run_dir)
    text = bcg_report.render(run)
    assert "CONFIDENCE DYNAMICS" in text
    assert run["aggregate"]["confidence_dynamics"]["nodes_changed"] >= 1


def test_interrupted_trial_without_result_json_is_found(run_dir):
    (run_dir / "trials" / "7" / "result.json").unlink()
    assert bcg_report.find_trials(run_dir) == [run_dir / "trials" / "7"]
    assert "prepare call 4" in bcg_report.dump_step_with_memory(run_dir, 4, {"7"})
