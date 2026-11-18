"""Interrupted-task viewing never grants permission to annotate or rerun."""
import json
from pathlib import Path

import pytest

from tools.case_study_studio.control import APIError
from tools.case_study_studio.tests.test_control import build_fixture, write
from workflows._shared.case_study import read_bundle


def test_interrupted_records_without_result_or_report(tmp_path: Path) -> None:
    control = build_fixture(tmp_path)
    root = control.results / "offline-smoke"
    trial = root / "drb2/trials/5"
    write(root, "drb2/_cases/5.json", {"question": "Interrupted task question", "rubric": "not evidence"})
    write(trial, "report.md", "")
    write(trial, "agent/trajectories/react_agent.jsonl",
          '{"t":"start","task_id":"internal-uuid","user_message":"Interrupted task question"}\n'
          '{"t":"llm","turn":1,"content":"Saved output before failure"}\n'
          '{"t":"end","turns":2,"stopped_by":"llm_error"}\n')
    write(trial, "agent/bcg/state.json", {"snapshot": {"nodes": [{"id": 1, "belief": "Saved belief"}], "relations": []}})
    write(trial, "agent/bcg/requests.jsonl", '{"call":1,"system_memory":"<belief_graph>B1: Saved belief</belief_graph>"}\n{"call":2')
    before = control.detail(1)
    task = next(t for t in before["tasks"] if t["id"] == "5")
    assert task["status"] == "interrupted" and task["viewable"]
    assert not task["annotatable"] and not task["report"]
    path, digest = control.records(1, "drb2", "5")
    bundle = read_bundle(path)
    case = bundle["cases"][0]
    meta = json.loads(next(f["text"] for f in case["files"] if f["path"] == "case.json"))
    assert meta["task_id"] == "5" and meta["benchmark"] == "deepresearch_bench_ii"
    assert meta["status"] == "interrupted" and meta["final_response"] == ""
    assert meta["question"] == "Interrupted task question"
    assert "rubric" not in meta
    assert any(f["path"].endswith("state.json") for f in case["files"])
    assert any("不完整" in w for w in case["warnings"])
    preview, preview_digest = control.records_preview(1, "drb2", "5")
    assert preview_digest == digest
    assert read_bundle(preview)["projection"]["omitted_files"]
    assert control.records(1, "drb2", "5") == (path, digest)
    assert control.detail(1) == before
    user = control.register("records-reader")["user"]
    for action in (lambda: control.evidence(1, "drb2", "5"),
                   lambda: control.review(1, "drb2", "5", user),
                   lambda: control.ai_reviews.enqueue(1, "drb2", "5", user),
                   lambda: control.save_review(1, "drb2", "5", user, {
                       "revision": 0, "evidence_sha": digest, "annotation": {
                           "status": "draft", "events": [], "dimensions": {}, "conclusion": {}}})):
        with pytest.raises(APIError, match="正常完成"):
            action()
    assert not (trial / "result.json").exists()
    assert (trial / "report.md").read_bytes() == b""


def test_no_records_for_unstarted_or_active_task(tmp_path: Path) -> None:
    control = build_fixture(tmp_path)
    with pytest.raises(APIError, match="尚无"):
        control.records(1, "drb2", "5")
    root = control.results / "offline-smoke"
    write(root, "drb2/trials/5/agent/trajectories/react_agent.jsonl", '{"t":"llm","turn":1,"content":"Working"}\n')
    with control.connect() as db:
        db.execute("UPDATE runs SET managed=1,status='running' WHERE id=1")
    with pytest.raises(APIError, match="尚无"):
        control.records(1, "drb2", "5")
