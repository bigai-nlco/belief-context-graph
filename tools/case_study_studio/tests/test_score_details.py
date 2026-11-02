"""Read-only Report Judge details: exact saved explanations, safe task boundaries."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.case_study_studio.control import APIError, Control
from tools.case_study_studio.tests.test_control import build_fixture, write


def save_score(control: Control, *, status: str = "scored") -> Path:
    return write(control.results, "offline-smoke/drb2/trials/2/judge/drb2_rubric.json", {
        "benchmark": "deepresearch_bench_ii", "task_id": "2", "status": status,
        "scores": {"total": 1/3, "inforecall": 0.5, "analysis": 0, "presentation": None},
        "judge": {"model": "offline-judge", "reasoning_effort": "high", "api_key": "private-key"},
        "judged_at": "2026-10-28T12:00:00+00:00", "duration_seconds": 20,
        "attempts": "/private/server/path.jsonl", "error": "private-error-detail",
        "result": {"task": "private rubric prompt", "scores": {
            "info_recall": {
                "criterion A": {"score": 1, "reason": "Reason A", "evidence": "Exact report quote\nLine 2", "secret": "hidden"},
                "criterion B": {"score": 0, "reason": "Reason B", "evidence": ""},
            },
            "analysis": {"criterion C": {"score": -1, "reason": "Blocked citation", "evidence": "Quote C"}},
            "presentation": {},
        }},
    })


@pytest.fixture
def control(tmp_path: Path) -> Control:
    return build_fixture(tmp_path)


def test_details_preserve_stored_scores_reasons_and_evidence_without_private_fields(control: Control) -> None:
    path = save_score(control)
    original = path.read_bytes()
    with patch("benchmarks.public.reports.drb2_judge.OpenAIChatClient") as model:
        result = control.score_details(1, "drb2", "2")
        model.assert_not_called()
    assert result["run"] == "RUN-0001" and result["task_id"] == "2"
    assert result["score"] == 1/3 and result["model"] == "offline-judge"
    assert result["dimensions"][0]["items"] == [
        {"criterion": "criterion A", "score": 1, "reason": "Reason A", "evidence": "Exact report quote\nLine 2"},
        {"criterion": "criterion B", "score": 0, "reason": "Reason B", "evidence": ""},
    ]
    assert result["dimensions"][1]["items"][0]["score"] == -1
    assert result["dimensions"][2]["score"] is None
    assert all(word not in json.dumps(result) for word in ("private", "secret", "hidden", "api_key", "attempts"))
    assert path.read_bytes() == original


def test_run_list_checks_availability_without_loading_details(control: Control) -> None:
    save_score(control)
    from tools.case_study_studio import control as module
    read = module.read_json

    def guarded(path: Path) -> dict:
        assert path.name != "drb2_rubric.json", "polling must not load full rubric details"
        return read(path)

    with patch.object(module, "read_json", guarded):
        tasks = control.detail(1)["tasks"]
    assert tasks[0]["score_details_available"] and not tasks[1]["score_details_available"]


def test_unavailable_or_mismatched_details_are_not_shown(control: Control) -> None:
    with pytest.raises(APIError) as missing:
        control.score_details(1, "drb2", "2")
    assert missing.value.status == 404
    path = save_score(control)
    data = json.loads(path.read_text())
    data["task_id"] = "5"
    write(path.parent, path.name, data)
    with pytest.raises(APIError) as mismatch:
        control.score_details(1, "drb2", "2")
    assert mismatch.value.status == 409
    for benchmark, task_id in (("drbench", "2"), ("drb2", "../2"), ("drb2", "999")):
        with pytest.raises(APIError):
            control.score_details(1, benchmark, task_id)


def test_score_paths_cannot_escape_trial(control: Control) -> None:
    path = save_score(control)
    path.unlink()
    private = write(control.state, "private-score.json", {"secret": "do not serve"})
    path.symlink_to(private)
    assert not control.detail(1)["tasks"][0]["score_details_available"]
    with pytest.raises(APIError):
        control.score_details(1, "drb2", "2")


def test_failed_judge_details_are_not_presented_as_a_score(control: Control) -> None:
    path = save_score(control, status="judge_error")
    data = json.loads(path.read_text())
    data.update(scores=None, result={"error": "batch failed"})
    write(path.parent, path.name, data)
    result = control.score_details(1, "drb2", "2")
    assert result["status"] == "judge_error" and result["score"] is None
    assert all(not dimension["items"] for dimension in result["dimensions"])
