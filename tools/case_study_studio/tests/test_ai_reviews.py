"""Integration tests use synthetic evidence and never call paid APIs."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.ai_review.client import ReviewConfig, ReviewError
from tools.case_study_studio.control import APIError, Control
from tools.case_study_studio.review_job import execute
from tools.case_study_studio.tests.test_control import build_fixture, write


@pytest.fixture
def control(tmp_path: Path) -> Iterator[Control]:
    with patch("tools.case_study_studio.review_queue.ReviewConfig.from_env", return_value=ReviewConfig(api_key="fake-key")):
        yield build_fixture(tmp_path)


def queued(control: Control) -> dict:
    return control.ai_reviews.enqueue(1, "drb2", "2", control.register("Reviewer")["user"])


def mark_running(control: Control, job_id: int) -> None:
    with control.connect() as db:
        db.execute("UPDATE ai_reviews SET status='running',pid=? WHERE id=?", (os.getpid(), job_id))


def test_no_automatic_backfill_and_no_unfinished_review(control: Control) -> None:
    user = control.register("Reviewer")["user"]
    assert control.detail(1)["tasks"][0]["ai_review"]["status"] == "not_started"
    with patch("tools.case_study_studio.review_queue.subprocess.Popen") as spawn:
        control.ai_reviews.tick()
        spawn.assert_not_called()
    with pytest.raises(APIError, match="正常完成"):
        control.ai_reviews.enqueue(1, "drb2", "5", user)
    write(control.results, "offline-smoke/drb2/trials/2/report.md", "\n ")
    with pytest.raises(APIError, match="产出报告"):
        control.ai_reviews.enqueue(1, "drb2", "2", user)


@pytest.mark.parametrize("status", [
    "failed", "timeout", "agent_error", "worker_error", "empty_report", "score_error",
    "graph_fallback", "interrupted", "cancelled", "queued", "running", "not_run", "unknown",
])
def test_unsuccessful_tasks_block_annotations_even_with_report_and_frozen_evidence(control: Control, status: str) -> None:
    user = control.register("Reviewer")["user"]
    source, digest = control.evidence(1, "drb2", "2")
    original_evidence = source.read_bytes()
    body = {"revision": 0, "evidence_sha": digest, "annotation": {
        "status": "draft", "events": [], "dimensions": {}, "conclusion": {}, "goal": "Keep existing work"}}
    control.save_review(1, "drb2", "2", user, body)
    job = control.ai_reviews.enqueue(1, "drb2", "2", user)
    trial = control.task(1, "drb2", "2")[0]
    write(trial, "result.json", {**json.loads((trial / "result.json").read_text()), "status": status})
    task = control.detail(1)["tasks"][0]
    assert task["report"] and not task["annotatable"]
    for action in (
        lambda: control.evidence(1, "drb2", "2"),
        lambda: control.review(1, "drb2", "2", user),
        lambda: control.save_review(1, "drb2", "2", user, body),
        lambda: control.ai_reviews.get(1, "drb2", "2"),
        lambda: control.ai_reviews.enqueue(1, "drb2", "2", user, retry=True),
    ):
        with pytest.raises(APIError, match="正常完成") as error:
            action()
        assert error.value.status == 409
    assert source.read_bytes() == original_evidence
    assert control.ai_reviews.row(job["id"])["attempts"] == 1
    with control.connect() as db:
        saved = db.execute("SELECT revision,annotation FROM reviews").fetchone()
        assert saved["revision"] == 1 and json.loads(saved["annotation"])["goal"] == "Keep existing work"
        assert db.execute("SELECT count(*) FROM ai_reviews").fetchone()[0] == 1


def test_worker_rechecks_task_success_before_any_model_request(control: Control) -> None:
    job = queued(control)
    mark_running(control, job["id"])
    trial = control.task(1, "drb2", "2")[0]
    write(trial, "result.json", {"status": "agent_error"})
    with patch("tools.case_study_studio.review_job.run_review") as reviewer:
        execute(control.ai_reviews, job["id"])
        reviewer.assert_not_called()
    assert control.ai_reviews.row(job["id"])["status"] == "failed"


def test_parallel_requests_share_one_job_and_preserve_human_review(control: Control) -> None:
    user = control.register("Reviewer")["user"]
    _, digest = control.evidence(1, "drb2", "2")
    control.save_review(1, "drb2", "2", user, {"revision": 0, "evidence_sha": digest,
        "annotation": {"status": "draft", "events": [], "dimensions": {}, "conclusion": {"problem": "Human text"}}})
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = list(pool.map(lambda _: control.ai_reviews.enqueue(1, "drb2", "2", user), range(12)))
    assert len({j["id"] for j in jobs}) == 1
    assert all(j["status"] == "queued" and j["attempts"] == 1 for j in jobs)
    with control.connect() as db:
        assert db.execute("SELECT count(*) FROM ai_reviews").fetchone()[0] == 1
    assert control.review(1, "drb2", "2", user)["annotation"]["conclusion"]["problem"] == "Human text"
    assert control.detail(1)["annotation_count"] == 1


def test_missing_key_fails_before_queueing(control: Control) -> None:
    user = control.register("Reviewer")["user"]
    with patch("tools.case_study_studio.review_queue.ReviewConfig.from_env", return_value=ReviewConfig()):
        with pytest.raises(APIError, match="LITELLM_OPENAI_API_KEY") as error:
            control.ai_reviews.enqueue(1, "drb2", "2", user)
        assert error.value.status == 503
    with control.connect() as db:
        assert db.execute("SELECT count(*) FROM ai_reviews").fetchone()[0] == 0


def test_serial_worker_is_detached_and_survives_service_restart(control: Control) -> None:
    job = queued(control)
    fake = SimpleNamespace(pid=12345, poll=lambda: None)
    with patch("tools.case_study_studio.review_queue.subprocess.Popen", return_value=fake) as spawn:
        control.ai_reviews.tick()
        assert spawn.call_args.kwargs["start_new_session"]
        assert "tools.case_study_studio.review_job" in spawn.call_args.args[0]
        assert "fake-key" not in str(spawn.call_args)
        assert control.ai_reviews.row(job["id"])["status"] == "running"
        restarted = Control(control.root, control.state)
        with patch("tools.case_study_studio.review_queue.alive", return_value=True):
            restarted.ai_reviews.tick()
        assert spawn.call_count == 1
        with patch("tools.case_study_studio.review_queue.alive", return_value=False):
            restarted.ai_reviews.tick()
        assert restarted.ai_reviews.row(job["id"])["status"] == "interrupted"
        assert spawn.call_count == 1  # never auto-retry paid calls


def test_failed_job_requires_explicit_retry(control: Control) -> None:
    job = queued(control)
    user = control.register("Second reviewer")["user"]
    with control.connect() as db:
        db.execute("UPDATE ai_reviews SET status='failed' WHERE id=?", (job["id"],))
    with pytest.raises(APIError, match="重试"):
        control.ai_reviews.enqueue(1, "drb2", "2", user)
    retried = control.ai_reviews.enqueue(1, "drb2", "2", user, retry=True)
    assert retried["id"] == job["id"] and retried["attempts"] == 2 and retried["status"] == "queued"


def test_worker_publishes_only_complete_review_on_same_frozen_evidence(control: Control) -> None:
    job = queued(control)
    source, digest = control.evidence(1, "drb2", "2")
    mark_running(control, job["id"])
    result = {"annotation": {"status": "draft", "events": [], "reviewer": "AI"},
              "usage": {"total_tokens": 123}, "coverage": {"agent_calls": 1}}

    def run(given_source: Path, output: Path, **kwargs: object) -> dict:
        assert source == given_source and kwargs["retry_failed"] is True
        write(output, "review.json", result)
        return result

    with patch("tools.case_study_studio.review_job.run_review", side_effect=run) as reviewer:
        execute(control.ai_reviews, job["id"])
        execute(control.ai_reviews, job["id"])
        reviewer.assert_called_once()
    published = control.ai_reviews.get(1, "drb2", "2")
    assert published["status"] == "completed" and published["evidence_sha"] == digest
    assert published["annotation"] == result["annotation"] and published["usage"]["total_tokens"] == 123
    assert control.detail(1)["tasks"][0]["ai_review"]["status"] == "completed"
    assert control.detail(1)["annotation_count"] == 0
    user = control.register("Second reviewer")["user"]
    assert control.ai_reviews.enqueue(1, "drb2", "2", user, retry=True)["attempts"] == 1
    assert control.review(1, "drb2", "2", user)["annotation"] is None
    assert "api_key" not in json.dumps(published)


def test_worker_error_is_safe_and_cannot_run_stale_job(control: Control) -> None:
    job = queued(control)
    with patch("tools.case_study_studio.review_job.run_review") as reviewer:
        execute(control.ai_reviews, job["id"])
        reviewer.assert_not_called()
    mark_running(control, job["id"])
    with patch("tools.case_study_studio.review_job.run_review", side_effect=ValueError("api_key=hidden")):
        execute(control.ai_reviews, job["id"])
    published = control.ai_reviews.get(1, "drb2", "2")
    assert published["status"] == "failed" and published["annotation"] is None
    assert "hidden" not in published["error"]
    mark_running(control, job["id"])
    with patch("tools.case_study_studio.review_job.run_review", side_effect=ReviewError("HTTP 429")):
        execute(control.ai_reviews, job["id"])
    assert control.ai_reviews.get(1, "drb2", "2")["error"] == "HTTP 429"
