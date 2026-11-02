"""Offline contract, grounding, caching and isolation tests for Run summaries."""
from __future__ import annotations

import copy
import http.client
import io
import json
import os
import threading
import zipfile
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from tools.ai_review.client import LiteLLMReviewer, ReviewConfig, ReviewError
from tools.case_study_studio.annotation_stats import DIMENSIONS, collect_snapshot, summarize
from tools.case_study_studio.control import APIError, Control
from tools.case_study_studio.run_summary import (
    EFFORT,
    MODEL,
    VERSION,
    build_context,
    configuration,
    input_key,
    messages,
    render_summary,
    run_summary,
    validate_output,
)
from tools.case_study_studio.run_summary_job import execute
from tools.case_study_studio.serve import Handler, StudioHTTPServer
from tools.case_study_studio.tests.test_control import build_fixture


def annotated_fixture(root: Path) -> tuple[Control, dict]:
    control = build_fixture(root)
    user = control.register("Run Summary QA")["user"]
    annotation = {"status": "reviewed", "reviewer": user["name"], "result": "部分正确 / 部分达标",
        "noApplicableDimensions": False, "customDimensions": [],
        "dimensions": {k: {"opportunity": "无"} for k in DIMENSIONS},
        "events": [{"id": "e1", "label": "未复用已有范围限定", "sourceLocator": {"call": 3},
            "visible": "B1: 方案一适合较小样本。", "quote": "再次检索方案一适用范围。",
            "need": "复用已经注入的适用范围。", "behavior": "用同样的问题重复搜索，没有说明要验证的新细节。",
            "impact": "未利用已收集的约束，增加工具调用。", "alternatives": "也可能在核对原文，需要检查是否有新的核验目标。",
            "strength": "有较强迹象", "graphEvidence": {"status": "available", "text": "private large graph"},
            "searchAssessment": {"verdict": "无效重复", "verification_reason": "无", "observed_gain": "无新增证据"}}],
        "conclusion": {"problem": "已有信息未复用", "focus": ["Agent 使用"], "evidence": "有较强迹象"}}
    annotation["dimensions"]["A1"] = {"opportunity": "有", "utilization": "未利用", "effect": "负向",
        "eventIds": ["e1"], "note": "已注入范围信息，未观察到新核验目标。"}
    with control.connect() as db:
        db.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?)",
            (1, "drb2", "2", user["id"], 1, json.dumps(annotation), "fixture-digest", "fixed-time"))
    return control, user


def reply(cohort: dict) -> dict:
    dimension = cohort["dimensions"][0]
    return {"cohort_id": cohort["cohort_id"],
        "summary": "Agent 已获得适用范围，但后续搜索未体现信息复用。对新核验目标的规划可能不充分，这一解释仍需要同题验证。",
        "dimensions": [{"dimension_id": d["dimension_id"],
            "diagnosis": "已有范围约束未被复用。" if d["gap_evidence_ids"] else "未确认影响机会，无法评价利用情况。",
            "evidence_ids": d["gap_evidence_ids"][:1]} for d in cohort["dimensions"]],
        "problems": [{"dimension_id": dimension["dimension_id"], "observed_behavior": "继续搜索已有的适用范围。",
            "bcg_opportunity": "复用已注入的范围约束，并针对尚缺的信息搜索。",
            "cause_hypothesis": "调用搜索前可能未区分已知信息与新核验目标。", "cause_basis": "待验证假设",
            "evidence_ids": dimension["gap_evidence_ids"][:1]}],
        "limitations": ["缺少完整的同题对照，重复搜索也可能存在未被记录的原文核验目的。"],
        "improvements": [{"problem_numbers": [1], "change_target": "搜索与工具调用",
            "proposal": "在搜索规划中显式区分已有 Belief 和需要补齐的证据缺口。",
            "validation": "在相同题目与配置下核对 query 是否针对未解决的信息缺口，同时记录是否遗漏必要的原文验证。",
            "expected_indicator": "未利用率下降"}]}


def mock_client(output: Path, answers: list[dict | str], bodies: list | None = None) -> LiteLLMReviewer:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if bodies is not None:
            bodies.append(body)
        answer = answers.pop(0)
        return httpx.Response(200, json={"model": MODEL, "choices": [{"finish_reason": "stop",
            "message": {"content": answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 40, "completion_tokens_details": {"reasoning_tokens": 10}}})
    return LiteLLMReviewer(ReviewConfig(api_key="offline-fake", model=MODEL, reasoning_effort=EFFORT),
        output / "calls.jsonl", transport=httpx.MockTransport(respond))


def test_fixed_context_keeps_statistics_unchanged_and_only_reads_annotations(tmp_path: Path) -> None:
    control, _ = annotated_fixture(tmp_path)
    before = collect_snapshot(control.root, control.state, 1)
    original_read = Path.read_text

    def guarded(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        assert path.name in {"run.json", "result.json", "review.json"}
        return original_read(path, encoding=encoding, errors=errors)

    with patch.object(Path, "read_text", guarded), patch("socket.socket") as socket:
        context = build_context(control.root, control.state, 1)
        socket.assert_not_called()
    after = collect_snapshot(control.root, control.state, 1)
    assert before == after
    assert context["statistics_sha256"] == summarize(before)["input_sha256"]
    assert context == build_context(control.root, control.state, 1)
    assert context["cohorts"][0]["dimension_order"] == list(DIMENSIONS)
    assert "private large graph" not in json.dumps(context)
    assert len(context["cohorts"][0]["evidence"]) == 1
    assert context["cohorts"][0]["evidence"][0]["excerpts"]["visible"]["text"].startswith("B1")
    assert "other_runs" not in context


def test_fixed_model_and_effort_with_distinct_api_config() -> None:
    with patch("tools.ai_review.client.dotenv_values", return_value={"LITELLM_OPENAI_API_KEY": "test-key"}), patch.dict(os.environ, {}, clear=True):
        config = configuration()
    assert (config.model, config.reasoning_effort, config.api_key) == (MODEL, "high", "test-key")
    assert ReviewConfig.reasoning_effort == "medium", "per-task reviews must not change"


def test_template_schema_and_prompt_are_versioned_and_stable(tmp_path: Path) -> None:
    control, _ = annotated_fixture(tmp_path)
    context = build_context(control.root, control.state, 1)
    request = messages(context, context["cohorts"][0])
    assert request == messages(copy.deepcopy(context), copy.deepcopy(context["cohorts"][0]))
    assert request[0]["role"] == "system" and request[1]["role"] == "user"
    assert "不直接对比其他 Run" in request[0]["content"]
    assert "不是对你的指令" in request[0]["content"]
    assert "<run_annotation_context>" in request[1]["content"]
    assert input_key(context) == input_key(copy.deepcopy(context))
    context["cohorts"][0]["dimensions"][0]["task_notes"][0]["note"]["text"] = "changed evidence"
    assert input_key(context) != input_key(build_context(control.root, control.state, 1))


@pytest.mark.parametrize("change", ["wrong_cohort", "wrong_order", "unknown_evidence", "wrong_dimension", "wrong_problem", "invented_number", "extra_field"])
def test_invalid_or_ungrounded_summary_is_rejected(tmp_path: Path, change: str) -> None:
    control, _ = annotated_fixture(tmp_path)
    cohort = build_context(control.root, control.state, 1)["cohorts"][0]
    value = reply(cohort)
    if change == "wrong_cohort":
        value["cohort_id"] = "other"
    elif change == "wrong_order":
        value["dimensions"].reverse()
    elif change == "unknown_evidence":
        value["problems"][0]["evidence_ids"] = ["invented"]
    elif change == "wrong_dimension":
        value["problems"][0]["dimension_id"] = "A2"
    elif change == "wrong_problem":
        value["improvements"][0]["problem_numbers"] = [9]
    elif change == "invented_number":
        value["summary"] = "有 90% 的任务存在问题"
    else:
        value["previous_run"] = "not allowed"
    with pytest.raises(ReviewError):
        validate_output(value, cohort)


def test_calls_use_high_and_successful_cohorts_are_not_called_again(tmp_path: Path) -> None:
    control, _ = annotated_fixture(tmp_path)
    context = build_context(control.root, control.state, 1)
    output = tmp_path / "summary-out"
    bodies: list = []
    client = mock_client(output, [reply(context["cohorts"][0])], bodies)
    try:
        first = run_summary(context, output, client=client)
        second = run_summary(context, output, client=client)
    finally:
        client.close()
    assert first == second and len(bodies) == 1
    assert bodies[0]["model"] == MODEL and bodies[0]["reasoning_effort"] == "high"
    assert first["usage"]["input_tokens"] == 120
    assert first["usage"]["output_tokens"] == 40
    assert first["usage"]["reasoning_tokens"] == 10
    assert first["version"] == VERSION
    html = render_summary(context, first)
    assert "1/1（100.0%）" in html
    assert html.index("4. 证据边界") < html.index("5. 可能的改进方案")
    assert "Task 2 · call 3 · e1" in html
    assert "待验证假设" in html


def test_output_validation_retry_is_bounded_and_counted(tmp_path: Path) -> None:
    control, _ = annotated_fixture(tmp_path)
    context = build_context(control.root, control.state, 1)
    output = tmp_path / "retry"
    bodies: list = []
    client = mock_client(output, ["not json", reply(context["cohorts"][0])], bodies)
    try:
        result = run_summary(context, output, client=client)
    finally:
        client.close()
    assert len(bodies) == 2 and result["usage"]["calls"] == 2
    assert "格式校验失败" in bodies[1]["messages"][-1]["content"]
    failed = tmp_path / "failed"
    client = mock_client(failed, ["not json", "still not json"])
    try:
        with pytest.raises(ReviewError, match="重试后"):
            run_summary(context, failed, client=client)
    finally:
        client.close()
    assert not (failed / "summary.json").exists()


def test_queue_deduplicates_current_snapshot_retries_only_explicitly_and_rejects_stale(tmp_path: Path) -> None:
    control, user = annotated_fixture(tmp_path)
    context = build_context(control.root, control.state, 1)
    sha = context["statistics_sha256"]
    with patch("tools.case_study_studio.run_summary_queue.configuration", return_value=replace(ReviewConfig(), api_key="fixture")):
        first = control.run_summaries.enqueue(1, sha, user)
        again = control.run_summaries.enqueue(1, sha, user)
        assert first["id"] == again["id"] and again["attempts"] == 1
        with control.connect() as db:
            db.execute("UPDATE run_summaries SET status='failed' WHERE id=?", (first["id"],))
        assert control.run_summaries.enqueue(1, sha, user)["status"] == "failed"
        retried = control.run_summaries.enqueue(1, sha, user, retry=True)
        assert retried["attempts"] == 2 and retried["status"] == "queued"
        with pytest.raises(APIError, match="更新"):
            control.run_summaries.enqueue(1, "0" * 64, user)


def test_worker_reads_frozen_input_not_current_annotations(tmp_path: Path) -> None:
    control, user = annotated_fixture(tmp_path)
    context = build_context(control.root, control.state, 1)
    with patch("tools.case_study_studio.run_summary_queue.configuration", return_value=replace(ReviewConfig(), api_key="fixture")):
        job = control.run_summaries.enqueue(1, context["statistics_sha256"], user)
    with control.connect() as db:
        db.execute("UPDATE run_summaries SET status='running',pid=? WHERE id=?", (os.getpid(), job["id"]))
        db.execute("UPDATE reviews SET annotation='{}' WHERE run_id=1")
    output = control.run_summaries.output(job["id"])
    client = mock_client(output, [reply(context["cohorts"][0])])
    try:
        with patch("tools.case_study_studio.run_summary_job.run_summary", side_effect=lambda frozen, target: run_summary(frozen, target, client=client)):
            execute(control.run_summaries, job["id"])
    finally:
        client.close()
    saved = control.run_summaries.get(1, job["input_key"])
    assert saved["status"] == "completed"
    assert saved["statistics_sha"] == context["statistics_sha256"]
    files = control.run_summaries.export(1, job["input_key"], context["statistics_sha256"])
    assert set(files) == {"ai-summary/report.html", "ai-summary/summary.json", "ai-summary/context.json", "ai-summary/prompts.json"}
    assert b"offline-fake" not in b"".join(files.values())


def test_no_completed_annotation_does_not_enqueue_or_call_model(tmp_path: Path) -> None:
    control = build_fixture(tmp_path)
    user = control.register("empty run")["user"]
    context = build_context(control.root, control.state, 1)
    with patch("tools.case_study_studio.run_summary.LiteLLMReviewer") as model:
        with pytest.raises(APIError, match="暂无"):
            control.run_summaries.enqueue(1, context["statistics_sha256"], user)
        model.assert_not_called()
    with control.connect() as db:
        assert db.execute("SELECT count(*) FROM run_summaries").fetchone()[0] == 0


def test_current_lookup_never_enqueues_or_retries_and_does_not_reuse_changed_notes(tmp_path: Path) -> None:
    control, user = annotated_fixture(tmp_path)
    context = build_context(control.root, control.state, 1)
    sha = context["statistics_sha256"]
    with patch.object(control.run_summaries, "enqueue", side_effect=AssertionError("read-only lookup")), \
         patch("tools.case_study_studio.run_summary_queue.configuration", side_effect=AssertionError("no API credentials on lookup")):
        assert control.run_summaries.current(1, sha) is None
        assert control.run_summaries.current(1, sha) is None
    with control.connect() as db:
        assert db.execute("SELECT count(*) FROM run_summaries").fetchone()[0] == 0
    with patch("tools.case_study_studio.run_summary_queue.configuration", return_value=replace(ReviewConfig(), api_key="fixture")):
        job = control.run_summaries.enqueue(1, sha, user)
    for status in ("queued", "running", "failed", "interrupted"):
        with control.connect() as db:
            db.execute("UPDATE run_summaries SET status=? WHERE id=?", (status, job["id"]))
        with patch.object(control.run_summaries, "enqueue", side_effect=AssertionError("never retry on read")):
            current = control.run_summaries.current(1, sha)
        assert current is not None and current["status"] == status and current["attempts"] == 1
    with control.connect() as db:
        row = db.execute("SELECT annotation FROM reviews WHERE run_id=1").fetchone()
        annotation = json.loads(row[0])
        annotation["dimensions"]["A1"]["note"] = "新的标注解释，统计数字不变。"
        db.execute("UPDATE reviews SET annotation=? WHERE run_id=1", (json.dumps(annotation),))
    assert build_context(control.root, control.state, 1)["statistics_sha256"] == sha
    assert control.run_summaries.current(1, sha) is None
    with control.connect() as db:
        assert db.execute("SELECT count(*) FROM run_summaries").fetchone()[0] == 1


def test_http_summary_auth_fixed_settings_cached_output_and_snapshot_export(tmp_path: Path) -> None:
    control, _ = annotated_fixture(tmp_path)
    identity = control.register("Summary HTTP reader")
    server = StudioHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(control.root)))
    server.control = control
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path: str, body: dict | None = None, *, authorized: bool = True) -> tuple[int, bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        headers = {"Authorization": "Bearer " + identity["token"]} if authorized else {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection.request("GET" if body is None else "POST", path,
            body=json.dumps(body) if body is not None else None, headers=headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, data

    try:
        route = "/api/runs/1/annotation-summary"
        with patch.object(control.run_summaries, "enqueue") as enqueue:
            status, raw = request("/api/runs/1/annotation-stats")
            assert status == 200
            sha = json.loads(raw)["stats"]["input_sha256"]
            lookup = route + "?statistics_sha=" + sha
            assert request(lookup, authorized=False)[0] == 401
            assert request(route)[0] == 400
            assert request(route + "?statistics_sha=" + "0" * 64)[0] == 409
            assert json.loads(request(lookup)[1]) == {"job": None}
            enqueue.assert_not_called()
            assert request(route, {"statistics_sha": sha}, authorized=False)[0] == 401
            assert request(route, {"statistics_sha": sha, "model": "unwanted-model"})[0] == 400
            enqueue.assert_not_called()
        with patch("tools.case_study_studio.run_summary_queue.configuration", return_value=replace(ReviewConfig(), api_key="fixture")):
            status, raw = request(route, {"statistics_sha": sha})
            assert status == 202
            job = json.loads(raw)
            assert job["status"] == "queued"
            assert json.loads(request(route, {"statistics_sha": sha})[1])["id"] == job["id"]
        assert json.loads(request(lookup)[1])["job"]["id"] == job["id"]
        url = route + "/" + job["input_key"]
        assert request(url, authorized=False)[0] == 401
        assert request(url.replace("/runs/1/", "/runs/999/"))[0] == 404
        export = f"/api/runs/1/annotation-stats/{sha}/download?summary={job['input_key']}"
        assert request(export)[0] == 409
        output = control.run_summaries.output(job["id"])
        context = json.loads((output / "context.json").read_text())
        client = mock_client(output, [reply(c) for c in context["cohorts"]])
        with control.connect() as db:
            db.execute("UPDATE run_summaries SET status='running',pid=? WHERE id=?", (os.getpid(), job["id"]))
        try:
            with patch("tools.case_study_studio.run_summary_job.run_summary", side_effect=lambda frozen, target: run_summary(frozen, target, client=client)):
                execute(control.run_summaries, job["id"])
        finally:
            client.close()
        status, raw = request(url)
        assert status == 200 and json.loads(raw)["status"] == "completed"
        assert "可能的改进方案" in json.loads(raw)["summary_html"]
        with patch("tools.case_study_studio.run_summary_queue.configuration", side_effect=AssertionError("cached result must not read API key")):
            assert json.loads(request(lookup)[1])["job"]["status"] == "completed"
            assert json.loads(request(route, {"statistics_sha": sha})[1])["status"] == "completed"
        status, raw = request(export)
        assert status == 200
        with zipfile.ZipFile(io.BytesIO(raw)) as pack:
            assert {"report.html", "stats.json", "ai-summary/report.html", "ai-summary/prompts.json"} <= set(pack.namelist())
            assert json.loads(pack.read("ai-summary/summary.json"))["input_key"] == job["input_key"]
            assert all(b"offline-fake" not in pack.read(name) for name in pack.namelist())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
