"""Offline coverage for shared runs, identity isolation and immutable evidence."""
from __future__ import annotations

import http.client
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.case_study_studio.control import DEFAULT_IDS, ROOT, APIError, Control, safe_child
from tools.case_study_studio.serve import Handler
from workflows._shared.case_study import export_case


def write(root: Path, name: str, data: object) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False))
    return path


def build_fixture(root: Path) -> Control:
    shutil.copytree(ROOT / "tools/case_study_studio/profiles", root / "tools/case_study_studio/profiles")
    batch = root / "results/offline-smoke"
    write(batch, "batch.json", {"status": "finished", "started_at": "2026-10-21T00:00:00Z",
        "finished_at": "2026-10-21T00:01:00Z", "workers": 1, "agent": {"model": "offline-fixture"}})
    write(batch, "drb2/run.json", {"task_ids": ["2", "5"]})
    trial = batch / "drb2/trials/2"
    write(trial, "result.json", {"task_id": "2", "benchmark": "deepresearch_bench_ii", "mode": "bcg",
        "question": "比较两个研究方案的适用场景", "status": "completed", "duration_seconds": 60,
        "usage": {"agent": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                  "graph": {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40}}})
    write(trial, "report.md", "# 研究结果\n\n方案一适合较小样本。方案二适合长期观察。\n")
    write(trial, "agent/trajectories/react_agent.jsonl",
          '{"t":"start","user_message":"比较两个研究方案的适用场景"}\n'
          '{"t":"llm","turn":1,"content":"检索到的证据支持方案一。",'
          '"usage":{"prompt_tokens":100,"completion_tokens":20}}\n')
    write(trial, "tools.jsonl", '{"tool":"web_search","args":{"q":["方案一","方案二"]}}\n')
    write(trial, "agent/bcg/state.json", {"problem_id": "fixture-2", "snapshot": {
        "nodes": [{"id": 1, "belief": "方案一适合较小样本。", "confidence": .8}], "relations": []}})
    write(trial, "agent/bcg/requests.jsonl",
          '{"call":1,"system_memory":"<belief_graph>\\nB1: 方案一适合较小样本。\\n</belief_graph>",'
          '"selected_node_ids":[1],"graph_usage_cumulative":{"llm_totals":{"total_tokens":40}}}\n')
    export_case(trial)
    return Control(root, root / "private-state")


@pytest.fixture
def control(tmp_path: Path) -> Control:
    return build_fixture(tmp_path)


def request_config(**changes: object) -> dict:
    return {"title": "Offline UI test", "profile": "qwen35-9b", "workers": 2,
            "tasks": {"drb2": ["2"]}, **changes}


def test_catalog_and_progress_are_stable(control: Control) -> None:
    user = control.register("甲")["user"]
    first = control.list_runs(user)
    assert len(first) == 1 and first[0]["number"] == "RUN-0001"
    assert control.list_runs(user) == first
    item = control.detail(1, user)
    assert (item["total"], item["ended"], item["reports"]) == (2, 1, 1)
    task = item["tasks"][0]
    assert (task["calls"], task["turn"], task["searches"]) == (1, 1, 2)
    assert (task["agent_tokens"], task["graph_tokens"]) == (120, 40)
    assert task["annotatable"]
    assert not item["tasks"][1]["annotatable"]
    assert "source" not in json.dumps(first)
    # Reading the run list must not consume any future launch numbers.
    assert control.create_run(request_config(), user)["number"] == "RUN-0002"


def test_deepresearch_only_preserves_archived_files_and_reviews(control: Control) -> None:
    user = control.register("archive-reviewer")["user"]
    mixed = control.results / "offline-smoke"
    archived = control.results / "legacy-drbench"
    for root in (mixed, archived):
        write(root, "drbench/run.json", {"task_ids": ["DR0001"]})
        write(root, "drbench/trials/DR0001/result.json", {
            "status": "completed", "usage": {"agent": {"total_tokens": 9999}}})
        write(root, "drbench/trials/DR0001/report.md", "Original DRBench report")
    write(archived, "batch.json", {"status": "finished"})
    report = archived / "drbench/trials/DR0001/report.md"
    before = report.read_bytes()
    with control.connect() as db:
        # Simulate a DRBench-only run cataloged before the filter was introduced.
        db.execute("INSERT INTO runs(source,title,created_at,status) VALUES (?,?,?,?)",
                   ("legacy-drbench", "Original run", "2026-10-21", "external"))
        for run_id in (1, 2):
            db.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?)",
                       (run_id, "drbench", "DR0001", user["id"], 1,
                        '{"status":"draft","goal":"Original annotation"}', "original-digest", "2026-10-21"))
    settings = control.settings()
    assert settings["default_tasks"] == DEFAULT_IDS
    runs = control.list_runs(user)
    assert [r["id"] for r in runs] == [1]
    item = control.detail(1, user)
    assert (item["total"], item["reports"], item["annotation_count"]) == (2, 1, 0)
    assert all(t["benchmark"] == "drb2" for t in item["tasks"])
    assert sum(t["agent_tokens"] or 0 for t in item["tasks"]) == 120
    assert report.read_bytes() == before
    with control.connect() as db:
        assert db.execute("SELECT count(*) FROM reviews WHERE benchmark='drbench'").fetchone()[0] == 2
        assert db.execute("SELECT title FROM runs WHERE id=2").fetchone()[0] == "Original run"
    with pytest.raises(APIError) as error:
        control.task(1, "drbench", "DR0001")
    assert error.value.status == 404
    # A new DRBench-only folder is ignored rather than consuming a run number.
    write(control.results / "new-drbench", "batch.json", {"status": "finished"})
    write(control.results / "new-drbench", "drbench/run.json", {"task_ids": ["DR0001"]})
    control.scan()
    with control.connect() as db:
        assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 2


def test_incremental_counters_ignore_incomplete_line(control: Control) -> None:
    path = control.results / "offline-smoke/drb2/trials/2/tools.jsonl"
    assert control._read_progress(path, "tools")["searches"] == 2
    with path.open("a") as f:
        f.write('{"tool":"web_search",')
    assert control._read_progress(path, "tools")["searches"] == 2
    with path.open("a") as f:
        f.write('"args":{"q":["第三条"]}}\n')
    assert control._read_progress(path, "tools")["searches"] == 3
    assert control._read_progress(path, "tools")["searches"] == 3


def test_identity_token_is_private_and_recoverable(control: Control) -> None:
    result = control.register("独立用户")
    assert control.authenticate(result["token"]) == result["user"]
    with control.connect() as db:
        row = dict(db.execute("SELECT * FROM users").fetchone())
    assert result["token"] not in str(row)
    assert set(control.authenticate(result["token"])) == {"id", "name"}
    with pytest.raises(APIError) as error:
        control.register("独立用户")
    assert error.value.status == 409
    with pytest.raises(APIError) as error:
        control.authenticate("invalid")
    assert error.value.status == 401
    assert control.database.stat().st_mode & 0o077 == 0
    assert not (control.state / "admin-code").exists()


def test_annotations_are_per_user_revisioned_and_evidence_bound(control: Control) -> None:
    a = control.register("甲")["user"]
    b = control.register("乙")["user"]
    path, digest = control.evidence(1, "drb2", "2")
    before = path.read_bytes()
    body = {"revision": 0, "evidence_sha": digest, "annotation": {
        "reviewer": "forged", "status": "draft", "events": [], "dimensions": {}, "conclusion": {}, "goal": "甲的分析"}}
    saved = control.save_review(1, "drb2", "2", a, body)
    assert saved["revision"] == 1
    assert control.review(1, "drb2", "2", a)["annotation"]["reviewer"] == "甲"
    assert control.review(1, "drb2", "2", b)["annotation"] is None
    body["annotation"]["goal"] = "乙的分析"
    control.save_review(1, "drb2", "2", b, body)
    assert control.review(1, "drb2", "2", a)["annotation"]["goal"] == "甲的分析"
    with pytest.raises(APIError) as error:
        control.save_review(1, "drb2", "2", a, body)
    assert error.value.status == 409
    body.update(revision=1, evidence_sha="wrong")
    with pytest.raises(APIError) as error:
        control.save_review(1, "drb2", "2", a, body)
    assert error.value.status == 409
    write(control.results / "offline-smoke/drb2/trials/2", "report.md", "Changed after annotation")
    assert control.evidence(1, "drb2", "2")[0].read_bytes() == before
    reviews = control.detail(1, a)["tasks"][0]["reviews"]
    assert len(reviews) == 2 and sum(r["mine"] for r in reviews) == 1


def test_concurrent_edit_has_one_winner(control: Control) -> None:
    user = control.register("same-user")["user"]
    digest = control.evidence(1, "drb2", "2")[1]
    body = {"revision": 0, "evidence_sha": digest, "annotation": {
        "status": "draft", "events": [], "dimensions": {}, "conclusion": {}}}
    def save(_: int) -> int:
        try:
            return control.save_review(1, "drb2", "2", user, body)["revision"]
        except APIError as error:
            return error.status
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(save, [0, 1])) == [1, 409]


@pytest.mark.parametrize("changes", [
    {"profile": {}}, {"profile": "arbitrary.py"}, {"workers": 3}, {"workers": True},
    {"tasks": {"drb2": ["../.env"]}}, {"tasks": {"drb2": [{}]}}, {"tasks": {"drb2": ["2", "2"]}},
    {"tasks": {"unknown": ["2"]}}, {"tasks": {"drb2": [str(i) for i in range(33)]}},
    {"tasks": {"drbench": ["DR0001"]}}, {"tasks": {"drb2": ["2"], "drbench": ["DR0001"]}},
    {"timeout": 0}, {"max_turns": 301}, {"search_calls": -1},
])
def test_launch_bounds(control: Control, changes: dict) -> None:
    user = control.register("operator")["user"]
    with pytest.raises(APIError) as error:
        control.create_run(request_config(**changes), user)
    assert error.value.status == 400
    assert len(control.list_runs(user)) == 1


def test_default_32_tasks_fit_the_launch_limit(control: Control) -> None:
    expected = ["2", "5", "8", "9", "23", "27", "30", "37", "38", "49", "52", "58", "63", "64",
                "76", "77", "81", "87", "88", "93", "96", "98", "100", "101", "103", "105", "108",
                "110", "119", "120", "121", "128"]
    settings = control.settings()
    assert settings["default_tasks"] == {"drb2": expected}
    assert settings["max_tasks"] == len(expected) == 32
    user = control.register("default-tasks-reviewer")["user"]
    result = control.create_run(request_config(tasks=settings["default_tasks"]), user)
    assert result["total"] == 32 and result["status"] == "queued"
    assert [task["id"] for task in result["tasks"]] == expected
    assert control.row(result["id"])["pid"] is None


def test_signed_in_reviewers_can_start_and_cancel(control: Control) -> None:
    user = control.register("operator")["user"]
    assert control.settings()["launch_requires_admin"] is False
    assert not (control.state / "admin-code").exists()
    result = control.create_run(request_config(), user)
    assert result["number"] == "RUN-0002" and result["status"] == "queued"
    other = control.register("another-operator")["user"]
    assert control.cancel_run(2, other)["status"] == "cancelled"
    with patch("tools.case_study_studio.control.subprocess.Popen") as spawn:
        control.tick()
        spawn.assert_not_called()


def test_obsolete_admin_file_does_not_gate_operations(control: Control) -> None:
    write(control.state, "admin-code", "obsolete")
    reopened = Control(control.root, control.state)
    user = reopened.register("returning-operator")["user"]
    result = reopened.create_run(request_config(), user)
    assert reopened.cancel_run(result["id"], user)["status"] == "cancelled"


def test_queue_serializes_runs_and_does_not_retry_dead_job(control: Control) -> None:
    user = control.register("operator")["user"]
    for _ in range(2):
        control.create_run(request_config(), user)
    with patch("tools.case_study_studio.control.subprocess.Popen", return_value=SimpleNamespace(pid=12345)) as spawn:
        control.tick()
        assert control.row(2)["status"] == "starting"
        assert control.row(3)["status"] == "queued"
        with patch("tools.case_study_studio.control.alive", return_value=True):
            control.tick()
        assert spawn.call_count == 1
        with patch("tools.case_study_studio.control.alive", return_value=False):
            control.tick()
        assert control.row(2)["status"] == "interrupted"
        assert control.row(3)["status"] == "starting"
        assert spawn.call_count == 2
        args = spawn.call_args.args[0]
        assert isinstance(args, list) and args[-1] == "3"
        assert spawn.call_args.kwargs.get("shell") is None


def test_path_boundary_and_unfinished_evidence(control: Control, tmp_path: Path) -> None:
    with pytest.raises(APIError):
        safe_child(control.results, "../private-state/admin-code")
    (control.results / "symlink").symlink_to(tmp_path / "private-state")
    with pytest.raises(APIError):
        safe_child(control.results, "symlink/admin-code")
    with pytest.raises(APIError) as error:
        control.evidence(1, "drb2", "5")
    assert error.value.status == 409
    with pytest.raises(APIError) as error:
        control.task(1, "drb2", "../2")
    assert error.value.status == 404


def test_http_permissions_and_static_boundary(control: Control) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(ROOT / "tools/case_study_studio")))
    server.control = control
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def call(path: str, *, method: str = "GET", body: object = None, headers: dict | None = None) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        h = headers or {}
        if body is not None:
            h = {"Content-Type": "application/json", **h}
        conn.request(method, path, json.dumps(body) if body is not None else None, h)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data
    try:
        assert call("/api/runs")[0] == 401
        assert call("/api/runs", method="POST", body=request_config())[0] == 401
        assert call("/api/runs/1/cancel", method="POST", body={})[0] == 401
        assert call("/api/runs/1/tasks/drb2/2/ai-review")[0] == 401
        assert call("/api/runs/1/tasks/drb2/2/evidence-preview")[0] == 401
        assert call("/api/runs/1/tasks/drb2/2/ai-review", method="POST", body={})[0] == 401
        assert call("/api/settings")[0] == 200
        assert b"runs.mjs" in call("/")[1]
        for name in ("/.env", "/control.py", "/evidence_preview.py", "/profiles/graph.yaml", "/runtime/case-study-studio/admin-code"):
            assert call(name)[0] == 404
        for name in ("evidence-loader.mjs", "evidence-worker.mjs", "evidence-cache.mjs"):
            assert call("/" + name)[0] == 200
        assert call("/api/settings", headers={"Host": "evil.example"})[0] == 403
        assert call("/api/settings", headers={"Origin": "http://evil.example"})[0] == 403
        status, raw = call("/api/session", method="POST", body={"name": "HTTP user"})
        assert status == 201
        identity = json.loads(raw)
        auth = {"Authorization": "Bearer " + identity["token"]}
        assert call("/api/runs", headers=auth)[0] == 200
        assert call("/api/runs", method="POST", body=request_config(), headers={**auth, "Origin": "http://evil.example"})[0] == 403
        status, raw = call("/api/runs", method="POST", body=request_config(), headers=auth)
        assert status == 201
        run_id = json.loads(raw)["id"]
        status, raw = call(f"/api/runs/{run_id}/cancel", method="POST", body={}, headers=auth)
        assert status == 200 and json.loads(raw)["status"] == "cancelled"
        assert call("/api/runs/1/tasks/drb2/2/evidence", headers=auth)[1][:2] == b"\x1f\x8b"
        assert call("/api/runs/1/tasks/drb2/2/evidence-preview", headers=auth)[1][:2] == b"\x1f\x8b"
        assert call("/api/session", method="POST", body=[], headers=auth)[0] == 400
        ai_path = "/api/runs/1/tasks/drb2/2/ai-review"
        assert json.loads(call(ai_path, headers=auth)[1])["status"] == "not_started"
        assert call(ai_path, method="POST", body={"retry": "yes"}, headers=auth)[0] == 400
        assert call(ai_path, method="POST", body={"input": "/etc/passwd"}, headers=auth)[0] == 400
        from tools.ai_review.client import ReviewConfig
        with patch("tools.case_study_studio.review_queue.ReviewConfig.from_env", return_value=ReviewConfig(api_key="fake")):
            assert call(ai_path, method="POST", body={}, headers=auth)[0] == 202
        assert json.loads(call(ai_path, headers=auth)[1])["status"] == "queued"
        assert call(ai_path, method="PUT", body={"annotation": {}}, headers=auth)[0] == 404
        trial = control.task(1, "drb2", "2")[0]
        write(trial, "result.json", {"status": "agent_error"})
        annotation_path = "/api/runs/1/tasks/drb2/2/annotation"
        body = {"revision": 0, "evidence_sha": "old-evidence", "annotation": {
            "status": "draft", "events": [], "dimensions": {}, "conclusion": {}}}
        for path, method, payload in (
            ("/api/runs/1/tasks/drb2/2/evidence", "GET", None),
            ("/api/runs/1/tasks/drb2/2/evidence-preview", "GET", None),
            (annotation_path, "GET", None), (annotation_path, "PUT", body),
            (ai_path, "GET", None), (ai_path, "POST", {"retry": True}),
        ):
            status, raw = call(path, method=method, body=payload, headers=auth)
            assert status == 409 and "正常完成" in json.loads(raw)["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
