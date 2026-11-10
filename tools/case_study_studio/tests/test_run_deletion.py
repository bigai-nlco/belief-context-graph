"""Offline checks: deletion is explicit, scoped, idle-only and rollback-safe."""
from __future__ import annotations

import http.client
import json
import sqlite3
import threading
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.case_study_studio.control import ROOT, APIError, Control
from tools.case_study_studio.serve import Handler, StudioHTTPServer
from tools.case_study_studio.tests.test_control import build_fixture, request_config, write


@pytest.fixture
def control(tmp_path: Path) -> Control:
    return build_fixture(tmp_path)


def seed_job(control: Control, user: dict, table: str, status: str = "completed") -> int:
    with control.connect() as db:
        if table == "ai_reviews":
            result = db.execute("INSERT INTO ai_reviews(run_id,benchmark,task_id,requested_by,status,created_at) "
                "VALUES (1,'drb2','2',?,?,?)", (user["id"], status, "2026-11-05"))
        else:
            result = db.execute("INSERT INTO run_summaries(run_id,input_key,statistics_sha,prompt_version,"
                "requested_by,status,cohorts,created_at) VALUES (1,'input','stats','v1',?,?,1,?)",
                (user["id"], status, "2026-11-05"))
        assert result.lastrowid is not None
        return result.lastrowid


def test_deletes_only_selected_run_and_all_associated_artifacts(control: Control) -> None:
    user = control.register("deleter")["user"]
    root = control.directory(control.row(1))
    evidence, digest = control.evidence(1, "drb2", "2")
    control.save_review(1, "drb2", "2", user, {"revision": 0, "evidence_sha": digest,
        "annotation": {"status": "reviewed", "events": [], "dimensions": {}, "conclusion": {}}})
    review_id = seed_job(control, user, "ai_reviews")
    summary_id = seed_job(control, user, "run_summaries")
    review = write(control.state, f"ai-reviews/{review_id}/review.json", {})
    summary = write(control.state, f"run-summaries/{summary_id}/summary.json", {})
    write(root, "graphs/test/graph.json", {"nodes": []})
    other = control.create_run(request_config(), user)
    other_file = write(control.directory(control.row(other["id"])), "keep.txt", "keep")
    control.detail(1)
    assert any(key.startswith(str(root)) for key in control.file_cache)
    result = control.delete_run(1, user, "RUN-0001")
    assert result == {"id": 1, "number": "RUN-0001", "deleted": True, "cleanup_pending": False}
    assert all(not path.exists() for path in (root, evidence, review, summary))
    assert other_file.read_text() == "keep"
    assert control.row(other["id"])["status"] == "queued"
    assert not any(key.startswith(str(root)) for key in control.file_cache)
    with control.connect() as db:
        for table in ("reviews", "ai_reviews", "run_summaries"):
            assert db.execute(f"SELECT count(*) FROM {table} WHERE run_id=1").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM users").fetchone()[0] == 1
    assert [run["id"] for run in control.list_runs(user)] == [2]
    assert control.create_run(request_config(), user)["id"] == 3
    with pytest.raises(APIError, match="找不到"):
        control.delete_run(1, user, "RUN-0001")


@pytest.mark.parametrize("confirmation", [None, "", "RUN-0002", "1", True])
def test_requires_exact_confirmation(control: Control, confirmation: object) -> None:
    user = control.register("user")["user"]
    with pytest.raises(APIError) as error:
        control.delete_run(1, user, confirmation)
    assert error.value.status == 400
    assert control.directory(control.row(1)).is_dir()


@pytest.mark.parametrize("status", ["queued", "starting", "running"])
def test_running_or_queued_test_cannot_be_deleted(control: Control, status: str) -> None:
    user = control.register("user")["user"]
    with control.connect() as db:
        db.execute("UPDATE runs SET managed=1,status=? WHERE id=1", (status,))
    assert not control.detail(1)["can_delete"]
    with pytest.raises(APIError, match="先停止测试"):
        control.delete_run(1, user, "RUN-0001")
    assert control.directory(control.row(1)).exists()


@pytest.mark.parametrize("table", ["ai_reviews", "run_summaries"])
@pytest.mark.parametrize("status", ["queued", "running"])
def test_blocks_active_ai_jobs(control: Control, table: str, status: str) -> None:
    user = control.register("user")["user"]
    seed_job(control, user, table, status)
    assert not control.detail(1)["can_delete"]
    with pytest.raises(APIError, match="AI"):
        control.delete_run(1, user, "RUN-0001")
    assert control.directory(control.row(1)).exists()


def test_terminal_status_is_not_enough_while_process_still_exiting(control: Control) -> None:
    user = control.register("user")["user"]
    with patch("tools.case_study_studio.run_deletion.alive", return_value=True), pytest.raises(APIError, match="后台进程"):
        control.delete_run(1, user, "RUN-0001")


def test_stopped_queued_run_without_files_can_be_deleted(control: Control) -> None:
    user = control.register("user")["user"]
    run = control.create_run(request_config(), user)
    control.cancel_run(run["id"], user)
    assert control.delete_run(run["id"], user, run["number"])["deleted"]
    assert control.directory(control.row(1)).is_dir()


def test_rename_failure_restores_files_and_database(control: Control) -> None:
    user = control.register("user")["user"]
    evidence, _ = control.evidence(1, "drb2", "2")
    original = Path.rename

    def fail_second(path: Path, target: Path) -> Path:
        if path == control.state / "evidence/1":
            raise PermissionError("test rename denied")
        return original(path, target)

    with patch.object(Path, "rename", fail_second), pytest.raises(PermissionError):
        control.delete_run(1, user, "RUN-0001")
    assert control.directory(control.row(1)).is_dir()
    assert evidence.is_file()
    assert not list(control.results.glob(".delete-*"))


def test_db_failure_rolls_back_staged_directories(control: Control) -> None:
    user = control.register("user")["user"]
    evidence, _ = control.evidence(1, "drb2", "2")
    with control.connect() as db:
        db.execute("CREATE TRIGGER refuse_delete BEFORE DELETE ON runs BEGIN SELECT RAISE(ABORT, 'test'); END")
    with pytest.raises(sqlite3.IntegrityError):
        control.delete_run(1, user, "RUN-0001")
    assert control.directory(control.row(1)).is_dir()
    assert evidence.is_file()


@pytest.mark.parametrize("source", [".", "", "studio", "../private-state", "/tmp", "offline-smoke/.."])
def test_refuses_root_or_outside_paths(control: Control, source: str) -> None:
    user = control.register("user")["user"]
    with control.connect() as db:
        db.execute("UPDATE runs SET source=? WHERE id=1", (source,))
    with pytest.raises(APIError):
        control.delete_run(1, user, "RUN-0001")
    assert (control.results / "offline-smoke/batch.json").is_file()


def test_refuses_root_symlink_but_does_not_follow_child_symlinks(control: Control) -> None:
    user = control.register("user")["user"]
    root = control.directory(control.row(1))
    (control.results / "alias").symlink_to(root, target_is_directory=True)
    with control.connect() as db:
        db.execute("UPDATE runs SET source='alias' WHERE id=1")
    with pytest.raises(APIError, match="软链接"):
        control.delete_run(1, user, "RUN-0001")
    with control.connect() as db:
        db.execute("UPDATE runs SET source='offline-smoke' WHERE id=1")
    outside = write(control.root, "shared-models/weights", "keep")
    (root / "model-link").symlink_to(outside.parent, target_is_directory=True)
    assert control.delete_run(1, user, "RUN-0001")["deleted"]
    assert outside.read_text() == "keep"


def test_rejects_overlapping_run_directories(control: Control) -> None:
    user = control.register("user")["user"]
    with control.connect() as db:
        db.execute("INSERT INTO runs(source,title,created_at,status) VALUES ('offline-smoke/nested','other','','external')")
    with pytest.raises(APIError, match="共用"):
        control.delete_run(1, user, "RUN-0001")


def test_cleanup_failure_is_visible_and_hidden_files_not_recataloged(control: Control) -> None:
    user = control.register("user")["user"]
    with patch("tools.case_study_studio.run_deletion.shutil.rmtree", side_effect=PermissionError):
        result = control.delete_run(1, user, "RUN-0001")
    assert result["cleanup_pending"]
    assert control.list_runs(user) == []
    assert len(list(control.results.glob(".delete-*/batch.json"))) == 1


def test_delete_endpoint_requires_auth_origin_and_confirmation(control: Control) -> None:
    server = StudioHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(ROOT / "tools/case_study_studio")))
    server.control = control
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    identity = control.register("http-user")

    def call(body: dict, *, token: str = "", origin: str | None = None) -> int:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + token}
        if origin:
            headers["Origin"] = origin
        connection.request("DELETE", "/api/runs/1", json.dumps(body), headers)
        response = connection.getresponse()
        response.read()
        connection.close()
        return response.status

    try:
        body = {"confirmation": "RUN-0001"}
        assert call(body) == 401
        assert call(body, token=identity["token"], origin="https://evil.example") == 403
        assert call({}, token=identity["token"]) == 400
        assert call({**body, "path": "/tmp"}, token=identity["token"]) == 400
        assert call(body, token=identity["token"]) == 200
        assert call(body, token=identity["token"]) == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
