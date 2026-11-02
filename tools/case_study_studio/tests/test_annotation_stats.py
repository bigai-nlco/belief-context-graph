"""Offline, deterministic annotation-only aggregation and authenticated exports."""
from __future__ import annotations

import copy
import http.client
import io
import json
import subprocess
import sys
import threading
import zipfile
from contextlib import closing
from functools import partial
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import pytest

from tools.case_study_studio.annotation_charts import render_report
from tools.case_study_studio.annotation_stats import (
    DIMENSIONS,
    MISSING,
    OPTIONS,
    ROOT,
    UNKNOWN,
    StatsError,
    archive,
    collect_snapshot,
    project_annotation,
    report_files,
    summarize,
)
from tools.case_study_studio.control import Control
from tools.case_study_studio.serve import Handler, StudioHTTPServer
from tools.case_study_studio.tests.test_control import build_fixture, write


def annotation() -> dict:
    return {"status": "reviewed", "result": "部分正确 / 部分达标", "noApplicableDimensions": False,
        "customDimensions": [], "dimensions": {k: {"opportunity": "无"} for k in DIMENSIONS},
        "events": [], "conclusion": {"focus": ["Agent 使用"], "evidence": "明确可见"}}


def record(task: str, reviewer: str = "甲", *, source: str = "human", state: str = "completed") -> dict:
    a = annotation()
    a["dimensions"]["A1"] = {"opportunity": "有", "utilization": "未利用", "effect": "负向", "eventIds": ["e1", "e2", "e1"]}
    a["events"] = [{"id": "e1", "label": "重复搜索", "ai_kind": "missed_opportunity", "strength": "明确可见"},
                   {"id": "e2", "label": "已有约束未采用", "strength": "有较强迹象"}]
    return {"task_id": task, "benchmark": "drb2", "source": source, "reviewer": reviewer,
        "reviewer_id": reviewer if source == "human" else "ai", "state": state,
        "model": "fixture-model", "prompt_version": "v1", "reasoning_effort": "medium",
        "annotation": project_annotation(a)}


def snapshot(records: list[dict] | None = None) -> dict:
    return {"format": "bcg-run-annotations", "version": 1, "run": {"id": 1, "number": "RUN-0001", "title": "Offline"},
        "tasks": [{"benchmark": "drb2", "task_id": str(n), "status": "completed"} for n in range(1, 5)],
        "records": records or [], "warnings": []}


def dimension(stats: dict, key: str) -> dict:
    return next(d for d in stats["cohorts"][0]["dimensions"] if d["id"] == key)


def test_one_vote_per_task_dimension_not_per_link_or_event() -> None:
    s = summarize(snapshot([record("1"), record("2")]))
    assert dimension(s, "A1")["utilization"]["未利用"] == 2
    assert dimension(s, "A1")["applicable_tasks"] == 2
    assert s["cohorts"][0]["events"] == 4
    assert s["dimension_rows"][0]["event_ids"] == ["e1", "e2"]
    assert s["coverage"]["human_only"] == 2


def test_unknown_blank_and_deselected_fields_are_not_negative() -> None:
    r = record("1")
    r["annotation"]["dimensions"].update({
        "A1": {"opportunity": "有", "utilization": "", "effect": "invented", "eventIds": ["e1"]},
        "A2": {"opportunity": "无", "utilization": "误用", "effect": "负向"},
        "A3": {"opportunity": "无法判断", "utilization": "充分利用", "effect": "正向"},
        "A4": {},
    })
    s = summarize(snapshot([r]))
    assert dimension(s, "A1")["utilization"][MISSING] == 1
    assert dimension(s, "A1")["utilization"]["未利用"] == 0
    assert dimension(s, "A1")["effect"][UNKNOWN] == 1
    assert dimension(s, "A2")["applicable_tasks"] == 0
    assert sum(dimension(s, "A2")["effect"].values()) == 0
    assert dimension(s, "A3")["opportunity"]["无法判断"] == 1
    assert sum(dimension(s, "A3")["utilization"].values()) == 0
    assert dimension(s, "A4")["opportunity"][MISSING] == 1
    assert any("未知值" in w for w in s["warnings"])


def test_reviewers_and_ai_prompt_versions_are_separate_without_duplicating_coverage() -> None:
    records = [record("1", "甲"), record("1", "乙"), record("1", source="ai"), record("2", source="ai")]
    records[-1]["prompt_version"] = "v2"
    s = summarize(snapshot(records))
    assert len(s["cohorts"]) == 4
    assert all(c["tasks"] == 1 for c in s["cohorts"])
    assert s["coverage"]["reviewed_tasks"] == 2
    assert s["coverage"]["both"] == 1
    assert s["coverage"]["ai_only"] == 1


def test_exclude_drafts_failed_jobs_out_of_run_and_failed_tasks() -> None:
    raw = snapshot([record("1", state="draft"), record("2", source="ai", state="failed"), record("8"), record("3")])
    raw["tasks"][2]["status"] = "timeout"
    s = summarize(raw)
    assert not s["cohorts"]
    assert len(s["excluded_records"]) == 4
    assert s["coverage"]["eligible_tasks"] == 3
    assert s["coverage"]["ineligible_tasks"] == 1
    assert s["coverage"]["unreviewed"] == 3


def test_duplicate_annotations_and_process_ids_rejected() -> None:
    with pytest.raises(StatsError, match="同一题"):
        summarize(snapshot([record("1"), record("1")]))
    a = annotation()
    a["events"] = [{"id": "x"}, {"id": "x"}]
    with pytest.raises(StatsError, match="ID"):
        project_annotation(a)
    with pytest.raises(StatsError):
        project_annotation({})


def test_custom_dimensions_use_exact_name_and_definition_not_local_ids() -> None:
    records = [record("1"), record("2"), record("3")]
    for r, local_id, title in zip(records, ["C1", "C2", "C1"], ["检查引用", "检查引用", "完成输出"], strict=True):
        r["annotation"]["customDimensions"] = [{"id": local_id, "title": title, "question": "轨迹中的具体行为"}]
        r["annotation"]["dimensions"][local_id] = {"opportunity": "有", "utilization": "部分利用", "effect": "待验证", "eventIds": ["e1"]}
    dims = [d for d in summarize(snapshot(records))["cohorts"][0]["dimensions"] if d["id"].startswith("custom-")]
    assert len(dims) == 2
    assert [(d["title"], d["available_tasks"], d["applicable_tasks"]) for d in dims] == [("检查引用", 2, 2), ("完成输出", 1, 1)]


def test_invalid_links_warn_but_never_reinterpret_the_rating() -> None:
    r = record("1")
    r["annotation"]["dimensions"]["A1"]["eventIds"] = ["lost", "e1", "lost"]
    s = summarize(snapshot([r]))
    assert dimension(s, "A1")["utilization"]["未利用"] == 1
    assert s["dimension_rows"][0]["missing_event_ids"] == ["lost"]
    assert len(s["warnings"]) == 1


def test_no_applicable_dimension_is_not_inferred_from_blank_controls() -> None:
    r = record("1")
    r["annotation"] = project_annotation(annotation())
    assert summarize(snapshot([r]))["cohorts"][0]["no_applicable_dimensions"] == 0
    r["annotation"]["noApplicableDimensions"] = True
    assert summarize(snapshot([r]))["cohorts"][0]["no_applicable_dimensions"] == 1
    r["annotation"]["noApplicableDimensions"] = None  # old all-no annotations
    assert summarize(snapshot([r]))["cohorts"][0]["no_applicable_dimensions"] == 1


def test_free_text_never_becomes_labels_and_projection_drops_large_evidence() -> None:
    a = annotation()
    a["conclusion"]["benefit"] = "发挥 BCG 优势，很多未利用，重复搜索"
    a["events"] = [{"id": "x", "label": "重复搜索", "quote": "private raw trace", "graphEvidence": {"text": "private graph"}}]
    projected = project_annotation(a)
    r = record("1")
    r["annotation"] = projected
    s = summarize(snapshot([r]))
    assert s["cohorts"][0]["event_kind"][MISSING] == 1
    assert s["cohorts"][0]["event_kind"]["missed_opportunity"] == 0
    assert s["cohorts"][0]["search"][MISSING] == 1
    assert "private" not in json.dumps(projected)


def test_deterministic_files_svg_html_and_archive_are_safe() -> None:
    raw = snapshot([record("1", "<script>alert(1)</script>")])
    raw["run"]["title"] = '<img src="x" onerror="alert(1)">'
    raw["records"][0]["annotation"]["events"][0]["label"] = '=HYPERLINK("x")'
    files = report_files(raw)
    assert files == report_files(copy.deepcopy(raw))
    assert archive(files) == archive(report_files(raw))
    assert b"<script>" not in files["report.html"]
    assert b'<img src="x"' not in files["report.html"]
    assert b"&lt;script&gt;" in files["report.html"]
    assert b"'=HYPERLINK" in files["events.csv"]
    for name, data in files.items():
        if name.endswith(".svg"):
            svg = ElementTree.fromstring(data)
            assert svg.tag.endswith("svg")
    with zipfile.ZipFile(io.BytesIO(archive(files))) as pack:
        assert set(pack.namelist()) == set(files)
        assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in pack.infolist())
        assert all(pack.read(name) == data for name, data in files.items())


def test_contract_matches_frontend_options() -> None:
    command = "import {DIMENSIONS,OPTIONS} from './tools/case_study_studio/model.mjs'; console.log(JSON.stringify({DIMENSIONS,OPTIONS}));"
    result = subprocess.run(["node", "--input-type=module", "-e", command], cwd=ROOT, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert {d["id"]: d["title"] for d in data["DIMENSIONS"]} == DIMENSIONS
    assert all(OPTIONS[k] == v for k, v in data["OPTIONS"].items())


def save_reviews(control: Control) -> None:
    user = control.register("统计测试者")["user"]
    a = annotation()
    with control.connect() as db:
        db.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?)", (1, "drb2", "2", user["id"], 1, json.dumps(a), "bundle-sha", "fixed-time"))
        db.execute("INSERT INTO ai_reviews(id,run_id,benchmark,task_id,requested_by,status,created_at,evidence_sha) VALUES (?,?,?,?,?,?,?,?)",
            (1, 1, "drb2", "2", user["id"], "completed", "fixed-time", "bundle-sha"))
    a["status"] = "draft"  # Completed AI files are intentionally saved as editable drafts.
    write(control.state, "ai-reviews/1/review.json", {"task_id": "2", "benchmark": "deepresearch_bench_ii",
        "evidence_sha256": "content-identity-sha", "model": {"model": "fixture-ai", "reasoning_effort": "medium",
        "api_key": "private-secret", "base_url": "https://private"}, "prompt_version": "v1", "annotation": a})


def test_collector_is_read_only_no_trajectory_model_or_private_config_access(tmp_path: Path) -> None:
    control = build_fixture(tmp_path)
    save_reviews(control)
    original_read = Path.read_text

    def guarded(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        assert path.name in {"run.json", "result.json", "review.json"}, f"unexpected file: {path.name}"
        return original_read(path, encoding=encoding, errors=errors)

    # Flush fixture writes first. A later GC of a fixture writer may otherwise
    # checkpoint its WAL while the read-only collector is running.
    with closing(control.connect()) as keeper:
        keeper.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        database_before = control.database.read_bytes()
        with patch.object(Path, "read_text", guarded), patch("socket.socket") as socket:
            raw = collect_snapshot(control.root, control.state, 1)
            stats = summarize(raw)
            report_files(raw)
            socket.assert_not_called()
        assert control.database.read_bytes() == database_before
    assert stats["coverage"]["both"] == 1
    assert len(stats["cohorts"]) == 2
    assert not any(v in json.dumps(raw) for v in ("private-secret", "https://private", "token_hash", "api_key"))
    assert next(r for r in raw["records"] if r["source"] == "ai")["annotation"]["status"] == "draft"


def test_collector_invalid_and_missing_ai_outputs_do_not_count(tmp_path: Path) -> None:
    control = build_fixture(tmp_path)
    save_reviews(control)
    path = control.state / "ai-reviews/1/review.json"
    saved = json.loads(path.read_text())
    saved["task_id"] = "5"
    write(path.parent, path.name, saved)
    stats = summarize(collect_snapshot(control.root, control.state, 1))
    assert stats["coverage"]["human_only"] == 1 and stats["coverage"]["both"] == 0
    assert stats["warnings"]
    path.unlink()
    assert summarize(collect_snapshot(control.root, control.state, 1))["excluded_records"][0]["reason"] == "invalid"
    with pytest.raises(StatsError):
        collect_snapshot(control.root, control.state, 1000)


def test_source_paths_cannot_escape_allowed_directories(tmp_path: Path) -> None:
    control = build_fixture(tmp_path)
    with control.connect() as db:
        db.execute("UPDATE runs SET source='../private-state' WHERE id=1")
    with pytest.raises(StatsError, match="超出"):
        collect_snapshot(control.root, control.state, 1)


def test_cli_export_and_replay_are_identical_without_overwriting(tmp_path: Path) -> None:
    control = build_fixture(tmp_path)
    save_reviews(control)
    first, second = tmp_path / "first", tmp_path / "second"
    command = [sys.executable, "-m", "tools.case_study_studio.annotation_stats"]
    subprocess.run([*command, "--root", str(control.root), "--state-dir", str(control.state), "--run-id", "1", "--output", str(first)], cwd=ROOT, check=True, capture_output=True)
    subprocess.run([*command, "--input", str(first / "input_snapshot.json"), "--output", str(second)], cwd=ROOT, check=True, capture_output=True)
    assert (first / "annotation-stats.zip").read_bytes() == (second / "annotation-stats.zip").read_bytes()
    again = subprocess.run([*command, "--input", str(first / "input_snapshot.json"), "--output", str(first)], cwd=ROOT, capture_output=True)
    assert again.returncode == 2
    assert "非空" in again.stderr.decode()


def test_empty_run_report_and_unsupported_snapshot() -> None:
    html, charts = render_report(summarize(snapshot()))
    assert "暂无已完成标注" in html
    assert set(charts) == {"coverage"}
    with pytest.raises(StatsError):
        summarize({"version": 2})


def test_http_report_requires_auth_and_zip_matches_displayed_snapshot(tmp_path: Path) -> None:
    control = build_fixture(tmp_path)
    save_reviews(control)
    identity = control.register("统计读取者")
    server = StudioHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(ROOT / "tools/case_study_studio")))
    server.control = control
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path: str, *, authorized: bool = True) -> tuple[int, bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        headers = {"Authorization": "Bearer " + identity["token"]} if authorized else {}
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, data

    try:
        route = "/api/runs/1/annotation-stats"
        assert request(route, authorized=False)[0] == 401
        assert request("/api/runs/999/annotation-stats")[0] == 404
        status, raw = request(route)
        assert status == 200
        result = json.loads(raw)
        assert "标注覆盖" in result["report_html"]
        assert b"private-secret" not in raw
        export = route + "/" + result["stats"]["input_sha256"] + "/download"
        assert request(export, authorized=False)[0] == 401
        status, zip_bytes = request(export)
        assert status == 200
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as pack:
            assert json.loads(pack.read("stats.json")) == result["stats"]
            assert pack.read("report.html").decode() == result["report_html"]
        with control.connect() as db:
            db.execute("UPDATE reviews SET revision=2 WHERE run_id=1")
        assert request(export)[0] == 409
        for asset in ("annotation-stats.html", "annotation-stats.css", "annotation-stats.mjs"):
            assert request("/" + asset, authorized=False)[0] == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
