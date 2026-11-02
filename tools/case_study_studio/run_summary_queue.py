"""Snapshot-keyed, durable Run summaries, separate from per-task AI reviews."""
from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

from tools.ai_review.client import ReviewError
from tools.case_study_studio.annotation_stats import StatsError, canonical, digest
from tools.case_study_studio.control import APIError, alive, now, read_json, safe_child
from tools.case_study_studio.run_summary import (
    EFFORT,
    MODEL,
    SUMMARY_STYLE,
    VERSION,
    build_context,
    configuration,
    input_key,
    messages,
    render_summary,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tools.case_study_studio.control import Control


class RunSummaries:
    def __init__(self, control: Control) -> None:
        self.control = control
        self.children: list[subprocess.Popen] = []
        with control.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS run_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL REFERENCES runs(id),
                input_key TEXT NOT NULL, statistics_sha TEXT NOT NULL, prompt_version TEXT NOT NULL,
                requested_by TEXT NOT NULL REFERENCES users(id), status TEXT NOT NULL, pid INTEGER,
                cohorts INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error TEXT, usage TEXT,
                UNIQUE(run_id,input_key))""")

    def output(self, job_id: int) -> Path:
        return safe_child(self.control.state, "run-summaries", str(job_id))

    def row(self, job_id: int) -> dict:
        with self.control.connect() as db:
            row = db.execute("SELECT * FROM run_summaries WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise APIError(404, "找不到这份 Run 总结")
        return dict(row)

    def public(self, row: dict) -> dict:
        return {key: row[key] for key in ("id", "run_id", "input_key", "statistics_sha", "prompt_version", "status",
            "attempts", "created_at", "started_at", "finished_at", "error")} | {
                "model": MODEL, "reasoning_effort": EFFORT, "cohorts": row["cohorts"],
                "completed_cohorts": len(list(self.output(row["id"]).glob("cohort-*.json"))),
                "usage": json.loads(row["usage"]) if row["usage"] else None}

    def get(self, run_id: int, key: str) -> dict:
        self.control.row(run_id)
        with self.control.connect() as db:
            row = db.execute("SELECT * FROM run_summaries WHERE run_id=? AND input_key=?", (run_id, key)).fetchone()
        if row is None:
            raise APIError(404, "找不到这份统计快照的 Run 总结")
        result = self.public(dict(row))
        if row["status"] == "completed":
            context, saved = self.saved(dict(row))
            result.update(summary=saved, summary_html=render_summary(context, saved), summary_style=SUMMARY_STYLE)
        return result

    def current(self, run_id: int, statistics_sha: str) -> dict | None:
        """Read an existing job for the current input; never enqueue or read API credentials."""
        self.control.row(run_id)
        try:
            context = build_context(self.control.root, self.control.state, run_id)
        except (StatsError, ReviewError) as exc:
            raise APIError(409, str(exc)) from exc
        if context["statistics_sha256"] != statistics_sha:
            raise APIError(409, "标注已有更新，请刷新统计后查看总结")
        key = input_key(context)
        with self.control.connect() as db:
            exists = db.execute("SELECT id FROM run_summaries WHERE run_id=? AND input_key=?", (run_id, key)).fetchone()
        return self.get(run_id, key) if exists else None

    def saved(self, row: dict) -> tuple[dict, dict]:
        output = self.output(row["id"])
        context, saved = read_json(output / "context.json"), read_json(output / "summary.json")
        if (not context or not saved or saved.get("context_sha256") != digest(context)
                or saved.get("input_key") != row["input_key"] or saved.get("version") != row["prompt_version"]
                or context.get("statistics_sha256") != row["statistics_sha"]):
            raise APIError(409, "Run 总结与固定输入不匹配，未展示")
        return context, saved

    def export(self, run_id: int, key: str, statistics_sha: str) -> dict[str, bytes]:
        with self.control.connect() as db:
            row = db.execute("SELECT * FROM run_summaries WHERE run_id=? AND input_key=?", (run_id, key)).fetchone()
        if row is None or row["status"] != "completed" or row["statistics_sha"] != statistics_sha:
            raise APIError(409, "所选 AI 总结尚未完成或不属于当前统计快照")
        context, saved = self.saved(dict(row))
        original_input = read_json(self.output(row["id"]) / "input.json")
        if original_input.get("input_key") != row["input_key"]:
            raise APIError(409, "缺少与当前总结匹配的原始请求记录")
        from tools.case_study_studio.annotation_charts import STYLE
        html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        html += f'<title>Agent × BCG Run 总结</title><style>{STYLE}{SUMMARY_STYLE}</style><main>{render_summary(context,saved)}</main></html>'
        return {"ai-summary/report.html": html.encode(), "ai-summary/summary.json": canonical(saved),
            "ai-summary/context.json": canonical(context),
            "ai-summary/prompts.json": canonical(original_input["requests"])}

    def enqueue(self, run_id: int, statistics_sha: str, user: dict, *, retry: bool = False) -> dict:
        self.control.row(run_id)
        try:
            context = build_context(self.control.root, self.control.state, run_id)
            if context["statistics_sha256"] != statistics_sha:
                raise APIError(409, "标注已有更新，请刷新统计后生成总结")
            if not context["cohorts"]:
                raise APIError(409, "暂无已完成标注，不调用总结模型")
            # Check the full input before enqueuing any paid work.
            for cohort in context["cohorts"]:
                messages(context, cohort)
        except (StatsError, ReviewError) as exc:
            raise APIError(409, str(exc)) from exc
        key = input_key(context)
        with self.control.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM run_summaries WHERE run_id=? AND input_key=?", (run_id, key)).fetchone()
            if row and (row["status"] in {"queued", "running", "completed"} or not retry):
                return self.public(dict(row))
            if not configuration().api_key:
                raise APIError(503, "服务器未配置 LITELLM_OPENAI_API_KEY，统计仍可正常查看")
            if db.execute("SELECT count(*) FROM run_summaries WHERE status IN ('queued','running')").fetchone()[0] >= 8:
                raise APIError(429, "Run 总结队列已满，请稍后重试")
            if row:
                db.execute("UPDATE run_summaries SET status='queued',pid=NULL,attempts=attempts+1,requested_by=?,"
                    "error=NULL,started_at=NULL,finished_at=NULL WHERE id=?", (user["id"], row["id"]))
                job_id = row["id"]
            else:
                cursor = db.execute("INSERT INTO run_summaries(run_id,input_key,statistics_sha,prompt_version,requested_by,"
                    "status,cohorts,created_at) VALUES (?,?,?,?,?,'queued',?,?)",
                    (run_id, key, statistics_sha, VERSION, user["id"], len(context["cohorts"]), now()))
                job_id = cursor.lastrowid
                assert job_id is not None
            output = self.output(job_id)
            output.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = output / "context.json"
            path.write_bytes(canonical(context))
            path.chmod(0o600)
        return self.public(self.row(job_id))

    def tick(self) -> None:
        self.children = [p for p in self.children if p.poll() is None]
        with self.control.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT * FROM run_summaries WHERE status='running'").fetchall():
                if alive(row["pid"], f"tools.case_study_studio.run_summary_job --job-id {row['id']} "):
                    return
                db.execute("UPDATE run_summaries SET status='interrupted',finished_at=?,error=? WHERE id=?",
                    (now(), "总结进程已退出；已完成的标注组保留，点击重试继续", row["id"]))
            row = db.execute("SELECT * FROM run_summaries WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return
            try:
                with (self.output(row["id"]) / "worker.log").open("ab") as stream:
                    process = subprocess.Popen([str(self.control.root / ".venv/bin/python"), "-m",
                        "tools.case_study_studio.run_summary_job", "--job-id", str(row["id"]),
                        "--root", str(self.control.root), "--state", str(self.control.state)], cwd=self.control.root,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.children.append(process)
            except OSError:
                db.execute("UPDATE run_summaries SET status='failed',finished_at=?,error=? WHERE id=?",
                    (now(), "无法启动 Run 总结进程，请检查服务器环境", row["id"]))
                return
            db.execute("UPDATE run_summaries SET status='running',pid=?,started_at=? WHERE id=?",
                (process.pid, now(), row["id"]))
