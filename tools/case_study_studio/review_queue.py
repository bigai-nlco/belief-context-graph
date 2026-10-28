"""Durable, on-demand AI reviews. Human annotations are never written here."""
from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

from tools.ai_review.client import ReviewConfig
from tools.case_study_studio.control import APIError, alive, now, read_json, safe_child

if TYPE_CHECKING:
    from pathlib import Path

    from tools.case_study_studio.control import Control


class AIReviews:
    def __init__(self, control: Control) -> None:
        self.control = control
        self.children: list[subprocess.Popen] = []
        with control.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS ai_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(id), benchmark TEXT NOT NULL,
                task_id TEXT NOT NULL, requested_by TEXT NOT NULL REFERENCES users(id),
                status TEXT NOT NULL, pid INTEGER, attempts INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                evidence_sha TEXT, error TEXT, usage TEXT,
                UNIQUE(run_id, benchmark, task_id))""")

    def output(self, job_id: int) -> Path:
        return safe_child(self.control.state, "ai-reviews", str(job_id))

    def row(self, job_id: int) -> dict:
        with self.control.connect() as db:
            row = db.execute("SELECT * FROM ai_reviews WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise APIError(404, "找不到这份 AI Review")
        return dict(row)

    def public(self, row: dict) -> dict:
        output = self.output(row["id"])
        manifest = read_json(output / "manifest.json")
        return {"id": row["id"], "status": row["status"], "model": "gpt-6.1-sol",
                "reasoning_effort": "medium", "attempts": row["attempts"],
                "created_at": row["created_at"], "started_at": row["started_at"],
                "finished_at": row["finished_at"], "error": row["error"],
                "batches": manifest.get("batches"),
                "completed_batches": len(list(output.glob("findings-*.json"))),
                "usage": json.loads(row["usage"]) if row["usage"] else manifest.get("usage")}

    def statuses(self, run_id: int) -> dict:
        with self.control.connect() as db:
            rows = db.execute("SELECT * FROM ai_reviews WHERE run_id=?", (run_id,)).fetchall()
        return {(row["benchmark"], row["task_id"]): self.public(dict(row)) for row in rows}

    def get(self, run_id: int, benchmark: str, task_id: str) -> dict:
        self.control.annotation_task(run_id, benchmark, task_id)
        with self.control.connect() as db:
            row = db.execute("SELECT * FROM ai_reviews WHERE run_id=? AND benchmark=? AND task_id=?",
                             (run_id, benchmark, task_id)).fetchone()
        if not row:
            return {"status": "not_started", "annotation": None}
        result = {**self.public(dict(row)), "annotation": None}
        if row["status"] == "completed":
            saved = read_json(self.output(row["id"]) / "review.json")
            if not saved.get("annotation") or not row["evidence_sha"]:
                raise APIError(409, "AI 标注文件不完整，请联系维护者检查")
            result.update(annotation=saved["annotation"], evidence_sha=row["evidence_sha"],
                          coverage=saved.get("coverage"), prompt_version=saved.get("prompt_version"))
        return result

    def enqueue(self, run_id: int, benchmark: str, task_id: str, user: dict, *, retry: bool = False) -> dict:
        trial, _ = self.control.annotation_task(run_id, benchmark, task_id)
        report = trial / "report.md"
        if not report.is_file() or not report.read_text().strip():
            raise APIError(409, "任务结束并产出报告后才能启动 AI Review")
        with self.control.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM ai_reviews WHERE run_id=? AND benchmark=? AND task_id=?",
                             (run_id, benchmark, task_id)).fetchone()
            if row and row["status"] in {"queued", "running", "completed"}:
                return self.public(dict(row))
            if row and not retry:
                raise APIError(409, "上次 AI Review 未完成，请点击重试")
            if not ReviewConfig.from_env().api_key:
                raise APIError(503, "服务器未配置 LITELLM_OPENAI_API_KEY")
            if db.execute("SELECT count(*) FROM ai_reviews WHERE status IN ('queued','running')").fetchone()[0] >= 32:
                raise APIError(429, "AI Review 队列已满，请稍后再试")
            if row:
                db.execute("UPDATE ai_reviews SET status='queued',pid=NULL,attempts=attempts+1,"
                           "requested_by=?,error=NULL,started_at=NULL,finished_at=NULL WHERE id=?",
                           (user["id"], row["id"]))
                job_id = row["id"]
            else:
                cursor = db.execute("INSERT INTO ai_reviews(run_id,benchmark,task_id,requested_by,status,created_at) "
                                    "VALUES (?,?,?,?,'queued',?)", (run_id, benchmark, task_id, user["id"], now()))
                job_id = cursor.lastrowid
                assert job_id is not None
        return self.public(self.row(job_id))

    def tick(self) -> None:
        # Only one paid Review runs at once, independently of the Agent queue.
        self.children = [p for p in self.children if p.poll() is None]
        with self.control.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT * FROM ai_reviews WHERE status='running'").fetchall()
            for row in active:
                marker = f"tools.case_study_studio.review_job --job-id {row['id']} "
                if alive(row["pid"], marker):
                    return
                # No automatic paid retry after a crash. Completed batches remain reusable.
                db.execute("UPDATE ai_reviews SET status='interrupted',finished_at=?,error=? WHERE id=?",
                           (now(), "Review 进程已退出，可点击重试继续已完成的批次", row["id"]))
            row = db.execute("SELECT * FROM ai_reviews WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
            if not row:
                return
            output = self.output(row["id"])
            output.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                with (output / "worker.log").open("ab") as log:
                    process = subprocess.Popen([str(self.control.root / ".venv/bin/python"), "-m",
                        "tools.case_study_studio.review_job", "--job-id", str(row["id"]),
                        "--root", str(self.control.root), "--state", str(self.control.state)],
                        cwd=self.control.root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.children.append(process)
            except OSError:
                db.execute("UPDATE ai_reviews SET status='failed',finished_at=?,error=? WHERE id=?",
                           (now(), "无法启动 Review 后台进程，请检查服务器 Python 环境", row["id"]))
                return
            db.execute("UPDATE ai_reviews SET status='running',pid=?,started_at=? WHERE id=?",
                       (process.pid, now(), row["id"]))
