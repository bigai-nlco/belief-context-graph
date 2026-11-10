"""Delete one idle run and its private artifacts, never a live worker or shared service."""
from __future__ import annotations

import logging
import secrets
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from tools.case_study_studio.control import APIError, alive, read_json, safe_child

if TYPE_CHECKING:
    from tools.case_study_studio.control import Control

ACTIVE = {"queued", "starting", "running"}


def blocked_reason(status: str, ai_statuses: list[str], summary_statuses: list[str]) -> str:
    if status in ACTIVE:
        return "请先停止测试，等待运行结束后再删除"
    if any(status in ACTIVE for status in ai_statuses):
        return "该运行还有 AI 标注正在执行或排队，请等待完成后再删除"
    if any(status in ACTIVE for status in summary_statuses):
        return "该运行还有 AI 总结正在执行或排队，请等待完成后再删除"
    return ""


def _directory(root: Path, *parts: str) -> Path:
    """Only exact owned directories; do not follow symlinks or accept a root itself."""
    relative = Path(*parts)
    if relative.is_absolute() or not relative.parts or any(p in {".", ".."} for p in relative.parts):
        raise APIError(409, "运行目录不安全，未执行删除")
    path = root
    for part in relative.parts:
        if path.is_symlink():
            raise APIError(409, "运行目录包含软链接，未执行删除")
        path /= part
    if path.is_symlink():
        raise APIError(409, "运行目录包含软链接，未执行删除")
    safe_child(root, *parts)
    if path.exists() and not path.is_dir():
        raise APIError(409, "运行目录格式异常，未执行删除")
    return path


def delete_run(control: Control, run_id: int, user: dict, confirmation: object) -> dict:
    number = control.label(run_id)
    if confirmation != number:
        raise APIError(400, f"请输入 {number} 确认删除")
    staged: list[tuple[Path, Path]] = []
    # Serialize against evidence creation and scheduler/queue writes. Check again
    # here: the page's last poll is not authority to delete a newly active job.
    with control.lock:
        try:
            with control.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if not db.execute("SELECT 1 FROM users WHERE id=?", (user.get("id"),)).fetchone():
                    raise APIError(401, "请先登录后再删除运行")
                row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
                if row is None:
                    raise APIError(404, "找不到这个运行编号")
                root = _directory(control.results, row["source"])
                if root == control.results / "studio":
                    raise APIError(409, "不能删除运行的公共目录")
                # Catalogs may contain legacy or nested paths. Never erase a
                # second run when removing an ancestor or an alias.
                for other in db.execute("SELECT source FROM runs WHERE id!=?", (run_id,)):
                    other_root = safe_child(control.results, other["source"])
                    if root.is_relative_to(other_root) or other_root.is_relative_to(root):
                        raise APIError(409, "该目录与其他运行共用，未执行删除")
                batch = read_json(root / "batch.json")
                status = row["status"] if row["managed"] else batch.get("status", "unknown")
                worker_alive = (alive(row["pid"], "case_study_studio/run_job.py")
                                or alive(batch.get("runner_pid"), str(root)))
                if not row["managed"] and status in {"starting", "running"} and not worker_alive:
                    status = "interrupted"
                reviews = db.execute("SELECT * FROM ai_reviews WHERE run_id=?", (run_id,)).fetchall()
                summaries = db.execute("SELECT * FROM run_summaries WHERE run_id=?", (run_id,)).fetchall()
                reason = blocked_reason(status, [r["status"] for r in reviews], [r["status"] for r in summaries])
                if reason:
                    raise APIError(409, reason)
                if worker_alive or any(
                    alive(job["pid"], f"tools.case_study_studio.{module} --job-id {job['id']} ")
                    for jobs, module in ((reviews, "review_job"), (summaries, "run_summary_job"))
                    for job in jobs
                ):
                    raise APIError(409, "后台进程仍在退出，请稍后再删除")
                paths = [root, _directory(control.state, "evidence", str(run_id))]
                paths.extend(_directory(control.state, "ai-reviews", str(job["id"])) for job in reviews)
                paths.extend(_directory(control.state, "run-summaries", str(job["id"])) for job in summaries)
                for path in paths:
                    if path.exists():
                        temporary = path.with_name(f".delete-{number}-{secrets.token_hex(8)}-{path.name}")
                        path.rename(temporary)
                        staged.append((path, temporary))
                for table in ("reviews", "ai_reviews", "run_summaries"):
                    db.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,))
                db.execute("DELETE FROM runs WHERE id=?", (run_id,))
        except Exception:
            # Failed validation/rename/DB commit must leave the run usable.
            for original, temporary in reversed(staged):
                temporary.rename(original)
            raise
        for key in list(control.file_cache):
            if any(key == str(path) or key.startswith(str(path) + "/") for path in paths):
                del control.file_cache[key]
    # Reclaim potentially large model-I/O files after releasing the DB lock.
    # Hidden staging directories are never rediscovered as historical runs.
    cleanup_pending = False
    for _, temporary in staged:
        try:
            shutil.rmtree(temporary)
        except OSError:
            cleanup_pending = True
            logging.getLogger(__name__).warning("%s deleted; artifact cleanup incomplete", number)
    return {"id": run_id, "number": number, "deleted": True, "cleanup_pending": cleanup_pending}
