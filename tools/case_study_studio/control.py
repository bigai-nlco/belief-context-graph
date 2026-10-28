"""Shared run catalog, bounded job queue and independent reviewer storage.

Only server-selected result trees and launch profiles are addressable. Browser
requests cannot submit commands, credential values, or filesystem paths.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The shared Studio exposes DeepResearch Bench II only. Keep archived files and
# database rows for other benchmarks intact; this is a catalog/launch filter.
DEFAULT_IDS = {"drb2": [
    "2", "5", "8", "9", "23", "27", "30", "37", "38", "49", "52", "58", "63", "64",
    "76", "77", "81", "87", "88", "93", "96", "98", "100", "101", "103", "105", "108",
    "110", "119", "120", "121", "128",
]}
MAX_TASKS = 32
TERMINAL = {"completed", "timeout", "agent_error", "worker_error", "empty_report",
            "score_error", "graph_fallback", "interrupted", "cancelled"}


class APIError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status, self.message = status, message
        super().__init__(message)


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def safe_child(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise APIError(400, "无效的结果路径")
    return path


def alive(pid: int | None, marker: str) -> bool:
    if not pid:
        return False
    try:
        command = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return marker in command
    except (OSError, ValueError):
        return False


class Control:
    def __init__(self, root: Path = ROOT, state: Path | None = None, *, schedule: bool = False) -> None:
        self.root = root.resolve()
        self.results = self.root / "results"
        self.state = (state or self.root / "runtime/case-study-studio").resolve()
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.database = self.state / "studio.sqlite3"
        self.lock = threading.RLock()
        self.file_cache: dict = {}
        self.closed = threading.Event()
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                    token_hash TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT UNIQUE,
                    title TEXT NOT NULL, owner TEXT, created_at TEXT NOT NULL,
                    config TEXT NOT NULL DEFAULT '{}', managed INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL, pid INTEGER, cancel INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT, finished_at TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS reviews (
                    run_id INTEGER NOT NULL REFERENCES runs(id), benchmark TEXT NOT NULL,
                    task_id TEXT NOT NULL, user_id TEXT NOT NULL REFERENCES users(id),
                    revision INTEGER NOT NULL, annotation TEXT NOT NULL,
                    evidence_sha TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, benchmark, task_id, user_id));
            """)
        self.database.chmod(0o600)
        from tools.case_study_studio.review_queue import AIReviews
        self.ai_reviews = AIReviews(self)
        self.scan()
        if schedule:
            threading.Thread(target=self._schedule, daemon=True, name="studio-queue").start()

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @staticmethod
    def label(run_id: int) -> str:
        return f"RUN-{run_id:04d}"

    def register(self, name: object) -> dict:
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 40 or any(ord(c) < 32 for c in name):
            raise APIError(400, "昵称需要 1–40 个字符")
        token = secrets.token_urlsafe(32)
        user = {"id": secrets.token_hex(12), "name": name.strip()}
        try:
            with self.connect() as db:
                db.execute("INSERT INTO users VALUES (?,?,?,?)", (
                    user["id"], user["name"], hashlib.sha256(token.encode()).hexdigest(), now()))
        except sqlite3.IntegrityError:
            raise APIError(409, "这个昵称已经注册，请使用个人身份码恢复，或换一个昵称") from None
        return {"user": user, "token": token}

    def authenticate(self, token: str) -> dict:
        if not isinstance(token, str) or not 20 <= len(token) <= 128:
            raise APIError(401, "请先填写标注昵称，或使用个人身份码登录")
        with self.connect() as db:
            row = db.execute("SELECT id,name FROM users WHERE token_hash=?",
                             (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        if not row:
            raise APIError(401, "个人身份码无效，请重新登录")
        return dict(row)

    def profiles(self) -> dict:
        return {
            "qwen35-9b": {"label": "Qwen3.5-9B · 本机 vLLM · Thinking 开启",
                "agent": "Qwen3.5-9B", "max_workers": 2,
                "source": "tools/case_study_studio/profiles/qwen35-9b.yaml"},
            "gpt6-luna": {"label": "GPT-6-Luna · LiteLLM · low",
                "agent": "gpt-6-luna", "max_workers": 6,
                "source": "tools/case_study_studio/profiles/gpt6-luna.yaml"},
        }

    def settings(self) -> dict:
        return {"shared": True, "profiles": [
            {"id": key, **{k: v for k, v in value.items() if k != "source"}}
            for key, value in self.profiles().items()
            if safe_child(self.root, value["source"]).is_file()],
            "default_tasks": DEFAULT_IDS, "graph_model": "gpt-6-luna", "graph_effort": "none",
            "max_tasks": MAX_TASKS, "poll_seconds": 3, "launch_requires_admin": False}

    def scan(self) -> None:
        candidates = []
        for path in self.results.glob("*/batch.json"):
            if path.is_symlink() or not path.resolve().is_relative_to(self.results.resolve()):
                continue
            data = read_json(path)
            if not any((path.parent / name / "run.json").is_file() for name in DEFAULT_IDS):
                continue
            candidates.append((str(data.get("started_at") or ""), path.parent))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for created, folder in sorted(candidates):
                source = str(folder.relative_to(self.results))
                # INSERT OR IGNORE advances SQLite AUTOINCREMENT even on duplicates.
                # Polling must not consume run numbers.
                if db.execute("SELECT 1 FROM runs WHERE source=?", (source,)).fetchone():
                    continue
                data = read_json(folder / "batch.json")
                model = (data.get("agent") or {}).get("model") or "历史实验"
                title = f"{model} · Deep Research · {created[:10]}"
                db.execute("INSERT INTO runs(source,title,created_at,status) VALUES (?,?,?,?)",
                    (source, title, created or now(), "external"))

    def row(self, run_id: int) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise APIError(404, "找不到这个运行编号")
        return dict(row)

    def directory(self, row: dict) -> Path:
        return safe_child(self.results, row["source"])

    def _read_progress(self, path: Path, kind: str) -> dict:
        """Incremental JSONL counters; incomplete final lines are read next poll."""
        with self.lock:
            try:
                stat = path.stat()
            except OSError:
                return {}
            key = str(path)
            item = self.file_cache.get(key)
            if not item or item["inode"] != stat.st_ino or stat.st_size < item["offset"]:
                item = {"inode": stat.st_ino, "offset": 0, "calls": 0, "turn": 0,
                        "input": 0, "output": 0, "searches": 0, "fetches": 0, "blocked": 0}
            with path.open("rb") as handle:
                handle.seek(item["offset"])
                while True:
                    line = handle.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    item["offset"] = handle.tell()
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if kind == "trace" and event.get("t") == "llm":
                        item["calls"] += 1
                        item["turn"] = event.get("turn", 0)
                        usage = event.get("usage") or {}
                        item["input"] += usage.get("prompt_tokens", 0) or 0
                        item["output"] += usage.get("completion_tokens", 0) or 0
                        item["updated"] = event.get("ts")
                    elif kind == "tools":
                        if event.get("tool") == "web_search":
                            queries = (event.get("args") or {}).get("q")
                            item["searches"] += len(queries) if isinstance(queries, list) else 1
                        item["fetches"] += event.get("tool") == "web_fetch"
                        item["blocked"] += str(event.get("result", "")).startswith("[BLOCKED]")
                    elif kind == "graph":
                        item["graph_usage"] = (event.get("graph_usage_cumulative") or {}).get("llm_totals", {})
                        item["graph_chars"] = event.get("graph_chars", 0)
            self.file_cache[key] = item
            return dict(item)

    def detail(self, run_id: int, user: dict | None = None) -> dict:
        row = self.row(run_id)
        root = self.directory(row)
        batch = read_json(root / "batch.json")
        config = json.loads(row["config"])
        ai_reviews = self.ai_reviews.statuses(run_id)
        status = row["status"] if row["managed"] else batch.get("status", "unknown")
        if not row["managed"] and status in {"running", "starting"} and not alive(batch.get("runner_pid"), str(root)):
            status = "interrupted"
        with self.connect() as db:
            reviews = [dict(x) for x in db.execute(
                "SELECT r.benchmark,r.task_id,r.user_id,r.revision,r.annotation,r.updated_at,u.name "
                "FROM reviews r JOIN users u ON u.id=r.user_id WHERE r.run_id=?", (run_id,))]
        tasks = []
        for benchmark in DEFAULT_IDS:
            run = read_json(root / benchmark / "run.json")
            ids = run.get("task_ids", config.get("tasks", {}).get(benchmark, []))
            for task_id in ids:
                if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", task_id):
                    continue
                trial = safe_child(root, benchmark, "trials", task_id)
                result = read_json(trial / "result.json")
                trace = self._read_progress(trial / "agent/trajectories/react_agent.jsonl", "trace")
                tool = self._read_progress(trial / "tools.jsonl", "tools")
                graph = self._read_progress(trial / "agent/bcg/requests.jsonl", "graph")
                task_status = result.get("status") or ("running" if trace else "queued")
                if not result and status not in {"running", "starting", "queued"}:
                    task_status = "interrupted" if trace else "not_run"
                usage = result.get("usage") or {}
                agent_usage = usage.get("agent") or {
                    "input_tokens": trace.get("input", 0), "output_tokens": trace.get("output", 0),
                    "total_tokens": trace.get("input", 0) + trace.get("output", 0)}
                graph_usage = usage.get("graph") or graph.get("graph_usage", {})
                task_reviews = [{"name": x["name"], "user_id": x["user_id"],
                    "status": json.loads(x["annotation"]).get("status", "draft"),
                    "mine": bool(user and x["user_id"] == user["id"]), "updated_at": x["updated_at"]}
                    for x in reviews if x["benchmark"] == benchmark and x["task_id"] == task_id]
                tasks.append({"benchmark": benchmark, "id": task_id, "status": task_status,
                    "calls": trace.get("calls", usage.get("agent_model_calls", 0)),
                    "turn": trace.get("turn", 0), "updated": trace.get("updated"),
                    "searches": tool.get("searches", 0), "fetches": tool.get("fetches", 0),
                    "blocked": tool.get("blocked", 0), "duration": result.get("duration_seconds"),
                    "agent_tokens": agent_usage.get("total_tokens"),
                    "graph_tokens": graph_usage.get("total_tokens"),
                    "agent_input": agent_usage.get("input_tokens"), "agent_output": agent_usage.get("output_tokens"),
                    "report": (trial / "report.md").is_file(), "reviews": task_reviews,
                    "ai_review": ai_reviews.get((benchmark, task_id), {"status": "not_started"}),
                    "annotatable": task_status == "completed"})
        ended = sum(t["status"] in TERMINAL for t in tasks)
        return {"id": run_id, "number": self.label(run_id), "title": row["title"], "status": status,
            "managed": bool(row["managed"]), "created_at": row["created_at"],
            "started_at": row["started_at"] or batch.get("started_at"),
            "finished_at": row["finished_at"] or batch.get("finished_at"),
            "agent": batch.get("agent", {}).get("model") or config.get("agent_model", ""),
            "graph": batch.get("graph", {}).get("model") or "gpt-6-luna",
            "workers": batch.get("workers") or config.get("workers"),
            "owner": row["owner"], "error": row["error"], "cancel_requested": bool(row["cancel"]),
            "tasks": tasks, "total": len(tasks), "ended": ended,
            "reports": sum(t["report"] for t in tasks),
            "failed": sum(t["status"] in TERMINAL and t["status"] != "completed" for t in tasks),
            "annotation_count": sum(len(t["reviews"]) for t in tasks)}

    def list_runs(self, user: dict) -> list[dict]:
        self.scan()
        with self.connect() as db:
            ids = [x[0] for x in db.execute("SELECT id FROM runs ORDER BY id DESC")]
        # Previously registered DRBench-only runs keep their IDs and annotations
        # in the database, but no longer appear as empty runs in the Studio.
        details = (self.detail(i, user) for i in ids)
        return [{k: v for k, v in item.items() if k != "tasks"}
                for item in details if item["tasks"]]

    def task(self, run_id: int, benchmark: str, task_id: str) -> tuple[Path, dict]:
        item = next((x for x in self.detail(run_id)["tasks"]
                     if x["benchmark"] == benchmark and x["id"] == task_id), None)
        if not item:
            raise APIError(404, "该运行中没有这个任务")
        return safe_child(self.directory(self.row(run_id)), benchmark, "trials", task_id), item

    def annotation_task(self, run_id: int, benchmark: str, task_id: str) -> tuple[Path, dict]:
        trial, task = self.task(run_id, benchmark, task_id)
        if not task["annotatable"]:
            raise APIError(409, "仅正常完成的任务可以标注")
        return trial, task

    def evidence(self, run_id: int, benchmark: str, task_id: str) -> tuple[Path, str]:
        trial, _ = self.annotation_task(run_id, benchmark, task_id)
        # Freeze one evidence revision for all reviewers. Never rewrite cited records.
        target = safe_child(self.state, "evidence", str(run_id), benchmark, task_id + ".json.gz")
        with self.lock:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                original = trial / "case-study.json.gz"
                if original.is_file() and not original.is_symlink():
                    # Generated bundles are already bounded, structured and redacted.
                    from workflows._shared.case_study import read_bundle
                    read_bundle(original)
                    temporary = target.with_suffix(".tmp")
                    temporary.write_bytes(original.read_bytes())
                    temporary.chmod(0o600)
                    temporary.replace(target)
                else:
                    from workflows._shared.case_study import export_case
                    root = self.directory(self.row(run_id))
                    source = read_json(root / benchmark / "_cases" / (task_id + ".json"))
                    export_case(trial, output=target, metadata={"question": source.get("question", "")},
                                graph_dir=root / "graphs")
            digest_path = target.with_suffix(".sha256")
            if not digest_path.exists():
                digest_path.write_text(hashlib.sha256(target.read_bytes()).hexdigest())
            digest = digest_path.read_text().strip()
        return target, digest

    def evidence_preview(self, run_id: int, benchmark: str, task_id: str) -> tuple[Path, str]:
        from tools.case_study_studio.evidence_preview import preview_bundle
        source, digest = self.evidence(run_id, benchmark, task_id)
        return preview_bundle(source, digest), digest

    def review(self, run_id: int, benchmark: str, task_id: str, user: dict) -> dict:
        self.annotation_task(run_id, benchmark, task_id)
        with self.connect() as db:
            row = db.execute("SELECT revision,annotation,evidence_sha,updated_at FROM reviews "
                "WHERE run_id=? AND benchmark=? AND task_id=? AND user_id=?",
                (run_id, benchmark, task_id, user["id"])).fetchone()
        return ({**dict(row), "annotation": json.loads(row["annotation"])} if row else
                {"revision": 0, "annotation": None})

    def save_review(self, run_id: int, benchmark: str, task_id: str, user: dict, data: dict) -> dict:
        annotation = data.get("annotation")
        revision = data.get("revision")
        if (not isinstance(annotation, dict) or type(revision) is not int or revision < 0
                or annotation.get("status") not in {"draft", "reviewed"}
                or not isinstance(annotation.get("events"), list)
                or not isinstance(annotation.get("dimensions"), dict)
                or not isinstance(annotation.get("conclusion"), dict)):
            raise APIError(400, "标注数据格式不正确")
        _, digest = self.evidence(run_id, benchmark, task_id)
        if data.get("evidence_sha") != digest:
            raise APIError(409, "证据版本不同，请重新打开该 Case")
        annotation = {**annotation, "reviewer": user["name"]}
        text = json.dumps(annotation, ensure_ascii=False)
        if len(text.encode()) > 1024 * 1024:
            raise APIError(413, "单份标注不能超过 1 MiB")
        stamp = now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT revision FROM reviews WHERE run_id=? AND benchmark=? AND task_id=? AND user_id=?",
                             (run_id, benchmark, task_id, user["id"])).fetchone()
            if (old["revision"] if old else 0) != revision:
                raise APIError(409, "另一页面已更新这份标注。请先导出当前草稿备份，再刷新加载服务器版本")
            db.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(run_id,benchmark,task_id,user_id) "
                       "DO UPDATE SET revision=excluded.revision,annotation=excluded.annotation,evidence_sha=excluded.evidence_sha,updated_at=excluded.updated_at",
                       (run_id, benchmark, task_id, user["id"], revision + 1, text, digest, stamp))
        return {"revision": revision + 1, "updated_at": stamp}

    def create_run(self, data: dict, user: dict) -> dict:
        profile_id = data.get("profile")
        profile = self.profiles().get(profile_id) if isinstance(profile_id, str) else None
        if not profile or not safe_child(self.root, profile["source"]).is_file():
            raise APIError(400, "请选择服务器已配置的 Agent 模型")
        title = data.get("title", "")
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 100:
            raise APIError(400, "填写 1–100 字的测试名称")
        workers = data.get("workers", 2)
        if type(workers) is not int or not 1 <= workers <= profile["max_workers"]:
            raise APIError(400, f"此模型的并发范围为 1–{profile['max_workers']}")
        tasks = data.get("tasks")
        if not isinstance(tasks, dict) or not tasks or set(tasks) - set(DEFAULT_IDS):
            raise APIError(400, "仅支持 DeepResearch Bench II，请填写该 Benchmark 的 task ID")
        for ids in tasks.values():
            if (not isinstance(ids, list) or not ids
                    or any(not isinstance(i, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", i) for i in ids)
                    or len(ids) != len(set(ids))):
                raise APIError(400, "task ID 格式错误或包含重复项")
        if sum(map(len, tasks.values())) > MAX_TASKS:
            raise APIError(400, f"单次最多启动 {MAX_TASKS} 个任务")
        config = {"profile": data["profile"], "agent_model": profile["agent"],
                  "tasks": tasks, "workers": workers, "pipeline": "stateful-react-agent"}
        for key, default, minimum, maximum in (("timeout", 3600, 60, 14400),
                ("max_turns", 200, 1, 300), ("search_calls", 0, 0, 2000), ("fetch_calls", 0, 0, 2000)):
            value = data.get(key, default)
            if type(value) is not int or not minimum <= value <= maximum:
                raise APIError(400, f"{key} 的范围为 {minimum}–{maximum}")
            config[key] = value
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT count(*) FROM runs WHERE managed=1 AND status IN ('queued','starting','running')").fetchone()[0] >= 5:
                raise APIError(409, "最多保留 5 个待运行或运行中的批次")
            cursor = db.execute("INSERT INTO runs(title,owner,created_at,config,managed,status) VALUES (?,?,?,?,1,'queued')",
                                (title.strip(), user["id"], now(), json.dumps(config)))
            run_id = cursor.lastrowid
            assert run_id is not None
            db.execute("UPDATE runs SET source=? WHERE id=?", (f"studio/{self.label(run_id)}", run_id))
        return self.detail(run_id, user)

    def cancel_run(self, run_id: int, user: dict) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise APIError(404, "找不到这个运行编号")
            if not row["managed"] or row["status"] not in {"queued", "starting", "running"}:
                raise APIError(409, "这次运行已结束，或不是由页面启动")
            db.execute("UPDATE runs SET cancel=1,status=?,finished_at=? WHERE id=?",
                       ("cancelled" if row["status"] == "queued" else row["status"],
                        now() if row["status"] == "queued" else None, run_id))
        return self.detail(run_id, user)

    def _schedule(self) -> None:
        while not self.closed.wait(2):
            for tick in (self.tick, self.ai_reviews.tick):
                try:
                    tick()
                except Exception:
                    # Do not expose exception strings containing configuration or secrets.
                    import logging
                    logging.getLogger(__name__).error("Studio queue tick failed; retrying")

    def tick(self) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT * FROM runs WHERE managed=1 AND status IN ('starting','running')").fetchall()
            for row in active:
                if alive(row["pid"], "case_study_studio/run_job.py"):
                    return
                db.execute("UPDATE runs SET status='interrupted',finished_at=?,error=? WHERE id=?",
                           (now(), "运行进程已退出；保留已有记录，未自动重跑", row["id"]))
            row = db.execute("SELECT * FROM runs WHERE managed=1 AND status='queued' ORDER BY id LIMIT 1").fetchone()
            if not row:
                return
            root = self.directory(dict(row))
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            with (root / "launcher.log").open("ab") as stream:
                proc = subprocess.Popen([str(self.root / ".venv/bin/python"),
                    str(self.root / "tools/case_study_studio/run_job.py"), "--root", str(self.root),
                    "--state", str(self.state), "--run-id", str(row["id"])], cwd=self.root,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            db.execute("UPDATE runs SET status='starting',pid=?,started_at=? WHERE id=?", (proc.pid, now(), row["id"]))
