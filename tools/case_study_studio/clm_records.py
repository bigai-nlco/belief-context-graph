"""Read-only access to explicitly published CLM recordings, never arbitrary paths."""
from __future__ import annotations

import json
from pathlib import Path
import re

from tools.case_study_studio.control import APIError, safe_child, read_json

TURN_FILES = (
    "input.json", "output.json", "harness.messages.json", "context.before.txt",
    "context.after.txt", "context.diff", "context.parsed.json", "step.json",
)
SETTINGS = (
    "model_name", "temperature", "top_p", "max_tokens", "max_steps", "lm_call_cap",
    "context_budget_tokens", "context_budget_reserve_tokens", "allow_edit_growth",
    "enable_thinking", "nudge_ratios", "persistent_nudge_ratio", "max_num_retry_on_limit",
    "retry_edit_margin_tokens", "observation_max_chars", "persistent_bash",
)


class CLMRecords:
    def __init__(self, roots: list[Path], control=None):
        self.control = control
        self.roots: dict[str, Path] = {}
        for path in roots:
            root = path.expanduser().resolve()
            if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,160}", root.name)
                    or not (root / "summary.json").is_file()):
                raise ValueError("CLM recording must be an existing run with summary.json")
            if root.name in self.roots:
                raise ValueError("CLM recording names must be unique")
            self.roots[root.name] = root

    def _managed(self) -> dict:
        if self.control is None:
            return {}
        with self.control.connect() as db:
            rows = db.execute("SELECT * FROM runs WHERE managed=1 ORDER BY id DESC").fetchall()
        found = {}
        for raw in rows:
            row = dict(raw)
            config = json.loads(row["config"])
            if config.get("mode") != "clm":
                continue
            root = self.control.directory(row)
            for task_id in config.get("tasks", {}).get("drb2", []):
                if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", task_id):
                    continue
                directory = safe_child(root, "drb2", "trials", task_id, "clm")
                if (directory / "summary.json").is_file():
                    found[f"{self.control.label(row['id'])}-task-{task_id}"] = (directory, row)
        return found

    def _root(self, run: str) -> Path:
        managed = self._managed()
        if run in managed:
            return managed[run][0]
        if run not in self.roots:
            raise APIError(404, "找不到这份 CLM 记录")
        return self.roots[run]

    @staticmethod
    def _text(root: Path, *parts: str) -> str | None:
        path = root.joinpath(*parts).resolve()
        if not path.is_relative_to(root):
            raise APIError(404, "记录文件不可用")
        if not path.is_file():
            return None
        if path.stat().st_size > 8 * 1024 * 1024:
            raise APIError(413, "单个记录文件超过 8 MiB")
        # Keep raw newlines, including CRLF, for display/download fidelity.
        return path.read_bytes().decode("utf-8")

    def _json(self, root: Path, *parts: str):
        text = self._text(root, *parts)
        return json.loads(text) if text else {}

    def summary(self, run: str) -> dict:
        managed = self._managed()
        if run in managed:
            root, row = managed[run]
        elif run in self.roots:
            root, row = self.roots[run], None
        else:
            raise APIError(404, "找不到这份 CLM 记录")
        return self._summary(run, root, row)

    def _summary(self, run: str, root: Path, row: dict | None) -> dict:
        summary = self._json(root, "summary.json")
        manifest = self._json(root, "manifest.json")
        settings = manifest.get("settings", {})
        if row is not None:
            result = read_json(root.parent / "result.json")
            if result.get("status"):
                summary["status"] = result["status"]
            elif row["status"] not in {"running", "starting", "queued"}:
                summary["status"] = "interrupted"
            observed = self.control._read_progress(root.parent / "agent/trajectories/react_agent.jsonl", "trace")
            if observed.get("calls"):
                summary["usage"] = {"prompt_tokens": observed["input"], "completion_tokens": observed["output"],
                                    "total_tokens": observed["input"] + observed["output"]}
        return {"id": run, "task_id": summary.get("task_id", "drb2:2"),
                "status": summary.get("status", "unknown"), "started_at": summary.get("started_at"),
                "wall_seconds": summary.get("wall_seconds"), "usage": summary.get("usage", {}),
                "model_calls": summary.get("model_calls", 0),
                "accepted_edits": summary.get("accepted_edits", 0),
                "substantial_edits": summary.get("substantial_edits", 0),
                "settings": {k: settings[k] for k in SETTINGS if k in settings},
                "clm_commit": manifest.get("clm_commit"), "tools": summary.get("tools", {}),
                "reasoning_effort": manifest.get("reasoning_effort"),
                "limits": manifest.get("limits", {"search_calls": 50, "fetch_calls": 50}),
                "managed": row is not None}

    def runs(self) -> list[dict]:
        managed = self._managed()
        return [self._summary(run, root, row) for run, (root, row) in managed.items()] + [
            self._summary(run, root, None) for run, root in self.roots.items() if run not in managed]

    def detail(self, run: str) -> dict:
        root = self._root(run)
        records = []
        for path in sorted((root / "turns").glob("[0-9][0-9][0-9][0-9]")):
            if not path.is_dir() or int(path.name) < 1:
                continue
            step = self._json(root, "turns", path.name, "step.json")
            response = self._json(root, "turns", path.name, "output.json")
            records.append({"number": int(path.name), "edited": bool(step.get("context_edit_applied")),
                            "file_changed": bool(step.get("file_changed")),
                            "submitted": bool(step.get("submitted")),
                            "command": str(step.get("command", ""))[:160],
                            "usage": response.get("usage", {}),
                            "has_output": bool(response)})
        return {**self.summary(run), "turns": records}

    def turn(self, run: str, number: str) -> dict:
        if not re.fullmatch(r"[0-9]{1,4}", number) or int(number) < 1:
            raise APIError(400, "轮次无效")
        root = self._root(run)
        name = f"{int(number):04d}"
        files = {file: self._text(root, "turns", name, file) for file in TURN_FILES}
        if files["input.json"] is None:
            raise APIError(404, "这一轮没有记录")
        return {"number": int(number), "files": files}

    def report(self, run: str) -> dict:
        root = self._root(run)
        return {"task": self._text(root, "task.txt"),
                "report": self._text(root, "report.md") if run in self._managed() else self._text(root, "workspace", "report.md")}
