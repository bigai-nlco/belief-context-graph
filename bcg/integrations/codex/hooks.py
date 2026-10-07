"""Non-blocking-by-design native hooks: local durable capture, bounded recall."""

from __future__ import annotations

import os
import re
import time

from bcg.core.env import read_env_file

from .config import Config
from .events import normalize_event
from .memory import context
from .store import Store


def handle(raw: dict, config: Config) -> dict:
    if (
        not config.enabled
        or not isinstance(raw, dict)
        or not raw.get("cwd")
        or not config.contains(str(raw["cwd"]))
    ):
        return {}
    session = raw.get("session_id", "")
    if not isinstance(session, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,160}", session
    ):
        raise ValueError("invalid session_id")
    environment = {**read_env_file(config.env_file), **os.environ}
    secrets = [
        value
        for name, value in environment.items()
        if re.search("KEY|TOKEN|SECRET|PASSWORD", name, re.I)
    ]
    store = Store(config.state_dir)
    event = normalize_event(raw, config, secrets)
    if event is not None:
        store.enqueue(event, config.max_events_per_session)
    kind = raw.get("hook_event_name")
    if kind == "PreCompact" and config.mode == "compact":
        # A bounded drain, not a compaction veto. MCP worker owns all remote
        # requests; this hook never starts an agent or contacts a provider.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = store.status(session)
            counts = status["events"]
            if not (counts.get("pending") or counts.get("processing")):
                break
            if (
                status["worker_last_seen_seconds"] is None
                or status["worker_last_seen_seconds"] > 60
            ):
                break
            time.sleep(0.2)
        return {}
    if kind not in {"SessionStart", "UserPromptSubmit", "PostToolUse"}:
        return {}
    if kind == "PostToolUse":
        if event is None:
            return {}
        marker = f"PostToolUse:{event['turn']}"
        with store.db() as db:
            seen = db.execute(
                "SELECT count(*) FROM injections WHERE session=? AND event=?",
                (session, marker),
            ).fetchone()[0]
            newest = db.execute(
                "SELECT max(updated) FROM graphs WHERE session=?", (session,)
            ).fetchone()[0]
            last = db.execute(
                "SELECT max(created) FROM injections WHERE session=?", (session,)
            ).fetchone()[0]
        # No repeated stale snapshots, at most two during a native user turn.
        if seen >= 2 or not newest or (last and newest <= last):
            return {}
    else:
        marker = str(kind)
    query = store.latest_question(session)
    text = context(store, session, query, config.context_chars)
    store.record_injection(session, marker, text)
    return {"hookSpecificOutput": {"hookEventName": kind, "additionalContext": text}}
