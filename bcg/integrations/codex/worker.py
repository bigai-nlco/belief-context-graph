"""One local graph worker per project; provider calls never delay native tools."""

from __future__ import annotations

import fcntl
import hashlib
import os
import threading
import uuid
from pathlib import Path

from .config import Config
from .events import graph_turns
from .store import Store


class Runtime:
    def __init__(self, config: Config):
        # Imports and embedding loads are lazy so MCP initialize/tools/list
        # stay responsive even while a model is loading.
        os.environ["BCG_ENV_FILE"] = config.env_file
        from bcg.core.env import load_project_env

        load_project_env(config.env_file, override=True)
        from bcg.construct.unified.online import SessionManager
        from bcg.construct.unified.stream import StreamOptions

        self.manager = SessionManager(
            config_path=config.graph_config,
            model_key="graph-model",
            output_root=Path(config.state_dir) / "graphs",
            options=StreamOptions(
                context_chars=12000, extraction_history_turns=2, max_previous_windows=3
            ),
        )
        self.segments: dict[str, str] = {}
        self.last_event: dict[str, int] = {}

    def apply(self, event: dict) -> dict:
        session = event["session"]
        if event["id"] <= self.last_event.get(session, -1):
            self.abandon(session)
        graph_id = self.segments.setdefault(
            session,
            f"codex_{hashlib.sha256(session.encode()).hexdigest()[:12]}_{uuid.uuid4().hex[:12]}",
        )
        turns = [
            {
                **turn,
                "problem_id": graph_id,
                "is_message_end": True,
                "is_trajectory_end": False,
            }
            for turn in graph_turns(event)
        ]
        snapshot = self.manager.push_many(turns)["latest"][graph_id]
        self.last_event[session] = event["id"]
        return snapshot

    def abandon(self, session: str) -> None:
        # No in-place retry after a possibly partial engine mutation. Published
        # old segments remain readable; new work uses a distinct graph ID.
        graph_id = self.segments.pop(session, None)
        if graph_id:
            self.manager._sessions.pop(graph_id, None)


def work(config_path: Path, stop: threading.Event, runtime_factory=Runtime) -> None:
    config = Config.load(config_path)
    store = Store(config.state_dir)
    lock_fd = os.open(store.root / "worker.lock", os.O_CREAT | os.O_WRONLY, 0o600)
    locked = False
    runtime = None
    try:
        while not stop.is_set():
            if not locked:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stop.wait(0.5)
                    continue
                locked = True
                store.recover()
            store.heartbeat()
            config = Config.load(config_path)
            if not config.enabled:
                stop.wait(0.5)
                continue
            event = store.next_event()
            if event is None:
                stop.wait(0.3)
                continue
            try:
                runtime = runtime or runtime_factory(config)
                snapshot = runtime.apply(event)
                store.complete(event, snapshot)
            except Exception as exc:
                # Exception messages from API clients can include keys, URLs,
                # or raw provider bodies. Only a class name reaches users/logs.
                store.fail(event["id"], type(exc).__name__)
                if runtime:
                    runtime.abandon(event["session"])
                stop.wait(2)
    finally:
        os.close(lock_fd)
