"""Private, durable inbox shared by fast hooks and the long-lived MCP worker."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "memory.sqlite3"
        fd = os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
        with self.db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, event_key TEXT UNIQUE NOT NULL,
                    session TEXT NOT NULL, turn TEXT NOT NULL, kind TEXT NOT NULL,
                    payload TEXT NOT NULL, created REAL NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending', error TEXT);
                CREATE INDEX IF NOT EXISTS events_session ON events(session,id);
                CREATE TABLE IF NOT EXISTS graphs (
                    graph_id TEXT PRIMARY KEY, session TEXT NOT NULL,
                    snapshot TEXT NOT NULL, event_id INTEGER NOT NULL,
                    updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS injections (
                    id INTEGER PRIMARY KEY, session TEXT, event TEXT,
                    created REAL, context TEXT);
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            """)

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def enqueue(self, event: dict, limit: int) -> bool:
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM events WHERE event_key=?", (event["key"],)
            ).fetchone():
                return False
            count = db.execute(
                "SELECT count(*) FROM events WHERE session=?", (event["session"],)
            ).fetchone()[0]
            if count >= limit:
                raise ValueError(
                    "BCG session capture limit reached; existing memories remain available"
                )
            db.execute(
                "INSERT INTO events(event_key,session,turn,kind,payload,created) VALUES(?,?,?,?,?,?)",
                (
                    event["key"],
                    event["session"],
                    event["turn"],
                    event["kind"],
                    json.dumps(event["payload"], ensure_ascii=False),
                    time.time(),
                ),
            )
        return True

    def recover(self) -> None:
        # Only the process holding the worker lock may call this. Published
        # snapshots and acknowledgements commit atomically; unfinished work is
        # retried in a NEW graph segment, never appended to an uncertain graph.
        with self.db() as db:
            db.execute("UPDATE events SET state='pending' WHERE state='processing'")

    def next_event(self) -> dict | None:
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM events WHERE state='pending' ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute("UPDATE events SET state='processing' WHERE id=?", (row["id"],))
            result = dict(row)
            result["payload"] = json.loads(result["payload"])
            return result

    def complete(self, event: dict, snapshot: dict) -> None:
        with self.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO graphs VALUES(?,?,?,?,?)",
                (
                    snapshot["problem_id"],
                    event["session"],
                    json.dumps(snapshot, ensure_ascii=False),
                    event["id"],
                    time.time(),
                ),
            )
            db.execute(
                "UPDATE events SET state='done',error=NULL WHERE id=?", (event["id"],)
            )

    def fail(self, event_id: int, error: str) -> None:
        with self.db() as db:
            db.execute(
                "UPDATE events SET state='error',error=? WHERE id=?", (error, event_id)
            )

    def retry(self) -> int:
        with self.db() as db:
            return db.execute(
                "UPDATE events SET state='pending',error=NULL WHERE state='error'"
            ).rowcount

    def snapshots(self, session: str) -> list[dict]:
        with self.db() as db:
            rows = db.execute(
                "SELECT snapshot FROM graphs WHERE session=? ORDER BY event_id DESC",
                (session,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def latest_question(self, session: str) -> str:
        with self.db() as db:
            row = db.execute(
                "SELECT payload FROM events WHERE session=? AND kind='UserPromptSubmit' ORDER BY id DESC LIMIT 1",
                (session,),
            ).fetchone()
        return json.loads(row[0])["text"] if row else ""

    def status(self, session: str | None = None) -> dict:
        with self.db() as db:
            clause, args = (" WHERE session=?", (session,)) if session else ("", ())
            counts = {
                row[0]: row[1]
                for row in db.execute(
                    "SELECT state,count(*) FROM events" + clause + " GROUP BY state",
                    args,
                )
            }
            graphs = db.execute(
                "SELECT count(*) FROM graphs" + clause, args
            ).fetchone()[0]
            heartbeat = db.execute(
                "SELECT value FROM meta WHERE key='worker'"
            ).fetchone()
            errors = [
                dict(r)
                for r in db.execute(
                    "SELECT id,kind,error FROM events"
                    + (clause + " AND" if clause else " WHERE")
                    + " state='error' ORDER BY id DESC LIMIT 3",
                    args,
                )
            ]
        return {
            "session_id": session,
            "events": counts,
            "graph_segments": graphs,
            "worker_last_seen_seconds": round(time.time() - float(heartbeat[0]), 1)
            if heartbeat
            else None,
            "errors": errors,
        }

    def heartbeat(self) -> None:
        with self.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO meta VALUES('worker',?)", (str(time.time()),)
            )

    def record_injection(self, session: str, event: str, context: str) -> None:
        with self.db() as db:
            db.execute(
                "INSERT INTO injections(session,event,created,context) VALUES(?,?,?,?)",
                (session, event, time.time(), context),
            )

    def evidence(self, session: str, graph_id: str, node_id: int) -> dict:
        for snapshot in self.snapshots(session):
            if snapshot["problem_id"] != graph_id:
                continue
            nodes = snapshot.get("nodes") or snapshot.get("beliefs", []) + snapshot.get(
                "decisions", []
            )
            for node in nodes:
                if node.get("id") == node_id:
                    evidence_ids = node.get("evidence_ids", [])
                    return {
                        "graph_id": graph_id,
                        "node": node,
                        "relations": [
                            e
                            for e in snapshot.get("relations", [])
                            if node_id in (e.get("from_id"), e.get("to_id"))
                        ],
                        "evidence": [
                            e
                            for e in snapshot.get("evidence", [])
                            if e.get("id") in evidence_ids
                        ],
                    }
        raise ValueError("No such belief in this session")
