"""Shared streaming trajectory session for both construct backends.

Extracted from the two backends' former ``online.py`` copies: the live
one-trajectory session state machine (``StreamingTrajectorySession``) plus
``resolve_dated_output_root`` and small helpers. Backend-specific wiring
(config loading, model clients, builder classes) is injected by the
backends; this module imports only shared primitives.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .._shared.llm import (
    TokenUsageTracker,
    bind_prompt_log_path,
    bind_usage_tracker,
    unbind_prompt_log_path,
    unbind_usage_tracker,
)
from .._shared.loaders import sanitize_name


class TrajectoryClosedError(RuntimeError):
    """Raised when a turn arrives for a trajectory that has already finalized."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


_DATED_OUTPUT_RE = re.compile(
    r"^(?P<prefix>outputs_)(?:(?P<year>\d{4})_)?(?P<month>\d{1,2})_(?P<day>\d{1,2})$"
)


def resolve_dated_output_root(output_root: Any, now: datetime | None = None) -> Path:
    """
    Resolve daily output roots at use time.

    Supported forms:
      * outputs_7_6       -> outputs_<today month>_<today day>
      * outputs_2026_7_6  -> outputs_<today year>_<today month>_<today day>
      * outputs_{Y}_{m}_{d} -> explicit template
      * outputs_{date}    -> e.g. outputs_2026-07-07

    Plain roots such as outputs_stream are left unchanged.
    """
    root = Path(output_root)
    local_now = now or datetime.now().astimezone()
    values = {
        "Y": f"{local_now.year:04d}",
        "y": f"{local_now.year % 100:02d}",
        "m": str(local_now.month),
        "mm": f"{local_now.month:02d}",
        "month": str(local_now.month),
        "d": str(local_now.day),
        "dd": f"{local_now.day:02d}",
        "day": str(local_now.day),
        "date": f"{local_now.year:04d}-{local_now.month:02d}-{local_now.day:02d}",
    }
    raw = str(root)
    if any(("{" + key + "}") in raw for key in values):
        try:
            return Path(raw.format(**values))
        except (KeyError, ValueError):
            return root

    m = _DATED_OUTPUT_RE.match(root.name)
    if not m:
        return root
    month = (
        f"{local_now.month:02d}" if len(m.group("month")) == 2 else str(local_now.month)
    )
    day = f"{local_now.day:02d}" if len(m.group("day")) == 2 else str(local_now.day)
    year = f"{local_now.year:04d}_" if m.group("year") else ""
    return root.with_name(f"{m.group('prefix')}{year}{month}_{day}")


def _optional_bool(value: Any) -> bool | None:
    """Coerce a present has_answer-like value while preserving missing as None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    return bool(value)


# ===========================================================================
# One trajectory  (one problem_id)
# ===========================================================================


class StreamingTrajectorySession:
    """
    Live belief-graph builder for ONE trajectory/item, fed one complete turn at
    a time. It is the online equivalent of ``pipeline.run_item``.
    """

    def __init__(
        self,
        problem_id: str,
        *,
        client,
        model: str,
        output_root: Any = "outputs_stream",
        options: Any | None = None,
        embedder=None,
        pricing: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        item_meta: dict[str, Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
        edge_generator=None,
        builder_cls: type | None = None,
        options_cls: type | None = None,
    ) -> None:
        self.problem_id = str(problem_id)
        self.client = client
        self.model = model
        self.options = options or (options_cls() if options_cls is not None else None)
        self.embedder = embedder
        self.pricing = pricing
        self.max_tokens = max_tokens
        self.edge_generator = edge_generator
        self.builder_cls = builder_cls
        self.options_cls = options_cls
        self.item_meta = dict(item_meta or {})
        self.extra_meta = dict(extra_meta or {})
        self.extra_meta.setdefault("order_sorted", False)

        self.out_dir = Path(output_root) / sanitize_name(self.problem_id)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir = self.out_dir / "logs"
        self._logs_dir.mkdir(parents=True, exist_ok=True)

        # output file paths
        self._stream_log_path = self.out_dir / "trajectory_stream.jsonl"
        self._trajectory_path = self.out_dir / "trajectory.json"
        self._graph_jsonl_path = self.out_dir / "belief_graph.jsonl"
        self._graph_latest_path = self.out_dir / "belief_graph_latest.json"
        # fresh start so re-runs do not append stale snapshots/log lines
        self._stream_log_path.write_text("", encoding="utf-8")
        self._graph_jsonl_path.write_text("", encoding="utf-8")

        # engine + per-session isolated state
        self._builder: Any | None = None
        # A persistent tracker for the WHOLE lifetime of this trajectory
        # (across every push()/finalize() call), bound into llm.py's
        # context-local USAGE for the duration of each engine call (see
        # _engine() below) instead of being swapped into a process-global.
        self._usage_tracker = TokenUsageTracker()

        # Re-entrant because push() can internally call finalize() (on
        # is_trajectory_end) while already holding this lock; RLock lets the
        # same thread re-acquire it without deadlocking. This is what keeps
        # ONE problem_id's turns strictly ordered even if two requests for it
        # land on different threads at nearly the same time, while placing NO
        # restriction on other problem_ids running concurrently on other
        # threads (they each have their own session, and thus their own lock).
        self._lock = threading.RLock()

        # assembled, ingested messages for trajectory.json
        self._messages: list[dict[str, Any]] = []
        # fragment buffer (only used when is_message_end=False is sent)
        self._buf_role: str | None = None
        self._buf_parts: list[str] = []
        self._buf_date: str | None = None
        self._buf_has_answer: bool | None = None

        self._n_received = 0  # raw dicts seen
        self._n_ingested = 0  # turns actually fed to the engine
        self._finalized = False
        self._result: dict[str, Any] | None = None

    # ------------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        return not self._finalized

    @property
    def n_turns(self) -> int:
        return self._n_ingested

    @property
    def result(self) -> dict[str, Any] | None:
        return self._result

    # ----------------------------------------------- engine-global isolation
    @contextmanager
    def _engine(self):
        """
        Bind THIS session's usage tracker + log paths as the active
        context-local engine state (see llm.py's bind_usage_tracker /
        bind_prompt_log_path / bind_embedding_log_path) for the duration of
        one engine call, then restore whatever was bound before.

        Because these are contextvars scoped to the current thread, two
        sessions running on two different threads at the same time (e.g. two
        different problem_ids under online_server.py, which no longer
        serializes them behind one global lock) never see or clobber each
        other's tracker or audit-log paths — unlike the old approach of
        mutating shared attributes on a process-global object.
        """
        usage_token = bind_usage_tracker(self._usage_tracker)
        prompt_token = bind_prompt_log_path(self._logs_dir / "prompts.jsonl")
        emb_token = None
        if self.embedder is not None:
            emb_token = self.embedder.set_log_path(
                self._logs_dir / "embedding_calls.jsonl"
            )
        try:
            yield
        finally:
            unbind_usage_tracker(usage_token)
            unbind_prompt_log_path(prompt_token)
            if emb_token is not None:
                self.embedder.unbind_log_path(emb_token)

    @contextmanager
    def exclusive(self):
        """
        Hold this session's lock for a whole multi-turn operation (e.g. one
        push_item() call ingesting a full item's turns) so those turns can
        never interleave with another concurrent request touching the SAME
        problem_id. push() and finalize() already take this lock individually
        for single-call safety; wrapping a whole batch in ``exclusive()`` on
        top of that makes the batch atomic too. Safe to nest (the lock is
        re-entrant).
        """
        with self._lock:
            yield self

    def _ensure_builder(self) -> Any:
        if self._builder is None:
            with self._engine():
                builder_kwargs = {
                    "client": self.client,
                    "model": self.model,
                    "item_id": self.problem_id,
                    "item_meta": self.item_meta,
                    "out_dir": self.out_dir,
                    "options": self.options,
                    "embedder": self.embedder,
                    "max_tokens": self.max_tokens,
                }
                if self.edge_generator is not None:
                    builder_kwargs["edge_generator"] = self.edge_generator
                self._builder = self.builder_cls(**builder_kwargs)
        return self._builder

    # ------------------------------------------------------------- file I/O
    def _append_stream_log(self, record: dict[str, Any]) -> None:
        with open(self._stream_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_trajectory(self, complete: bool) -> None:
        doc = {
            "problem_id": self.problem_id,
            "item_id": self.problem_id,
            "complete": complete,
            "n_turns": len(self._messages),
            "updated_at": _now(),
            "meta": dict(self.item_meta),
            "order_sorted": bool(self.extra_meta.get("order_sorted", False)),
            # run.py-compatible: run.py can read this back
            "trajectory": list(self._messages),
        }
        with open(self._trajectory_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

    def _snapshot(self, *, stage: str) -> dict[str, Any]:
        """Current graph snapshot, tagged with streaming metadata."""
        builder = self._builder
        if builder is None:
            return {
                "problem_id": self.problem_id,
                "item_id": self.problem_id,
                "stage": stage,
                "finalized": self._finalized,
                "n_nodes": 0,
                "n_beliefs": 0,
                "n_decisions": 0,
                "nodes": [],
                "beliefs": [],
                "decisions": [],
                "evidence": [],
                "relations": [],
                "merges": [],
                "sessions": [],
                "generated_at": _now(),
            }
        return builder.graph.snapshot(
            extra={
                "problem_id": self.problem_id,
                "item_id": self.problem_id,
                "stage": stage,
                "finalized": self._finalized,
                "stream_turn_index": max(0, self._n_ingested - 1),
                "n_turns_ingested": self._n_ingested,
            }
        )

    def _emit_snapshot(self, *, stage: str) -> dict[str, Any]:
        """Snapshot, append to belief_graph.jsonl, refresh belief_graph_latest.json."""
        snap = self._snapshot(stage=stage)
        with open(self._graph_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snap, ensure_ascii=False) + "\n")
        with open(self._graph_latest_path, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
        return snap

    # ------------------------------------------------------------- ingestion
    def _ingest(
        self,
        role: str,
        content: str,
        *,
        date: str | None = None,
        has_answer: bool | None = None,
    ) -> None:
        """Feed one assembled turn to the engine and record it for trajectory.json."""
        builder = self._ensure_builder()
        with self._engine():
            builder.ingest_turn(role, content, date=date, has_answer=has_answer)
        msg: dict[str, Any] = {"role": role, "content": content}
        if date is not None:
            msg["date"] = date
        if has_answer is not None:
            msg["has_answer"] = bool(has_answer)
        self._messages.append(msg)
        self._n_ingested += 1

    def _flush_buffer(self) -> None:
        if self._buf_parts:
            role = self._buf_role or "user"
            self._ingest(
                role,
                "".join(self._buf_parts),
                date=self._buf_date,
                has_answer=self._buf_has_answer,
            )
        self._buf_role = None
        self._buf_parts = []
        self._buf_date = None
        self._buf_has_answer = None

    # ------------------------------------------------------------------ push
    def push(self, turn: dict[str, Any]) -> dict[str, Any]:
        """
        Process one incoming dict. Returns the current belief-graph snapshot.
        On ``is_trajectory_end`` the trajectory is finalized and the returned
        snapshot is the complete graph.

        Thread-safe with respect to OTHER calls on this SAME session (guarded
        by self._lock, an RLock so the internal call to self.finalize() below
        doesn't deadlock); unrelated sessions (other problem_ids) are
        untouched by this lock and may run fully concurrently.
        """
        with self._lock:
            return self._push_locked(turn)

    def _push_locked(self, turn: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(turn, dict):
            raise TypeError(f"push() expects a dict, got {type(turn).__name__}")
        if self._finalized:
            raise TrajectoryClosedError(
                f"trajectory {self.problem_id!r} is already finalized; no more turns accepted."
            )

        pid = turn.get("problem_id")
        if pid is not None and str(pid) != self.problem_id:
            raise ValueError(
                f"problem_id mismatch: session is {self.problem_id!r}, turn carries {pid!r}."
            )
        role = turn.get("role") or "user"
        content = turn.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)

        is_traj_end = bool(turn.get("is_trajectory_end", False))
        # a trajectory cannot end mid-message
        is_msg_end = True if is_traj_end else bool(turn.get("is_message_end", True))
        date = turn.get("date") or turn.get("session_date")
        has_answer = (
            _optional_bool(turn.get("has_answer")) if "has_answer" in turn else None
        )

        # 1) RAW stream log (real time) — every received dict, ingested or not
        self._n_received += 1
        self._append_stream_log(
            {
                "recv_ts": _now(),
                "recv_index": self._n_received - 1,
                "ingested": is_msg_end,
                "is_message_end": is_msg_end,
                "is_trajectory_end": is_traj_end,
                "turn": turn,
            }
        )

        # 2) fragment buffering (no-op under the default one-dict-one-turn contract)
        if not is_msg_end:
            if self._buf_role is not None and self._buf_role != role:
                # role changed mid-message: flush what we have, then start over
                print(
                    f"  [warn] {self.problem_id}: role changed mid-message "
                    f"({self._buf_role!r} -> {role!r}); flushing buffered fragment",
                    file=sys.stderr,
                )
                self._flush_buffer()
            self._buf_role = role
            self._buf_parts.append(content)
            if date is not None:
                self._buf_date = date
            if has_answer is not None:
                self._buf_has_answer = has_answer
            return self._snapshot(stage="buffered")

        # 3) assemble + ingest one complete turn
        if self._buf_parts:
            if self._buf_role and self._buf_role != role:
                print(
                    f"  [warn] {self.problem_id}: message-end role {role!r} differs "
                    f"from buffered role {self._buf_role!r}; using buffered role",
                    file=sys.stderr,
                )
            role = self._buf_role or role
            content = "".join(self._buf_parts) + content
            date = date or self._buf_date
            has_answer = has_answer if has_answer is not None else self._buf_has_answer
            self._buf_role = None
            self._buf_parts = []
            self._buf_date = None
            self._buf_has_answer = None

        self._ingest(role, content, date=date, has_answer=has_answer)
        self._write_trajectory(complete=is_traj_end)
        snap = self._emit_snapshot(stage="turn")

        # 4) finalize on the last turn
        if is_traj_end:
            return self.finalize()
        return snap

    # -------------------------------------------------------------- finalize
    def finalize(self) -> dict[str, Any]:
        """
        Run the same session-end merge pass used by ``pipeline.run_item`` and
        write result.json. Safe to call explicitly if the producer forgot
        ``is_trajectory_end``.

        Thread-safe the same way push() is (see there); also safe to call
        from WITHIN a push() on this same session, since self._lock is
        re-entrant.
        """
        with self._lock:
            return self._finalize_locked()

    def _finalize_locked(self) -> dict[str, Any]:
        if self._finalized:
            return self._snapshot(stage="final")
        # flush any dangling fragment first
        self._flush_buffer()
        builder = self._ensure_builder()
        meta: dict[str, Any] = dict(self.extra_meta)
        with self._engine():
            self._result = builder.finalize(extra_meta=meta, pricing=self.pricing)
        self._finalized = True
        self._write_trajectory(complete=True)
        snap = self._emit_snapshot(stage="final")
        print(
            f"  [online] trajectory {self.problem_id!r} finalized: "
            f"{snap.get('n_nodes')} node(s), "
            f"{len(snap.get('relations', []))} relation(s), "
            f"{len(snap.get('merges', []))} merge(s) -> {self.out_dir / 'result.json'}"
        )
        return snap


# ===========================================================================
# Many trajectories  (multiplex by problem_id)
# ===========================================================================
