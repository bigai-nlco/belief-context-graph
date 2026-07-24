"""
online.py  (construct_beliefs v3)
=================================
Streaming / online driver for the same pipeline used by ``run.py``.

The batch CLI (``run.py``) and the HTTP/online service both ultimately drive
``StreamingBeliefBuilder`` in ``stream.py``. This module keeps the online path
aligned with the batch path by:

  * using the same config wiring as ``pipeline.run_input``;
  * passing the same per-turn fields into ``ingest_turn`` (role/content/date/
    has_answer);
  * carrying item metadata and ``order_sorted`` into ``finalize`` like
    ``pipeline.run_item``;
  * optionally accepting already-normalised run.py-style input through
    ``SessionManager.push_input``.

Streaming contract
------------------
Each incoming turn dict should carry:

    {
      "problem_id": "<trajectory/item id>",
      "role":       "system" | "user" | "assistant" | "tool" | "function",
      "content":    "<complete message text>"
    }

Optional keys:

    "date" / "session_date"   passed through to evidence/source metadata
    "has_answer"              passed through exactly like run.py session data
    "is_message_end"          default True; set False only for fragments
    "is_trajectory_end"       default False; True triggers finalization

Outputs are written under ``<output_root>/<problem_id>/``:

    trajectory_stream.jsonl   raw received stream log
    trajectory.json           reconstructed run.py-compatible trajectory
    belief_graph.jsonl        one snapshot per turn plus final snapshot
    belief_graph_latest.json  latest snapshot

The engine has per-session state (a token usage tracker and audit log paths)
that used to be process-global and swapped in/out around every call — safe
only because online_server.py serialized ALL engine work behind one lock.
Each session now binds its own tracker/paths through llm.py's context-local
helpers (contextvars), and each StreamingTrajectorySession guards its own
mutable state (the belief graph, its builder, per-turn counters, output
files) with its own re-entrant lock. Together this means: turns for the SAME
problem_id are still processed strictly in order (whichever thread got there
first), while DIFFERENT problem_ids can be processed fully concurrently, on
different threads, with no shared mutable state between them. See
online_server.py, which no longer needs a single global lock because of this.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import llm
from .llm import (
    load_config,
    load_belief_graph_config,
    load_embedding_config,
    make_client,
    make_embedder,
)
from .loaders import iter_items, sanitize_name, select_items
from .stream import StreamOptions, StreamingBeliefBuilder


class TrajectoryClosedError(RuntimeError):
    """Raised when a turn arrives for a trajectory that has already finalized."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional_bool(value: Any) -> Optional[bool]:
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
        options: Optional[StreamOptions] = None,
        embedder=None,
        edge_generator=None,
        pricing: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        item_meta: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.problem_id = str(problem_id)
        self.client = client
        self.model = model
        self.options = options or StreamOptions()
        self.embedder = embedder
        self.edge_generator = edge_generator
        self.pricing = pricing
        self.max_tokens = max_tokens
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
        self._builder: Optional[StreamingBeliefBuilder] = None
        # A persistent tracker for the WHOLE lifetime of this trajectory
        # (across every push()/finalize() call), bound into llm.py's
        # context-local USAGE for the duration of each engine call (see
        # _engine() below) instead of being swapped into a process-global.
        self._usage_tracker = llm.TokenUsageTracker()

        # Re-entrant because push() can internally call finalize() (on
        # is_trajectory_end) while already holding this lock; RLock lets the
        # same thread re-acquire it without deadlocking. This is what keeps
        # ONE problem_id's turns strictly ordered even if two requests for it
        # land on different threads at nearly the same time, while placing NO
        # restriction on other problem_ids running concurrently on other
        # threads (they each have their own session, and thus their own lock).
        self._lock = threading.RLock()

        # assembled, ingested messages for trajectory.json
        self._messages: List[Dict[str, Any]] = []
        # fragment buffer (only used when is_message_end=False is sent)
        self._buf_role: Optional[str] = None
        self._buf_parts: List[str] = []
        self._buf_date: Optional[str] = None
        self._buf_has_answer: Optional[bool] = None

        self._n_received = 0       # raw dicts seen
        self._n_ingested = 0       # turns actually fed to the engine
        self._finalized = False
        self._result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        return not self._finalized

    @property
    def n_turns(self) -> int:
        return self._n_ingested

    @property
    def result(self) -> Optional[Dict[str, Any]]:
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
        usage_token = llm.bind_usage_tracker(self._usage_tracker)
        prompt_token = llm.bind_prompt_log_path(self._logs_dir / "prompts.jsonl")
        emb_token = None
        if self.embedder is not None:
            emb_token = self.embedder.set_log_path(self._logs_dir / "embedding_calls.jsonl")
        try:
            yield
        finally:
            llm.unbind_usage_tracker(usage_token)
            llm.unbind_prompt_log_path(prompt_token)
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


    def _ensure_builder(self) -> StreamingBeliefBuilder:
        if self._builder is None:
            with self._engine():
                self._builder = StreamingBeliefBuilder(
                    client=self.client,
                    model=self.model,
                    item_id=self.problem_id,
                    item_meta=self.item_meta,
                    out_dir=self.out_dir,
                    options=self.options,
                    embedder=self.embedder,
                    edge_generator=self.edge_generator,
                    max_tokens=self.max_tokens,
                )
        return self._builder

    # ------------------------------------------------------------- file I/O
    def _append_stream_log(self, record: Dict[str, Any]) -> None:
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

    def _snapshot(self, *, stage: str) -> Dict[str, Any]:
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
        return builder.graph.snapshot(extra={
            "problem_id": self.problem_id,
            "item_id": self.problem_id,
            "stage": stage,
            "finalized": self._finalized,
            "stream_turn_index": max(0, self._n_ingested - 1),
            "n_turns_ingested": self._n_ingested,
        })

    def _emit_snapshot(self, *, stage: str) -> Dict[str, Any]:
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
        date: Optional[str] = None,
        has_answer: Optional[bool] = None,
    ) -> None:
        """Feed one assembled turn to the engine and record it for trajectory.json."""
        builder = self._ensure_builder()
        with self._engine():
            builder.ingest_turn(role, content, date=date, has_answer=has_answer)
        msg: Dict[str, Any] = {"role": role, "content": content}
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
    def push(self, turn: Dict[str, Any]) -> Dict[str, Any]:
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

    def _push_locked(self, turn: Dict[str, Any]) -> Dict[str, Any]:
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
        has_answer = _optional_bool(turn.get("has_answer")) if "has_answer" in turn else None

        # 1) RAW stream log (real time) — every received dict, ingested or not
        self._n_received += 1
        self._append_stream_log({
            "recv_ts": _now(),
            "recv_index": self._n_received - 1,
            "ingested": is_msg_end,
            "is_message_end": is_msg_end,
            "is_trajectory_end": is_traj_end,
            "turn": turn,
        })

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
    def finalize(self) -> Dict[str, Any]:
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

    def _finalize_locked(self) -> Dict[str, Any]:
        if self._finalized:
            return self._snapshot(stage="final")
        # flush any dangling fragment first
        self._flush_buffer()
        builder = self._ensure_builder()
        meta: Dict[str, Any] = dict(self.extra_meta)
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

class SessionManager:
    """
    Thin multiplexer over ``StreamingTrajectorySession``, keyed by problem_id.

    It can process raw online turn dicts (``push``) or run.py-compatible inputs
    (``push_input``), both through the same engine and options.
    """

    def __init__(
        self,
        *,
        config_path: str = "model_config.json",
        model_key: Optional[str] = None,
        embedding_key: str = "embedding",
        output_root: Any = "outputs_stream",
        options: Optional[StreamOptions] = None,
        # pre-built injection (e.g. offline tests / custom wiring); when given,
        # config_path is NOT read.
        client: Any = "__from_config__",
        model: Optional[str] = None,
        embedder: Any = "__from_config__",
        edge_generator=None,
        pricing: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.options = options or StreamOptions()
        self.pricing = pricing
        self.edge_generator = edge_generator
        self.max_tokens: Optional[int] = None
        self._sessions: Dict[str, StreamingTrajectorySession] = {}
        # Guards ONLY the check-and-create step in get_session(): without it,
        # two threads racing to push the FIRST turn of a brand-new problem_id
        # could each construct their own StreamingTrajectorySession (each
        # creating/clearing the same output files), with one silently
        # clobbering the other in self._sessions. This is a short critical
        # section, separate from each session's own (potentially long-held)
        # self._lock, so it never becomes a bottleneck for concurrent
        # problem_ids once their sessions already exist.
        self._sessions_lock = threading.Lock()

        injected = (client != "__from_config__") or (model is not None)
        if injected:
            self.client = None if client == "__from_config__" else client
            if options is None:
                raise ValueError("injected SessionManager requires config-populated options")
            self.model = model or "model"
            self.embedder = None if embedder == "__from_config__" else embedder
        else:
            self._wire_from_config(config_path, model_key, embedding_key)

    # ---- config wiring (mirrors pipeline.run_input) -----------------------
    def _wire_from_config(self, config_path, model_key, embedding_key) -> None:
        cfg = load_config(config_path, model_key=model_key)
        bg_cfg = load_belief_graph_config(config_path, model_key=model_key)
        if bg_cfg:
            self.options = copy.deepcopy(self.options)
            self.options.apply_belief_graph_config(bg_cfg)
        self.client = make_client(cfg)
        self.model = cfg["model"]
        self.max_tokens = cfg.get("max_tokens")
        if self.pricing is None:
            self.pricing = cfg.get("pricing")
        masked = (cfg.get("api_key", "") or "")
        masked = (masked[:6] + "…" + masked[-3:]) if len(masked) > 10 else "***"
        print(
            f"[online] model={self.model}  base_url={cfg['base_url']}  api_key={masked}"
            + (f"  max_tokens={self.max_tokens}" if self.max_tokens else "")
        )
        extractor_cfg = self.options.to_dict()["extractor"]
        if extractor_cfg.get("enabled", True):
            print(
                f"[online] extractor={extractor_cfg['model']}  "
                f"base_url={extractor_cfg['base_url']}  "
                f"max_concurrency={extractor_cfg['max_concurrency']}  "
                f"context_scope={extractor_cfg['context_scope']}"
            )
        else:
            print(
                "[warn] generative extractor disabled; turns will produce no nodes",
                file=sys.stderr,
            )
        entity_cfg = self.options.to_dict()["entities"]
        print(
            f"[online] entities method={entity_cfg['method']}  "
            f"spaCy={entity_cfg['spacy_model']}  "
            "stage=post-merge"
        )
        edge_cfg = self.options.to_dict()["edge_generation"]
        print(
            f"[online] edge_generator={edge_cfg['model']}  "
            f"base_url={edge_cfg['base_url']}  "
            f"non_thinking={not edge_cfg['enable_thinking']}"
        )
        stance_cfg = self.options.to_dict()["stance"]
        print(
            f"[online] stance_model={stance_cfg['model_path']}  "
            f"device={stance_cfg['device']}  labels={','.join(stance_cfg['labels'])} "
            "(weights load lazily on first use)"
        )

        emb_cfg = load_embedding_config(config_path, embedding_key=embedding_key)
        self.embedder = None
        if emb_cfg is not None:
            self.embedder = make_embedder(emb_cfg)
            if emb_cfg.get("provider") == "local":
                print(
                    f"[online] embedding provider=local  model={emb_cfg['model']}  "
                    f"(weights load lazily on first use)"
                )
            else:
                print(
                    f"[online] embedding model={emb_cfg['model']}  base_url={emb_cfg['base_url']}"
                )
        else:
            # Match pipeline.run_input fallback behaviour.
            if self.options.chunking_enabled:
                print(
                    f"[warn] semantic chunking needs an {embedding_key!r} config entry; "
                    "each turn will fall back to one chunk",
                    file=sys.stderr,
                )
            if self.options.incremental_merge:
                print(
                    f"[warn] incremental merge needs an {embedding_key!r} config entry; "
                    "incremental merge passes will be skipped",
                    file=sys.stderr,
                )

    # ---- routing ----------------------------------------------------------
    def get_session(
        self,
        problem_id: str,
        *,
        create: bool = True,
        item_meta: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[StreamingTrajectorySession]:
        pid = str(problem_id)
        with self._sessions_lock:
            sess = self._sessions.get(pid)
            if sess is None and create:
                sess = StreamingTrajectorySession(
                    pid,
                    client=self.client,
                    model=self.model,
                    output_root=self.output_root,
                    options=self.options,
                    embedder=self.embedder,
                    edge_generator=self.edge_generator,
                    pricing=self.pricing,
                    max_tokens=self.max_tokens,
                    item_meta=item_meta,
                    extra_meta=extra_meta,
                )
                self._sessions[pid] = sess
        return sess

    def push(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        """Route one incoming dict to its trajectory and return the live graph."""
        pid = turn.get("problem_id")
        if pid is None:
            raise ValueError("each streamed turn must carry a 'problem_id'")
        sess = self.get_session(str(pid))
        assert sess is not None
        return sess.push(turn)

    def push_many(self, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Push several turns (as used by the /turns HTTP endpoint), keeping each
        problem_id's own turns strictly in arrival order and atomic relative
        to any other concurrent request touching that SAME problem_id (via
        session.exclusive()), while turns for DIFFERENT problem_ids are
        pushed concurrently on a small thread pool.
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}
        order: List[str] = []
        for t in turns:
            pid = t.get("problem_id")
            if pid is None:
                raise ValueError("each streamed turn must carry a 'problem_id'")
            pid = str(pid)
            if pid not in groups:
                groups[pid] = []
                order.append(pid)
            groups[pid].append(t)

        latest: Dict[str, Any] = {}
        finalized: List[str] = []
        result_lock = threading.Lock()

        def _run_group(pid: str) -> None:
            sess = self.get_session(pid)
            assert sess is not None
            group_latest = None
            group_finalized = False
            with sess.exclusive():
                for t in groups[pid]:
                    group_latest = sess.push(t)
                    if group_latest.get("finalized"):
                        group_finalized = True
            with result_lock:
                if group_latest is not None:
                    latest[pid] = group_latest
                if group_finalized:
                    finalized.append(pid)

        if len(order) <= 1:
            for pid in order:
                _run_group(pid)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(order), 8)) as ex:
                list(ex.map(_run_group, order))

        return {"pushed": len(turns), "finalized": finalized, "latest": latest}

    def push_item(self, item: Dict[str, Any], *, finalize: bool = True) -> Dict[str, Any]:
        """
        Process one normalised loader item exactly like ``pipeline.run_item``:
        ingest its turns in order, then finalize at the end.

        The whole turn loop runs under ``sess.exclusive()`` so this item's
        turns can never interleave with another concurrent request (e.g. a
        stray /turn call) touching the same problem_id.
        """
        item_id = str(item.get("item_id") or item.get("problem_id") or "trajectory")
        sess = self.get_session(
            item_id,
            item_meta=item.get("meta") or {},
            extra_meta={"order_sorted": item.get("order_sorted", False)},
        )
        assert sess is not None
        turns = item.get("turns") or []
        with sess.exclusive():
            latest: Optional[Dict[str, Any]] = None
            for idx, raw_turn in enumerate(turns):
                if not isinstance(raw_turn, dict):
                    continue
                t = dict(raw_turn)
                t["problem_id"] = item_id
                # Full-item mode should behave like run_item: the finalization point is
                # the end of the item, not whatever was present in the original data.
                t["is_trajectory_end"] = bool(finalize and idx == len(turns) - 1)
                latest = sess.push(t)
            if latest is None:
                latest = sess.finalize() if finalize else sess._snapshot(stage="empty")
            elif finalize and not latest.get("finalized"):
                latest = sess.finalize()
        return latest

    def push_input(
        self,
        data: Any,
        *,
        keep_order: bool = False,
        item_selector: Optional[str] = None,
        finalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Accept the same data shapes as run.py/loaders.py, normalise them into
        items, and process them through the online sessions.

        Different items (problem_ids) are processed concurrently on a small
        thread pool — safe because each resolves to its own
        StreamingTrajectorySession, guarded by that session's own lock (see
        push_item's use of sess.exclusive()). Items are expected to carry
        distinct ids; if two items in one payload happen to share an id, their
        turns funnel into the same session and are still individually safe,
        but may interleave across the two concurrent push_item() calls.
        """
        items = select_items(iter_items(data, keep_order=keep_order), item_selector)
        latest: Dict[str, Any] = {}
        finalized: List[str] = []
        result_lock = threading.Lock()

        def _run(item: Dict[str, Any]) -> None:
            snap = self.push_item(item, finalize=finalize)
            item_id = str(item.get("item_id") or item.get("problem_id") or "trajectory")
            with result_lock:
                latest[item_id] = snap
                if snap.get("finalized"):
                    finalized.append(item_id)

        if len(items) <= 1:
            for item in items:
                _run(item)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(items), 8)) as ex:
                list(ex.map(_run, items))

        return {"items": len(items), "finalized": finalized, "latest": latest}

    def finalize(self, problem_id: str) -> Dict[str, Any]:
        sess = self.get_session(problem_id, create=False)
        if sess is None:
            raise KeyError(f"no active trajectory for problem_id {problem_id!r}")
        return sess.finalize()

    def get_graph(self, problem_id: str) -> Optional[Dict[str, Any]]:
        sess = self.get_session(problem_id, create=False)
        return None if sess is None else sess._snapshot(stage="query")

    def active_problem_ids(self) -> List[str]:
        return [pid for pid, s in self._sessions.items() if s.active]

    def all_problem_ids(self) -> List[str]:
        return list(self._sessions.keys())
