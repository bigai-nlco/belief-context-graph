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
import sys
import threading
from pathlib import Path
from typing import Any

from .._shared.loaders import iter_items, select_items
from .._shared.session import (  # noqa: F401 - compat re-exports for legacy imports
    StreamingTrajectorySession,
    TrajectoryClosedError,
    _now,
    _optional_bool,
    resolve_dated_output_root,
)
from .llm import (
    load_belief_graph_config,
    load_config,
    load_embedding_config,
    make_client,
    make_embedder,
    resolve_reasoning_effort,
)
from .stream import StreamingBeliefBuilder, StreamOptions


class SessionManager:
    """
    Thin multiplexer over ``StreamingTrajectorySession``, keyed by problem_id.

    It can process raw online turn dicts (``push``) or run.py-compatible inputs
    (``push_input``), both through the same engine and options.
    """

    def __init__(
        self,
        *,
        config_path: str | None = None,  # None = layered YAML configuration
        model_key: str | None = None,
        embedding_key: str = "embedding",
        output_root: Any = "outputs_stream",
        options: StreamOptions | None = None,
        # pre-built injection (e.g. offline tests / custom wiring); when given,
        # config_path is NOT read.
        client: Any = "__from_config__",
        model: str | None = None,
        embedder: Any = "__from_config__",
        pricing: dict[str, Any] | None = None,
    ) -> None:
        self.output_root_template = Path(output_root)
        self.output_root = resolve_dated_output_root(self.output_root_template)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.options = options or StreamOptions()
        self.pricing = pricing
        self.max_tokens: int | None = None
        self._sessions: dict[str, StreamingTrajectorySession] = {}
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
            self.model = model or "model"
            self.embedder = None if embedder == "__from_config__" else embedder
        else:
            self._wire_from_config(config_path, model_key, embedding_key)

    def current_output_root(self) -> Path:
        root = resolve_dated_output_root(self.output_root_template)
        root.mkdir(parents=True, exist_ok=True)
        self.output_root = root
        return root

    # ---- config wiring (mirrors pipeline.run_input) -----------------------
    def _wire_from_config(self, config_path, model_key, embedding_key) -> None:
        cfg = load_config(config_path, model_key=model_key)
        bg_cfg = load_belief_graph_config(config_path, model_key=model_key)
        if bg_cfg:
            self.options.apply_belief_graph_config(bg_cfg)
        self.client = make_client(cfg)
        self.model = cfg.get("model") or cfg.get("model_name") or "gpt-4o-mini"
        if cfg.get("reasoning_effort") is not None:
            self.options.reasoning_effort = str(cfg["reasoning_effort"])
        self.max_tokens = cfg.get("max_tokens")
        if self.pricing is None:
            self.pricing = cfg.get("pricing")
        masked = cfg.get("api_key", "") or ""
        masked = (masked[:6] + "…" + masked[-3:]) if len(masked) > 10 else "***"
        effective_reasoning = resolve_reasoning_effort(
            self.model, self.options.reasoning_effort
        )
        print(
            f"[online] model={self.model}  base_url={cfg['base_url']}  api_key={masked}"
            f"  reasoning_effort={effective_reasoning}"
            + (f"  max_tokens={self.max_tokens}" if self.max_tokens else "")
        )
        print(
            f"[online] stance/entities model={self.model}  metadata_source=graph_model"
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
            print(
                f"[warn] no {embedding_key!r} config entry — the incremental merge "
                f"pass will be skipped",
                file=sys.stderr,
            )

    # ---- routing ----------------------------------------------------------
    def get_session(
        self,
        problem_id: str,
        *,
        create: bool = True,
        item_meta: dict[str, Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> StreamingTrajectorySession | None:
        pid = str(problem_id)
        with self._sessions_lock:
            sess = self._sessions.get(pid)
            if sess is None and create:
                sess = StreamingTrajectorySession(
                    pid,
                    client=self.client,
                    model=self.model,
                    output_root=self.current_output_root(),
                    options=self.options,
                    embedder=self.embedder,
                    pricing=self.pricing,
                    max_tokens=self.max_tokens,
                    item_meta=item_meta,
                    extra_meta=extra_meta,
                    builder_cls=StreamingBeliefBuilder,
                    options_cls=StreamOptions,
                )
                self._sessions[pid] = sess
        return sess

    def push(self, turn: dict[str, Any]) -> dict[str, Any]:
        """Route one incoming dict to its trajectory and return the live graph."""
        pid = turn.get("problem_id")
        if pid is None:
            raise ValueError("each streamed turn must carry a 'problem_id'")
        sess = self.get_session(str(pid))
        assert sess is not None
        return sess.push(turn)

    def push_many(self, turns: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Push several turns (as used by the /turns HTTP endpoint), keeping each
        problem_id's own turns strictly in arrival order and atomic relative
        to any other concurrent request touching that SAME problem_id (via
        session.exclusive()), while turns for DIFFERENT problem_ids are
        pushed concurrently on a small thread pool.
        """
        groups: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for t in turns:
            pid = t.get("problem_id")
            if pid is None:
                raise ValueError("each streamed turn must carry a 'problem_id'")
            pid = str(pid)
            if pid not in groups:
                groups[pid] = []
                order.append(pid)
            groups[pid].append(t)

        latest: dict[str, Any] = {}
        finalized: list[str] = []
        result_lock = threading.Lock()

        def _run_group(pid: str) -> None:
            sess = self.get_session(pid)
            assert sess is not None
            group_latest = None
            group_finalized = False
            with sess.exclusive():
                batch_push = getattr(sess, "push_many", None)
                if callable(batch_push):
                    group_latest = batch_push(groups[pid])
                    group_finalized = bool(group_latest.get("finalized"))
                else:
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
                max_workers=min(len(order), 8)
            ) as ex:
                list(ex.map(_run_group, order))

        return {"pushed": len(turns), "finalized": finalized, "latest": latest}

    def push_item(
        self, item: dict[str, Any], *, finalize: bool = True
    ) -> dict[str, Any]:
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
            latest: dict[str, Any] | None = None
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
        item_selector: str | None = None,
        finalize: bool = True,
    ) -> dict[str, Any]:
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
        latest: dict[str, Any] = {}
        finalized: list[str] = []
        result_lock = threading.Lock()

        def _run(item: dict[str, Any]) -> None:
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
                max_workers=min(len(items), 8)
            ) as ex:
                list(ex.map(_run, items))

        return {"items": len(items), "finalized": finalized, "latest": latest}

    def finalize(self, problem_id: str) -> dict[str, Any]:
        sess = self.get_session(problem_id, create=False)
        if sess is None:
            raise KeyError(f"no active trajectory for problem_id {problem_id!r}")
        return sess.finalize()

    def release(self, problem_id: str) -> dict[str, Any]:
        """Drop one completed producer session while preserving its disk artifacts."""
        pid = str(problem_id)
        with self._sessions_lock:
            released = self._sessions.pop(pid, None) is not None
        return {"problem_id": pid, "released": released}

    def get_graph(self, problem_id: str) -> dict[str, Any] | None:
        sess = self.get_session(problem_id, create=False)
        return None if sess is None else sess._snapshot(stage="query")

    def select_context(
        self,
        problem_id: str,
        query: str,
        *,
        strategy: str = "connected",
        focus_query: str | None = None,
        question: str | None = None,
        node_char_budget: int = 6_600,
        max_depth: int = 4,
    ) -> dict[str, Any]:
        sess = self.get_session(problem_id, create=False)
        if sess is None:
            raise KeyError(f"no active trajectory for problem_id {problem_id!r}")
        return sess.select_context(
            query,
            strategy=strategy,
            focus_query=focus_query,
            question=question,
            node_char_budget=node_char_budget,
            max_depth=max_depth,
        )

    def active_problem_ids(self) -> list[str]:
        return [pid for pid, s in self._sessions.items() if s.active]

    def all_problem_ids(self) -> list[str]:
        return list(self._sessions.keys())
