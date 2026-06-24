"""
online.py  (construct_beliefs v2.1)
===================================
Streaming / online interface (role-only, scenario-free).

Background
----------
`StreamingBeliefBuilder` (stream.py) was *already* the online engine — the
README Roadmap said only the **driver** was missing. This module is that
driver. It receives a research trajectory ONE COMPLETE TURN AT A TIME (as the
agent generates it) and maintains the live belief graph incrementally, exactly
as a batch `run_research(...)` call would, but turn-by-turn.

Per the agreed design:

  * each complete turn extracts new belief/decision nodes and typed relations;
  * merge/dedup runs when the whole trajectory has been seen (i.e. on
    `is_trajectory_end`);
  * each incoming dict is one complete turn (an optional `is_message_end`
    flag also supports token-level fragment streaming — see below).

What you feed it
----------------
Each incoming item is a dict. REQUIRED keys:

    {
      "problem_id": "<id of the trajectory this turn belongs to>",
      "role":       "system" | "user" | "assistant" | "tool" | "function",
      "content":    "<full message text; tags included for assistant/tool>"
    }

OPTIONAL keys (all may be omitted):

    "is_message_end"     bool  (default True)  Set False ONLY when one message
                               is streamed in fragments (token streaming).
                               Consecutive fragments of the same problem_id are
                               buffered and concatenated until a fragment with
                               is_message_end=True arrives; only THEN is the
                               assembled turn ingested. With the default
                               "one dict == one turn" contract this is always
                               True and no buffering happens.
    "is_trajectory_end"  bool  (default False) True on the LAST turn of the
                               trajectory. Triggers finalize() (merge + result.json). Implies is_message_end.
    "turn_index"         int   informational only; the engine assigns its own.
    "session_date"       str   informational (research has no real sessions).
    ... any other keys are preserved verbatim in the raw stream log.

What you get back / what is written
-----------------------------------
`push(turn)` RETURNS the current belief-graph snapshot (a dict) and, per turn,
writes four artifacts under  <output_root>/<problem_id>/ :

    trajectory_stream.jsonl   append-only RAW log: every received dict +
                              server timestamp, in real time (this is the
                              "log file" that records the partial trajectory).
    trajectory.json           the reconstructed COMPLETE trajectory, rebuilt
                              incrementally and marked complete on
                              is_trajectory_end. Shape is run.py-compatible
                              ({"trajectory":[{role,content},...]}) so it can
                              be replayed through scripts/run.py directly.
    belief_graph.jsonl        append-only: ONE graph snapshot per turn
                              (stage="turn"), plus one final snapshot
                              (stage="final") after finalize — the evolution,
                              turn by turn.
    belief_graph_latest.json  the latest full snapshot (overwritten each turn).

Plus the engine's own native outputs land in the SAME directory (because the
builder's out_dir points here): result.json, events.jsonl,
token_usage.{json,txt}, and logs/ (prompts.jsonl, embedding_calls.jsonl,
merge_final.{json,log}).

Isolation / concurrency
-----------------------
Token accounting (`llm.USAGE`), the prompt-audit log path
(`llm.PROMPT_LOG_PATH`) and the embedder's log path are process-global /
shared-instance state in the engine. To keep multiple trajectories from
cross-contaminating each other, every engine call is wrapped in a context
manager that swaps in this session's own usage list + log paths and restores
the previous state afterwards. Calls are synchronous, so interleaving turns
from different problem_ids is safe.

Typical use
-----------
    mgr = SessionManager(config_path="model_config.json", model_key="gpt-5.5")
    for turn in incoming_stream:            # dicts as described above
        graph = mgr.push(turn)              # returns the live graph

    # or drive a single trajectory directly:
    sess = StreamingTrajectorySession("prob_42", client=c, model=m, ...)
    sess.push({"problem_id": "prob_42", "role": "user", "content": "..."})
    final = sess.push({..., "is_trajectory_end": True})
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import llm
from .llm import (
    load_config,
    load_embedding_config,
    make_client,
    make_embedder,
)
from .loaders import sanitize_name
from .stream import StreamOptions, StreamingBeliefBuilder


class TrajectoryClosedError(RuntimeError):
    """Raised when a turn arrives for a trajectory that has already finalized."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# One trajectory  (one problem_id)
# ===========================================================================

class StreamingTrajectorySession:
    """
    Live belief-graph builder for ONE research trajectory, fed one turn at a
    time. Self-contained: owns its output files and isolates its engine state,
    so it can be used standalone or multiplexed by a SessionManager.
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
        pricing: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        item_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.problem_id = str(problem_id)
        self.client = client
        self.model = model
        self.options = options or StreamOptions()
        self.embedder = embedder
        self.pricing = pricing
        self.max_tokens = max_tokens
        self.item_meta = dict(item_meta or {})
        self.item_meta.setdefault("problem_id", self.problem_id)

        self.out_dir = Path(output_root) / sanitize_name(self.problem_id)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir = self.out_dir / "logs"
        self._logs_dir.mkdir(parents=True, exist_ok=True)

        # output file paths
        self._stream_log_path = self.out_dir / "trajectory_stream.jsonl"
        self._trajectory_path = self.out_dir / "trajectory.json"
        self._graph_jsonl_path = self.out_dir / "belief_graph.jsonl"
        self._graph_latest_path = self.out_dir / "belief_graph_latest.json"
        # fresh start (truncate) so re-runs don't accumulate stale lines
        self._stream_log_path.write_text("", encoding="utf-8")
        self._graph_jsonl_path.write_text("", encoding="utf-8")

        # engine + per-session isolated state
        self._builder: Optional[StreamingBeliefBuilder] = None
        self._usage_records: List[Dict[str, Any]] = []
        self._usage_label: str = "unlabeled"

        # assembled, ingested messages (role+content only) for trajectory.json
        self._messages: List[Dict[str, str]] = []
        # fragment buffer (only used when is_message_end=False is sent)
        self._buf_role: Optional[str] = None
        self._buf_parts: List[str] = []
        self._buf_date: Optional[str] = None

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
        Install THIS session's usage list + log paths into the engine's
        shared/global state for the duration of an engine call, then restore
        the previous state. Safe because pushes are synchronous.
        """
        prev_records = llm.USAGE.records
        prev_label = llm.USAGE._label
        prev_prompt = llm.PROMPT_LOG_PATH
        prev_emb_log = getattr(self.embedder, "_log_path", None) if self.embedder else None

        # mutate the shared tracker in place so stream.py's `from .llm import USAGE`
        # reference sees this session's records too.
        llm.USAGE.records = self._usage_records
        llm.USAGE._label = self._usage_label
        llm.set_prompt_log_path(self._logs_dir / "prompts.jsonl")
        if self.embedder is not None:
            self.embedder.set_log_path(self._logs_dir / "embedding_calls.jsonl")
        try:
            yield
        finally:
            self._usage_records = llm.USAGE.records
            self._usage_label = llm.USAGE._label
            llm.USAGE.records = prev_records
            llm.USAGE._label = prev_label
            llm.PROMPT_LOG_PATH = prev_prompt
            if self.embedder is not None and prev_emb_log is not None:
                self.embedder.set_log_path(prev_emb_log)

    def _ensure_builder(self) -> StreamingBeliefBuilder:
        if self._builder is None:
            with self._engine():
                self._builder = StreamingBeliefBuilder(
                    client=self.client, model=self.model,
                    item_id=self.problem_id, out_dir=self.out_dir,
                    options=self.options, embedder=self.embedder,
                    item_meta=self.item_meta, max_tokens=self.max_tokens,
                )
        return self._builder

    # ------------------------------------------------------------- file I/O
    def _append_stream_log(self, record: Dict[str, Any]) -> None:
        with open(self._stream_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_trajectory(self, complete: bool) -> None:
        doc = {
            "problem_id": self.problem_id,
            "complete": complete,
            "n_turns": len(self._messages),
            "updated_at": _now(),
            # run.py-compatible: scripts/run.py can read this back
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
                "stage": stage, "finalized": self._finalized,
                "n_nodes": 0, "n_beliefs": 0, "n_decisions": 0,
                "nodes": [], "beliefs": [], "decisions": [],
                "relations": [], "merges": [], "sessions": [],
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
    def _ingest(self, role: str, content: str, date: Optional[str] = None) -> None:
        """Feed one assembled turn to the engine and record it for trajectory.json."""
        builder = self._ensure_builder()
        with self._engine():
            builder.ingest_turn(role, content, date=date)
        msg = {"role": role, "content": content}
        if date is not None:
            msg["date"] = date
        self._messages.append(msg)
        self._n_ingested += 1

    def _flush_buffer(self) -> None:
        if self._buf_parts:
            role = self._buf_role or "user"
            self._ingest(role, "".join(self._buf_parts), date=self._buf_date)
        self._buf_role = None
        self._buf_parts = []
        self._buf_date = None

    # ------------------------------------------------------------------ push
    def push(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one incoming dict. Returns the current belief-graph snapshot.
        On is_trajectory_end the trajectory is finalized (merge + result.json)
        and the returned snapshot is the COMPLETE graph.
        """
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
        role = (turn.get("role") or "user")
        content = turn.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)

        is_traj_end = bool(turn.get("is_trajectory_end", False))
        # a trajectory cannot end mid-message
        is_msg_end = True if is_traj_end else bool(turn.get("is_message_end", True))
        date = turn.get("date") or turn.get("session_date")

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
                print(f"  [warn] {self.problem_id}: role changed mid-message "
                      f"({self._buf_role!r} -> {role!r}); flushing buffered fragment",
                      file=sys.stderr)
                self._flush_buffer()
            self._buf_role = role
            self._buf_parts.append(content)
            if date is not None:
                self._buf_date = date
            return self._snapshot(stage="buffered")

        # 3) assemble + ingest one complete turn
        if self._buf_parts:
            if self._buf_role and self._buf_role != role:
                print(f"  [warn] {self.problem_id}: message-end role {role!r} differs "
                      f"from buffered role {self._buf_role!r}; using buffered role",
                      file=sys.stderr)
            role = self._buf_role or role
            content = "".join(self._buf_parts) + content
            date = date or self._buf_date
            self._buf_role = None
            self._buf_parts = []
            self._buf_date = None

        self._ingest(role, content, date=date)
        self._write_trajectory(complete=is_traj_end)
        snap = self._emit_snapshot(stage="turn")

        # 4) finalize on the last turn
        if is_traj_end:
            return self.finalize()
        return snap

    # -------------------------------------------------------------- finalize
    def finalize(self) -> Dict[str, Any]:
        """
        Run the session-end merge pass and write result.json. Safe to call
        explicitly if a producer forgot is_trajectory_end. Returns the FINAL
        graph snapshot with merges applied.
        """
        if self._finalized:
            return self._snapshot(stage="final")
        # flush any dangling fragment first
        self._flush_buffer()
        builder = self._ensure_builder()
        with self._engine():
            self._result = builder.finalize(
                extra_meta={"mode": "stream-online", "problem_id": self.problem_id},
                pricing=self.pricing,
            )
        self._finalized = True
        self._write_trajectory(complete=True)
        snap = self._emit_snapshot(stage="final")
        print(f"  [online] trajectory {self.problem_id!r} finalized: "
              f"{snap.get('n_nodes')} node(s), "
              f"{len(snap.get('relations', []))} relation(s), "
              f"{len(snap.get('merges', []))} merge(s) -> {self.out_dir/'result.json'}")
        return snap


# ===========================================================================
# Many trajectories  (multiplex by problem_id)
# ===========================================================================

class SessionManager:
    """
    Thin multiplexer over StreamingTrajectorySession, keyed by problem_id.

    Builds ONE shared chat client + embedder (so an in-process / local
    embedding model is loaded only once) and hands them to every per-trajectory
    session. Turns for different problem_ids may interleave freely.
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
        pricing: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.options = options or StreamOptions()
        self.pricing = pricing
        self.max_tokens: Optional[int] = None
        self._sessions: Dict[str, StreamingTrajectorySession] = {}

        injected = (client != "__from_config__") or (model is not None)
        if injected:
            self.client = None if client == "__from_config__" else client
            self.model = model or "model"
            self.embedder = None if embedder == "__from_config__" else embedder
        else:
            self._wire_from_config(config_path, model_key, embedding_key)

    # ---- config wiring (mirrors pipeline.run_input) -----------------------
    def _wire_from_config(self, config_path, model_key, embedding_key) -> None:
        cfg = load_config(config_path, model_key=model_key)
        self.client = make_client(cfg)
        self.model = cfg.get("model") or cfg.get("model_name") or "gpt-4o-mini"
        self.max_tokens = cfg.get("max_tokens")
        if self.pricing is None:
            self.pricing = cfg.get("pricing")
        masked = (cfg.get("api_key", "") or "")
        masked = (masked[:6] + "…" + masked[-3:]) if len(masked) > 10 else "***"
        print(f"[online] model={self.model}  base_url={cfg['base_url']}  api_key={masked}"
              + (f"  max_tokens={self.max_tokens}" if self.max_tokens else ""))

        emb_cfg = load_embedding_config(config_path, embedding_key=embedding_key)
        self.embedder = None
        if emb_cfg is not None:
            self.embedder = make_embedder(emb_cfg)
            if emb_cfg.get("provider") == "local":
                print(f"[online] embedding provider=local  model={emb_cfg['model']}  "
                      f"(weights load lazily on first use)")
            else:
                print(f"[online] embedding model={emb_cfg['model']}  base_url={emb_cfg['base_url']}")
        else:
            if self.options.use_clustering:
                print(f"[warn] --use-clustering needs an {embedding_key!r} config entry — "
                      f"clustering DISABLED", file=sys.stderr)
                self.options.use_clustering = False
            if self.options.merge_strategy == "embedding":
                print(f"[warn] merge-strategy=embedding needs an {embedding_key!r} entry — "
                      f"falling back to merge-strategy=llm", file=sys.stderr)
                self.options.merge_strategy = "llm"

    # ---- routing ----------------------------------------------------------
    def get_session(self, problem_id: str, *, create: bool = True) -> Optional[StreamingTrajectorySession]:
        pid = str(problem_id)
        sess = self._sessions.get(pid)
        if sess is None and create:
            sess = StreamingTrajectorySession(
                pid, client=self.client, model=self.model,
                output_root=self.output_root, options=self.options,
                embedder=self.embedder, pricing=self.pricing,
                max_tokens=self.max_tokens,
            )
            self._sessions[pid] = sess
        return sess

    def push(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        """Route one incoming dict to its trajectory and return the live graph."""
        pid = turn.get("problem_id")
        if pid is None:
            raise ValueError("each streamed turn must carry a 'problem_id'")
        return self.get_session(str(pid)).push(turn)

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
