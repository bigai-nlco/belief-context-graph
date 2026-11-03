"""Public SDK orchestration backed by :mod:`bcg.construct` (unified or hybrid)."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bcg.construct.backends import resolve_backend
from bcg.core.client_adapter import ConstructClientAdapter
from bcg.core.contracts import (
    BeliefGraphRunPaths,
    BeliefGraphRunResult,
    ConstructBackend,
    ConstructSession,
    RunOptions,
    SessionSpec,
)
from bcg.core.errors import BCGUsageError
from bcg.core.graph import BCG, BeliefPayload, BeliefSource, EvidenceExcerpt
from bcg.core.memory import BCGMemory
from bcg.core.utils import new_run_id, save_json, utc_now

# Private compatibility aliases retained for callers that imported test helpers.
_ConstructClientAdapter = ConstructClientAdapter
_resolve_backend = resolve_backend


@dataclass(slots=True)
class BCGRunner:
    """Own a belief run while delegating all construction to either backend."""

    memory: BCGMemory
    llm: Any
    output_root: str | Path = ".bcg/runs"
    backend: str = "unified"
    backend_adapter: ConstructBackend | None = field(default=None, repr=False)
    graph: BCG | None = field(default=None, init=False, repr=False)
    run_id: str | None = field(default=None, init=False)
    model: str | None = field(default=None, init=False, repr=False)
    max_tokens: int | None = field(default=None, init=False, repr=False)
    scenario: str = field(default="research", init=False)
    item_id: str = field(default="trajectory", init=False)
    # Either a BeliefGraphOptions (unified) or a hybrid StreamOptions,
    # depending on which backend built this run.
    options: Any = field(default=None, init=False, repr=False)
    embedder: Any | None = field(default=None, init=False, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    paths: BeliefGraphRunPaths | None = field(default=None, init=False, repr=False)
    trajectory: list[dict[str, Any]] = field(default_factory=list, init=False)
    _backend: ConstructBackend = field(init=False, repr=False)
    _engine: ConstructSession | None = field(default=None, init=False, repr=False)
    _session: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _sessions: list[dict[str, Any]] = field(default_factory=list, init=False)
    _active: bool = field(default=False, init=False, repr=False)
    _finalized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._backend = self.backend_adapter or resolve_backend(self.backend)
        self.backend = self._backend.name

    async def observe_trajectory(
        self,
        trajectory: list[dict[str, Any]],
        *,
        run_id: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        scenario: str = "research",
        item_id: str = "trajectory",
        backend: str | None = None,
        evidence_mode: str = "sentence",
        incremental_merge: bool = True,
        incremental_merge_threshold: float = 0.86,
        verify_merge: bool = False,
        context_chars: int = 100000,
        io_context_chars: int = 6000,
        min_content_len: int = 0,
        belief_graph_config: dict[str, Any] | None = None,
        embedder: Any | None = None,
        metadata: dict[str, Any] | None = None,
        options: Any | None = None,
    ) -> BeliefGraphRunResult:
        self.begin_belief_run(
            run_id=run_id,
            model=model,
            max_tokens=max_tokens,
            scenario=scenario,
            item_id=item_id,
            backend=backend,
            evidence_mode=evidence_mode,
            incremental_merge=incremental_merge,
            incremental_merge_threshold=incremental_merge_threshold,
            verify_merge=verify_merge,
            context_chars=context_chars,
            io_context_chars=io_context_chars,
            min_content_len=min_content_len,
            belief_graph_config=belief_graph_config,
            embedder=embedder,
            metadata=metadata,
            options=options,
        )
        self.start_session("session-0", None)
        for turn in trajectory:
            await self.observe_turn(
                str(turn.get("role") or "user"),
                str(turn.get("content") or ""),
                date=turn.get("date") or turn.get("session_date"),
                has_answer=turn.get("has_answer"),
            )
        await self.end_session()
        return await self.finalize()

    def begin_belief_run(
        self,
        *,
        run_id: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        scenario: str = "research",
        item_id: str = "trajectory",
        backend: str | None = None,
        # --- unified-only knobs, mapped onto BeliefGraphOptions ---
        evidence_mode: str = "sentence",
        incremental_merge: bool = True,
        incremental_merge_threshold: float = 0.86,
        verify_merge: bool = False,
        context_chars: int = 100000,
        io_context_chars: int = 6000,
        min_content_len: int = 0,
        # --- hybrid-only knob: the `belief_graph` section of model_config.json
        # (see bcg/model_config.example.json), applied via StreamOptions
        # .apply_belief_graph_config(). Omit it to fall back to that
        # StreamOptions' own built-in defaults.
        belief_graph_config: dict[str, Any] | None = None,
        embedder: Any | None = None,
        metadata: dict[str, Any] | None = None,
        options: Any | None = None,
    ) -> str:
        if self._active:
            raise BCGUsageError("A belief run is already active")
        if backend is not None:
            self._backend = _resolve_backend(backend)
            self.backend = self._backend.name

        self.run_id = run_id or new_run_id()
        self.model = model or _model_name(self.llm)
        self.max_tokens = max_tokens
        self.scenario = scenario
        self.item_id = item_id

        if options is not None:
            self.options = options
        else:
            self.options = self._backend.build_options(
                RunOptions(
                    evidence_mode=evidence_mode,
                    incremental_merge=incremental_merge,
                    incremental_merge_threshold=incremental_merge_threshold,
                    verify_merge=verify_merge,
                    context_chars=context_chars,
                    io_context_chars=io_context_chars,
                    min_content_len=min_content_len,
                ),
                belief_graph_config=belief_graph_config,
            )

        self.embedder = embedder
        self.metadata = dict(metadata or {})
        self.paths = _run_paths(Path(self.output_root), self.run_id)
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.graph = BCG(
            metadata={
                "run_id": self.run_id,
                "namespace": self.memory.namespace,
                "engine": f"bcg.construct.{self._backend.name}",
            }
        )
        self.memory.graph = self.graph
        self.trajectory = []
        self._sessions = []
        self._session = None

        self._engine = self._backend.create_session(
            SessionSpec(
                run_id=self.run_id,
                llm=self.llm,
                model=self.model,
                output_root=Path(self.output_root),
                options=self._backend.session_options(self.options),
                embedder=self.embedder,
                max_tokens=self.max_tokens,
                item_meta={"scenario": self.scenario, "item_id": self.item_id},
                extra_meta={"metadata": self.metadata},
            )
        )
        self._active = True
        self._finalized = False
        return self.run_id

    def start_session(self, session_id: str, date: str | None = None) -> None:
        self._require_active()
        if self._session is not None:
            raise BCGUsageError("A session is already active")
        self._session = {
            "session_id": str(session_id),
            "session_index": len(self._sessions),
            "date": date,
            "start_turn": len(self.trajectory),
            "n_turns": 0,
        }

    async def observe_turn(
        self,
        role: str,
        content: str,
        *,
        date: str | None = None,
        has_answer: bool | None = None,
    ) -> dict[str, Any]:
        self._require_active()
        assert self._engine is not None
        if self._session is None:
            self.start_session(f"session-{len(self._sessions)}", date)
        assert self._session is not None
        effective_date = date or self._session.get("date")
        turn = {
            "problem_id": self.run_id,
            "role": role,
            "content": content,
            "date": effective_date,
        }
        if has_answer is not None:
            turn["has_answer"] = bool(has_answer)
        snapshot = await _invoke_engine(self.llm, self._engine.push, turn)
        self.trajectory.append(
            {
                "role": role,
                "content": content,
                "date": effective_date,
                "has_answer": has_answer,
                "session_id": self._session["session_id"],
                "session_index": self._session["session_index"],
                "turn_index": len(self.trajectory),
            }
        )
        self._session["n_turns"] += 1
        self._sync_graph(snapshot)
        return snapshot

    async def end_session(self) -> dict[str, Any] | None:
        self._require_active()
        if self._session is None:
            return None
        summary = dict(self._session)
        summary["end_turn"] = len(self.trajectory)
        self._sessions.append(summary)
        self._session = None
        if self.graph is not None:
            self.graph.sessions = list(self._sessions)
        return summary

    async def finalize(self) -> BeliefGraphRunResult:
        self._require_active()
        assert self._engine is not None
        assert self.paths is not None
        assert self.run_id is not None
        assert self.options is not None
        if self._finalized:
            raise BCGUsageError("finalize() called twice")
        if self._session is not None:
            await self.end_session()
        snapshot = await _invoke_engine(self.llm, self._backend.finalize, self._engine)
        native = self._backend.result(self._engine)
        self._sync_graph(snapshot)
        assert self.graph is not None
        self.graph.sessions = list(self._sessions)
        self.graph.metadata.update(
            {
                "scenario": self.scenario,
                "item_id": self.item_id,
                "generated_at": native.get("generated_at"),
            }
        )
        options_dict = self._backend.serialize_options(self.options)
        memory = _memory_document(
            self.graph,
            native=native,
            run_id=self.run_id,
            trajectory=self.trajectory,
            options_dict=options_dict,
            metadata=self.metadata,
            engine=f"bcg.construct.{self._backend.name}",
        )
        token_usage = dict(native.get("token_usage") or {})
        counts = dict(memory["counts"])
        _write_compatibility_outputs(
            self.paths,
            graph=self.graph,
            memory=memory,
            native=native,
        )
        self.memory.graph = self.graph
        self._finalized = True
        self._active = False
        return BeliefGraphRunResult(
            run_id=self.run_id,
            graph=self.graph,
            memory=memory,
            output_paths=self.paths,
            token_usage=token_usage,
            counts=counts,
            construct_result=native,
        )

    def _sync_graph(self, snapshot: dict[str, Any]) -> None:
        self.graph = _bcg_from_construct(
            snapshot,
            sessions=self._sessions,
            metadata={
                "run_id": self.run_id,
                "namespace": self.memory.namespace,
                "engine": f"bcg.construct.{self._backend.name}",
            },
        )
        self.memory.graph = self.graph

    def _require_active(self) -> None:
        if not self._active:
            raise BCGUsageError("No active belief run. Call begin_belief_run() first.")


async def _invoke_engine(llm: Any, func: Any, *args: Any) -> Any:
    """Keep sync test/custom clients inline; isolate async SDK clients in a worker."""
    generate = getattr(llm, "generate", None) or getattr(llm, "generate_text", None)
    if generate is not None and inspect.iscoroutinefunction(generate):
        return await asyncio.to_thread(func, *args)
    return func(*args)


def _source(raw: dict[str, Any], role: str) -> BeliefSource:
    turn_id = int(raw.get("turn_id", -1))
    return BeliefSource(
        type=role,
        role=role,
        trajectory_index=turn_id,
        segment_index=0,
        segment_type=role,
        item_id=raw.get("item_id"),
        turn_index=turn_id,
        session_date=raw.get("date"),
        has_answer=raw.get("has_answer"),
        **{
            key: value
            for key, value in raw.items()
            if key not in {"has_answer", "item_id", "role", "type"}
        },
    )


def _bcg_from_construct(
    snapshot: dict[str, Any],
    *,
    sessions: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> BCG:
    graph = BCG(
        merges=list(snapshot.get("merges") or []),
        sessions=list(sessions),
        evidence=list(snapshot.get("evidence") or []),
        factors=list(snapshot.get("factors") or []),
        metadata=dict(metadata),
    )
    evidence_by_id = {
        int(item["id"]): item
        for item in graph.evidence
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    for node in snapshot.get("nodes") or []:
        if not isinstance(node, dict) or not isinstance(node.get("id"), int):
            continue
        node_type = str(node.get("node_type") or "belief")
        text = str(node.get("belief") or node.get("decision") or "").strip()
        if not text:
            continue
        role = str(
            node.get("role") or (node.get("source") or {}).get("role") or "unknown"
        )
        source = _source(dict(node.get("source") or {}), role)
        evidence = []
        for evidence_id in node.get("evidence_ids") or []:
            raw_evidence = evidence_by_id.get(int(evidence_id))
            if raw_evidence is None or not raw_evidence.get("text"):
                continue
            evidence.append(
                EvidenceExcerpt(
                    text=str(raw_evidence["text"]),
                    start=raw_evidence.get("start"),
                    end=raw_evidence.get("end"),
                    match=raw_evidence.get("match"),
                    via=raw_evidence.get("via"),
                    source=_source(
                        dict(raw_evidence.get("source") or {}),
                        str(raw_evidence.get("role") or role),
                    ),
                )
            )
        payload = BeliefPayload(
            id=int(node["id"]),
            node_type="decision" if node_type == "decision" else "belief",
            belief=text,
            decision=str(node.get("decision")) if node.get("decision") else None,
            stance=node.get("stance") or "asserted",
            layer="io" if role in {"user", "tool", "function"} else "reasoning",
            role=role,
            entities=list(node.get("entities") or []),
            source=source,
            evidence_ids=[int(value) for value in node.get("evidence_ids") or []],
            factor_ids=[int(value) for value in node.get("factor_ids") or []],
            supporting_excerpts=list(node.get("supporting_excerpts") or []),
            evidence=evidence,
            event_time=node.get("event_time"),
            time_text=node.get("time_text"),
            merged_from=[int(value) for value in node.get("merged_from") or []],
            belief_original=node.get("belief_original"),
            confidence=float(node.get("confidence", 1.0)),
            initial_confidence=node.get("initial_confidence"),
            evidence_confidence=float(node.get("evidence_confidence", 0.0)),
            factor_confidence=float(node.get("factor_confidence", 0.0)),
            confidence_history=list(node.get("confidence_history") or []),
            decision_history=[
                int(value) for value in node.get("decision_history") or []
            ],
            metadata={"construct": dict(node)},
        )
        graph.add_belief(payload)
    graph.add_relations(list(snapshot.get("relations") or []))
    return graph


def _memory_document(
    graph: BCG,
    *,
    native: dict[str, Any],
    run_id: str,
    trajectory: list[dict[str, Any]],
    options_dict: dict[str, Any],
    metadata: dict[str, Any],
    engine: str,
) -> dict[str, Any]:
    base = graph.to_memory_dict()
    all_nodes = graph.belief_dicts()
    beliefs = [node for node in all_nodes if node.get("node_type") == "belief"]
    decisions = [node for node in all_nodes if node.get("node_type") == "decision"]
    base.update(
        {
            "schema": "bcg.memory.v2",
            "engine": engine,
            "run_id": run_id,
            "prompt_name": native.get("prompt_name", "construct_beliefs"),
            "model": native.get("model"),
            "generated_at": native.get("generated_at") or utc_now().isoformat(),
            "mode": "stream",
            "options": options_dict,
            "trajectory": trajectory,
            "nodes": all_nodes,
            "beliefs": beliefs,
            "all_beliefs": beliefs,
            "decisions": decisions,
            "io_beliefs": [node for node in beliefs if node.get("layer") == "io"],
            "reasoning_beliefs": [
                node for node in beliefs if node.get("layer") == "reasoning"
            ],
            "token_usage": native.get("token_usage") or {},
            "timing": native.get("timing") or {},
            "metadata": metadata,
        }
    )
    base["counts"].update(
        {
            "nodes": len(all_nodes),
            "beliefs": len(beliefs),
            "decisions": len(decisions),
            "relations": len(base["relations"]),
        }
    )
    return base


def _write_compatibility_outputs(
    paths: BeliefGraphRunPaths,
    *,
    graph: BCG,
    memory: dict[str, Any],
    native: dict[str, Any],
) -> None:
    save_json(graph.model_dump(mode="json"), paths.graph)
    save_json(memory, paths.memory)
    save_json(native.get("token_usage") or {}, paths.token_usage)
    save_json(
        {"schema": "bcg.segments.v2", "trajectory": memory["trajectory"]},
        paths.segments,
    )
    save_json({"beliefs": memory["io_beliefs"]}, paths.io_beliefs)
    save_json({"beliefs": memory["reasoning_beliefs"]}, paths.reasoning_beliefs)
    save_json({"relations": memory["forward_relations"]}, paths.forward_relations)
    save_json({"relations": memory["backward_relations"]}, paths.backward_relations)
    save_json({"merges": memory["merges"]}, paths.merges)


def _model_name(llm: Any) -> str:
    config = getattr(llm, "config", None)
    model = getattr(config, "model", None)
    return model if isinstance(model, str) and model else "unknown"


def _run_paths(output_root: Path, run_id: str) -> BeliefGraphRunPaths:
    run_dir = output_root / run_id
    artifacts = run_dir / "artifacts"
    return BeliefGraphRunPaths(
        run_dir=run_dir,
        artifacts_dir=artifacts,
        graph=run_dir / "graph.json",
        memory=run_dir / "memory.json",
        token_usage=run_dir / "token_usage.json",
        events=run_dir / "events.jsonl",
        segments=artifacts / "segments.json",
        io_beliefs=artifacts / "io_beliefs.json",
        reasoning_beliefs=artifacts / "reasoning_beliefs.json",
        forward_relations=artifacts / "forward_relations.json",
        backward_relations=artifacts / "backward_relations.json",
        merges=artifacts / "merges.json",
        result=run_dir / "result.json",
        final_graph=run_dir / "final_graph.json",
        trajectory=run_dir / "trajectory.json",
        graph_stream=run_dir / "belief_graph.jsonl",
    )


__all__ = ["BCGRunner"]
