"""Public SDK orchestration backed exclusively by :mod:`bcg.construct`."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bcg.construct import llm as construct_llm
from bcg.construct.online import StreamingTrajectorySession
from bcg.construct.pipeline import (
    BeliefGraphOptions,
    BeliefGraphRunPaths,
    BeliefGraphRunResult,
)
from bcg.construct.utils import new_run_id, save_json
from bcg.graph import BCG, BeliefPayload, BeliefSource, EvidenceExcerpt
from bcg.memory import BCGMemory
from bcg.utils import utc_now


@dataclass(slots=True)
class BCGRunner:
    """Own a belief run while delegating all construction to the v3 engine."""

    memory: BCGMemory
    llm: Any
    output_root: str | Path = ".bcg/runs"
    graph: BCG | None = field(default=None, init=False, repr=False)
    run_id: str | None = field(default=None, init=False)
    model: str | None = field(default=None, init=False, repr=False)
    max_tokens: int | None = field(default=None, init=False, repr=False)
    scenario: str = field(default="research", init=False)
    item_id: str = field(default="trajectory", init=False)
    options: BeliefGraphOptions | None = field(default=None, init=False, repr=False)
    embedder: Any | None = field(default=None, init=False, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    paths: BeliefGraphRunPaths | None = field(default=None, init=False, repr=False)
    trajectory: list[dict[str, Any]] = field(default_factory=list, init=False)
    _engine: StreamingTrajectorySession | None = field(
        default=None, init=False, repr=False
    )
    _session: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _sessions: list[dict[str, Any]] = field(default_factory=list, init=False)
    _active: bool = field(default=False, init=False, repr=False)
    _finalized: bool = field(default=False, init=False, repr=False)

    async def observe_trajectory(
        self,
        trajectory: list[dict[str, Any]],
        *,
        run_id: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        scenario: str = "research",
        item_id: str = "trajectory",
        evidence_mode: str = "sentence",
        use_split: bool = False,
        split_threshold: float = 0.6,
        split_min_sentences: int = 4,
        split_buffer: int = 0,
        merge_strategy: str = "off",
        merge_threshold: float = 0.86,
        incremental_merge: bool = True,
        incremental_merge_threshold: float = 0.8,
        verify_merge: bool = False,
        factor_similarity_threshold: float = 0.85,
        factor_input_confidence_threshold: float = 0.5,
        context_chars: int = 9000,
        io_context_chars: int = 6000,
        min_content_len: int = 0,
        min_segment_len: dict[str, int] | None = None,
        embedder: Any | None = None,
        confidence_config: Any | None = None,
        metadata: dict[str, Any] | None = None,
        options: BeliefGraphOptions | None = None,
    ) -> BeliefGraphRunResult:
        del min_segment_len, confidence_config
        self.begin_belief_run(
            run_id=run_id,
            model=model,
            max_tokens=max_tokens,
            scenario=scenario,
            item_id=item_id,
            evidence_mode=evidence_mode,
            use_split=use_split,
            split_threshold=split_threshold,
            split_min_sentences=split_min_sentences,
            split_buffer=split_buffer,
            merge_strategy=merge_strategy,
            merge_threshold=merge_threshold,
            incremental_merge=incremental_merge,
            incremental_merge_threshold=incremental_merge_threshold,
            verify_merge=verify_merge,
            factor_similarity_threshold=factor_similarity_threshold,
            factor_input_confidence_threshold=factor_input_confidence_threshold,
            context_chars=context_chars,
            io_context_chars=io_context_chars,
            min_content_len=min_content_len,
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
        evidence_mode: str = "sentence",
        use_split: bool = False,
        split_threshold: float = 0.6,
        split_min_sentences: int = 4,
        split_buffer: int = 0,
        merge_strategy: str = "off",
        merge_threshold: float = 0.86,
        incremental_merge: bool = True,
        incremental_merge_threshold: float = 0.8,
        verify_merge: bool = False,
        factor_similarity_threshold: float = 0.85,
        factor_input_confidence_threshold: float = 0.5,
        context_chars: int = 9000,
        io_context_chars: int = 6000,
        min_content_len: int = 0,
        min_segment_len: dict[str, int] | None = None,
        embedder: Any | None = None,
        confidence_config: Any | None = None,
        metadata: dict[str, Any] | None = None,
        options: BeliefGraphOptions | None = None,
    ) -> str:
        del min_segment_len, confidence_config
        if self._active:
            raise RuntimeError("A belief run is already active")
        self.run_id = run_id or new_run_id()
        self.model = model or _model_name(self.llm)
        self.max_tokens = max_tokens
        self.scenario = scenario
        self.item_id = item_id
        self.options = options or BeliefGraphOptions(
            evidence_mode=evidence_mode,
            use_split=use_split,
            split_threshold=split_threshold,
            split_min_sentences=split_min_sentences,
            split_buffer=split_buffer,
            merge_strategy=merge_strategy,
            merge_threshold=merge_threshold,
            incremental_merge=incremental_merge,
            incremental_merge_threshold=incremental_merge_threshold,
            verify_merge=verify_merge,
            factor_similarity_threshold=factor_similarity_threshold,
            factor_input_confidence_threshold=factor_input_confidence_threshold,
            context_chars=context_chars,
            io_context_chars=io_context_chars,
            min_content_len=min_content_len,
        )
        self.embedder = embedder
        self.metadata = dict(metadata or {})
        self.paths = _run_paths(Path(self.output_root), self.run_id)
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.graph = BCG(
            metadata={
                "run_id": self.run_id,
                "namespace": self.memory.namespace,
                "engine": "bcg.construct",
            }
        )
        self.memory.graph = self.graph
        self.trajectory = []
        self._sessions = []
        self._session = None
        self._engine = StreamingTrajectorySession(
            self.run_id,
            client=_ConstructClientAdapter(self.llm),
            model=self.model,
            output_root=Path(self.output_root),
            options=self.options.to_stream_options(),
            embedder=self.embedder,
            max_tokens=self.max_tokens,
            item_meta={"scenario": self.scenario, "item_id": self.item_id},
            extra_meta={"metadata": self.metadata},
        )
        self._active = True
        self._finalized = False
        return self.run_id

    def start_session(self, session_id: str, date: str | None = None) -> None:
        self._require_active()
        if self._session is not None:
            raise RuntimeError("A session is already active")
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
            raise RuntimeError("finalize() called twice")
        if self._session is not None:
            await self.end_session()
        snapshot = await _invoke_engine(self.llm, self._engine.finalize)
        native = dict(self._engine.result or {})
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
        memory = _memory_document(
            self.graph,
            native=native,
            run_id=self.run_id,
            trajectory=self.trajectory,
            options=self.options,
            metadata=self.metadata,
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
                "engine": "bcg.construct",
            },
        )
        self.memory.graph = self.graph

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("No active belief run. Call begin_belief_run() first.")


class _ConstructClientAdapter:
    """Expose an OpenAI chat-completions shape over the public SDK LLM client."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs: Any) -> Any:
        messages = list(kwargs.get("messages") or [])
        prompt = str(messages[-1].get("content") or "") if messages else ""
        model = kwargs.get("model")
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        if hasattr(self.llm, "generate"):
            call_kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if model and _accepts_keyword(self.llm.generate, "model"):
                call_kwargs["model"] = model
            response = _resolve_sync(self.llm.generate(messages, **call_kwargs))
            content = _response_content(response)
            usage = getattr(response, "usage", {})
        elif hasattr(self.llm, "generate_text"):
            call_kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if _accepts_keyword(self.llm.generate_text, "label"):
                tracker = construct_llm.current_usage_tracker()
                call_kwargs["label"] = getattr(tracker, "_label", "unlabeled")
            if model and _accepts_keyword(self.llm.generate_text, "model"):
                call_kwargs["model"] = model
            content = str(_resolve_sync(self.llm.generate_text(prompt, **call_kwargs)))
            usage = {}
        else:
            raise TypeError("BCGRunner requires an LLM with generate or generate_text")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=_chat_usage(usage),
        )


def _resolve_sync(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


async def _invoke_engine(llm: Any, func: Any, *args: Any) -> Any:
    """Keep sync test/custom clients inline; isolate async SDK clients in a worker."""
    generate = getattr(llm, "generate", None) or getattr(llm, "generate_text", None)
    if generate is not None and inspect.iscoroutinefunction(generate):
        return await asyncio.to_thread(func, *args)
    return func(*args)


def _response_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    if isinstance(response, dict):
        return str(response.get("content") or response.get("text") or "")
    return str(response)


def _chat_usage(usage: Any) -> Any:
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    usage = usage if isinstance(usage, dict) else {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = int(prompt) + int(completion)
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


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
    options: BeliefGraphOptions,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    base = graph.to_memory_dict()
    all_nodes = graph.belief_dicts()
    beliefs = [node for node in all_nodes if node.get("node_type") == "belief"]
    decisions = [node for node in all_nodes if node.get("node_type") == "decision"]
    base.update(
        {
            "schema": "bcg.memory.v2",
            "engine": "bcg.construct",
            "run_id": run_id,
            "prompt_name": native.get("prompt_name", "construct_beliefs"),
            "model": native.get("model"),
            "generated_at": native.get("generated_at") or utc_now().isoformat(),
            "mode": "stream",
            "options": options.to_dict(),
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


def _accepts_keyword(func: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


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
