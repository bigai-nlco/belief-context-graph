"""Run orchestration for incremental belief graph construction."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bcg.belief_graph.confidence import (
    ConfidenceConfig,
    apply_relations_incremental,
    init_belief_confidence,
)
from bcg.belief_graph.constants import (
    DEFAULT_MIN_SEGMENT_LEN,
    IO_TYPES,
    REASONING_TYPES,
    SEGMENT_SOURCE_TYPES,
)
from bcg.belief_graph.evidence import (
    evidence_from_excerpt,
    evidence_from_sentence,
    source_descriptor,
)
from bcg.belief_graph.extraction import (
    extract_segment,
    format_graph_context,
    format_io_context,
)
from bcg.belief_graph.linking import (
    link_backward_session,
    link_forward_incremental,
)
from bcg.belief_graph.merge import run_merge_pass
from bcg.belief_graph.pipeline import (
    BeliefGraphOptions,
    BeliefGraphRunPaths,
    BeliefGraphRunResult,
)
from bcg.belief_graph.segment import (
    Segment,
    segment_turn,
    summarize_segments,
)
from bcg.belief_graph.split import cluster_sentences, split_sentences
from bcg.belief_graph.utils import count_by, new_run_id, save_json
from bcg.graph import BCG, BeliefPayload
from bcg.llm import LLMResponse, TokenUsageTracker
from bcg.memory import BCGMemory
from bcg.utils import maybe_await, utc_now


@dataclass(slots=True)
class BCGRunner:
    """Owns belief-run lifecycle, run ids, sessions, and output paths."""

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
    confidence_config: ConfidenceConfig | None = field(
        default=None,
        init=False,
        repr=False,
    )
    metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    paths: BeliefGraphRunPaths | None = field(default=None, init=False, repr=False)
    usage: TokenUsageTracker | None = field(default=None, init=False, repr=False)
    generator: _TrackedTextGenerator | None = field(
        default=None,
        init=False,
        repr=False,
    )
    trajectory: list[dict[str, Any]] = field(default_factory=list, init=False)
    segment_reports: list[dict[str, Any]] = field(default_factory=list, init=False)
    forward_reports: list[dict[str, Any]] = field(default_factory=list, init=False)
    backward_reports: list[dict[str, Any]] = field(default_factory=list, init=False)
    _session: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _session_count: int = field(default=0, init=False, repr=False)
    _flat_turn_index: int = field(default=0, init=False, repr=False)
    _finalized: bool = field(default=False, init=False, repr=False)
    _started_at: Any | None = field(default=None, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)

    async def observe_trajectory(
        self,
        trajectory: list[dict[str, Any]],
        *,
        run_id: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        scenario: str = "research",
        item_id: str = "trajectory",
        use_split: bool = False,
        split_threshold: float = 0.6,
        split_min_sentences: int = 4,
        split_buffer: int = 0,
        merge_strategy: str = "off",
        merge_threshold: float = 0.86,
        context_chars: int = 9000,
        io_context_chars: int = 6000,
        min_segment_len: dict[str, int] | None = None,
        embedder: Any | None = None,
        confidence_config: ConfidenceConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BeliefGraphRunResult:
        """Run the belief-graph pipeline over a complete trajectory."""

        self.memory.graph = BCG(metadata={"namespace": self.memory.namespace})
        self.begin_belief_run(
            run_id=run_id,
            model=model,
            max_tokens=max_tokens,
            scenario=scenario,
            item_id=item_id,
            use_split=use_split,
            split_threshold=split_threshold,
            split_min_sentences=split_min_sentences,
            split_buffer=split_buffer,
            merge_strategy=merge_strategy,
            merge_threshold=merge_threshold,
            context_chars=context_chars,
            io_context_chars=io_context_chars,
            min_segment_len=min_segment_len,
            embedder=embedder,
            confidence_config=confidence_config,
            metadata=metadata,
        )
        self.start_session(session_id="session_0")
        for message in trajectory:
            await self.observe_turn(
                str(message.get("role") or "user"),
                str(message.get("content") or ""),
                has_answer=message.get("has_answer"),
            )
        return await self.finalize()

    def begin_belief_run(
        self,
        *,
        run_id: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        scenario: str = "research",
        item_id: str = "trajectory",
        use_split: bool = False,
        split_threshold: float = 0.6,
        split_min_sentences: int = 4,
        split_buffer: int = 0,
        merge_strategy: str = "off",
        merge_threshold: float = 0.86,
        context_chars: int = 9000,
        io_context_chars: int = 6000,
        min_segment_len: dict[str, int] | None = None,
        embedder: Any | None = None,
        confidence_config: ConfidenceConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BCGRunner:
        """Start an incremental belief graph run."""

        segment_lengths = dict(DEFAULT_MIN_SEGMENT_LEN)
        segment_lengths.update(min_segment_len or {})
        self.run_id = run_id or new_run_id()
        self.graph = self.memory._graph()
        self.graph.metadata.update(
            {"namespace": self.memory.namespace, "run_id": self.run_id}
        )
        self.model = model
        self.max_tokens = max_tokens
        self.scenario = scenario
        self.item_id = item_id
        self.options = BeliefGraphOptions(
            use_split=use_split,
            split_threshold=split_threshold,
            split_min_sentences=split_min_sentences,
            split_buffer=split_buffer,
            merge_strategy=merge_strategy,
            merge_threshold=merge_threshold,
            context_chars=context_chars,
            io_context_chars=io_context_chars,
            min_segment_len=segment_lengths,
        )
        self.embedder = embedder
        self.confidence_config = confidence_config or self.memory.confidence_config
        self.metadata = {"namespace": self.memory.namespace, **(metadata or {})}
        self.paths = _run_paths(Path(self.output_root), self.run_id)
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.paths.events.write_text("", encoding="utf-8")
        self.usage = TokenUsageTracker()
        self.generator = _TrackedTextGenerator(
            self.llm,
            usage=self.usage,
            model=self.model,
        )
        self.trajectory = []
        self.segment_reports = []
        self.forward_reports = []
        self.backward_reports = []
        self._session = None
        self._session_count = 0
        self._flat_turn_index = 0
        self._finalized = False
        self._started_at = utc_now()
        self._active = True
        return self

    def start_session(
        self,
        session_id: str | None = None,
        session_date: str | None = None,
    ) -> dict[str, Any]:
        """Start an incremental memory session."""

        self._require_active()
        assert self.graph is not None
        if self._session is not None:
            raise RuntimeError("end the active session before starting another")
        session_index = self._session_count
        self._session_count += 1
        self._session = {
            "session_id": session_id or f"session_{session_index}",
            "session_index": session_index,
            "date": session_date,
            "start_belief_id": self.graph.next_belief_id,
            "start_flat_turn": self._flat_turn_index,
            "n_turns": 0,
            "forward_added": 0,
        }
        event = {
            "session_id": self._session["session_id"],
            "session_index": session_index,
            "date": session_date,
        }
        self._event("session_start", event)
        return event

    async def observe_turn(
        self,
        role: str,
        content: str,
        *,
        has_answer: bool | None = None,
    ) -> dict[str, Any]:
        """Ingest one turn into the active incremental belief run."""

        self._require_active()
        assert self.graph is not None
        assert self.generator is not None
        assert self.options is not None
        if self._session is None:
            self.start_session()
        assert self._session is not None
        session = self._session
        turn_index = int(session["n_turns"])
        flat_index = self._flat_turn_index
        content = content or ""
        trajectory_entry: dict[str, Any] = {
            "role": role,
            "content": content,
            "session_id": session["session_id"],
            "session_index": session["session_index"],
            "session_date": session["date"],
            "turn_index": turn_index,
        }
        if has_answer is not None:
            trajectory_entry["has_answer"] = bool(has_answer)
        self.trajectory.append(trajectory_entry)

        segments = segment_turn(
            turn_index,
            role,
            content,
            trajectory_index=flat_index,
        )
        new_beliefs: list[dict[str, Any]] = []
        turn_reports: list[dict[str, Any]] = []
        for segment in segments:
            report = await self._extract_segment(
                segment,
                turn_content=content,
                session=session,
                flat_index=flat_index,
                has_answer=has_answer,
            )
            turn_reports.append(report)
            new_beliefs.extend(report.pop("_beliefs", []))

        existing = [
            belief
            for belief in self.graph.belief_dicts()
            if int(belief["id"]) not in {int(item["id"]) for item in new_beliefs}
        ]
        forward_result = await link_forward_incremental(
            self.generator,
            existing,
            new_beliefs,
            max_tokens=self.max_tokens,
            label=f"s{session['session_index']}.t{turn_index}.forward",
        )
        forward_added = self.graph.add_relations(forward_result.relations)
        session["forward_added"] += forward_added
        self.forward_reports.append(
            {
                "session_index": session["session_index"],
                "turn_index": turn_index,
                "trajectory_index": flat_index,
                "relations": forward_result.relations,
                "raw_output": forward_result.raw_output,
                "skipped": forward_result.skipped,
                "skip_reason": forward_result.skip_reason,
            }
        )

        session["n_turns"] += 1
        self._flat_turn_index += 1
        event = {
            "session_index": session["session_index"],
            "turn_index": turn_index,
            "trajectory_index": flat_index,
            "role": role,
            "content_chars": len(content),
            "segments": summarize_segments(segments),
            "segment_reports": turn_reports,
            "new_belief_ids": [belief["id"] for belief in new_beliefs],
            "forward_added": forward_added,
            "forward_skip_reason": forward_result.skip_reason,
        }
        self._event("turn", event)
        return event

    async def end_session(self) -> dict[str, Any] | None:
        """End the active incremental session and apply backward updates."""

        self._require_active()
        assert self.graph is not None
        assert self.generator is not None
        assert self.options is not None
        session = self._session
        if session is None:
            return None
        session_index = int(session["session_index"])
        start_belief_id = int(session["start_belief_id"])
        active = self.graph.belief_dicts()
        session_beliefs = [
            belief for belief in active if int(belief["id"]) >= start_belief_id
        ]
        earlier = [belief for belief in active if int(belief["id"]) < start_belief_id]
        backward_result = await link_backward_session(
            self.generator,
            earlier,
            session_beliefs,
            max_tokens=self.max_tokens,
            label=f"s{session_index}.backward",
        )
        backward_added = self.graph.add_relations(backward_result.relations)
        by_id = {
            int(belief["id"]): belief
            for belief in self.graph.belief_dicts()
            if "id" in belief
        }
        confidence_updates = apply_relations_incremental(
            by_id,
            backward_result.relations,
            config=self.confidence_config,
        )
        for belief in by_id.values():
            self.graph.update_belief(belief)

        merge_result = await run_merge_pass(
            graph=self.graph,
            strategy=self.options.merge_strategy,
            generator=self.generator,
            embedder=self.embedder,
            threshold=self.options.merge_threshold,
            max_tokens=self.max_tokens,
            pass_label=f"session_{session_index:02d}",
            log_dir=None,
        )
        summary = {
            "session_id": session["session_id"],
            "session_index": session_index,
            "date": session["date"],
            "n_turns": session["n_turns"],
            "new_beliefs": len(session_beliefs),
            "forward_added": session["forward_added"],
            "backward_relations": backward_added,
            "backward_skip_reason": backward_result.skip_reason,
            "confidence_updates": confidence_updates,
            "merges_applied": (merge_result.applied or []),
        }
        self.backward_reports.append(
            {
                "session_index": session_index,
                "relations": backward_result.relations,
                "raw_output": backward_result.raw_output,
                "skipped": backward_result.skipped,
                "skip_reason": backward_result.skip_reason,
            }
        )
        self.graph.sessions.append(summary)
        self._event("session_end", summary)
        self._session = None
        return summary

    async def finalize(self) -> BeliefGraphRunResult:
        """Finalize the active incremental run and write local outputs."""

        self._require_active()
        assert self.graph is not None
        assert self.generator is not None
        assert self.options is not None
        assert self.paths is not None
        assert self.run_id is not None
        assert self.usage is not None
        assert self._started_at is not None
        if self._finalized:
            raise RuntimeError("finalize() called twice")
        if self._session is not None:
            await self.end_session()
        self._finalized = True

        ended_at = utc_now()
        beliefs = self.graph.belief_dicts()
        forward_relations = self.graph.relation_dicts("informs")
        backward_relations = [
            relation.model_dump(mode="json")
            for relation in self.graph.relations()
            if relation.type in {"confirms", "contradicts", "extends"}
        ]
        io_beliefs = [belief for belief in beliefs if belief.get("layer") == "io"]
        reasoning_beliefs = [
            belief for belief in beliefs if belief.get("layer") == "reasoning"
        ]
        counts = _counts(beliefs, forward_relations, backward_relations, self.graph)
        memory = {
            "schema": "bcg.memory.v1",
            "run_id": self.run_id,
            "prompt_name": "construct_beliefs",
            "model": self.generator.model,
            "scenario": self.scenario,
            "item_id": self.item_id,
            "generated_at": ended_at.isoformat(),
            "mode": "incremental",
            "options": self.options.to_dict(),
            "timing": {
                "start": self._started_at.isoformat(),
                "end": ended_at.isoformat(),
                "duration_seconds": (ended_at - self._started_at).total_seconds(),
            },
            "trajectory": self.trajectory,
            "sessions": list(self.graph.sessions),
            "beliefs": beliefs,
            "all_beliefs": beliefs,
            "io_beliefs": io_beliefs,
            "reasoning_beliefs": reasoning_beliefs,
            "forward_relations": forward_relations,
            "backward_relations": backward_relations,
            "merges": list(self.graph.merges),
            "counts": counts,
            "metadata": self.metadata,
            "artifacts": {
                key: value
                for key, value in self.paths.to_dict().items()
                if key
                in {
                    "segments",
                    "io_beliefs",
                    "reasoning_beliefs",
                    "forward_relations",
                    "backward_relations",
                    "merges",
                }
            },
        }
        token_usage = self.usage.to_dict()
        save_json(self.graph.model_dump(mode="json"), self.paths.graph)
        save_json(memory, self.paths.memory)
        save_json(token_usage, self.paths.token_usage)
        save_json(
            {
                "schema": "bcg.segments.v1",
                "run_id": self.run_id,
                "trajectory": self.trajectory,
                "segments": self.segment_reports,
            },
            self.paths.segments,
        )
        save_json(
            {
                "schema": "bcg.io_beliefs.v1",
                "run_id": self.run_id,
                "beliefs": io_beliefs,
            },
            self.paths.io_beliefs,
        )
        save_json(
            {
                "schema": "bcg.reasoning_beliefs.v1",
                "run_id": self.run_id,
                "beliefs": reasoning_beliefs,
            },
            self.paths.reasoning_beliefs,
        )
        save_json(
            {
                "schema": "bcg.forward_relations.v1",
                "run_id": self.run_id,
                "forward_relations": forward_relations,
                "reports": self.forward_reports,
            },
            self.paths.forward_relations,
        )
        save_json(
            {
                "schema": "bcg.backward_relations.v1",
                "run_id": self.run_id,
                "backward_relations": backward_relations,
                "reports": self.backward_reports,
            },
            self.paths.backward_relations,
        )
        save_json(
            {
                "schema": "bcg.merges.v1",
                "run_id": self.run_id,
                "merges": self.graph.merges,
            },
            self.paths.merges,
        )
        self._event(
            "finalize",
            {
                "beliefs": len(beliefs),
                "forward_relations": len(forward_relations),
                "backward_relations": len(backward_relations),
                "merges": len(self.graph.merges),
            },
        )
        result = BeliefGraphRunResult(
            run_id=self.run_id,
            graph=self.graph,
            memory=memory,
            output_paths=self.paths,
            token_usage=token_usage,
            counts=counts,
        )
        self.memory.graph = self.graph
        self._active = False
        return result

    async def _extract_segment(
        self,
        segment: Segment,
        *,
        turn_content: str,
        session: dict[str, Any],
        flat_index: int,
        has_answer: bool | None,
    ) -> dict[str, Any]:
        assert self.graph is not None
        assert self.generator is not None
        assert self.options is not None
        min_len = self.options.min_segment_len.get(segment.type, 0)
        base_report: dict[str, Any] = {
            "segment": segment.to_dict(),
            "split": None,
            "belief_ids": [],
            "n_beliefs": 0,
        }
        if len(segment.content) < min_len:
            base_report.update(
                {
                    "skipped": True,
                    "skip_reason": "too short",
                    "_beliefs": [],
                }
            )
            self.segment_reports.append(dict(base_report, _beliefs=[]))
            return base_report

        source = source_descriptor(
            source_type=SEGMENT_SOURCE_TYPES.get(segment.type, segment.type),
            scenario=self.scenario,
            item_id=self.item_id,
            session_id=session.get("session_id"),
            session_index=int(session["session_index"]),
            session_date=session.get("date"),
            turn_index=int(session["n_turns"]),
            trajectory_index=flat_index,
            role=segment.role,
            segment=segment,
            has_answer=has_answer,
        )
        graph_context = format_graph_context(
            self.graph.belief_dicts(),
            char_budget=self.options.context_chars,
        )
        io_context = (
            format_io_context(
                self.graph.belief_dicts(),
                char_budget=self.options.io_context_chars,
            )
            if segment.type in REASONING_TYPES
            else None
        )
        produced: list[dict[str, Any]] = []
        clusters = None
        if self.options.use_split and self.embedder is not None:
            sentences = split_sentences(segment.content)
            if len(sentences) >= self.options.split_min_sentences:
                try:
                    clusters, split_info = cluster_sentences(
                        sentences,
                        self.embedder,
                        similarity_threshold=self.options.split_threshold,
                        buffer_size=self.options.split_buffer,
                        purpose=(
                            f"split:s{session['session_index']}.t"
                            f"{session['n_turns']}.seg{segment.seg_idx}"
                        ),
                    )
                    base_report["split"] = split_info
                except Exception as exc:
                    base_report["split"] = {"error": str(exc)}
                    clusters = None

        if clusters:
            cluster_outputs: list[dict[str, Any]] = []
            for cluster in clusters:
                result = await extract_segment(
                    self.generator,
                    segment,
                    scenario=self.scenario,
                    mode="sentences",
                    sentences=[sentence.text for sentence in cluster.sentences],
                    graph_context=graph_context,
                    io_context=io_context,
                    session_date=session.get("date"),
                    max_tokens=self.max_tokens,
                    label=(
                        f"s{session['session_index']}.t{session['n_turns']}"
                        f".extract:{segment.type}[c{cluster.cluster_id}]"
                    ),
                )
                cluster_outputs.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "raw_output": result.raw_output,
                        "skipped": result.skipped,
                        "skip_reason": result.skip_reason,
                    }
                )
                for cleaned in result.beliefs:
                    indices = cleaned.pop("supporting_sentence_indices", None)
                    chosen = (
                        [
                            cluster.sentences[index]
                            for index in indices
                            if 0 <= index < len(cluster.sentences)
                        ]
                        if indices
                        else list(cluster.sentences)
                    )
                    evidence = [
                        evidence_from_sentence(
                            sentence.text,
                            sentence.start,
                            sentence.end,
                            segment,
                            turn_content,
                            source,
                        )
                        for sentence in chosen
                    ]
                    produced.append(self._make_belief(cleaned, source, evidence))
            base_report["cluster_outputs"] = cluster_outputs
        else:
            result = await extract_segment(
                self.generator,
                segment,
                scenario=self.scenario,
                mode="excerpt",
                graph_context=graph_context,
                io_context=io_context,
                session_date=session.get("date"),
                max_tokens=self.max_tokens,
                label=(
                    f"s{session['session_index']}.t{session['n_turns']}"
                    f".extract:{segment.type}"
                ),
            )
            base_report["raw_output"] = result.raw_output
            base_report["skipped"] = result.skipped
            if result.skip_reason:
                base_report["skip_reason"] = result.skip_reason
            for cleaned in result.beliefs:
                evidence = [
                    evidence_from_excerpt(excerpt, segment, turn_content, source)
                    for excerpt in cleaned.pop("supporting_excerpts", [])
                ]
                produced.append(self._make_belief(cleaned, source, evidence))

        base_report["n_beliefs"] = len(produced)
        base_report["belief_ids"] = [belief["id"] for belief in produced]
        stored_report = dict(base_report)
        self.segment_reports.append(stored_report)
        base_report["_beliefs"] = produced
        return base_report

    def _make_belief(
        self,
        cleaned: dict[str, Any],
        source: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        assert self.graph is not None
        belief = {
            "id": self.graph.allocate_belief_id(),
            "layer": "io" if source.get("segment_type") in IO_TYPES else "reasoning",
            "belief": cleaned["belief"],
            "stance": cleaned["stance"],
            "event_time": cleaned.get("event_time"),
            "time_text": cleaned.get("time_text"),
            "source": dict(source),
            "evidence": evidence,
            "supporting_excerpts": [
                item["text"] for item in evidence if item.get("text")
            ],
        }
        init_belief_confidence(belief, config=self.confidence_config)
        payload = BeliefPayload.model_validate(belief)
        self.graph.add_belief(payload)
        return payload.model_dump(mode="json")

    def _event(self, kind: str, payload: dict[str, Any]) -> None:
        assert self.paths is not None
        record = {"ts": utc_now().isoformat(), "event": kind, **payload}
        with self.paths.events.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("No active belief run. Call begin_belief_run() first.")


class _TrackedTextGenerator:
    def __init__(
        self,
        llm: Any,
        *,
        usage: TokenUsageTracker,
        model: str | None,
    ) -> None:
        self.llm = llm
        self.usage = usage
        self.model = model or _model_name(llm)

    async def generate_text(
        self,
        prompt: str,
        *,
        label: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        label = label or "unlabeled"
        if hasattr(self.llm, "generate"):
            kwargs: dict[str, Any] = {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if self.model != "unknown" and _accepts_keyword(self.llm.generate, "model"):
                kwargs["model"] = self.model
            response = await maybe_await(
                self.llm.generate(
                    [{"role": "user", "content": prompt}],
                    **kwargs,
                )
            )
            if isinstance(response, LLMResponse):
                self.usage.record_response(
                    label=label,
                    model=self.model,
                    prompt=prompt,
                    response=response,
                )
                return response.content
            content = _response_content(response)
            self.usage.record(
                label=label,
                model=self.model,
                usage=getattr(response, "usage", {}),
                prompt=prompt,
                completion=content,
            )
            return content

        if not hasattr(self.llm, "generate_text"):
            raise TypeError("belief graph pipeline requires generate or generate_text")
        text = await maybe_await(
            self.llm.generate_text(
                prompt,
                label=label,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        content = str(text)
        self.usage.record(
            label=label,
            model=self.model,
            prompt=prompt,
            completion=content,
        )
        return content


def _response_content(response: Any) -> str:
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    if isinstance(response, dict):
        return str(response.get("content") or response.get("text") or "")
    return str(response)


def _model_name(llm: Any) -> str:
    config = getattr(llm, "config", None)
    model = getattr(config, "model", None)
    if isinstance(model, str) and model:
        return model
    return "unknown"


def _accepts_keyword(func: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    return keyword in signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _run_paths(output_root: Path, run_id: str) -> BeliefGraphRunPaths:
    run_dir = output_root / run_id
    artifacts_dir = run_dir / "artifacts"
    return BeliefGraphRunPaths(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        graph=run_dir / "graph.json",
        memory=run_dir / "memory.json",
        token_usage=run_dir / "token_usage.json",
        events=run_dir / "events.jsonl",
        segments=artifacts_dir / "segments.json",
        io_beliefs=artifacts_dir / "io_beliefs.json",
        reasoning_beliefs=artifacts_dir / "reasoning_beliefs.json",
        forward_relations=artifacts_dir / "forward_relations.json",
        backward_relations=artifacts_dir / "backward_relations.json",
        merges=artifacts_dir / "merges.json",
    )


def _counts(
    beliefs: list[dict[str, Any]],
    forward_relations: list[dict[str, Any]],
    backward_relations: list[dict[str, Any]],
    graph: BCG,
) -> dict[str, Any]:
    return {
        "beliefs": len(beliefs),
        "forward_relations": len(forward_relations),
        "backward_relations": len(backward_relations),
        "merges": len(graph.merges),
        "sessions": len(graph.sessions),
        "source_counts": count_by(
            beliefs,
            lambda belief: (belief.get("source") or {}).get("type"),
        ),
        "layer_counts": count_by(beliefs, lambda belief: belief.get("layer")),
        "stance_counts": count_by(beliefs, lambda belief: belief.get("stance")),
    }
