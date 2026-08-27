"""
stream.py
=========
Streaming belief/decision graph engine.

Each turn is routed only by role (user / assistant / tool; system is recorded
but yields no nodes; function == tool). For every non-skipped turn, one LLM call
extracts new belief nodes and new decision nodes. Assistant relation extraction
judges up to ``max_previous_windows`` prior Graph layers together and selects at
most one; other roles retain the sequential current-turn / prior-turn search.

Node schema additions:
  * node_type: "belief" | "decision"
  * entities: list[str]
  * event_time: UTC node-creation timestamp assigned by the graph builder
  * evidence_ids: list[int]
  * confidence / initial_confidence / evidence_confidence / factor_confidence

Relation schema:
  * depends_on | supplements | contradicts
  * depends_on / contradicts carry weight + activated_condition
  * supplements keeps weight and activated_condition as null

  * merging is incremental only (per-turn, embedding-based); there is no
    trajectory-end global merge/dedup pass.
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .._shared.roles import normalize_role
from .._shared.tool_queries import extract_tool_calls, strip_valid_tool_calls
from .._shared.tool_results import extract_tool_results
from .._shared.writers import ArtifactWriter, EventRecorder
from ..hybrid.named_entities import normalize_entity_config
from ..hybrid.stance import normalize_stance_config
from . import llm
from .confidence import (
    init_belief_confidence,
    normalize_confidence_config,
    propagate_relation_confidences,
    recompute_evidence_confidence_from_node,
    relation_output_node_id,
)
from .evidence import (
    evidence_from_excerpt,
    evidence_from_sentence,
    source_descriptor,
)
from .extract import (
    extract_compact_tool_result_nodes,
    extract_compact_tool_result_nodes_batch,
    extract_layered_relations,
    extract_nodes,
    extract_relations,
    extract_rule_tool_result_nodes,
    format_extraction_nodes,
    format_graph_edges,
    format_graph_nodes,
    format_relation_nodes,
)
from .graph import BeliefGraph
from .llm import USAGE
from .merge import run_merge_pass
from .split import split_sentences

BELIEF_ROLES = {"user", "assistant", "tool"}


@dataclass
class StreamOptions:
    # evidence mode: "sentence" (whole-sentence evidence) | "excerpt" (free spans)
    evidence_mode: str = "sentence"
    # incremental embedding-ONLY merge after EACH turn's new nodes/edges
    # (no LLM verification by default).
    incremental_merge: bool = True
    incremental_merge_threshold: float = 0.86
    # When True, the per-turn incremental merge additionally calls the LLM once per
    # embedding-flagged candidate group to (1) verify the merge is reasonable
    # (apply-time gate: only LLM-confirmed groups are merged) and (2) rewrite the
    # surviving node's content so it covers the full meaning of all merged nodes.
    # Needs an embedder (same as incremental_merge).
    verify_merge: bool = False
    # prompt budgets
    context_chars: int = 100000  # existing-nodes context budget
    # Limit node-extraction history to the latest N non-empty Graph turns before
    # applying ``context_chars``. Zero preserves the character-budget-only path.
    extraction_history_turns: int = 2
    # Maximum number of non-empty historical turn windows considered while
    # looking for a current-to-prior relation. Assistant turns bundle these
    # windows into one request; other roles inspect them sequentially.
    max_previous_windows: int = 3
    # None selects the model-aware default (GPT-5.6-Luna -> none; otherwise
    # medium). A configured value is forwarded to every graph-model call.
    reasoning_effort: str | None = None
    # skip turns whose content is shorter than this (0 = never skip)
    min_content_len: int = 0
    # Bounded, history-free semantic extraction for structured Tool Results.
    tool_result_max_search_results: int = 10
    tool_result_max_snippet_chars: int = 240
    tool_result_semantic_extraction: bool = True
    tool_result_max_facts: int = 3
    tool_result_max_semantic_calls: int = 12
    # Retained for configuration compatibility with the hybrid backend. Unified
    # extraction asks the graph model to emit stance and entities with each node.
    stance_config: dict[str, Any] = field(default_factory=dict)
    entity_config: dict[str, Any] = field(default_factory=dict)
    confidence_config: dict[str, Any] = field(default_factory=dict)

    def apply_belief_graph_config(self, cfg: dict[str, Any] | None) -> None:
        """Apply shared relation-propagation settings from model_config.json."""
        if not isinstance(cfg, dict):
            return
        conf_cfg = cfg.get("confidence") or {}
        if isinstance(conf_cfg, dict):
            self.confidence_config = normalize_confidence_config(conf_cfg)
        else:
            self.confidence_config = normalize_confidence_config(self.confidence_config)
        edge_cfg = cfg.get("edge_generation") or {}
        if (
            isinstance(edge_cfg, dict)
            and edge_cfg.get("max_previous_windows") is not None
        ):
            self.max_previous_windows = max(1, int(edge_cfg["max_previous_windows"]))
        runtime_cfg = cfg.get("runtime") or {}
        if (
            isinstance(runtime_cfg, dict)
            and runtime_cfg.get("extraction_history_turns") is not None
        ):
            self.extraction_history_turns = max(
                0, int(runtime_cfg["extraction_history_turns"])
            )
        tool_result_cfg = cfg.get("tool_results") or {}
        if isinstance(tool_result_cfg, dict):
            if tool_result_cfg.get("max_search_results") is not None:
                self.tool_result_max_search_results = max(
                    1, int(tool_result_cfg["max_search_results"])
                )
            if tool_result_cfg.get("max_snippet_chars") is not None:
                self.tool_result_max_snippet_chars = max(
                    40, int(tool_result_cfg["max_snippet_chars"])
                )
            if tool_result_cfg.get("semantic_extraction") is not None:
                self.tool_result_semantic_extraction = bool(
                    tool_result_cfg["semantic_extraction"]
                )
            if tool_result_cfg.get("max_facts") is not None:
                self.tool_result_max_facts = max(1, int(tool_result_cfg["max_facts"]))
            if tool_result_cfg.get("max_semantic_calls") is not None:
                self.tool_result_max_semantic_calls = max(
                    0, int(tool_result_cfg["max_semantic_calls"])
                )
        stance_cfg = cfg.get("stance") or {}
        if isinstance(stance_cfg, dict):
            merged_stance_cfg = dict(self.stance_config or {})
            if isinstance(stance_cfg.get("labels"), dict):
                merged_labels = dict(merged_stance_cfg.get("labels") or {})
                merged_labels.update(stance_cfg.get("labels") or {})
                merged_stance_cfg["labels"] = merged_labels
                stance_cfg = {
                    key: value for key, value in stance_cfg.items() if key != "labels"
                }
            merged_stance_cfg.update(stance_cfg)
            self.stance_config = normalize_stance_config(merged_stance_cfg)
        else:
            self.stance_config = normalize_stance_config(self.stance_config)
        entity_cfg = cfg.get("entities") or {}
        if isinstance(entity_cfg, dict):
            merged_entity_cfg = dict(self.entity_config or {})
            merged_entity_cfg.update(entity_cfg)
            self.entity_config = normalize_entity_config(merged_entity_cfg)
        else:
            self.entity_config = normalize_entity_config(self.entity_config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_mode": self.evidence_mode,
            "incremental_merge": self.incremental_merge,
            "incremental_merge_threshold": self.incremental_merge_threshold,
            "verify_merge": self.verify_merge,
            "context_chars": self.context_chars,
            "extraction_history_turns": self.extraction_history_turns,
            "max_previous_windows": self.max_previous_windows,
            "reasoning_effort": self.reasoning_effort,
            "min_content_len": self.min_content_len,
            "tool_result_max_search_results": self.tool_result_max_search_results,
            "tool_result_max_snippet_chars": self.tool_result_max_snippet_chars,
            "tool_result_semantic_extraction": self.tool_result_semantic_extraction,
            "tool_result_max_facts": self.tool_result_max_facts,
            "tool_result_max_semantic_calls": self.tool_result_max_semantic_calls,
            "stance": normalize_stance_config(self.stance_config),
            "entities": normalize_entity_config(self.entity_config),
            "confidence_config": normalize_confidence_config(self.confidence_config),
        }


class StreamingBeliefBuilder:
    def __init__(
        self,
        *,
        client,
        model: str,
        item_id: str,
        out_dir: Path,
        options: StreamOptions | None = None,
        embedder=None,
        stance_classifier=None,
        entity_recognizer=None,
        item_meta: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.item_id = item_id
        self.item_meta = item_meta or {}
        self.max_tokens = max_tokens
        self.options = options or StreamOptions()
        self.options.max_previous_windows = max(
            1, int(self.options.max_previous_windows)
        )
        self.options.reasoning_effort = llm.resolve_reasoning_effort(
            model, self.options.reasoning_effort
        )
        self.options.confidence_config = normalize_confidence_config(
            self.options.confidence_config
        )
        self.options.stance_config = normalize_stance_config(self.options.stance_config)
        self.options.entity_config = normalize_entity_config(self.options.entity_config)
        # These optional arguments remain accepted for API compatibility, but
        # Unified now obtains both metadata fields from the graph-model response.
        self.stance_classifier = stance_classifier
        self.entity_recognizer = entity_recognizer
        self.embedder = embedder

        self.graph = BeliefGraph(confidence_config=self.options.confidence_config)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.out_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self.embedder is not None:
            self.embedder.set_log_path(self.logs_dir / "embedding_calls.jsonl")
        llm.set_prompt_log_path(self.logs_dir / "prompts.jsonl")

        self._events = EventRecorder(self.out_dir / "events.jsonl")
        self._artifacts = ArtifactWriter(self.out_dir)

        self._trajectory: list[dict[str, Any]] = []  # flat, ALL turns (incl. system)
        self._flat_turn = 0
        # The first user turn establishes the problem beliefs.  Those beliefs
        # intentionally remain an unconnected root layer: relation generation
        # starts only when a later turn can connect new evidence/reasoning back
        # to an existing layer.
        self._has_extracted_user_beliefs = False
        self._finalized = False
        self._start_time = datetime.now(UTC)
        self._end_time: datetime | None = None

        # Per-turn sub-step timing (seconds). One record per NON-skipped turn:
        #   {turn_index, role, node_generation, merging, llm_check,
        #    entity_extraction, edge_generation, turn_total}. Populated in ingest_turn.
        self._turn_timings: list[dict[str, Any]] = []
        self._semantic_tool_result_calls = 0
        # Results prepared by one batch extraction call and consumed later by
        # the normal per-turn pipeline. Keys are future flat turn indices;
        # values retain independent nodes/evidence for exactly one tool turn.
        self._prepared_tool_results: dict[
            int,
            tuple[
                dict[str, Any] | None,
                float,
            ],
        ] = {}
        # Final (trajectory-end) merge timing; filled in finalize(). Always
        # zero now that the trajectory-end global merge has been removed.
        self._final_merge_timing: dict[str, Any] = {
            "merging": 0.0,
            "llm_check": 0.0,
            "total": 0.0,
        }

    def _node_extraction_history(self) -> list[dict[str, Any]]:
        """Return active nodes from the configured latest Graph-turn window."""
        nodes = self.graph.active()
        turn_limit = int(self.options.extraction_history_turns)
        if turn_limit <= 0 or not nodes:
            return nodes

        turn_ids = sorted(
            {
                int(turn_id)
                for node in nodes
                if (turn_id := (node.get("source") or {}).get("turn_id")) is not None
            }
        )
        if len(turn_ids) <= turn_limit:
            return nodes

        retained_turn_ids = set(turn_ids[-turn_limit:])
        return [
            node
            for node in nodes
            if (node.get("source") or {}).get("turn_id") in retained_turn_ids
        ]

    def _formatted_node_extraction_history(self) -> str:
        return format_extraction_nodes(
            self._node_extraction_history(),
            char_budget=self.options.context_chars,
        )

    # ------------------------------------------------------------------ events
    def _propagate_relation_confidences(
        self,
        *,
        seed_output_node_ids: list[int] | None = None,
        step: str = "relation_propagation",
    ) -> dict[str, Any]:
        return propagate_relation_confidences(
            self.graph.beliefs,
            self.graph.relations,
            config=self.options.confidence_config,
            seed_output_node_ids=seed_output_node_ids,
            step=step,
        )

    # ------------------------------------------------------------- turn ingest
    def ingest_turn(
        self,
        role: str,
        content: str,
        date: str | None = None,
        has_answer: bool | None = None,
    ) -> dict[str, Any]:
        """Process one incoming turn: extract nodes, merge, then link local edges."""
        flat_idx = self._flat_turn
        turn_idx = flat_idx  # no sessions: turn index == flat index
        content = content or ""
        raw_role = (role or "user").strip().lower()
        eff_role = normalize_role(raw_role)

        traj_entry: dict[str, Any] = {
            "role": raw_role,
            "content": content,
            "turn_index": turn_idx,
        }
        if date is not None:
            traj_entry["date"] = date
        if has_answer is not None:
            traj_entry["has_answer"] = bool(has_answer)
        self._trajectory.append(traj_entry)

        new_nodes: list[dict[str, Any]] = []
        relations_added = 0
        skip_reason: str | None = None
        report: dict[str, Any] = {"split": None}

        skip = (
            raw_role == "system"
            or not content.strip()
            or eff_role not in BELIEF_ROLES
            or len(content) < self.options.min_content_len
        )
        if skip:
            if raw_role == "system":
                skip_reason = "system turn"
            elif not content.strip():
                skip_reason = "empty content"
            elif eff_role not in BELIEF_ROLES:
                skip_reason = f"non-belief role {raw_role!r}"
            else:
                skip_reason = "too short"
        else:
            _t_turn = time.perf_counter()
            new_nodes, relations_added, report = self._update_from_turn(
                eff_role, content, turn_idx, flat_idx, date, has_answer
            )
            turn_total = (
                time.perf_counter()
                - _t_turn
                + float(report.pop("_prepared_batch_seconds", 0.0) or 0.0)
            )
            # Normalise + round the sub-step timing produced by _update_from_turn,
            # attach turn_total, and record one wide-table row for this turn.
            st = report.get("timing") or {}
            st = {
                k: round(float(st.get(k, 0.0) or 0.0), 6)
                for k in (
                    "node_generation",
                    "merging",
                    "llm_check",
                    "entity_extraction",
                    "edge_generation",
                )
            }
            st["turn_total"] = round(turn_total, 6)
            report["timing"] = st
            self._turn_timings.append({"turn_index": turn_idx, "role": raw_role, **st})

        self._flat_turn += 1
        n_merged = len((report.get("incremental_merge") or {}).get("applied", []))
        print(
            f"  t{turn_idx} role={raw_role:<9} -> {len(new_nodes)} node(s), "
            f"{relations_added} relation(s)"
            + (f", {n_merged} merge(s)" if n_merged else "")
            + (f"  [skip: {skip_reason}]" if skip_reason else "")
        )
        return self._events.record(
            "turn",
            {
                "turn_index": turn_idx,
                "trajectory_index": flat_idx,
                "role": raw_role,
                "effective_role": eff_role,
                "content_chars": len(content),
                "skip_reason": skip_reason,
                "semantic_tool_result_calls": self._semantic_tool_result_calls,
                "split": report.get("split"),
                "raw_output": report.get("raw_output"),
                "stance_classification": report.get("stance_classification"),
                "raw_relation_output": report.get("raw_relation_output"),
                "new_node_ids": [b["id"] for b in new_nodes],
                "new_belief_ids": [
                    b["id"]
                    for b in new_nodes
                    if b.get("node_type", "belief") == "belief"
                ],
                "new_decision_ids": [
                    b["id"] for b in new_nodes if b.get("node_type") == "decision"
                ],
                "relations_added": relations_added,
                "edge_attempts": report.get("edge_attempts"),
                "edge_linked_previous_trajectory_index": report.get(
                    "edge_linked_previous_trajectory_index"
                ),
                "edge_window_limit_reached": report.get(
                    "edge_window_limit_reached", False
                ),
                "edge_skip_reason": report.get("edge_skip_reason"),
                "incremental_merge": report.get("incremental_merge"),
                "entity_extraction": report.get("entity_extraction"),
                "timing": report.get("timing"),
            },
        )

    def prepare_tool_result_batch(self, contents: list[str]) -> int:
        """Prepare consecutive tool results with at most one graph-model call.

        The prepared outputs are consumed by subsequent ``ingest_turn`` calls,
        so node allocation, evidence, merge behavior, source turn indices, and
        deterministic query provenance stay identical to sequential ingestion.
        Returns the number of future turns prepared.
        """

        opt = self.options
        if len(contents) < 2:
            return 0
        if self._prepared_tool_results:
            # A caller must consume one batch before preparing another one.
            return 0

        remaining = max(
            0,
            int(opt.tool_result_max_semantic_calls) - self._semantic_tool_result_calls,
        )
        semantic_count = (
            min(len(contents), remaining) if opt.tool_result_semantic_extraction else 0
        )
        active_nodes = self.graph.active()
        batch_items: list[dict[str, Any]] = []
        sentence_lists: list[list[str] | None] = []
        for offset, content in enumerate(contents):
            sentences = (
                [sentence.text for sentence in split_sentences(content)]
                if opt.evidence_mode == "sentence"
                else None
            )
            sentence_lists.append(sentences)
            batch_items.append(
                {
                    "content": content,
                    "sentences": sentences,
                    "query": self._tool_result_query_text(
                        active_nodes,
                        self._flat_turn,
                        additional_prior_tool_results=offset,
                    ),
                }
            )

        _t_batch = time.perf_counter()
        semantic_results: list[dict[str, Any] | None] = []
        if semantic_count:
            USAGE.set_label(f"t{self._flat_turn}.extract:tool_batch:{semantic_count}")
            semantic_results = extract_compact_tool_result_nodes_batch(
                self.client,
                self.model,
                items=batch_items[:semantic_count],
                mode=opt.evidence_mode,
                max_results=opt.tool_result_max_search_results,
                max_snippet_chars=opt.tool_result_max_snippet_chars,
                max_facts=opt.tool_result_max_facts,
                max_tokens=self.max_tokens,
                reasoning_effort=opt.reasoning_effort,
            )
        prepared: list[dict[str, Any] | None] = list(semantic_results)
        for offset in range(semantic_count, len(contents)):
            prepared.append(
                extract_rule_tool_result_nodes(
                    role="tool",
                    content=contents[offset],
                    mode=opt.evidence_mode,
                    sentences=sentence_lists[offset],
                    max_results=opt.tool_result_max_search_results,
                    max_snippet_chars=opt.tool_result_max_snippet_chars,
                )
            )
        elapsed = time.perf_counter() - _t_batch

        self._semantic_tool_result_calls += sum(
            1
            for result in semantic_results
            if result is not None
            and result.get("extraction_method") == "compact_llm_tool_result"
        )
        for offset, result in enumerate(prepared):
            self._prepared_tool_results[self._flat_turn + offset] = (
                result,
                elapsed if offset == 0 else 0.0,
            )
        return len(prepared)

    def prepare_assistant_tool_result_batch(
        self,
        assistant_content: str,
        tool_contents: list[str],
    ) -> int:
        """Prepare ``Assistant -> Tool Result(s)`` with split extraction paths.

        Assistant reasoning is extracted independently with the canonical node
        prompt and bounded historical graph context.  A pure Tool Call Assistant
        turn is handled deterministically without a model call.  Tool Results
        are distilled together by the compact history-free prompt.  Prepared
        results are still consumed by sequential ``ingest_turn`` calls, keeping
        every node on its original Assistant or Tool layer.
        """

        if not tool_contents or self._prepared_tool_results:
            return 0

        opt = self.options
        assistant_sentences = (
            [sentence.text for sentence in split_sentences(assistant_content)]
            if opt.evidence_mode == "sentence"
            else None
        )
        calls = extract_tool_calls(assistant_content)
        query_by_call_id = {
            str(call.tool_call_id): call.query
            for call in calls
            if call.tool_call_id is not None and call.query is not None
        }

        flat_items: list[dict[str, Any]] = []
        flat_sources: list[dict[str, Any]] = []
        for tool_turn_offset, content in enumerate(tool_contents):
            turn_sentences = (
                [sentence.text for sentence in split_sentences(content)]
                if opt.evidence_mode == "sentence"
                else None
            )
            parsed_results = extract_tool_results(content)
            if parsed_results:
                for result_index, parsed in enumerate(parsed_results):
                    flat_index = len(flat_items)
                    query = (
                        query_by_call_id.get(str(parsed.tool_call_id))
                        if parsed.tool_call_id is not None
                        else None
                    )
                    if query is None and calls:
                        query = calls[min(flat_index, len(calls) - 1)].query
                    flat_items.append(
                        {
                            "content": parsed.source_block or content,
                            "sentences": turn_sentences,
                            "query": query,
                        }
                    )
                    flat_sources.append(
                        {
                            "tool_turn_offset": tool_turn_offset,
                            "result_index": result_index,
                            "parsed": parsed,
                            "grouped": True,
                        }
                    )
                continue

            flat_index = len(flat_items)
            query = calls[min(flat_index, len(calls) - 1)].query if calls else None
            flat_items.append(
                {
                    "content": content,
                    "sentences": turn_sentences,
                    "query": query,
                }
            )
            flat_sources.append(
                {
                    "tool_turn_offset": tool_turn_offset,
                    "result_index": 0,
                    "parsed": None,
                    "grouped": False,
                }
            )

        remaining = max(
            0,
            int(opt.tool_result_max_semantic_calls) - self._semantic_tool_result_calls,
        )
        semantic_count = (
            min(len(flat_items), remaining)
            if opt.tool_result_semantic_extraction
            else 0
        )

        # Extract Assistant reasoning separately only when the Agent supplied a
        # visible <thinking> block. Tool Call beliefs are always deterministic
        # and are merged back into the same Assistant turn by code; raw Tool Call
        # JSON never enters the semantic node-extraction prompt.
        _t_assistant = time.perf_counter()
        USAGE.set_label(f"t{self._flat_turn}.extract:assistant")
        thinking_blocks = re.findall(
            r"<thinking>.*?</thinking>",
            assistant_content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        tool_call_content = "\n".join(call.excerpt for call in calls)
        if calls:
            tool_call_result = extract_nodes(
                self.client,
                self.model,
                role="assistant",
                mode=opt.evidence_mode,
                content=tool_call_content,
                sentences=assistant_sentences,
                graph_nodes_str="[]",
                graph_edges_str="[]",
                max_tokens=self.max_tokens,
                reasoning_effort=opt.reasoning_effort,
            )
        else:
            tool_call_result = {
                "nodes": [],
                "beliefs": [],
                "decisions": [],
                "relations": [],
                "raw_output": None,
                "skipped": False,
                "extraction_method": "no_tool_calls",
            }
        if thinking_blocks:
            thinking_content = "\n".join(thinking_blocks)
            thinking_sentences = (
                [sentence.text for sentence in split_sentences(thinking_content)]
                if opt.evidence_mode == "sentence"
                else None
            )
            thinking_result = extract_nodes(
                self.client,
                self.model,
                role="assistant",
                mode=opt.evidence_mode,
                content=thinking_content,
                sentences=thinking_sentences,
                graph_nodes_str=self._formatted_node_extraction_history(),
                graph_edges_str="[]",
                max_tokens=self.max_tokens,
                reasoning_effort=opt.reasoning_effort,
            )
            thinking_nodes = [dict(node) for node in thinking_result.get("nodes", [])]
            tool_call_nodes = [dict(node) for node in tool_call_result.get("nodes", [])]
            for node in tool_call_nodes:
                node["tmp_id"] = f"n{len(thinking_nodes)}"
                thinking_nodes.append(node)
            assistant_result = {
                "nodes": thinking_nodes,
                "beliefs": [
                    node
                    for node in thinking_nodes
                    if node.get("node_type", "belief") == "belief"
                ],
                "decisions": list(thinking_result.get("decisions", [])),
                "relations": [],
                "raw_output": thinking_result.get("raw_output"),
                "skipped": bool(thinking_result.get("skipped", False)),
                "extraction_method": "split_thinking_and_rule_tool_call",
            }
        else:
            assistant_result = tool_call_result
        assistant_elapsed = time.perf_counter() - _t_assistant

        # Tool Results use the compact history-free extractor.  Multiple
        # parallel results share this one request but remain partitioned by the
        # code-owned item_index returned in the response.
        _t_tools = time.perf_counter()
        semantic_results: list[dict[str, Any] | None] = []
        if semantic_count:
            USAGE.set_label(
                f"t{self._flat_turn + 1}.extract:tool_batch:{semantic_count}"
            )
            semantic_results = extract_compact_tool_result_nodes_batch(
                self.client,
                self.model,
                items=flat_items[:semantic_count],
                mode=opt.evidence_mode,
                max_results=opt.tool_result_max_search_results,
                max_snippet_chars=opt.tool_result_max_snippet_chars,
                max_facts=opt.tool_result_max_facts,
                max_tokens=self.max_tokens,
                reasoning_effort=opt.reasoning_effort,
            )
        flat_results: list[dict[str, Any] | None] = list(semantic_results)
        for item in flat_items[semantic_count:]:
            flat_results.append(
                extract_rule_tool_result_nodes(
                    role="tool",
                    content=str(item.get("content") or ""),
                    mode=opt.evidence_mode,
                    sentences=item.get("sentences"),
                    max_results=opt.tool_result_max_search_results,
                    max_snippet_chars=opt.tool_result_max_snippet_chars,
                )
            )
        tool_elapsed = time.perf_counter() - _t_tools
        self._semantic_tool_result_calls += sum(
            1
            for result in semantic_results
            if result is not None
            and result.get("extraction_method") == "compact_llm_tool_result"
        )

        per_turn_nodes: list[list[dict[str, Any]]] = [[] for _ in tool_contents]
        per_turn_raw: list[list[str]] = [[] for _ in tool_contents]
        per_turn_grouped = [False for _ in tool_contents]
        per_turn_fallback: list[dict[str, Any] | None] = [None for _ in tool_contents]
        for source, result in zip(flat_sources, flat_results, strict=True):
            turn_offset = int(source["tool_turn_offset"])
            if result is None:
                continue
            if not source["grouped"]:
                per_turn_fallback[turn_offset] = result
                continue
            per_turn_grouped[turn_offset] = True
            raw = result.get("raw_output")
            if isinstance(raw, str) and raw not in per_turn_raw[turn_offset]:
                per_turn_raw[turn_offset].append(raw)
            parsed = source["parsed"]
            result_index = int(source["result_index"])
            for fact_index, original in enumerate(result.get("nodes") or []):
                node = dict(original)
                node["tmp_id"] = f"n{len(per_turn_nodes[turn_offset])}"
                node["tool_name"] = parsed.tool_name
                node["tool_result_index"] = result_index
                node["tool_result_fact_index"] = fact_index
                if parsed.tool_call_id is not None:
                    node["tool_call_id"] = parsed.tool_call_id
                per_turn_nodes[turn_offset].append(node)

        prepared_tools: list[dict[str, Any] | None] = []
        for offset in range(len(tool_contents)):
            if per_turn_grouped[offset]:
                nodes = per_turn_nodes[offset]
                prepared_tools.append(
                    {
                        "nodes": nodes,
                        "beliefs": nodes,
                        "decisions": [],
                        "relations": [],
                        "raw_output": "\n".join(per_turn_raw[offset]),
                        "skipped": False,
                        "extraction_method": "grouped_tool_result",
                    }
                )
            else:
                prepared_tools.append(per_turn_fallback[offset])

        self._prepared_tool_results[self._flat_turn] = (
            assistant_result,
            assistant_elapsed,
        )
        for offset, result in enumerate(prepared_tools, start=1):
            self._prepared_tool_results[self._flat_turn + offset] = (
                result,
                tool_elapsed if offset == 1 else 0.0,
            )
        return 1 + len(tool_contents)

    def _extract_grouped_tool_results(
        self,
        *,
        content: str,
        mode: str,
        sentences: list[str] | None,
        flat_idx: int,
    ) -> dict[str, Any] | None:
        """Extract one canonical parallel-result turn with at most one LLM call.

        The model sees bounded snippets and a code-owned ``item_index`` only.
        Exact tool names and call IDs are copied back from the wire payload, so
        every extracted fact remains pairable with its originating Tool Call.
        """

        parsed_results = extract_tool_results(content)
        if not parsed_results:
            return None

        opt = self.options
        active_nodes = self.graph.active()
        query_by_call_id = {
            str(node["tool_call_id"]): str(node["query"])
            for node in active_nodes
            if node.get("extraction_method") == "rule_tool_call"
            and isinstance(node.get("tool_call_id"), str)
            and isinstance(node.get("query"), str)
        }
        items: list[dict[str, Any]] = []
        for result_index, parsed in enumerate(parsed_results):
            query = (
                query_by_call_id.get(parsed.tool_call_id)
                if parsed.tool_call_id is not None
                else None
            )
            if query is None:
                query = self._tool_result_query_text(
                    active_nodes,
                    flat_idx,
                    additional_prior_tool_results=result_index,
                )
            items.append(
                {
                    "content": parsed.source_block or content,
                    "sentences": sentences,
                    "query": query,
                }
            )

        remaining = max(
            0,
            int(opt.tool_result_max_semantic_calls) - self._semantic_tool_result_calls,
        )
        semantic_count = (
            min(len(items), remaining) if opt.tool_result_semantic_extraction else 0
        )
        extracted: list[dict[str, Any] | None] = []
        if semantic_count:
            USAGE.set_label(f"t{flat_idx}.extract:tool_group:{semantic_count}")
            extracted.extend(
                extract_compact_tool_result_nodes_batch(
                    self.client,
                    self.model,
                    items=items[:semantic_count],
                    mode=mode,
                    max_results=opt.tool_result_max_search_results,
                    max_snippet_chars=opt.tool_result_max_snippet_chars,
                    max_facts=opt.tool_result_max_facts,
                    max_tokens=self.max_tokens,
                    reasoning_effort=opt.reasoning_effort,
                )
            )
        for result_index in range(semantic_count, len(items)):
            extracted.append(
                extract_rule_tool_result_nodes(
                    role="tool",
                    content=str(items[result_index]["content"]),
                    mode=mode,
                    sentences=sentences,
                    max_results=opt.tool_result_max_search_results,
                    max_snippet_chars=opt.tool_result_max_snippet_chars,
                )
            )

        self._semantic_tool_result_calls += sum(
            1
            for result in extracted[:semantic_count]
            if result is not None
            and result.get("extraction_method") == "compact_llm_tool_result"
        )

        nodes: list[dict[str, Any]] = []
        raw_outputs: list[str] = []
        for result_index, (parsed, result) in enumerate(
            zip(parsed_results, extracted, strict=True)
        ):
            if result is None:
                continue
            raw = result.get("raw_output")
            if isinstance(raw, str) and raw not in raw_outputs:
                raw_outputs.append(raw)
            for fact_index, original in enumerate(result.get("nodes") or []):
                node = dict(original)
                node["tmp_id"] = f"n{len(nodes)}"
                node["tool_name"] = parsed.tool_name
                node["tool_result_index"] = result_index
                node["tool_result_fact_index"] = fact_index
                if parsed.tool_call_id is not None:
                    node["tool_call_id"] = parsed.tool_call_id
                nodes.append(node)

        return {
            "nodes": nodes,
            "beliefs": nodes,
            "decisions": [],
            "relations": [],
            "raw_output": "\n".join(raw_outputs),
            "skipped": False,
            "extraction_method": "grouped_tool_result",
        }

    # ------------------------------------------ per-turn three-phase pipeline
    def _update_from_turn(
        self,
        role: str,
        content: str,
        turn_idx: int,
        flat_idx: int,
        date: str | None,
        has_answer: bool | None,
    ):
        """Three-phase per-turn update:
        Phase 1 — extract belief/decision text, stance, and entities together
        Phase 2 — incremental merge for belief nodes only (embedding, optionally
                  LLM-verified); decision nodes are preserved as extracted
        Phase 3 — extract relations on the post-merge graph (one LLM call)
        """
        opt = self.options
        report: dict[str, Any] = {"split": None}
        # Per-turn sub-step wall clocks (seconds), filled across the three phases.
        # merging/llm_check come from the incremental merge (merge.py); a turn may
        # issue several relation calls, whose times accumulate into edge_generation.
        timing = {
            "node_generation": 0.0,
            "merging": 0.0,
            "llm_check": 0.0,
            "entity_extraction": 0.0,
            "edge_generation": 0.0,
        }
        report["timing"] = timing

        src = source_descriptor(
            role=role,
            item_id=self.item_id,
            turn_index=turn_idx,
            flat_turn_index=flat_idx,
            date=date,
            has_answer=has_answer,
        )

        # Node extraction receives the newest historical nodes that fit the same
        # context budget used by relation extraction. Historical edges belong
        # exclusively to the separate relation phase and are omitted here.
        graph_nodes_str = self._formatted_node_extraction_history()
        graph_edges_str = "[]"

        # ---- prepare evidence mode
        sentences = None
        clusters_idx = None
        if opt.evidence_mode == "sentence":
            sents = split_sentences(content)
            sentences = [s.text for s in sents]
            self._last_sentences = sents
        else:
            self._last_sentences = []

        # ---- PHASE 1: extract nodes only (beliefs + decisions, no relations)
        # Structured tool results use a bounded, history-free semantic extractor
        # in both construction modes. Other roles retain the canonical extractor.
        USAGE.set_label(f"t{turn_idx}.extract:{role}")
        _t_nodes = time.perf_counter()
        node_res = None
        prepared = self._prepared_tool_results.pop(flat_idx, None)
        if prepared is not None:
            node_res, prepared_seconds = prepared
            timing["node_generation"] = prepared_seconds
            report["_prepared_batch_seconds"] = prepared_seconds
        if prepared is None and role == "tool" and "<tool_result>" in content.lower():
            node_res = self._extract_grouped_tool_results(
                content=content,
                mode=opt.evidence_mode,
                sentences=sentences,
                flat_idx=flat_idx,
            )
        if role == "tool" and prepared is None and node_res is None:
            tool_result_query = self._tool_result_query_text(
                self.graph.active(), flat_idx
            )
            semantic_enabled = (
                opt.tool_result_semantic_extraction
                and self._semantic_tool_result_calls
                < opt.tool_result_max_semantic_calls
            )
            if semantic_enabled:
                node_res = extract_compact_tool_result_nodes(
                    self.client,
                    self.model,
                    role=role,
                    content=content,
                    mode=opt.evidence_mode,
                    query=tool_result_query,
                    sentences=sentences,
                    max_results=opt.tool_result_max_search_results,
                    max_snippet_chars=opt.tool_result_max_snippet_chars,
                    max_facts=opt.tool_result_max_facts,
                    max_tokens=self.max_tokens,
                    reasoning_effort=opt.reasoning_effort,
                )
                if (
                    node_res is not None
                    and node_res.get("extraction_method") == "compact_llm_tool_result"
                ):
                    self._semantic_tool_result_calls += 1
            else:
                node_res = extract_rule_tool_result_nodes(
                    role=role,
                    content=content,
                    mode=opt.evidence_mode,
                    sentences=sentences,
                    max_results=opt.tool_result_max_search_results,
                    max_snippet_chars=opt.tool_result_max_snippet_chars,
                )
        used_fallback_extractor = node_res is None
        if used_fallback_extractor:
            node_res = extract_nodes(
                self.client,
                self.model,
                role=role,
                mode=opt.evidence_mode,
                content=content,
                sentences=sentences,
                clusters=clusters_idx,
                graph_nodes_str=graph_nodes_str,
                graph_edges_str=graph_edges_str,
                current_date=date,
                max_tokens=self.max_tokens,
                reasoning_effort=opt.reasoning_effort,
            )
        local_node_seconds = time.perf_counter() - _t_nodes
        if prepared is None:
            timing["node_generation"] = local_node_seconds
        elif used_fallback_extractor:
            timing["node_generation"] += local_node_seconds
        report["raw_output"] = node_res.get("raw_output")
        report["skipped"] = node_res.get("skipped", False)
        if node_res.get("skip_reason"):
            report["skip_reason"] = node_res["skip_reason"]

        # Semantic nodes already carry stance and entities from the same graph
        # model response that extracted their text. Deterministic Tool Call nodes
        # remain asserted and retain their code-owned exact tool-name entity.
        extracted_nodes = list(node_res.get("nodes", []))
        for node in extracted_nodes:
            if node.get("extraction_method") == "rule_tool_call":
                node["stance_confidence"] = 1.0
                node["stance_scores"] = {
                    "asserted": 1.0,
                    "recalled": 0.0,
                    "judged": 0.0,
                    "speculated": 0.0,
                }
                node["stance_model"] = "rule_tool_call"
            else:
                # The extraction contract intentionally asks for a label but no
                # numerical probability; confidence remains code-owned.
                node["stance_confidence"] = 0.0
                node["stance_scores"] = {}
                node["stance_model"] = f"graph_model:{self.model}"
        report["stance_classification"] = {
            "source": "graph_model",
            "model": self.model,
            "stances": [node.get("stance", "asserted") for node in extracted_nodes],
        }

        # ---- allocate ids + attach evidence (in output order, so n0<n1<… in id)
        tmp_to_gid: dict[str, int] = {}
        new_nodes: list[dict[str, Any]] = []
        new_node_ids: set = set()
        for cb in extracted_nodes:
            evid = self._evidence_for(cb, content, src, opt.evidence_mode, role)
            node = self._make_node(cb, src, evid, role)
            tmp_to_gid[cb["tmp_id"]] = node["id"]
            new_nodes.append(node)
            new_node_ids.add(node["id"])
        # ---- PHASE 2: incremental merge (before relation extraction so that
        #      relations are drawn against a deduplicated graph). Decision nodes
        #      are excluded from incremental merge entirely: intermediate answers
        #      should not absorb each other during generation, because only the
        #      final decision is retained as a decision at trajectory end.
        if new_nodes and self.options.incremental_merge and self.embedder is not None:
            USAGE.set_label(f"t{turn_idx}.merge")
            verify_merge = self.options.verify_merge
            decision_ids = {
                node["id"]
                for node in self.graph.active()
                if isinstance(node.get("id"), int)
                and node.get("node_type") == "decision"
            }
            # Provenance nodes are stable anchors in every construction mode.
            # Merging them would erase exact call/result identity.
            decision_ids |= {
                node["id"]
                for node in self.graph.active()
                if isinstance(node.get("id"), int)
                and node.get("extraction_method")
                in {"rule_tool_call", "rule_tool_result"}
            }
            inc = run_merge_pass(
                graph=self.graph,
                strategy="embedding",
                verify=verify_merge,
                verify_rewrite=verify_merge,
                client=self.client,
                model=self.model,
                embedder=self.embedder,
                threshold=self.options.incremental_merge_threshold,
                max_tokens=self.max_tokens,
                pass_label=f"turn_{turn_idx}",
                log_dir=(self.logs_dir if verify_merge else None),
                exclude_node_ids=decision_ids,
                reasoning_effort=opt.reasoning_effort,
            )
            # embedding candidate time -> merging; LLM verify time -> llm_check.
            # Present even when the pass is internally skipped (defaults to 0).
            _mt = inc.get("timing") or {}
            timing["merging"] += float(_mt.get("embedding_seconds", 0.0) or 0.0)
            timing["llm_check"] += float(_mt.get("llm_verify_seconds", 0.0) or 0.0)
            if not inc.get("skipped"):
                report["incremental_merge"] = {
                    "applied": inc.get("applied", []),
                    "excluded_decision_ids": inc.get("excluded_node_ids", []),
                    "relation_rewire": inc.get("relation_rewire"),
                }
                for m in inc.get("applied", []):
                    for aid in m.get("absorbed_ids") or []:
                        new_node_ids.discard(aid)
                if inc.get("applied"):
                    report.setdefault("confidence_propagation", []).append(
                        {
                            "trigger": "incremental_merge",
                            "report": self._propagate_relation_confidences(
                                seed_output_node_ids=None,
                                step="relation_propagation_after_merge",
                            ),
                        }
                    )

        # ---- PHASE 3: extract relations inside a local edge window.
        #
        # Assistant turns judge all bounded prior layers in one request and may
        # select at most one. Other roles retain the original policy: inspect one
        # previous non-empty layer at a time and stop after the first accepted
        # cross-turn edge. Old <-> old edges remain impossible because every
        # accepted relation must include a current surviving new node.
        relations_added = 0
        active_nodes = self.graph.active()
        active_ids = set(self.graph.ids())
        surviving_new_ids = new_node_ids & active_ids
        report["edge_attempts"] = []
        active_nodes_by_id = {
            node.get("id"): node
            for node in active_nodes
            if isinstance(node.get("id"), int)
        }
        initial_user_belief_turn = (
            role == "user"
            and bool(surviving_new_ids)
            and not self._has_extracted_user_beliefs
        )

        if initial_user_belief_turn:
            report["edge_skip_reason"] = (
                "initial user belief turn does not create internal relations"
            )
        elif role == "tool" and any(
            active_nodes_by_id.get(node_id, {}).get("extraction_method")
            in {"rule_tool_result", "compact_llm_tool_result"}
            for node_id in surviving_new_ids
        ):
            relations_added, deterministic_report = self._add_tool_result_relations(
                flat_idx=flat_idx,
                active_nodes=active_nodes,
                surviving_new_ids=surviving_new_ids,
            )
            report["edge_attempts"] = [deterministic_report]
            report["edge_linked_previous_trajectory_index"] = deterministic_report.get(
                "previous_trajectory_index"
            )
            thinking_ids = set(deterministic_report.get("thinking_node_ids") or [])
            if surviving_new_ids and thinking_ids:
                _t_edge = time.perf_counter()
                added, _added_cross, attempt = self._extract_relations_for_edge_window(
                    role=role,
                    content=content,
                    turn_idx=turn_idx,
                    previous_trajectory_index=deterministic_report.get(
                        "previous_trajectory_index"
                    ),
                    active_nodes=active_nodes,
                    active_ids=active_ids,
                    surviving_new_ids=surviving_new_ids,
                    previous_node_ids=thinking_ids,
                    date=date,
                    context_chars=opt.context_chars,
                    allow_current_to_current=False,
                )
                timing["edge_generation"] += time.perf_counter() - _t_edge
                attempt["strategy"] = "tool_results_to_prior_thinking"
                report["edge_attempts"].append(attempt)
                report["raw_relation_output"] = attempt.get("raw_relation_output")
                relations_added += added
            if not surviving_new_ids:
                report["edge_skip_reason"] = (
                    "no active current-turn nodes after incremental merge"
                    if new_nodes
                    else "no current-turn nodes extracted"
                )
        elif surviving_new_ids and role == "assistant":
            candidate_layers: list[tuple[int, set[int]]] = []
            for candidate_idx in range(flat_idx - 1, -1, -1):
                previous_node_ids = self._node_ids_from_trajectory_index(
                    active_nodes, candidate_idx
                )
                if not previous_node_ids:
                    continue
                candidate_layers.append((candidate_idx, previous_node_ids))
                if len(candidate_layers) >= opt.max_previous_windows:
                    break

            if candidate_layers:
                _t_edge = time.perf_counter()
                added, attempt = self._extract_relations_for_layered_edge_window(
                    role=role,
                    content=content,
                    turn_idx=turn_idx,
                    active_nodes=active_nodes,
                    active_ids=active_ids,
                    surviving_new_ids=surviving_new_ids,
                    candidate_layers=candidate_layers,
                    context_chars=opt.context_chars,
                )
                timing["edge_generation"] += time.perf_counter() - _t_edge
                report["edge_attempts"].append(attempt)
                report["raw_relation_output"] = attempt.get("raw_relation_output")
                report["edge_linked_previous_trajectory_index"] = attempt.get(
                    "selected_previous_trajectory_index"
                )
                relations_added += added
            else:
                report["edge_skip_reason"] = (
                    "no non-empty previous Graph layer available for Assistant relations"
                )
        elif surviving_new_ids:
            tried_prior_window = False
            linked_prior_turn = None
            attempted_windows = 0

            for candidate_idx in range(flat_idx - 1, -1, -1):
                previous_node_ids = self._node_ids_from_trajectory_index(
                    active_nodes, candidate_idx
                )
                if not previous_node_ids:
                    report["edge_attempts"].append(
                        {
                            "previous_trajectory_index": candidate_idx,
                            "previous_node_ids": [],
                            "relations_added": 0,
                            "cross_turn_relations_added": 0,
                            "skip_reason": "no active nodes from this turn after merge",
                        }
                    )
                    continue

                if attempted_windows >= opt.max_previous_windows:
                    report["edge_window_limit_reached"] = True
                    break
                attempted_windows += 1

                tried_prior_window = True
                _t_edge = time.perf_counter()
                added, added_cross, attempt = self._extract_relations_for_edge_window(
                    role=role,
                    content=content,
                    turn_idx=turn_idx,
                    previous_trajectory_index=candidate_idx,
                    active_nodes=active_nodes,
                    active_ids=active_ids,
                    surviving_new_ids=surviving_new_ids,
                    previous_node_ids=previous_node_ids,
                    date=date,
                    context_chars=opt.context_chars,
                )
                timing["edge_generation"] += time.perf_counter() - _t_edge
                report["edge_attempts"].append(attempt)
                report["raw_relation_output"] = attempt.get("raw_relation_output")
                relations_added += added

                if added_cross > 0:
                    linked_prior_turn = candidate_idx
                    break

            # If this is the first node-producing turn, or every earlier turn's
            # nodes have been merged away, still make one current-only attempt so
            # valid current new <-> current new edges are not lost.
            if not tried_prior_window and role != "assistant":
                _t_edge = time.perf_counter()
                added, _added_cross, attempt = self._extract_relations_for_edge_window(
                    role=role,
                    content=content,
                    turn_idx=turn_idx,
                    previous_trajectory_index=None,
                    active_nodes=active_nodes,
                    active_ids=active_ids,
                    surviving_new_ids=surviving_new_ids,
                    previous_node_ids=set(),
                    date=date,
                    context_chars=opt.context_chars,
                )
                timing["edge_generation"] += time.perf_counter() - _t_edge
                report["edge_attempts"].append(attempt)
                report["raw_relation_output"] = attempt.get("raw_relation_output")
                relations_added += added

            report["edge_linked_previous_trajectory_index"] = linked_prior_turn
        else:
            report["edge_skip_reason"] = (
                "no active current-turn nodes after incremental merge"
                if new_nodes
                else "no current-turn nodes extracted"
            )

        if role == "user" and new_nodes:
            self._has_extracted_user_beliefs = True

        return new_nodes, relations_added, report

    def _add_tool_result_relations(
        self,
        *,
        flat_idx: int,
        active_nodes: list[dict[str, Any]],
        surviving_new_ids: set,
    ) -> tuple[int, dict[str, Any]]:
        """Pair every result with its exact call and expose prior thinking ids."""

        previous_idx: int | None = None
        previous_nodes: list[dict[str, Any]] = []
        for candidate_idx in range(flat_idx - 1, -1, -1):
            if (
                normalize_role(str(self._trajectory[candidate_idx].get("role") or ""))
                != "assistant"
            ):
                continue
            candidate_ids = self._node_ids_from_trajectory_index(
                active_nodes, candidate_idx
            )
            if candidate_ids:
                previous_idx = candidate_idx
                previous_nodes = [
                    node for node in active_nodes if node.get("id") in candidate_ids
                ]
                break

        calls_by_id = {
            str(node.get("tool_call_id")): int(node["id"])
            for node in previous_nodes
            if node.get("extraction_method") == "rule_tool_call"
            and isinstance(node.get("tool_call_id"), str)
            and isinstance(node.get("id"), int)
        }
        calls_by_index = sorted(
            (
                node
                for node in previous_nodes
                if node.get("extraction_method") == "rule_tool_call"
                and isinstance(node.get("id"), int)
            ),
            key=lambda node: int(node.get("tool_call_index") or 0),
        )
        thinking_ids = {
            int(node["id"])
            for node in previous_nodes
            if node.get("source_component") == "thinking"
            and isinstance(node.get("id"), int)
        }
        proposed: list[dict[str, Any]] = []
        pairings: list[dict[str, Any]] = []
        for result_id in sorted(surviving_new_ids):
            result = next(
                (node for node in active_nodes if node.get("id") == result_id), None
            )
            if result is None or result.get("extraction_method") not in {
                "rule_tool_result",
                "compact_llm_tool_result",
            }:
                continue
            raw_call_id = result.get("tool_call_id")
            call_id = (
                calls_by_id.get(str(raw_call_id))
                if isinstance(raw_call_id, str)
                else None
            )
            if call_id is None and not isinstance(raw_call_id, str):
                _legacy_idx, legacy_ids = self._query_nodes_for_tool_result(
                    active_nodes, flat_idx
                )
                if len(legacy_ids) == 1:
                    call_id = next(iter(legacy_ids))
            if call_id is None and calls_by_index:
                result_index = int(result.get("tool_result_index") or 0)
                call_id = int(
                    calls_by_index[min(result_index, len(calls_by_index) - 1)]["id"]
                )
            if call_id is None or call_id == result_id:
                continue
            proposed.append(
                {
                    "from_id": result_id,
                    "to_id": call_id,
                    "type": "depends_on",
                    "note": "The tool result was produced by the preceding tool call.",
                    # Deterministic edges are provenance only. A zero weight keeps
                    # later global confidence recomputations from treating them as
                    # epistemic support.
                    "weight": 0.0,
                    "activated_condition": {"input_conf_threshold": 1.0},
                }
            )
            pairings.append(
                {
                    "tool_call_id": raw_call_id,
                    "result_node_id": result_id,
                    "call_node_id": call_id,
                }
            )
        added = self.graph.add_relations(proposed)
        # These edges encode provenance (result came from query / action used
        # prior evidence), not epistemic support. Propagating confidence across
        # them made unverified search snippets approach 1.0 and encouraged the
        # Agent to over-trust noisy retrieval results.
        report: dict[str, Any] = {
            "strategy": "deterministic_provenance",
            "pairing_strategy": "tool_call_id",
            "previous_trajectory_index": previous_idx,
            "previous_node_ids": sorted(
                int(node["id"])
                for node in previous_nodes
                if isinstance(node.get("id"), int)
            ),
            "new_node_ids": sorted(surviving_new_ids),
            "thinking_node_ids": sorted(thinking_ids),
            "pairings": pairings,
            "relations_added": added,
            "cross_turn_relations_added": added,
        }
        if not proposed:
            report["skip_reason"] = "no deterministic provenance target"
        return added, report

    def _query_nodes_for_tool_result(
        self,
        active_nodes: list[dict[str, Any]],
        flat_idx: int,
        *,
        additional_prior_tool_results: int = 0,
    ) -> tuple[int | None, set[int]]:
        """Match consecutive tool results to their source call by stable order."""

        prior_tool_results = max(0, int(additional_prior_tool_results))
        for candidate_idx in range(flat_idx - 1, -1, -1):
            candidate_role = normalize_role(
                str(self._trajectory[candidate_idx].get("role") or "")
            )
            if candidate_role == "tool":
                prior_tool_results += 1
                continue
            if candidate_role != "assistant":
                continue
            candidates = sorted(
                (
                    node
                    for node in active_nodes
                    if isinstance(node.get("id"), int)
                    and (node.get("source") or {}).get("turn_id") == candidate_idx
                    and node.get("extraction_method") == "rule_tool_call"
                ),
                key=lambda node: int(node.get("tool_call_index") or 0),
            )
            if not candidates:
                continue
            selected = candidates[min(prior_tool_results, len(candidates) - 1)]
            return candidate_idx, {int(selected["id"])}
        return None, set()

    def _tool_result_query_text(
        self,
        active_nodes: list[dict[str, Any]],
        flat_idx: int,
        *,
        additional_prior_tool_results: int = 0,
    ) -> str | None:
        _previous_idx, node_ids = self._query_nodes_for_tool_result(
            active_nodes,
            flat_idx,
            additional_prior_tool_results=additional_prior_tool_results,
        )
        if not node_ids:
            return None
        selected_id = min(node_ids)
        selected = next(
            (node for node in active_nodes if node.get("id") == selected_id), None
        )
        query = selected.get("query") if selected is not None else None
        return str(query) if isinstance(query, str) and query.strip() else None

    def _extract_relations_for_layered_edge_window(
        self,
        *,
        role: str,
        content: str,
        turn_idx: int,
        active_nodes: list[dict[str, Any]],
        active_ids: set,
        surviving_new_ids: set,
        candidate_layers: list[tuple[int, set[int]]],
        context_chars: int,
    ) -> tuple[int, dict[str, Any]]:
        """Judge several prior Assistant edge windows in one model request.

        Candidate layer 1 is the nearest non-empty prior Graph turn. The model
        may link the current nodes to at most one candidate layer. A response
        violating that invariant is retried twice. If all three attempts are
        invalid, only relations to the most-used layer in the final response
        survive (ties prefer the nearest layer).
        """
        previous_ids = set().union(*(ids for _idx, ids in candidate_layers))
        edge_window_ids = surviving_new_ids | previous_ids
        graph_nodes_post = format_relation_nodes(
            [node for node in active_nodes if node.get("id") in edge_window_ids],
            char_budget=context_chars,
        )
        # ``format_relation_nodes`` drops the oldest nodes when over budget. Keep
        # the layer manifest and edge window exactly aligned with what the model
        # can actually see.
        displayed_ids = {
            int(match) for match in re.findall(r'"id"\s*:\s*(\d+)', graph_nodes_post)
        }
        displayed_new_ids = surviving_new_ids & displayed_ids
        displayed_layers: list[dict[str, Any]] = []
        layer_to_trajectory: dict[int, int] = {}
        layer_to_ids: dict[int, set[int]] = {}
        for layer_number, (trajectory_index, node_ids) in enumerate(
            candidate_layers, start=1
        ):
            retained_ids = node_ids & displayed_ids
            if not retained_ids:
                continue
            displayed_layers.append(
                {
                    "layer": layer_number,
                    "node_ids": sorted(retained_ids),
                }
            )
            layer_to_trajectory[layer_number] = trajectory_index
            layer_to_ids[layer_number] = retained_ids

        displayed_previous_ids = set().union(*layer_to_ids.values())
        displayed_window_ids = displayed_new_ids | displayed_previous_ids
        graph_edges_post = format_graph_edges(
            self.graph.relations, keep_ids=displayed_window_ids
        )
        node_to_layer = {
            node_id: layer_number
            for layer_number, node_ids in layer_to_ids.items()
            for node_id in node_ids
        }

        def normalize_selected(value: Any) -> int | None | str:
            if value is None:
                return None
            if isinstance(value, bool):
                return "invalid"
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str):
                stripped = value.strip().lower()
                if stripped in {"", "null", "none"}:
                    return None
                match = re.fullmatch(r"(?:previous[_ -]?layer[_ -]?)?(\d+)", stripped)
                if match:
                    return int(match.group(1))
            return "invalid"

        def cross_layer(relation: dict[str, Any]) -> int | None:
            from_id = relation.get("from_id")
            to_id = relation.get("to_id")
            if from_id in displayed_new_ids and to_id in node_to_layer:
                return node_to_layer[to_id]
            if to_id in displayed_new_ids and from_id in node_to_layer:
                return node_to_layer[from_id]
            return None

        feedback: str | None = None
        model_attempts: list[dict[str, Any]] = []
        final_resolved: list[dict[str, Any]] = []
        final_touched_layers: set[int] = set()
        selected_layer: int | None = None
        valid_response = False

        for attempt_number in range(1, 4):
            USAGE.set_label(
                f"t{turn_idx}.relations:{role}:layer_bundle:attempt{attempt_number}"
            )
            rel_res = extract_layered_relations(
                self.client,
                self.model,
                role=role,
                # Tool calls already have deterministic belief nodes. Keep only
                # Thinking/plain reasoning as semantic evidence for relations.
                content=strip_valid_tool_calls(content),
                graph_nodes_str=graph_nodes_post,
                graph_edges_str=graph_edges_post,
                new_node_ids=displayed_new_ids,
                candidate_layers=displayed_layers,
                validation_feedback=feedback,
                max_tokens=self.max_tokens,
                reasoning_effort=self.options.reasoning_effort,
            )
            resolved = self._resolve_relations(
                rel_res.get("relations", []),
                {},
                active_ids,
                new_node_ids=displayed_new_ids,
                previous_node_ids=displayed_previous_ids,
            )
            touched_layers = {
                layer
                for relation in resolved
                if (layer := cross_layer(relation)) is not None
            }
            normalized = normalize_selected(rel_res.get("selected_previous_layer"))
            reasons: list[str] = []
            if normalized == "invalid" or (
                isinstance(normalized, int) and normalized not in layer_to_ids
            ):
                reasons.append("selected_previous_layer is not an available layer")
            if len(touched_layers) > 1:
                reasons.append(
                    "cross-turn relations touch multiple previous layers: "
                    + ", ".join(str(layer) for layer in sorted(touched_layers))
                )
            elif touched_layers:
                only_layer = next(iter(touched_layers))
                if normalized != only_layer:
                    reasons.append(
                        "selected_previous_layer does not match the layer used by "
                        f"cross-turn relations ({only_layer})"
                    )
            elif normalized is not None:
                reasons.append(
                    "selected_previous_layer must be null when no cross-turn relation exists"
                )

            model_attempts.append(
                {
                    "attempt": attempt_number,
                    "selected_previous_layer": rel_res.get("selected_previous_layer"),
                    "resolved_cross_layers": sorted(touched_layers),
                    "valid": not reasons,
                    "validation_errors": reasons,
                    "raw_relation_output": rel_res.get("raw_output"),
                    "skipped": bool(rel_res.get("skipped")),
                }
            )
            final_resolved = resolved
            final_touched_layers = touched_layers
            if not reasons:
                valid_response = True
                selected_layer = normalized if isinstance(normalized, int) else None
                break
            feedback = (
                "The previous response violated the single-layer contract: "
                + "; ".join(reasons)
                + ". Select zero or one previous layer and regenerate all relations."
            )

        fallback_pruned = False
        if not valid_response:
            fallback_pruned = True
            counts = {
                layer: sum(
                    1 for relation in final_resolved if cross_layer(relation) == layer
                )
                for layer in final_touched_layers
            }
            selected_layer = (
                min(counts, key=lambda layer: (-counts[layer], layer))
                if counts
                else None
            )
            final_resolved = [
                relation
                for relation in final_resolved
                if cross_layer(relation) in {None, selected_layer}
            ]

        existing_keys = {
            (relation.get("from_id"), relation.get("to_id"), relation.get("type"))
            for relation in self.graph.relations
        }
        selected_previous_ids = layer_to_ids.get(selected_layer, set())
        resolved_keys = {
            (relation.get("from_id"), relation.get("to_id"), relation.get("type"))
            for relation in final_resolved
        }
        new_keys = resolved_keys - existing_keys
        cross_turn_relations_added = sum(
            1
            for relation in final_resolved
            if (
                relation.get("from_id"),
                relation.get("to_id"),
                relation.get("type"),
            )
            in new_keys
            and (
                (
                    relation.get("from_id") in displayed_new_ids
                    and relation.get("to_id") in selected_previous_ids
                )
                or (
                    relation.get("to_id") in displayed_new_ids
                    and relation.get("from_id") in selected_previous_ids
                )
            )
        )
        before_relation_count = len(self.graph.relations)
        relations_added = self.graph.add_relations(final_resolved)
        added_relations = self.graph.relations[before_relation_count:]
        seed_output_node_ids = [
            node_id
            for node_id in (
                relation_output_node_id(relation) for relation in added_relations
            )
            if node_id is not None
        ]
        propagation_report = (
            self._propagate_relation_confidences(
                seed_output_node_ids=seed_output_node_ids,
                step="relation_propagation_after_layer_bundle",
            )
            if seed_output_node_ids
            else None
        )

        attempt_report: dict[str, Any] = {
            "strategy": "assistant_previous_layer_bundle",
            "candidate_layers": displayed_layers,
            "new_node_ids": sorted(displayed_new_ids),
            "edge_window_ids": sorted(displayed_window_ids),
            "model_attempts": model_attempts,
            "validation_passed": valid_response,
            "fallback_pruned": fallback_pruned,
            "selected_previous_layer": selected_layer,
            "selected_previous_trajectory_index": layer_to_trajectory.get(
                selected_layer
            ),
            "relations_added": relations_added,
            "cross_turn_relations_added": cross_turn_relations_added,
            "raw_relation_output": model_attempts[-1]["raw_relation_output"],
        }
        if propagation_report is not None:
            attempt_report["confidence_propagation"] = propagation_report
        return relations_added, attempt_report

    def _extract_relations_for_edge_window(
        self,
        *,
        role: str,
        content: str,
        turn_idx: int,
        previous_trajectory_index: int | None,
        active_nodes: list[dict[str, Any]],
        active_ids: set,
        surviving_new_ids: set,
        previous_node_ids: set,
        date: str | None,
        context_chars: int,
        allow_current_to_current: bool = True,
    ):
        """Run one relation-extraction attempt for current nodes + one prior turn.

        Returns ``(relations_added, cross_turn_relations_added, attempt_report)``.
        ``relations_added`` counts every newly inserted relation accepted by the
        graph, including current new <-> current new.
        ``cross_turn_relations_added`` counts only newly inserted relations that
        actually connect a current surviving node to the candidate prior turn;
        this is the backward-search stop condition.
        """
        edge_window_ids = surviving_new_ids | previous_node_ids
        window_nodes = [
            node for node in active_nodes if node.get("id") in edge_window_ids
        ]
        graph_nodes_post = (
            format_relation_nodes(window_nodes, char_budget=context_chars)
            if normalize_role(role) == "tool"
            else format_graph_nodes(window_nodes, char_budget=context_chars)
        )
        graph_edges_post = format_graph_edges(
            self.graph.relations, keep_ids=edge_window_ids
        )

        previous_label = (
            "current_only"
            if previous_trajectory_index is None
            else f"prev{previous_trajectory_index}"
        )
        USAGE.set_label(f"t{turn_idx}.relations:{role}:{previous_label}")
        rel_res = extract_relations(
            self.client,
            self.model,
            role=role,
            content=content,
            graph_nodes_str=graph_nodes_post,
            graph_edges_str=graph_edges_post,
            new_node_ids=surviving_new_ids,
            current_date=date,
            max_tokens=self.max_tokens,
            reasoning_effort=self.options.reasoning_effort,
        )

        resolved = self._resolve_relations(
            rel_res.get("relations", []),
            {},
            active_ids,
            new_node_ids=surviving_new_ids,
            previous_node_ids=previous_node_ids,
            allow_current_to_current=allow_current_to_current,
        )

        existing_keys = {
            (r.get("from_id"), r.get("to_id"), r.get("type"))
            for r in self.graph.relations
        }
        resolved_keys = {
            (r.get("from_id"), r.get("to_id"), r.get("type")) for r in resolved
        }
        new_keys = resolved_keys - existing_keys
        cross_turn_relations_added = sum(
            1
            for r in resolved
            if (r.get("from_id"), r.get("to_id"), r.get("type")) in new_keys
            and (
                (
                    r.get("from_id") in surviving_new_ids
                    and r.get("to_id") in previous_node_ids
                )
                or (
                    r.get("from_id") in previous_node_ids
                    and r.get("to_id") in surviving_new_ids
                )
            )
        )
        before_relation_count = len(self.graph.relations)
        relations_added = self.graph.add_relations(resolved)
        added_relations = self.graph.relations[before_relation_count:]
        seed_output_node_ids = [
            node_id
            for node_id in (
                relation_output_node_id(relation) for relation in added_relations
            )
            if node_id is not None
        ]
        propagation_report = (
            self._propagate_relation_confidences(
                seed_output_node_ids=seed_output_node_ids,
                step="relation_propagation_after_relation_add",
            )
            if seed_output_node_ids
            else None
        )

        attempt = {
            "previous_trajectory_index": previous_trajectory_index,
            "previous_node_ids": sorted(previous_node_ids),
            "new_node_ids": sorted(surviving_new_ids),
            "edge_window_ids": sorted(edge_window_ids),
            "relations_added": relations_added,
            "cross_turn_relations_added": cross_turn_relations_added,
            "raw_relation_output": rel_res.get("raw_output"),
        }
        if propagation_report is not None:
            attempt["confidence_propagation"] = propagation_report
        if rel_res.get("skipped"):
            attempt["skip_reason"] = (
                rel_res.get("skip_reason") or "relation extraction skipped"
            )
        return relations_added, cross_turn_relations_added, attempt

    @staticmethod
    def _node_ids_from_trajectory_index(
        nodes: list[dict[str, Any]], trajectory_index: int
    ) -> set:
        """Return active node ids whose source belongs to one trajectory turn."""
        return {
            node["id"]
            for node in nodes
            if isinstance(node.get("id"), int)
            and (node.get("source") or {}).get("turn_id") == trajectory_index
        }

    def _evidence_for(self, cb, content, src, mode, role) -> list[dict[str, Any]]:
        stance = cb.get("stance", "asserted")
        if mode == "sentence":
            sents = getattr(self, "_last_sentences", []) or []
            idxs = cb.get("supporting_sentence_indices")
            chosen = (
                [sents[i] for i in idxs if 0 <= i < len(sents)] if idxs else list(sents)
            )
            if not chosen:
                chosen = list(sents)
            return [
                evidence_from_sentence(
                    s.start, s.end, content, src, stance=stance, role=role
                )
                for s in chosen
            ]
        excerpts = cb.get("supporting_excerpts", [])
        return [
            evidence_from_excerpt(ex, content, src, stance=stance, role=role)
            for ex in excerpts
        ]

    def _resolve_relations(
        self,
        raw_relations,
        tmp_to_gid,
        existing_ids,
        *,
        new_node_ids=None,
        previous_node_ids=None,
        allow_current_to_current: bool = True,
    ):
        """Keep only current-turn ↔ previous-turn or current-turn ↔ current-turn edges."""
        new_gids = set(tmp_to_gid.values())
        if new_node_ids:
            new_gids |= set(new_node_ids)
        previous_gids = set(previous_node_ids or ())
        valid_types = {"depends_on", "supplements", "contradicts"}

        def _gid(ref):
            if isinstance(ref, str):
                if ref in tmp_to_gid:
                    return tmp_to_gid[ref]
                try:
                    return int(ref)
                except ValueError:
                    return None
            if isinstance(ref, (int, float)):
                return int(ref)
            return None

        out: list[dict[str, Any]] = []
        seen = set()
        active_ids = set(existing_ids) | new_gids | previous_gids
        edge_window_ids = new_gids | previous_gids

        for r in raw_relations or []:
            fid = _gid(r.get("from"))
            tid = _gid(r.get("to"))
            if fid is None or tid is None or fid == tid:
                continue
            if fid not in active_ids or tid not in active_ids:
                continue
            # The relation candidate window is exactly {current new} ∪ {previous old}.
            # Therefore every accepted edge is new→old, old→new, or new→new.
            if fid not in edge_window_ids or tid not in edge_window_ids:
                continue
            if fid not in new_gids and tid not in new_gids:
                continue
            if not allow_current_to_current and fid in new_gids and tid in new_gids:
                continue
            rtype = r.get("type")
            if rtype not in valid_types:
                continue
            key = (fid, tid, rtype)
            if key in seen:
                continue
            seen.add(key)
            note = r.get("note", "") or ""
            out.append(
                {
                    "from_id": fid,
                    "to_id": tid,
                    "type": rtype,
                    "note": note if isinstance(note, str) else str(note),
                }
            )
        return out

    @staticmethod
    def _primary_text(node: dict[str, Any]) -> str:
        if node.get("node_type") == "decision":
            return str(node.get("decision") or node.get("belief") or "")
        return str(node.get("belief") or node.get("decision") or "")

    @staticmethod
    def _set_primary_text_field(
        node: dict[str, Any],
        *,
        text_key: str,
        text: str,
    ) -> None:
        """Keep exactly one primary text field, placed after node_type.

        Decision nodes use only ``decision``; belief nodes use only ``belief``.
        This also preserves the old primary-text position in JSON output:
        id, node_type, decision/belief, stance, ...
        """
        original = dict(node)
        node.clear()
        inserted = False
        had_primary_key = "belief" in original or "decision" in original
        for key, value in original.items():
            if key in {"belief", "decision"}:
                if not inserted:
                    node[text_key] = text
                    inserted = True
                continue
            node[key] = value
            if key == "node_type" and not inserted and not had_primary_key:
                node[text_key] = text
                inserted = True
        if not inserted:
            node[text_key] = text

    def _make_node(self, cleaned, src, evid, role) -> dict[str, Any]:
        node_type = cleaned.get("node_type", "belief")
        evidence_ids = [self.graph.add_evidence(ev) for ev in evid]
        primary_text_key = "decision" if node_type == "decision" else "belief"
        primary_text = cleaned.get(primary_text_key)
        if not primary_text:
            primary_text = cleaned.get("belief") or cleaned.get("decision") or ""
        node: dict[str, Any] = {
            "id": self.graph.allocate_id(),
            "node_type": node_type,
            primary_text_key: primary_text,
            "stance": cleaned["stance"],
            "stance_confidence": float(cleaned.get("stance_confidence") or 0.0),
            "stance_scores": dict(cleaned.get("stance_scores") or {}),
            "stance_model": str(cleaned.get("stance_model") or ""),
            "role": role,
            "entities": list(cleaned.get("entities") or []),
            # Match the hybrid backend: event_time records graph-node creation
            # time and is never accepted from model output.
            "event_time": datetime.now(UTC).isoformat(),
            "source": dict(src),
            "evidence_ids": evidence_ids,
            "supporting_excerpts": [ev["text"] for ev in evid if ev.get("text")],
        }
        if cleaned.get("query"):
            node["tool_name"] = str(cleaned.get("tool_name") or "tool")
            node["query"] = str(cleaned["query"])
        if cleaned.get("source_component"):
            node["source_component"] = str(cleaned["source_component"])
        if cleaned.get("extraction_method") == "rule_tool_call":
            node["tool_name"] = str(cleaned.get("tool_name") or "tool")
            node["tool_arguments"] = dict(cleaned.get("tool_arguments") or {})
            node["tool_call_index"] = int(cleaned.get("tool_call_index") or 0)
            node["extraction_method"] = "rule_tool_call"
            node["tool_call_id"] = str(
                cleaned.get("tool_call_id")
                or (
                    f"{src.get('item_id', self.item_id)}:"
                    f"t{src.get('turn_id', -1)}:c{node['tool_call_index']}"
                )
            )
        if cleaned.get("extraction_method") in {
            "rule_tool_result",
            "compact_llm_tool_result",
        }:
            node["tool_name"] = str(cleaned.get("tool_name") or "tool")
            node["tool_result_count"] = int(cleaned.get("tool_result_count") or 0)
            node["tool_result_items"] = list(cleaned.get("tool_result_items") or [])
            node["tool_result_truncated_count"] = int(
                cleaned.get("tool_result_truncated_count") or 0
            )
            node["extraction_method"] = str(cleaned["extraction_method"])
            if cleaned.get("tool_call_id"):
                node["tool_call_id"] = str(cleaned["tool_call_id"])
            node["tool_result_index"] = int(cleaned.get("tool_result_index") or 0)
            node["tool_result_fact_index"] = int(
                cleaned.get("tool_result_fact_index") or 0
            )
            result_suffix = (
                f":{node['tool_result_index']}"
                if cleaned.get("tool_call_id") or node["tool_result_index"] > 0
                else ""
            )
            if node["tool_result_fact_index"] > 0:
                result_suffix += f":{node['tool_result_fact_index']}"
            node["tool_result_id"] = (
                f"{src.get('item_id', self.item_id)}:t{src.get('turn_id', -1)}:"
                f"result{result_suffix}"
            )
        if node_type == "decision":
            node["decision_history"] = []
        init_belief_confidence(node)
        recompute_evidence_confidence_from_node(
            node,
            self.graph.evidence,
            record_history=True,
            step="initial_evidence",
        )
        self.graph.add_belief(node)
        return node

    @staticmethod
    def _as_int(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _decision_generation_key(self, node: dict[str, Any]) -> tuple:
        """Sort key for finding the latest generated decision after merging.

        Node ids are allocated monotonically as turns are processed. A final merge
        can absorb a newer decision into an older canonical node, so `merged_from`
        is included when deciding which active decision represents the latest
        generated decision.
        """
        generated_ids = [self._as_int(node.get("id"))]
        generated_ids.extend(
            self._as_int(mid)
            for mid in (node.get("merged_from") or [])
            if self._as_int(mid) >= 0
        )
        src = node.get("source") or {}
        return (
            max(generated_ids) if generated_ids else -1,
            self._as_int(src.get("turn_id")),
            self._as_int(src.get("turn_id")),
            self._as_int(node.get("id")),
        )

    def _keep_only_latest_decision(self) -> dict[str, Any]:
        """Demote intermediate decisions to beliefs without changing edges.

        This runs after all turns have produced their nodes but before the final
        global merge pass. The latest active decision is selected by the newest
        generated node id it represents (`id` plus any `merged_from` ids). All
        other active decision nodes keep their id, source, evidence_ids, confidence,
        and incident relations; only their node type is changed to `belief`, so
        they participate in the final global belief merge.
        """
        decisions = [
            node for node in self.graph.active() if node.get("node_type") == "decision"
        ]
        report: dict[str, Any] = {
            "kept_decision_id": None,
            "kept_latest_generated_id": None,
            "converted_to_belief_ids": [],
        }
        if not decisions:
            return report

        final_decision = max(decisions, key=self._decision_generation_key)
        final_decision["node_type"] = "decision"
        self._set_primary_text_field(
            final_decision,
            text_key="decision",
            text=self._primary_text(final_decision),
        )

        report["kept_decision_id"] = final_decision.get("id")
        report["kept_latest_generated_id"] = self._decision_generation_key(
            final_decision
        )[0]

        for node in decisions:
            if node.get("id") == final_decision.get("id"):
                continue
            # A demoted decision becomes an ordinary belief. Its id is still
            # recorded later in the retained final decision's decision_history.
            demoted_text = self._primary_text(node)
            node.pop("decision_history", None)
            node["node_type"] = "belief"
            self._set_primary_text_field(
                node,
                text_key="belief",
                text=demoted_text,
            )
            report["converted_to_belief_ids"].append(node.get("id"))

        report["converted_to_belief_ids"] = sorted(
            i for i in report["converted_to_belief_ids"] if isinstance(i, int)
        )
        return report

    # ------------------------------------------------------------------ result
    def finalize(
        self,
        extra_meta: dict[str, Any] | None = None,
        pricing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("finalize() called twice")
        self._finalized = True

        # Before the final global merge, keep only the latest generated decision
        # as a decision. Earlier/intermediate decisions become ordinary beliefs
        # and therefore participate in final deduplication. The retained final
        # decision itself is then excluded from the global merge pass.
        decision_normalization = self._keep_only_latest_decision()
        final_decision_id = decision_normalization.get("kept_decision_id")
        final_merge_excluded_ids = (
            {final_decision_id} if isinstance(final_decision_id, int) else set()
        )

        _t_fm = time.perf_counter()
        # Trajectory-end global merge/dedup has been removed; only the
        # per-turn incremental merge (StreamOptions.incremental_merge) runs.
        merge_report = {
            "skipped": True,
            "skip_reason": "final merge removed",
            "applied": [],
            "excluded_node_ids": sorted(final_merge_excluded_ids),
        }
        _fm_wall = time.perf_counter() - _t_fm
        self._final_merge_timing = {"merging": 0.0, "llm_check": 0.0, "total": 0.0}

        # Record former decisions centrally only after the optional final merge.
        # The history keeps all demoted decision ids, even if one is absorbed by
        # a later merge, because it records generation history rather than active
        # graph membership.
        if isinstance(final_decision_id, int):
            final_decision = self.graph.beliefs.get(final_decision_id)
            if (
                final_decision is not None
                and final_decision.get("node_type") == "decision"
            ):
                final_decision["decision_history"] = sorted(
                    node_id
                    for node_id in decision_normalization.get(
                        "converted_to_belief_ids", []
                    )
                    if isinstance(node_id, int)
                )

        # Final snapshot
        snap_path = self.out_dir / "final_graph.json"
        self.graph.save_snapshot(snap_path, extra={"item_id": self.item_id})

        summary = {
            "n_nodes": len(self.graph.active()),
            "n_beliefs": sum(
                1
                for n in self.graph.active()
                if n.get("node_type", "belief") == "belief"
            ),
            "n_decisions": sum(
                1 for n in self.graph.active() if n.get("node_type") == "decision"
            ),
            "final_decision_id": final_decision_id,
            "relations": len(self.graph.relations),
            "n_final_merges": len(merge_report.get("applied", [])),
            "final_merge_strategy": "off",  # trajectory-end global merge removed
            "snapshot": snap_path.name,
        }
        self.graph.sessions.append(summary)

        self._end_time = datetime.now(UTC)
        duration = (self._end_time - self._start_time).total_seconds()
        nodes = self.graph.active()

        # Aggregate the per-turn sub-step timings across the whole trajectory.
        _steps = (
            "node_generation",
            "merging",
            "llm_check",
            "entity_extraction",
            "edge_generation",
            "turn_total",
        )
        by_step: dict[str, Any] = {}
        for _s in _steps:
            _vals = [float(t.get(_s, 0.0) or 0.0) for t in self._turn_timings]
            by_step[_s] = {"total_seconds": round(sum(_vals), 6), "n_turns": len(_vals)}

        result: dict[str, Any] = {
            "prompt_name": "construct_beliefs",
            "model": self.model,
            "item_id": self.item_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "mode": "stream",
            "options": self.options.to_dict(),
            "embedding_model": getattr(self.embedder, "model", None),
            "timing": {
                "start": self._start_time.isoformat(),
                "end": self._end_time.isoformat(),
                "duration_seconds": duration,
                "per_turn": self._turn_timings,
                "by_step": by_step,
                "final_merge": self._final_merge_timing,
            },
        }
        if self.item_meta:
            result["meta"] = dict(self.item_meta)
        if extra_meta:
            result.update(extra_meta)
        result.update(
            {
                "trajectory": self._trajectory,
                "final": summary,
                "all_nodes": nodes,
                "all_beliefs": [
                    n for n in nodes if n.get("node_type", "belief") == "belief"
                ],
                "all_decisions": [n for n in nodes if n.get("node_type") == "decision"],
                "evidence": [
                    self.graph.evidence[i] for i in sorted(self.graph.evidence.keys())
                ],
                "relations": self.graph.relations,
                "merges": self.graph.merges,
                "source_counts": _count_by(
                    nodes,
                    lambda b: (
                        b.get("role")
                        or (b.get("source") or {}).get("role")
                        or (b.get("source") or {}).get("type")
                    ),
                ),
                "stance_counts": _count_by(nodes, lambda b: b.get("stance")),
                "node_type_counts": _count_by(
                    nodes, lambda b: b.get("node_type", "belief")
                ),
                "token_usage": USAGE.summary(pricing),
            }
        )

        self._artifacts.write_json("result.json", result)
        self._events.record("finalize", summary)
        self._events.record(
            "timing",
            {
                "start": self._start_time.isoformat(),
                "end": self._end_time.isoformat(),
                "duration_seconds": duration,
                "by_step": by_step,
                "final_merge": self._final_merge_timing,
            },
        )
        # timing.csv — WIDE table, one file PER ITEM in this item's logs/ folder,
        # all times in SECONDS. Rows are tagged by `row_type` so a downstream viz
        # script can split them cleanly (e.g. pandas: df[df.row_type == "turn"]):
        #   row_type="turn"        one row per built turn: the four sub-steps +
        #                          turn_total; item-level count/duration columns
        #                          are left blank.
        #   row_type="final_merge" the trajectory-end merge pass, kept as an
        #                          always-zero row (the pass itself was removed;
        #                          only the per-turn incremental merge runs now).
        #   row_type="item"        one summary row per trajectory: the four
        #                          sub-step totals, the summed turn_total, and the
        #                          full-trajectory counts + duration_seconds.
        try:
            n_beliefs = sum(
                1 for n in nodes if n.get("node_type", "belief") == "belief"
            )
            n_decisions = sum(1 for n in nodes if n.get("node_type") == "decision")
            timing_path = self.logs_dir / "timing.csv"
            header = [
                "row_type",
                "item_id",
                "turn_index",
                "role",
                "node_generation",
                "merging",
                "llm_check",
                "entity_extraction",
                "edge_generation",
                "turn_total",
                "n_nodes",
                "n_beliefs",
                "n_decisions",
                "n_relations",
                "n_merges",
                "duration_seconds",
                "result_path",
            ]

            def _f(x: Any) -> str:
                return f"{float(x or 0.0):.6f}"

            # One self-contained file per item -> write fresh (overwrite) with the
            # header every time finalize runs for this trajectory.
            with open(timing_path, "w", newline="", encoding="utf-8") as csvf:
                writer = csv.writer(csvf)
                writer.writerow(header)
                # one row per built turn
                for t in self._turn_timings:
                    writer.writerow(
                        [
                            "turn",
                            self.item_id,
                            t.get("turn_index"),
                            t.get("role"),
                            _f(t.get("node_generation")),
                            _f(t.get("merging")),
                            _f(t.get("llm_check")),
                            _f(t.get("entity_extraction")),
                            _f(t.get("edge_generation")),
                            _f(t.get("turn_total")),
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                        ]
                    )
                # trajectory-end merge pass — removed, always 0
                fm = self._final_merge_timing or {}
                writer.writerow(
                    [
                        "final_merge",
                        self.item_id,
                        "",
                        "",
                        _f(0.0),
                        _f(fm.get("merging")),
                        _f(fm.get("llm_check")),
                        _f(0.0),
                        _f(0.0),
                        _f(fm.get("total")),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                # per-trajectory summary
                writer.writerow(
                    [
                        "item",
                        self.item_id,
                        "",
                        "",
                        _f(by_step["node_generation"]["total_seconds"]),
                        _f(by_step["merging"]["total_seconds"]),
                        _f(by_step["llm_check"]["total_seconds"]),
                        _f(by_step["entity_extraction"]["total_seconds"]),
                        _f(by_step["edge_generation"]["total_seconds"]),
                        _f(by_step["turn_total"]["total_seconds"]),
                        len(nodes),
                        n_beliefs,
                        n_decisions,
                        len(self.graph.relations),
                        len(self.graph.merges),
                        _f(duration),
                        str(self.out_dir / "result.json"),
                    ]
                )
        except Exception:
            self._events.record(
                "timing_csv_error", {"error": "failed to write timing.csv"}
            )

        USAGE.save_json(self.out_dir / "token_usage.json", pricing=pricing)
        USAGE.save_text(self.out_dir / "token_usage.txt", pricing=pricing)
        print(
            f"  [finalize] {len(nodes)} node(s); "
            f"{len(self.graph.relations)} relation(s); "
            f"{len(self.graph.merges)} merge record(s); {duration:.3f}s"
        )
        print(f"  saved -> {self.out_dir / 'result.json'}")
        return result


def _count_by(items: list[dict[str, Any]], key_fn) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        k = key_fn(it) or "unknown"
        out[k] = out.get(k, 0) + 1
    return out
