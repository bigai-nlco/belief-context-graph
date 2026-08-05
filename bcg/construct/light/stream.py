"""
stream.py
=========
Streaming belief/decision graph engine.

Each non-system turn is split into adjacent-distance semantic chunks. Once the
complete chunk list is available, a small generative model (Qwen behind an
OpenAI-compatible endpoint) extracts each chunk concurrently into zero or more
role-anchored belief/decision nodes. Relations are generated conservatively by
a non-thinking Qwen model after each incremental merge. Belief nodes are
incrementally merged by embeddings; decision nodes never participate in merges.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import llm
from .confidence import (
    init_belief_confidence,
    normalize_confidence_config,
    propagate_relation_confidences,
    recompute_evidence_confidence_from_node,
    relation_output_node_id,
)
from .constants import VALID_STANCES
from .evidence import evidence_from_chunk, source_descriptor
from .graph import BeliefGraph
from .llm import USAGE
from .merge import run_merge_pass
from .named_entities import NamedEntityRecognizer, normalize_entity_config
from .edge_generation import (
    VALID_RELATION_TYPES,
    get_edge_generator,
    normalize_edge_config,
)
from .stance import (
    StancePrediction,
    get_stance_classifier,
    normalize_stance_config,
)
from .._shared.roles import normalize_role
from .split import (
    semantic_breakpoint_chunks,
    semantic_chunks_isolating_tool_calls,
    single_fallback_chunk,
    split_sentences,
)
from .extractor import (
    ExtractedNode,
    extracted_nodes_as_json,
    get_extractor,
    normalize_extractor_config,
)

BELIEF_ROLES = {"user", "assistant", "tool"}


@dataclass
class StreamOptions:
    """Runtime options populated from ``model_config.json > belief_graph``."""

    evidence_mode: Optional[str] = None
    chunking_enabled: Optional[bool] = None
    breakpoint_percentile_threshold: Optional[float] = None
    chunk_buffer_size: Optional[int] = None
    min_chunk_sentences: Optional[int] = None
    isolate_tool_calls: Optional[bool] = None
    extractor_config: Dict[str, Any] = field(default_factory=dict)
    incremental_merge: Optional[bool] = None
    incremental_merge_threshold: Optional[float] = None
    merge_keep_newest_text: Optional[bool] = None
    context_chars: Optional[int] = None
    min_content_len: Optional[int] = None
    stance_config: Dict[str, Any] = field(default_factory=dict)
    confidence_config: Dict[str, Any] = field(default_factory=dict)
    entity_config: Dict[str, Any] = field(default_factory=dict)
    edge_config: Dict[str, Any] = field(default_factory=dict)

    def apply_belief_graph_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        """Apply shared model_config.json > belief_graph settings in-place."""
        if not isinstance(cfg, dict):
            raise ValueError("model config must contain a belief_graph object")

        runtime_cfg = cfg.get("runtime")
        if not isinstance(runtime_cfg, dict):
            raise ValueError("belief_graph.runtime must be an object")
        for key in ("evidence_mode", "context_chars", "min_content_len"):
            if key not in runtime_cfg:
                raise ValueError(f"belief_graph.runtime.{key} is required")
        self.evidence_mode = str(runtime_cfg["evidence_mode"])
        self.context_chars = max(0, int(runtime_cfg["context_chars"]))
        self.min_content_len = max(0, int(runtime_cfg["min_content_len"]))

        merge_cfg = cfg.get("incremental_merge")
        if not isinstance(merge_cfg, dict):
            raise ValueError("belief_graph.incremental_merge must be an object")
        for key in ("enabled", "threshold", "keep_newest_text"):
            if key not in merge_cfg:
                raise ValueError(f"belief_graph.incremental_merge.{key} is required")
        self.incremental_merge = bool(merge_cfg["enabled"])
        self.incremental_merge_threshold = float(merge_cfg["threshold"])
        self.merge_keep_newest_text = bool(merge_cfg["keep_newest_text"])

        chunk_cfg = cfg.get("chunking")
        if not isinstance(chunk_cfg, dict):
            raise ValueError("belief_graph.chunking must be an object")
        for key in (
            "enabled", "breakpoint_percentile_threshold", "buffer_size",
            "min_chunk_sentences", "isolate_tool_calls",
        ):
            if key not in chunk_cfg:
                raise ValueError(f"belief_graph.chunking.{key} is required")
        if isinstance(chunk_cfg, dict):
            if "enabled" in chunk_cfg:
                self.chunking_enabled = bool(chunk_cfg.get("enabled"))
            if chunk_cfg.get("breakpoint_percentile_threshold") is not None:
                self.breakpoint_percentile_threshold = min(
                    100.0,
                    max(0.0, float(chunk_cfg.get("breakpoint_percentile_threshold"))),
                )
            if chunk_cfg.get("buffer_size") is not None:
                self.chunk_buffer_size = max(0, int(chunk_cfg.get("buffer_size") or 0))
            if chunk_cfg.get("min_chunk_sentences") is not None:
                self.min_chunk_sentences = max(
                    1, int(chunk_cfg.get("min_chunk_sentences") or 1)
                )
            if "isolate_tool_calls" in chunk_cfg:
                self.isolate_tool_calls = bool(chunk_cfg.get("isolate_tool_calls"))

        extractor_cfg = cfg.get("extractor") or {}
        if isinstance(extractor_cfg, dict):
            merged_extractor_cfg = dict(self.extractor_config or {})
            merged_extractor_cfg.update(extractor_cfg)
            self.extractor_config = normalize_extractor_config(merged_extractor_cfg)
        else:
            self.extractor_config = normalize_extractor_config(self.extractor_config)

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

        self.edge_config = normalize_edge_config(cfg.get("edge_generation"))

        conf_cfg = cfg.get("confidence") or {}
        if isinstance(conf_cfg, dict):
            self.confidence_config = normalize_confidence_config(conf_cfg)
        else:
            self.confidence_config = normalize_confidence_config(self.confidence_config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunking": {
                "enabled": self.chunking_enabled,
                "breakpoint_percentile_threshold": self.breakpoint_percentile_threshold,
                "buffer_size": self.chunk_buffer_size,
                "min_chunk_sentences": self.min_chunk_sentences,
                "isolate_tool_calls": self.isolate_tool_calls,
            },
            "extractor": normalize_extractor_config(self.extractor_config),
            "runtime": {
                "evidence_mode": self.evidence_mode,
                "context_chars": self.context_chars,
                "min_content_len": self.min_content_len,
            },
            "incremental_merge": {
                "enabled": self.incremental_merge,
                "threshold": self.incremental_merge_threshold,
                "keep_newest_text": self.merge_keep_newest_text,
            },
            "stance": normalize_stance_config(self.stance_config),
            "entities": normalize_entity_config(self.entity_config),
            "edge_generation": normalize_edge_config(self.edge_config),
            "confidence_config": normalize_confidence_config(self.confidence_config),
        }

    def to_public_dict(self) -> Dict[str, Any]:
        """Serialize options without resolved runtime credentials."""

        data = self.to_dict()
        for section in ("extractor", "edge_generation"):
            value = data.get(section)
            if isinstance(value, dict):
                value.pop("api_key", None)
        return data


class StreamingBeliefBuilder:
    def __init__(
        self,
        *,
        client,
        model: str,
        item_id: str,
        out_dir: Path,
        options: Optional[StreamOptions] = None,
        embedder=None,
        extractor=None,
        entity_recognizer=None,
        stance_classifier=None,
        edge_generator=None,
        item_meta: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.item_id = item_id
        self.item_meta = item_meta or {}
        self.max_tokens = max_tokens
        if options is None:
            raise ValueError("StreamingBeliefBuilder requires config-populated StreamOptions")
        self.options = options
        self.embedder = embedder
        self.extractor = extractor
        self.options.extractor_config = normalize_extractor_config(self.options.extractor_config)
        self.options.stance_config = normalize_stance_config(self.options.stance_config)
        self.options.entity_config = normalize_entity_config(self.options.entity_config)
        self.options.edge_config = normalize_edge_config(self.options.edge_config)
        self.stance_classifier = (
            stance_classifier
            if stance_classifier is not None
            else get_stance_classifier(self.options.stance_config)
        )
        self.entity_recognizer = entity_recognizer or NamedEntityRecognizer(
            config=self.options.entity_config
        )
        self.edge_generator = (
            edge_generator
            if edge_generator is not None
            else get_edge_generator(self.options.edge_config)
        )
        if self.extractor is None and self.options.extractor_config.get("enabled", True):
            self.extractor = get_extractor(self.options.extractor_config)

        # Eagerly load in-process weights before any per-turn timer starts.
        # Remote Qwen endpoints load their own weights outside this process.
        stance_loader = getattr(self.stance_classifier, "_ensure_loaded", None)
        if callable(stance_loader):
            stance_loader()
        embedding_loader = getattr(self.embedder, "_ensure_model", None)
        if callable(embedding_loader):
            embedding_loader()
        entity_warmup = getattr(self.entity_recognizer, "extract_entity_texts", None)
        if callable(entity_warmup):
            entity_warmup("OpenAI is based in California.")

        self.options.confidence_config = normalize_confidence_config(self.options.confidence_config)
        self.graph = BeliefGraph(
            confidence_config=self.options.confidence_config,
            valid_relation_types=VALID_RELATION_TYPES,
        )
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.out_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self.embedder is not None:
            self.embedder.set_log_path(self.logs_dir / "embedding_calls.jsonl")
        llm.set_prompt_log_path(self.logs_dir / "prompts.jsonl")

        self._events_path = self.out_dir / "events.jsonl"
        self._events_path.write_text("", encoding="utf-8")

        self._trajectory: List[Dict[str, Any]] = []   # flat, ALL turns (incl. system)
        self._flat_turn = 0
        self._finalized = False
        self._start_time = datetime.now(timezone.utc)
        self._end_time: Optional[datetime] = None

        # Exact per-turn timing/count schema consumed downstream.
        self._turn_timings: List[Dict[str, Any]] = []
        # Compact chunking output written to result.json. Keep only the fields
        # needed to inspect how each processed turn was split.
        self._turn_chunks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ events
    def _event(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": kind}
        rec.update(payload)
        with open(self._events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def _propagate_relation_confidences(
        self,
        *,
        seed_output_node_ids: Optional[List[int]] = None,
        step: str = "relation_propagation",
    ) -> Dict[str, Any]:
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
        date: Optional[str] = None,
        has_answer: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Process one incoming turn: extract nodes, merge, then link local edges."""
        flat_idx = self._flat_turn
        turn_idx = flat_idx                       # no sessions: turn index == flat index
        content = content or ""
        raw_role = (role or "user").strip().lower()
        eff_role = normalize_role(raw_role)

        traj_entry: Dict[str, Any] = {
            "role": raw_role, "content": content, "turn_index": turn_idx,
        }
        if date is not None:
            traj_entry["date"] = date
        if has_answer is not None:
            traj_entry["has_answer"] = bool(has_answer)
        self._trajectory.append(traj_entry)

        new_nodes: List[Dict[str, Any]] = []
        relations_added = 0
        skip_reason: Optional[str] = None
        report: Dict[str, Any] = {"split": None}

        skip = (raw_role == "system" or not content.strip()
                or eff_role not in BELIEF_ROLES
                or len(content) < self.options.min_content_len)
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
                eff_role, content, turn_idx, flat_idx, date, has_answer)
            turn_total = time.perf_counter() - _t_turn
            # Normalise + round the sub-step timing produced by _update_from_turn,
            # attach turn_total, and record one wide-table row for this turn.
            st = report.get("timing") or {}
            st = {k: round(float(st.get(k, 0.0) or 0.0), 6) for k in
                  ("node_generation", "merging", "entity_extraction",
                   "edge_generation")}
            st["turn_total"] = round(turn_total, 6)
            report["timing"] = st
            active_after_turn = self.graph.active()
            self._turn_timings.append({
                "item_id": self.item_id,
                "turn_index": turn_idx,
                "role": raw_role,
                **st,
                "n_nodes": len(active_after_turn),
                "n_beliefs": sum(
                    1 for node in active_after_turn
                    if node.get("node_type", "belief") == "belief"
                ),
                "n_decisions": sum(
                    1 for node in active_after_turn
                    if node.get("node_type") == "decision"
                ),
                "n_relations": len(self.graph.relations),
                "n_merges": len(self.graph.merges),
            })

        self._flat_turn += 1
        n_merged = len((report.get("incremental_merge") or {}).get("applied", []))
        print(f"  t{turn_idx} role={raw_role:<9} -> {len(new_nodes)} node(s), "
              f"{relations_added} relation(s)"
              + (f", {n_merged} merge(s)" if n_merged else "")
              + (f"  [skip: {skip_reason}]" if skip_reason else ""))
        return self._event("turn", {
            "turn_index": turn_idx,
            "trajectory_index": flat_idx,
            "role": raw_role,
            "effective_role": eff_role,
            "content_chars": len(content),
            "skip_reason": skip_reason,
            "split": report.get("split"),
            "raw_output": report.get("raw_output"),
            "extractor_error": report.get("extractor_error"),
            "stance_classification": report.get("stance_classification"),
            "raw_relation_output": report.get("raw_relation_output"),
            "new_node_ids": [b["id"] for b in new_nodes],
            "new_belief_ids": [b["id"] for b in new_nodes if b.get("node_type", "belief") == "belief"],
            "new_decision_ids": [b["id"] for b in new_nodes if b.get("node_type") == "decision"],
            "relations_added": relations_added,
            "edge_attempts": report.get("edge_attempts"),
            "edge_linked_previous_trajectory_index": report.get(
                "edge_linked_previous_trajectory_index"),
            "edge_window_limit_reached": report.get(
                "edge_window_limit_reached", False),
            "edge_skip_reason": report.get("edge_skip_reason"),
            "incremental_merge": report.get("incremental_merge"),
            "edge_new_node_ids": report.get("edge_new_node_ids"),
            "cross_turn_anchor_ids": report.get("cross_turn_anchor_ids"),
            "entity_extraction": report.get("entity_extraction"),
            "timing": report.get("timing"),
        })

    # ------------------------------------------- per-turn four-phase pipeline
    def _classify_texts_stances(self, texts) -> List[StancePrediction]:
        """Infer one required stance per extracted node text, in batch."""
        clean = [str(text or "").strip() for text in texts]
        predictions = self.stance_classifier.classify_texts(clean)
        if len(predictions) != len(clean):
            raise RuntimeError(
                f"stance classifier returned {len(predictions)} predictions for "
                f"{len(clean)} node texts"
            )
        for prediction in predictions:
            if prediction.stance not in VALID_STANCES:
                raise RuntimeError(
                    f"stance classifier returned unsupported label {prediction.stance!r}"
                )
        return predictions

    def _graph_nodes_context_snapshot(self) -> List[Dict[str, Any]]:
        """Prior-turn nodes shown to the extractor as read-only context.

        Phase 1 runs before the current turn's nodes are created, so the active
        graph here contains only historical (earlier-turn) nodes. Relations are
        intentionally excluded from the extraction context.
        """
        return list(self.graph.active())

    def _extract_entities_for_node_ids(self, node_ids) -> Dict[str, Any]:
        """Extract entities only from active nodes whose final text is stable."""
        processed: List[int] = []
        errors: List[Dict[str, Any]] = []
        for raw_id in sorted(set(node_ids or [])):
            try:
                node_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            node = self.graph.beliefs.get(node_id)
            if node is None:
                continue
            try:
                node["entities"] = self.entity_recognizer.extract_entity_texts(
                    self._primary_text(node)
                )
                processed.append(node_id)
            except Exception as exc:
                node["entities"] = []
                errors.append({"node_id": node_id, "error": str(exc)})
        report: Dict[str, Any] = {
            "method": normalize_entity_config(self.options.entity_config)["method"],
            "processed_node_ids": processed,
            "errors": errors,
        }
        load_errors = getattr(self.entity_recognizer, "load_errors", None)
        if load_errors:
            report["fallback_errors"] = load_errors
        return report

    def _update_from_turn(
        self, role: str, content: str, turn_idx: int, flat_idx: int,
        date: Optional[str], has_answer: Optional[bool],
    ):
        """Four-phase per-turn update.

        Phase 1: semantic chunking, concurrent generative extraction, and model stance inference.
        Phase 2: complete incremental merge, evidence dedup (canonical text kept as-is).
        Phase 3: local NER on stable surviving node identities.
        Phase 4: non-thinking Qwen edge generation on the post-merge window.
        """
        opt = self.options
        report: Dict[str, Any] = {"split": None}
        timing = {
            "node_generation": 0.0,
            "merging": 0.0,
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

        # ---- PHASE 1: split the complete turn, then extract all chunks together.
        _t_nodes = time.perf_counter()
        if opt.isolate_tool_calls:
            chunks, split_info = semantic_chunks_isolating_tool_calls(
                content,
                self.embedder,
                enabled=opt.chunking_enabled,
                breakpoint_percentile_threshold=opt.breakpoint_percentile_threshold,
                buffer_size=opt.chunk_buffer_size,
                min_chunk_sentences=opt.min_chunk_sentences,
                purpose=f"chunk:t{turn_idx}",
            )
        else:
            sentences = split_sentences(content)
            if not opt.chunking_enabled:
                chunks, split_info = single_fallback_chunk(
                    sentences, content, reason="chunking disabled"
                )
            else:
                try:
                    chunks, split_info = semantic_breakpoint_chunks(
                        sentences,
                        content,
                        self.embedder,
                        breakpoint_percentile_threshold=opt.breakpoint_percentile_threshold,
                        buffer_size=opt.chunk_buffer_size,
                        min_chunk_sentences=opt.min_chunk_sentences,
                        purpose=f"chunk:t{turn_idx}",
                    )
                except Exception as exc:
                    chunks, split_info = single_fallback_chunk(
                        sentences, content, reason=str(exc)
                    )
        report["split"] = split_info
        self._turn_chunks.append({
            "turn_id": turn_idx,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "sentences": [sentence.text for sentence in chunk.sentences],
                    "content": chunk.text,
                }
                for chunk in chunks
            ],
        })

        # Generative extraction: all chunks of this turn are submitted together
        # and may each yield zero, one, or several nodes. Historical graph nodes
        # (no relations) are handed to the model as read-only context.
        graph_nodes_context = (
            self._graph_nodes_context_snapshot()
            if self.extractor is not None
            and self.options.extractor_config.get("context_scope") == "graph"
            else []
        )
        try:
            if self.extractor is None:
                per_chunk_nodes: List[List[ExtractedNode]] = [[] for _ in chunks]
                report["extractor_error"] = "extractor disabled"
            else:
                per_chunk_nodes = self.extractor.extract_turn(
                    chunks,
                    role,
                    turn_content=content,
                    graph_nodes=graph_nodes_context,
                    context_chars=opt.context_chars,
                    turn_index=turn_idx,
                )
            report["skipped"] = False
        except Exception as exc:
            # A whole-turn extractor failure yields zero nodes for the turn
            # (allowed). Stance inference is only needed if nodes exist.
            per_chunk_nodes = [[] for _ in chunks]
            report["extractor_error"] = str(exc)
            report["skipped"] = False
        report["raw_output"] = extracted_nodes_as_json(per_chunk_nodes)

        # Flatten to (chunk, extracted_node) pairs, preserving order.
        flat: List[tuple] = [
            (chunk, extracted)
            for chunk, group in zip(chunks, per_chunk_nodes)
            for extracted in group
        ]

        # Stance is inferred per extracted node TEXT (not per chunk).
        if flat:
            try:
                stance_predictions = self._classify_texts_stances(
                    [extracted.text for _, extracted in flat]
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Local stance classification failed for turn={turn_idx}: {exc}"
                ) from exc
        else:
            stance_predictions = []
        report["stance_classification"] = {
            "model_path": self.options.stance_config["model_path"],
            "predictions": [prediction.to_dict() for prediction in stance_predictions],
        }
        timing["node_generation"] = time.perf_counter() - _t_nodes

        new_nodes: List[Dict[str, Any]] = []
        current_node_ids: set = set()
        for (chunk, extracted), stance_prediction in zip(flat, stance_predictions):
            node_type = extracted.node_type if extracted.node_type in {"belief", "decision"} else "belief"
            # Decisions are honoured only for assistant turns.
            if node_type == "decision" and role != "assistant":
                node_type = "belief"
            text_key = "decision" if node_type == "decision" else "belief"
            cleaned = {
                "node_type": node_type,
                text_key: extracted.text,
                "stance": stance_prediction.stance,
                "stance_confidence": stance_prediction.confidence,
                "stance_scores": dict(stance_prediction.scores),
                "stance_model": stance_prediction.model_path,
            }
            # Evidence granularity: every node from a chunk shares that chunk's
            # full contiguous span (one evidence record per node).
            evidence = evidence_from_chunk(
                chunk.start,
                chunk.end,
                content,
                src,
                chunk_index=chunk.chunk_id,
                sentence_indices=chunk.sentence_indices,
                stance=stance_prediction.stance,
                stance_confidence=stance_prediction.confidence,
                stance_scores=stance_prediction.scores,
                stance_model=stance_prediction.model_path,
                role=role,
            )
            node = self._make_node(cleaned, src, [evidence], role)
            new_nodes.append(node)
            current_node_ids.add(node["id"])

        generated_node_ids = set(current_node_ids)

        # ---- PHASE 2: belief-only embedding merge. A canonical historical node
        # produced by absorbing a current node is treated as a new node for this
        # round's relation classification. Cross-turn stop detection is handled by a
        # separate set of current-turn ids that actually survive the merge.
        if new_nodes and opt.incremental_merge and self.embedder is not None:
            USAGE.set_label(f"t{turn_idx}.merge")
            inc = run_merge_pass(
                graph=self.graph,
                strategy="embedding",
                verify=False,
                client=self.client,
                model=self.model,
                embedder=self.embedder,
                threshold=opt.incremental_merge_threshold,
                max_tokens=self.max_tokens,
                pass_label=f"turn_{turn_idx}",
                log_dir=None,
                incremental_new_ids=set(current_node_ids),
                summary_regenerator=None,
                keep_newest_text=opt.merge_keep_newest_text,
            )
            merge_timing = inc.get("timing") or {}
            timing["merging"] += float(
                merge_timing.get("embedding_seconds", 0.0) or 0.0
            )
            report["incremental_merge"] = {
                "applied": inc.get("applied", []),
                "excluded_decision_ids": inc.get("excluded_decision_ids", []),
                "relation_rewire": inc.get("relation_rewire"),
                "evidence_prune": inc.get("evidence_prune"),
                "summary_regeneration": inc.get("summary_regeneration"),
                "skip_reason": inc.get("skip_reason"),
            }
            for merge in inc.get("applied", []):
                canonical_id = merge.get("canonical_id")
                absorbed_ids = {
                    int(value) for value in (merge.get("absorbed_ids") or [])
                    if isinstance(value, int)
                }
                touched_current = bool(current_node_ids & absorbed_ids)
                if isinstance(canonical_id, int) and canonical_id in current_node_ids:
                    touched_current = True
                current_node_ids.difference_update(absorbed_ids)
                if touched_current and isinstance(canonical_id, int):
                    current_node_ids.add(canonical_id)
            if inc.get("applied"):
                report.setdefault("confidence_propagation", []).append({
                    "trigger": "incremental_merge",
                    "report": self._propagate_relation_confidences(
                        seed_output_node_ids=None,
                        step="relation_propagation_after_merge",
                    ),
                })

        relations_added = 0
        active_nodes = self.graph.active()
        active_ids = set(self.graph.ids())
        edge_new_node_ids = current_node_ids & active_ids
        cross_turn_anchor_ids = generated_node_ids & active_ids

        # ---- PHASE 3: extract entities only after all incremental merges and
        # canonical summary regeneration for this turn have completed.
        _t_entities = time.perf_counter()
        report["entity_extraction"] = self._extract_entities_for_node_ids(
            edge_new_node_ids
        )
        timing["entity_extraction"] += time.perf_counter() - _t_entities

        # ---- PHASE 4: conservative Qwen edge generation on stable identities.
        report["edge_new_node_ids"] = sorted(edge_new_node_ids)
        report["cross_turn_anchor_ids"] = sorted(cross_turn_anchor_ids)
        report["edge_attempts"] = []

        if edge_new_node_ids:
            tried_prior_window = False
            linked_prior_turn = None
            attempted_windows = 0

            candidate_indices = range(flat_idx - 1, -1, -1)
            if not self.options.edge_config["search_previous_turns"]:
                candidate_indices = range(flat_idx - 1, max(-1, flat_idx - 2), -1)
            for candidate_idx in candidate_indices:
                previous_node_ids = self._node_ids_from_trajectory_index(
                    active_nodes, candidate_idx
                )
                # A historical canonical node that absorbed current evidence is a
                # current identity for this round, not simultaneously a prior node.
                previous_node_ids.difference_update(edge_new_node_ids)

                if not previous_node_ids:
                    report["edge_attempts"].append({
                        "previous_trajectory_index": candidate_idx,
                        "previous_node_ids": [],
                        "relations_added": 0,
                        "cross_turn_relations_added": 0,
                        "skip_reason": "no active prior nodes outside current merge identities",
                    })
                    continue

                if (
                    attempted_windows
                    >= self.options.edge_config["max_previous_windows"]
                ):
                    report["edge_window_limit_reached"] = True
                    break
                attempted_windows += 1
                tried_prior_window = True
                _t_edge = time.perf_counter()
                added, added_cross, attempt = self._extract_relations_for_edge_window(
                    turn_idx=turn_idx,
                    previous_trajectory_index=candidate_idx,
                    active_nodes=active_nodes,
                    active_ids=active_ids,
                    edge_new_node_ids=edge_new_node_ids,
                    cross_turn_anchor_ids=cross_turn_anchor_ids,
                    previous_node_ids=previous_node_ids,
                )
                edge_elapsed = time.perf_counter() - _t_edge
                timing["edge_generation"] += edge_elapsed
                report["edge_attempts"].append(attempt)
                report["raw_relation_output"] = attempt.get("raw_relation_output")
                relations_added += added

                if added_cross > 0:
                    linked_prior_turn = candidate_idx
                    break

            if not tried_prior_window:
                report["edge_skip_reason"] = "no earlier turn with active nodes"

            report["edge_linked_previous_trajectory_index"] = linked_prior_turn
        else:
            report["edge_skip_reason"] = (
                "no active current-turn node identity after incremental merge"
                if new_nodes else "no current-turn nodes generated"
            )

        return new_nodes, relations_added, report

    def _extract_relations_for_edge_window(
        self,
        *,
        turn_idx: int,
        previous_trajectory_index: Optional[int],
        active_nodes: List[Dict[str, Any]],
        active_ids: set,
        edge_new_node_ids: set,
        cross_turn_anchor_ids: set,
        previous_node_ids: set,
    ):
        """Run one Qwen relation-generation attempt for one edge window."""
        edge_window_ids = edge_new_node_ids | previous_node_ids
        window_nodes = [
            node for node in active_nodes if node.get("id") in edge_window_ids
        ]

        if self.edge_generator is None:
            rel_res = {
                "relations": [],
                "diagnostics": {
                    "skipped": True,
                    "skip_reason": "edge generation disabled",
                },
            }
        else:
            try:
                rel_res = self.edge_generator.generate_window(
                    window_nodes,
                    current_node_ids=set(edge_new_node_ids),
                    turn_index=turn_idx,
                    previous_turn_index=previous_trajectory_index,
                )
            except Exception as exc:
                if self.options.edge_config["fail_on_error"]:
                    raise RuntimeError(
                        "Qwen edge generation failed for "
                        f"turn={turn_idx}, previous_turn={previous_trajectory_index}: {exc}"
                    ) from exc
                rel_res = {
                    "relations": [],
                    "diagnostics": {
                        "skipped": True,
                        "skip_reason": str(exc),
                        "error_type": type(exc).__name__,
                    },
                }

        resolved = self._resolve_relations(
            rel_res.get("relations", []), {}, active_ids,
            new_node_ids=edge_new_node_ids,
            previous_node_ids=previous_node_ids)

        existing_keys = {
            (r.get("from_id"), r.get("to_id"), r.get("type"))
            for r in self.graph.relations
        }
        resolved_keys = {
            (r.get("from_id"), r.get("to_id"), r.get("type"))
            for r in resolved
        }
        new_keys = resolved_keys - existing_keys
        cross_turn_relations_added = sum(
            1
            for r in resolved
            if (r.get("from_id"), r.get("to_id"), r.get("type")) in new_keys
            and (
                (r.get("from_id") in cross_turn_anchor_ids
                 and r.get("to_id") in previous_node_ids)
                or
                (r.get("from_id") in previous_node_ids
                 and r.get("to_id") in cross_turn_anchor_ids)
            )
        )
        before_relation_count = len(self.graph.relations)
        relations_added = self.graph.add_relations(resolved)
        added_relations = self.graph.relations[before_relation_count:]
        seed_output_node_ids = [
            node_id for node_id in
            (relation_output_node_id(relation) for relation in added_relations)
            if node_id is not None
        ]
        propagation_report = (
            self._propagate_relation_confidences(
                seed_output_node_ids=seed_output_node_ids,
                step="relation_propagation_after_relation_add",
            )
            if seed_output_node_ids else None
        )

        diagnostics = rel_res.get("diagnostics") or {}
        attempt = {
            "previous_trajectory_index": previous_trajectory_index,
            "previous_node_ids": sorted(previous_node_ids),
            "new_node_ids": sorted(edge_new_node_ids),
            "cross_turn_anchor_ids": sorted(cross_turn_anchor_ids),
            "edge_window_ids": sorted(edge_window_ids),
            "relations_added": relations_added,
            "cross_turn_relations_added": cross_turn_relations_added,
            "raw_relation_output": json.dumps(diagnostics, ensure_ascii=False),
        }
        if propagation_report is not None:
            attempt["confidence_propagation"] = propagation_report
        if diagnostics.get("skipped"):
            attempt["skip_reason"] = diagnostics.get("skip_reason") or "edge generation skipped"
        return relations_added, cross_turn_relations_added, attempt

    @staticmethod
    def _node_ids_from_trajectory_index(
        nodes: List[Dict[str, Any]], trajectory_index: int
    ) -> set:
        """Return active node ids whose source belongs to one trajectory turn."""
        return {
            node["id"]
            for node in nodes
            if isinstance(node.get("id"), int)
            and (node.get("source") or {}).get("turn_id") == trajectory_index
        }

    def _resolve_relations(
        self,
        raw_relations,
        tmp_to_gid,
        existing_ids,
        *,
        new_node_ids=None,
        previous_node_ids=None,
    ):
        """Keep only current-turn ↔ previous-turn or current-turn ↔ current-turn edges."""
        new_gids = set(tmp_to_gid.values())
        if new_node_ids:
            new_gids |= set(new_node_ids)
        previous_gids = set(previous_node_ids or ())
        valid_types = set(self.graph.valid_relation_types)

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

        out: List[Dict[str, Any]] = []
        seen = set()
        active_ids = set(existing_ids) | new_gids | previous_gids
        edge_window_ids = new_gids | previous_gids

        for r in raw_relations or []:
            fid = _gid(r.get("from", r.get("from_id")))
            tid = _gid(r.get("to", r.get("to_id")))
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
            rtype = r.get("type")
            if rtype not in valid_types:
                continue
            key = (fid, tid, rtype)
            if key in seen:
                continue
            seen.add(key)
            note = r.get("note", "") or ""
            clean = {"from_id": fid, "to_id": tid, "type": rtype,
                     "note": note if isinstance(note, str) else str(note)}
            out.append(clean)
        return out

    @staticmethod
    def _primary_text(node: Dict[str, Any]) -> str:
        if node.get("node_type") == "decision":
            return str(node.get("decision") or node.get("belief") or "")
        return str(node.get("belief") or node.get("decision") or "")

    @staticmethod
    def _set_primary_text_field(
        node: Dict[str, Any],
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

    def _make_node(self, cleaned, src, evid, role) -> Dict[str, Any]:
        node_type = cleaned.get("node_type", "belief")
        primary_text_key = "decision" if node_type == "decision" else "belief"
        primary_text = cleaned.get(primary_text_key)
        if not primary_text:
            primary_text = cleaned.get("belief") or cleaned.get("decision") or ""

        stance = str(cleaned.get("stance") or "").strip().lower()
        if stance not in VALID_STANCES:
            raise ValueError(
                f"Every generated node requires a valid model-inferred stance; got {stance!r}"
            )
        for evidence in evid:
            evidence_stance = str(evidence.get("stance") or "").strip().lower()
            if evidence_stance != stance:
                raise ValueError(
                    "Node/evidence stance mismatch during node creation: "
                    f"node={stance!r}, evidence={evidence_stance!r}"
                )

        evidence_ids = [self.graph.add_evidence(ev) for ev in evid]
        node: Dict[str, Any] = {
            "id": self.graph.allocate_id(),
            "node_type": node_type,
            primary_text_key: primary_text,
            "stance": stance,
            "stance_confidence": float(cleaned.get("stance_confidence") or 0.0),
            "stance_scores": dict(cleaned.get("stance_scores") or {}),
            "stance_model": str(cleaned.get("stance_model") or ""),
            "role": role,
            "entities": [],
            "event_time": datetime.now(timezone.utc).isoformat(),
            "source": dict(src),
            "evidence_ids": evidence_ids,
            "initial_evidence_count": len(evidence_ids),
            "supporting_excerpts": [ev["text"] for ev in evid if ev.get("text")],
        }
        if node_type == "decision":
            node["decision_history"] = []
        init_belief_confidence(node, config=self.options.confidence_config)
        recompute_evidence_confidence_from_node(
            node, self.graph.evidence,
            record_history=True,
            step="initial_evidence",
            config=self.options.confidence_config,
        )
        self.graph.add_belief(node)
        return node

    def _keep_only_latest_decision(self) -> Dict[str, Any]:
        """At trajectory end, keep only the active decision with max id.

        Decision nodes are excluded from incremental merge passes. At
        trajectory end, every older active decision is converted in place to a
        belief; its id is preserved in the retained decision's decision_history.
        """
        decisions = [
            node for node in self.graph.active()
            if node.get("node_type") == "decision"
            and isinstance(node.get("id"), int)
        ]
        report: Dict[str, Any] = {
            "kept_decision_id": None,
            "converted_to_belief_ids": [],
        }
        if not decisions:
            return report

        final_decision = max(decisions, key=lambda node: int(node["id"]))
        final_decision_id = int(final_decision["id"])
        prior_history = {
            int(value)
            for value in (final_decision.get("decision_history") or [])
            if isinstance(value, int)
        }

        final_decision["node_type"] = "decision"
        self._set_primary_text_field(
            final_decision,
            text_key="decision",
            text=self._primary_text(final_decision),
        )
        report["kept_decision_id"] = final_decision_id

        converted_ids: List[int] = []
        for node in decisions:
            node_id = int(node["id"])
            if node_id == final_decision_id:
                continue
            demoted_text = self._primary_text(node)
            node.pop("decision_history", None)
            node["node_type"] = "belief"
            self._set_primary_text_field(
                node,
                text_key="belief",
                text=demoted_text,
            )
            converted_ids.append(node_id)

        converted_ids.sort()
        final_decision["decision_history"] = sorted(prior_history | set(converted_ids))
        report["converted_to_belief_ids"] = converted_ids
        report["decision_history"] = list(final_decision["decision_history"])
        return report

    # ------------------------------------------------------------------ result
    def finalize(
        self,
        extra_meta: Optional[Dict[str, Any]] = None,
        pricing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._finalized:
            raise RuntimeError("finalize() called twice")
        self._finalized = True

        decision_normalization = self._keep_only_latest_decision()
        final_decision_id = decision_normalization.get("kept_decision_id")

        # Final snapshot
        snap_path = self.out_dir / "final_graph.json"
        self.graph.save_snapshot(snap_path, extra={"item_id": self.item_id})

        summary = {
            "n_nodes": len(self.graph.active()),
            "n_beliefs": sum(1 for n in self.graph.active() if n.get("node_type", "belief") == "belief"),
            "n_decisions": sum(1 for n in self.graph.active() if n.get("node_type") == "decision"),
            "final_decision_id": final_decision_id,
            "relations": len(self.graph.relations),
            "decision_normalization": decision_normalization,
            "snapshot": snap_path.name,
        }
        self.graph.sessions.append(summary)

        self._end_time = datetime.now(timezone.utc)
        duration = (self._end_time - self._start_time).total_seconds()
        nodes = self.graph.active()

        result: Dict[str, Any] = {
            "prompt_name": "construct_beliefs",
            "model": self.model,
            "item_id": self.item_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "stream",
            "options": self.options.to_public_dict(),
            "embedding_model": getattr(self.embedder, "model", None),
            "timing": self._turn_timings,
            "turn_chunks": self._turn_chunks,
        }
        if self.item_meta:
            result["meta"] = dict(self.item_meta)
        if extra_meta:
            result.update(extra_meta)
        result.update({
            "trajectory": self._trajectory,
            "final": summary,
            "all_nodes": nodes,
            "all_beliefs": [n for n in nodes if n.get("node_type", "belief") == "belief"],
            "all_decisions": [n for n in nodes if n.get("node_type") == "decision"],
            "evidence": [self.graph.evidence[i] for i in sorted(self.graph.evidence.keys())],
            "relations": self.graph.relations,
            "merges": self.graph.merges,
            "source_counts": _count_by(nodes, lambda b: b.get("role") or (b.get("source") or {}).get("role") or (b.get("source") or {}).get("type")),
            "stance_counts": _count_by(nodes, lambda b: b.get("stance")),
            "node_type_counts": _count_by(nodes, lambda b: b.get("node_type", "belief")),
            "token_usage": USAGE.summary(pricing),
        })

        with open(self.out_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        self._event("finalize", summary)
        # timing.csv mirrors result.json's timing rows exactly.
        try:
            timing_path = self.logs_dir / "timing.csv"
            header = [
                "item_id", "turn_index", "role", "node_generation", "merging",
                "entity_extraction", "edge_generation", "turn_total", "n_nodes",
                "n_beliefs", "n_decisions", "n_relations", "n_merges",
            ]
            with open(timing_path, "w", newline='', encoding="utf-8") as csvf:
                writer = csv.DictWriter(csvf, fieldnames=header, extrasaction="ignore")
                writer.writeheader()
                for t in self._turn_timings:
                    writer.writerow(t)
        except Exception:
            self._event("timing_csv_error", {"error": "failed to write timing.csv"})

        USAGE.save_json(self.out_dir / "token_usage.json", pricing=pricing)
        USAGE.save_text(self.out_dir / "token_usage.txt", pricing=pricing)
        print(f"  [finalize] {len(nodes)} node(s); "
              f"{len(self.graph.relations)} relation(s); "
              f"{len(self.graph.merges)} merge record(s); {duration:.3f}s")
        print(f"  saved -> {self.out_dir / 'result.json'}")
        return result


def _count_by(items: List[Dict[str, Any]], key_fn) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        k = key_fn(it) or "unknown"
        out[k] = out.get(k, 0) + 1
    return out
