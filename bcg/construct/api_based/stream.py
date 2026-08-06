"""
stream.py
=========
Streaming belief/decision graph engine.

Each turn is routed only by role (user / assistant / tool; system is recorded
but yields no nodes; function == tool). For every non-skipped turn, one LLM call
extracts new belief nodes and new decision nodes. Relation extraction then runs
on the current-turn / prior-turn edge window; if the nearest prior turn cannot
yield a cross-turn edge, the window walks backward one turn at a time until a
current-to-prior edge is added or no earlier active turn remains.

Node schema additions:
  * node_type: "belief" | "decision"
  * entities: list[str]
  * evidence_ids: list[int]
  * confidence / initial_confidence / evidence_confidence

Relation schema:
  * depends_on | supplements | contradicts

  * merging is incremental only (per-turn, embedding-based); there is no
    trajectory-end global merge/dedup pass.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .._shared.roles import normalize_role
from .._shared.writers import ArtifactWriter, EventRecorder
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
    extract_nodes,
    extract_relations,
    format_graph_edges,
    format_graph_nodes,
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
    incremental_merge_threshold: float = 0.8
    # When True, the per-turn incremental merge additionally calls the LLM once per
    # embedding-flagged candidate group to (1) verify the merge is reasonable
    # (apply-time gate: only LLM-confirmed groups are merged) and (2) rewrite the
    # surviving node's content so it covers the full meaning of all merged nodes.
    # Needs an embedder (same as incremental_merge).
    verify_merge: bool = False
    # prompt budgets
    context_chars: int = 9000  # existing-nodes context budget
    # skip turns whose content is shorter than this (0 = never skip)
    min_content_len: int = 0
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_mode": self.evidence_mode,
            "incremental_merge": self.incremental_merge,
            "incremental_merge_threshold": self.incremental_merge_threshold,
            "verify_merge": self.verify_merge,
            "context_chars": self.context_chars,
            "min_content_len": self.min_content_len,
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
        item_meta: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.item_id = item_id
        self.item_meta = item_meta or {}
        self.max_tokens = max_tokens
        self.options = options or StreamOptions()
        self.options.confidence_config = normalize_confidence_config(
            self.options.confidence_config
        )
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
        self._finalized = False
        self._start_time = datetime.now(UTC)
        self._end_time: datetime | None = None

        # Per-turn sub-step timing (seconds). One record per NON-skipped turn:
        #   {turn_index, role, node_generation, merging, llm_check,
        #    edge_generation, turn_total}. Populated in ingest_turn.
        self._turn_timings: list[dict[str, Any]] = []
        # Final (trajectory-end) merge timing; filled in finalize(). Always
        # zero now that the trajectory-end global merge has been removed.
        self._final_merge_timing: dict[str, Any] = {
            "merging": 0.0,
            "llm_check": 0.0,
            "total": 0.0,
        }

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
            turn_total = time.perf_counter() - _t_turn
            # Normalise + round the sub-step timing produced by _update_from_turn,
            # attach turn_total, and record one wide-table row for this turn.
            st = report.get("timing") or {}
            st = {
                k: round(float(st.get(k, 0.0) or 0.0), 6)
                for k in ("node_generation", "merging", "llm_check", "edge_generation")
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
                "split": report.get("split"),
                "raw_output": report.get("raw_output"),
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
                "edge_skip_reason": report.get("edge_skip_reason"),
                "incremental_merge": report.get("incremental_merge"),
                "timing": report.get("timing"),
            },
        )

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
        Phase 1 — extract belief/decision nodes (one LLM call)
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

        graph_nodes_str = format_graph_nodes(
            self.graph.active(), char_budget=opt.context_chars
        )
        graph_edges_str = format_graph_edges(
            self.graph.relations, keep_ids=set(self.graph.ids())
        )

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
        USAGE.set_label(f"t{turn_idx}.extract:{role}")
        _t_nodes = time.perf_counter()
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
        )
        timing["node_generation"] = time.perf_counter() - _t_nodes
        report["raw_output"] = node_res.get("raw_output")
        report["skipped"] = node_res.get("skipped", False)
        if node_res.get("skip_reason"):
            report["skip_reason"] = node_res["skip_reason"]

        # ---- allocate ids + attach evidence (in output order, so n0<n1<… in id)
        tmp_to_gid: dict[str, int] = {}
        new_nodes: list[dict[str, Any]] = []
        new_node_ids: set = set()
        for cb in node_res.get("nodes", []):
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
        # The original online policy links the current surviving new nodes against
        # the immediately previous turn's surviving nodes, plus current new <->
        # current new. If that adjacent turn cannot yield a cross-turn edge
        # (because it has no surviving nodes, or because the relation extractor
        # finds no accepted edge), walk backward one turn at a time and stop as
        # soon as a real current-turn <-> prior-turn edge is added. old <-> old
        # edges remain impossible because every accepted relation must include a
        # current surviving new node.
        relations_added = 0
        active_nodes = self.graph.active()
        active_ids = set(self.graph.ids())
        surviving_new_ids = new_node_ids & active_ids
        report["edge_attempts"] = []

        if surviving_new_ids:
            tried_prior_window = False
            linked_prior_turn = None

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
            if not tried_prior_window:
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

        return new_nodes, relations_added, report

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
        graph_nodes_post = format_graph_nodes(
            [node for node in active_nodes if node.get("id") in edge_window_ids],
            char_budget=context_chars,
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
        )

        resolved = self._resolve_relations(
            rel_res.get("relations", []),
            {},
            active_ids,
            new_node_ids=surviving_new_ids,
            previous_node_ids=previous_node_ids,
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
            "role": role,
            "entities": list(cleaned.get("entities") or []),
            "event_time": cleaned.get("event_time"),
            "time_text": cleaned.get("time_text"),
            "source": dict(src),
            "evidence_ids": evidence_ids,
            "supporting_excerpts": [ev["text"] for ev in evid if ev.get("text")],
        }
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
