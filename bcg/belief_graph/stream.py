"""
stream.py  (v3)
===============
The streaming belief-graph engine. One builder handles ONE trajectory (one
item / one problem_id) and is fed turns in order:

    builder = StreamingBeliefBuilder(...)
    builder.ingest_turn(role, content)        # repeat per turn (role-only)
    ...
    result = builder.finalize()               # writes result.json

There are NO scenarios and NO sessions. Turns are routed only by role
(user / assistant / tool; system is recorded but yields no beliefs;
function == tool). Tags inside content are NOT used to split the turn — the
whole content is handed to the model, which decides how many beliefs to extract.

Per TURN (`ingest_turn`) — ONE LLM call:
    1. (sentence mode) split the content into COMPLETE sentences with exact
       offsets; optionally group them by topic cluster (still one call).
    2. ONE call to extract.update_graph: existing graph (nodes + forward edges)
       as read-only context  →  NEW nodes (temp ids) + NEW forward edges.
    3. allocate monotonic ids, resolve temp ids, attach evidence (whole
       sentences, or located excerpts), add nodes + validated forward edges.

At FINALIZE (called once, e.g. on is_trajectory_end):
    1. ONE backward call over the FULL graph → confirms / contradicts / extends;
    2. incremental confidence update from those relations;
    3. merge / dedup pass over the full graph;
    4. final snapshot → final_graph.json, then result.json.

Every step is appended to events.jsonl for replay/debugging.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .confidence import apply_relations_incremental, init_belief_confidence
from .evidence import (
    evidence_from_excerpt,
    evidence_from_sentence,
    source_descriptor,
)
from .extract import (
    format_graph_edges,
    format_graph_nodes,
    update_graph,
)
from .graph import BeliefGraph
from .link import link_backward_all
from . import llm
from .llm import USAGE
from .merge import run_merge_pass
from .split import cluster_sentences, split_sentences

# roles that produce beliefs; "function" is treated as "tool".
_ROLE_ALIASES = {"function": "tool"}
BELIEF_ROLES = {"user", "assistant", "tool"}


@dataclass
class StreamOptions:
    # evidence mode: "sentence" (whole-sentence evidence) | "excerpt" (free spans)
    evidence_mode: str = "sentence"
    # optional topic clustering of sentences (sentence mode only; needs embedder)
    use_clustering: bool = False
    cluster_threshold: float = 0.6
    cluster_buffer: int = 0
    cluster_min_sentences: int = 4        # below this, skip clustering (flat call)
    # merge / dedup (runs once at finalize)
    merge_strategy: str = "embedding"     # embedding | llm | off
    merge_threshold: float = 0.86
    # prompt budgets
    context_chars: int = 9000             # existing-nodes context budget
    # skip turns whose content is shorter than this (0 = never skip)
    min_content_len: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_mode": self.evidence_mode,
            "use_clustering": self.use_clustering,
            "cluster_threshold": self.cluster_threshold,
            "cluster_buffer": self.cluster_buffer,
            "cluster_min_sentences": self.cluster_min_sentences,
            "merge_strategy": self.merge_strategy,
            "merge_threshold": self.merge_threshold,
            "context_chars": self.context_chars,
            "min_content_len": self.min_content_len,
        }


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
        item_meta: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.item_id = item_id
        self.item_meta = item_meta or {}
        self.max_tokens = max_tokens
        self.options = options or StreamOptions()
        self.embedder = embedder

        self.graph = BeliefGraph()
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

    # ------------------------------------------------------------------ events
    def _event(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": kind}
        rec.update(payload)
        with open(self._events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    # ------------------------------------------------------------- turn ingest
    def ingest_turn(
        self,
        role: str,
        content: str,
        date: Optional[str] = None,
        has_answer: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Process one incoming turn: ONE call → new nodes + new forward edges."""
        flat_idx = self._flat_turn
        turn_idx = flat_idx                       # no sessions: turn index == flat index
        content = content or ""
        raw_role = (role or "user").strip().lower()
        eff_role = _ROLE_ALIASES.get(raw_role, raw_role)

        traj_entry: Dict[str, Any] = {
            "role": raw_role, "content": content, "turn_index": turn_idx,
        }
        if date is not None:
            traj_entry["date"] = date
        if has_answer is not None:
            traj_entry["has_answer"] = bool(has_answer)
        self._trajectory.append(traj_entry)

        new_beliefs: List[Dict[str, Any]] = []
        forward_added = 0
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
            new_beliefs, forward_added, report = self._update_from_turn(
                eff_role, content, turn_idx, flat_idx, date, has_answer)

        self._flat_turn += 1
        print(f"  t{turn_idx} role={raw_role:<9} -> {len(new_beliefs)} belief(s), "
              f"{forward_added} informs edge(s)"
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
            "new_belief_ids": [b["id"] for b in new_beliefs],
            "forward_added": forward_added,
        })

    # ----------------------------------------------------- per-turn single call
    def _update_from_turn(
        self, role: str, content: str, turn_idx: int, flat_idx: int,
        date: Optional[str], has_answer: Optional[bool],
    ):
        opt = self.options
        report: Dict[str, Any] = {"split": None}

        src = source_descriptor(
            role=role, item_id=self.item_id, turn_index=turn_idx,
            flat_turn_index=flat_idx, date=date, has_answer=has_answer)

        graph_nodes_str = format_graph_nodes(
            self.graph.active(), char_budget=opt.context_chars)
        graph_edges_str = format_graph_edges(
            self.graph.forward_relations, keep_ids=set(self.graph.ids()))

        # ---- prepare evidence mode
        sentences = None
        clusters_idx = None
        if opt.evidence_mode == "sentence":
            sents = split_sentences(content)
            sentences = [s.text for s in sents]
            self._last_sentences = sents
            if (opt.use_clustering and self.embedder is not None
                    and len(sents) >= opt.cluster_min_sentences):
                try:
                    clusters, split_info = cluster_sentences(
                        sents, self.embedder,
                        similarity_threshold=opt.cluster_threshold,
                        buffer_size=opt.cluster_buffer,
                        purpose=f"split:t{turn_idx}")
                    clusters_idx = [c.sentence_indices for c in clusters]
                    report["split"] = split_info
                except Exception as e:
                    report["split"] = {"error": str(e)}
                    clusters_idx = None
        else:
            self._last_sentences = []

        USAGE.set_label(f"t{turn_idx}.update:{role}")
        res = update_graph(
            self.client, self.model,
            role=role, mode=opt.evidence_mode,
            content=content, sentences=sentences, clusters=clusters_idx,
            graph_nodes_str=graph_nodes_str, graph_edges_str=graph_edges_str,
            current_date=date, max_tokens=self.max_tokens)
        report["raw_output"] = res.get("raw_output")
        report["skipped"] = res.get("skipped", False)
        if res.get("skip_reason"):
            report["skip_reason"] = res["skip_reason"]

        # ---- allocate ids + attach evidence (in output order, so n0<n1<… in id)
        tmp_to_gid: Dict[str, int] = {}
        new_beliefs: List[Dict[str, Any]] = []
        for cb in res["beliefs"]:
            evid = self._evidence_for(cb, content, src, opt.evidence_mode)
            belief = self._make_belief(cb, src, evid)
            tmp_to_gid[cb["tmp_id"]] = belief["id"]
            new_beliefs.append(belief)

        # ---- resolve + add forward edges
        forward_added = 0
        if new_beliefs:
            existing_ids = {b["id"] for b in self.graph.active()
                            if b["id"] not in tmp_to_gid.values()}
            resolved = self._resolve_forward(
                res.get("forward_relations", []), tmp_to_gid, existing_ids)
            forward_added = self.graph.add_forward(resolved)
        return new_beliefs, forward_added, report

    def _evidence_for(self, cb, content, src, mode) -> List[Dict[str, Any]]:
        if mode == "sentence":
            sents = getattr(self, "_last_sentences", []) or []
            idxs = cb.get("supporting_sentence_indices")
            chosen = ([sents[i] for i in idxs if 0 <= i < len(sents)]
                      if idxs else list(sents))
            if not chosen:
                chosen = list(sents)
            return [evidence_from_sentence(s.start, s.end, content, src) for s in chosen]
        excerpts = cb.get("supporting_excerpts", [])
        return [evidence_from_excerpt(ex, content, src) for ex in excerpts]

    def _resolve_forward(self, raw_fwd, tmp_to_gid, existing_ids):
        new_gids = set(tmp_to_gid.values())

        def _gid(ref):
            if isinstance(ref, str):
                if ref in tmp_to_gid:
                    return tmp_to_gid[ref]
                # maybe a stringified int
                try:
                    return int(ref)
                except ValueError:
                    return None
            if isinstance(ref, (int, float)):
                return int(ref)
            return None

        out: List[Dict[str, Any]] = []
        seen = set()
        for r in raw_fwd or []:
            tid = _gid(r.get("to"))
            fid = _gid(r.get("from"))
            if tid is None or fid is None:
                continue
            if tid not in new_gids:           # "to" must be a NEW belief
                continue
            if fid not in new_gids and fid not in existing_ids:
                continue
            if fid == tid or not fid < tid:    # monotonic ids → from<to
                continue
            key = (fid, tid, "informs")
            if key in seen:
                continue
            seen.add(key)
            note = r.get("note", "") or ""
            out.append({"from_id": fid, "to_id": tid, "type": "informs",
                        "note": note if isinstance(note, str) else str(note)})
        return out

    def _make_belief(self, cleaned, src, evid) -> Dict[str, Any]:
        belief: Dict[str, Any] = {
            "id": self.graph.allocate_id(),
            "belief": cleaned["belief"],
            "stance": cleaned["stance"],
            "event_time": cleaned.get("event_time"),
            "time_text": cleaned.get("time_text"),
            "source": dict(src),
            "evidence": evid,
            "supporting_excerpts": [ev["text"] for ev in evid if ev.get("text")],
        }
        init_belief_confidence(belief)
        self.graph.add_belief(belief)
        return belief

    # ------------------------------------------------------------------ result
    def finalize(
        self,
        extra_meta: Optional[Dict[str, Any]] = None,
        pricing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._finalized:
            raise RuntimeError("finalize() called twice")
        self._finalized = True

        # 1) backward linking over the FULL graph + 2) confidence update
        beliefs_now = self.graph.active()
        backward_rels: List[Dict[str, Any]] = []
        conf_updates: List[Dict[str, Any]] = []
        bwd_skip = None
        if len(beliefs_now) >= 2:
            USAGE.set_label("final.backward")
            res = link_backward_all(self.client, self.model, beliefs_now,
                                    max_tokens=self.max_tokens)
            if res.get("skipped"):
                bwd_skip = res.get("skip_reason")
            else:
                backward_rels = res["relations"]
                self.graph.add_backward(backward_rels)
                conf_updates = apply_relations_incremental(
                    self.graph.beliefs, backward_rels)

        # 3) merge / dedup over the full graph
        USAGE.set_label("final.merge")
        merge_report = run_merge_pass(
            graph=self.graph, strategy=self.options.merge_strategy,
            client=self.client, model=self.model, embedder=self.embedder,
            threshold=self.options.merge_threshold, max_tokens=self.max_tokens,
            pass_label="final", log_dir=self.logs_dir)

        # 4) final snapshot
        snap_path = self.out_dir / "final_graph.json"
        self.graph.save_snapshot(snap_path, extra={"item_id": self.item_id})

        summary = {
            "n_beliefs": len(self.graph.active()),
            "forward_relations": len(self.graph.forward_relations),
            "backward_relations": len(backward_rels),
            "backward_skip_reason": bwd_skip,
            "confidence_updates": conf_updates,
            "merges_applied": merge_report.get("applied", []),
            "snapshot": snap_path.name,
        }
        self.graph.sessions.append(summary)

        self._end_time = datetime.now(timezone.utc)
        duration = (self._end_time - self._start_time).total_seconds()
        beliefs = self.graph.active()

        result: Dict[str, Any] = {
            "prompt_name": "construct_beliefs",
            "model": self.model,
            "item_id": self.item_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "stream",
            "options": self.options.to_dict(),
            "embedding_model": getattr(self.embedder, "model", None),
            "timing": {
                "start": self._start_time.isoformat(),
                "end": self._end_time.isoformat(),
                "duration_seconds": duration,
            },
        }
        if self.item_meta:
            result["meta"] = dict(self.item_meta)
        if extra_meta:
            result.update(extra_meta)
        result.update({
            "trajectory": self._trajectory,
            "final": summary,
            "all_beliefs": beliefs,
            "forward_relations": self.graph.forward_relations,
            "backward_relations": self.graph.backward_relations,
            "merges": self.graph.merges,
            "source_counts": _count_by(beliefs, lambda b: (b.get("source") or {}).get("type")),
            "stance_counts": _count_by(beliefs, lambda b: b.get("stance")),
            "token_usage": USAGE.summary(pricing),
        })

        with open(self.out_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        self._event("finalize", summary)
        self._event("timing", {"start": self._start_time.isoformat(),
                               "end": self._end_time.isoformat(),
                               "duration_seconds": duration})
        try:
            agg_path = self.out_dir.parent / "timing.csv"
            write_header = not agg_path.exists()
            with open(agg_path, "a", newline='', encoding="utf-8") as csvf:
                writer = csv.writer(csvf)
                if write_header:
                    writer.writerow(["item_id", "start", "end", "duration_seconds",
                                     "n_beliefs", "n_forward_relations",
                                     "n_backward_relations", "n_merges", "result_path"])
                writer.writerow([
                    self.item_id, self._start_time.isoformat(), self._end_time.isoformat(),
                    f"{duration:.6f}", len(beliefs),
                    len(self.graph.forward_relations), len(self.graph.backward_relations),
                    len(self.graph.merges), str(self.out_dir / "result.json"),
                ])
        except Exception:
            self._event("timing_csv_error", {"error": "failed to append timing.csv"})

        USAGE.save_json(self.out_dir / "token_usage.json", pricing=pricing)
        USAGE.save_text(self.out_dir / "token_usage.txt", pricing=pricing)
        print(f"  [finalize] {len(beliefs)} belief(s); "
              f"{len(self.graph.forward_relations)} forward + "
              f"{len(self.graph.backward_relations)} backward relation(s); "
              f"{len(self.graph.merges)} merge record(s); {duration:.3f}s")
        print(f"  saved -> {self.out_dir / 'result.json'}")
        return result


def _count_by(items: List[Dict[str, Any]], key_fn) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        k = key_fn(it) or "unknown"
        out[k] = out.get(k, 0) + 1
    return out
