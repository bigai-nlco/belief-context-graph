"""
stream.py
=========
Streaming belief/decision graph engine.

Each turn is routed only by role (user / assistant / tool; system is recorded
but yields no nodes; function == tool). For every non-skipped turn, one LLM call
extracts new belief nodes, new decision nodes, and typed relations between new
and existing nodes.

Node schema additions:
  * node_type: "belief" | "decision"
  * entities: list[str]

Relation schema:
  * causal | depends_on | supplements | contradicts

There is no forward/backward pass. Finalize only runs merge/dedup and writes the
final graph/result.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .confidence import init_belief_confidence
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
    # incremental embedding-ONLY merge after EACH turn's new nodes/edges
    # (no LLM verification). Independent of merge_strategy; needs an embedder.
    incremental_merge: bool = True
    incremental_merge_threshold: float = 0.8
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
            "incremental_merge": self.incremental_merge,
            "incremental_merge_threshold": self.incremental_merge_threshold,
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
        """Process one incoming turn: ONE call → new nodes + typed relations."""
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
            new_nodes, relations_added, report = self._update_from_turn(
                eff_role, content, turn_idx, flat_idx, date, has_answer)

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
            "new_node_ids": [b["id"] for b in new_nodes],
            "new_belief_ids": [b["id"] for b in new_nodes if b.get("node_type", "belief") == "belief"],
            "new_decision_ids": [b["id"] for b in new_nodes if b.get("node_type") == "decision"],
            "relations_added": relations_added,
            "incremental_merge": report.get("incremental_merge"),
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
            self.graph.relations, keep_ids=set(self.graph.ids()))

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
        new_nodes: List[Dict[str, Any]] = []
        for cb in res.get("nodes", []):
            evid = self._evidence_for(cb, content, src, opt.evidence_mode)
            node = self._make_node(cb, src, evid)
            tmp_to_gid[cb["tmp_id"]] = node["id"]
            new_nodes.append(node)

        # ---- resolve + add typed relations
        relations_added = 0
        if new_nodes:
            existing_ids = {b["id"] for b in self.graph.active()
                            if b["id"] not in tmp_to_gid.values()}
            resolved = self._resolve_relations(
                res.get("relations", []), tmp_to_gid, existing_ids)
            relations_added = self.graph.add_relations(resolved)

        # ---- incremental embedding-ONLY merge of the freshly-updated graph
        #      (threshold = incremental_merge_threshold, NO LLM verification).
        #      canonical = smallest id (earliest content); evidence is unioned
        #      onto it. log_dir=None → no per-turn merge_*.json; the applied
        #      merges are recorded in the turn event (events.jsonl) instead.
        #      The final merge in finalize() is unchanged.
        if (new_nodes and self.options.incremental_merge
                and self.embedder is not None):
            USAGE.set_label(f"t{turn_idx}.merge")
            inc = run_merge_pass(
                graph=self.graph, strategy="embedding", verify=False,
                client=self.client, model=self.model, embedder=self.embedder,
                threshold=self.options.incremental_merge_threshold,
                max_tokens=self.max_tokens,
                pass_label=f"turn_{turn_idx}", log_dir=None)
            if not inc.get("skipped"):
                report["incremental_merge"] = {
                    "applied": inc.get("applied", []),
                    "relation_rewire": inc.get("relation_rewire"),
                }
        return new_nodes, relations_added, report

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

    def _resolve_relations(self, raw_relations, tmp_to_gid, existing_ids):
        new_gids = set(tmp_to_gid.values())
        valid_types = {"causal", "depends_on", "supplements", "contradicts"}

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
        allowed_ids = set(existing_ids) | new_gids
        for r in raw_relations or []:
            fid = _gid(r.get("from"))
            tid = _gid(r.get("to"))
            if fid is None or tid is None:
                continue
            if fid == tid:
                continue
            if fid not in allowed_ids or tid not in allowed_ids:
                continue
            # Avoid re-emitting old existing-existing relations during an update call.
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
            out.append({"from_id": fid, "to_id": tid, "type": rtype,
                        "note": note if isinstance(note, str) else str(note)})
        return out

    def _make_node(self, cleaned, src, evid) -> Dict[str, Any]:
        node_type = cleaned.get("node_type", "belief")
        node: Dict[str, Any] = {
            "id": self.graph.allocate_id(),
            "node_type": node_type,
            "belief": cleaned["belief"],
            "stance": cleaned["stance"],
            "entities": list(cleaned.get("entities") or []),
            "event_time": cleaned.get("event_time"),
            "time_text": cleaned.get("time_text"),
            "source": dict(src),
            "evidence": evid,
            "supporting_excerpts": [ev["text"] for ev in evid if ev.get("text")],
        }
        if node_type == "decision":
            node["decision"] = cleaned.get("decision") or cleaned["belief"]
        init_belief_confidence(node)
        self.graph.add_belief(node)
        return node

    # ------------------------------------------------------------------ result
    def finalize(
        self,
        extra_meta: Optional[Dict[str, Any]] = None,
        pricing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._finalized:
            raise RuntimeError("finalize() called twice")
        self._finalized = True

        # Merge / dedup over the full graph
        conf_updates: List[Dict[str, Any]] = []
        USAGE.set_label("final.merge")
        merge_report = run_merge_pass(
            graph=self.graph, strategy=self.options.merge_strategy,
            client=self.client, model=self.model, embedder=self.embedder,
            threshold=self.options.merge_threshold, max_tokens=self.max_tokens,
            pass_label="final", log_dir=self.logs_dir)

        # Final snapshot
        snap_path = self.out_dir / "final_graph.json"
        self.graph.save_snapshot(snap_path, extra={"item_id": self.item_id})

        summary = {
            "n_nodes": len(self.graph.active()),
            "n_beliefs": sum(1 for n in self.graph.active() if n.get("node_type", "belief") == "belief"),
            "n_decisions": sum(1 for n in self.graph.active() if n.get("node_type") == "decision"),
            "relations": len(self.graph.relations),
            "confidence_updates": conf_updates,
            "merges_applied": merge_report.get("applied", []),
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
            "all_nodes": nodes,
            "all_beliefs": [n for n in nodes if n.get("node_type", "belief") == "belief"],
            "all_decisions": [n for n in nodes if n.get("node_type") == "decision"],
            "relations": self.graph.relations,
            "merges": self.graph.merges,
            "source_counts": _count_by(nodes, lambda b: (b.get("source") or {}).get("type")),
            "stance_counts": _count_by(nodes, lambda b: b.get("stance")),
            "node_type_counts": _count_by(nodes, lambda b: b.get("node_type", "belief")),
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
                                     "n_nodes", "n_beliefs", "n_decisions",
                                     "n_relations", "n_merges", "result_path"])
                writer.writerow([
                    self.item_id, self._start_time.isoformat(), self._end_time.isoformat(),
                    f"{duration:.6f}", len(nodes),
                    sum(1 for n in nodes if n.get("node_type", "belief") == "belief"),
                    sum(1 for n in nodes if n.get("node_type") == "decision"),
                    len(self.graph.relations), len(self.graph.merges), str(self.out_dir / "result.json"),
                ])
        except Exception:
            self._event("timing_csv_error", {"error": "failed to append timing.csv"})

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
