"""
graph.py
========
The in-memory belief graph mutated by the streaming engine.

Current schema:
  * nodes / beliefs  — dict id -> active node. A node has node_type="belief" or
                       node_type="decision". The public attribute is still named
                       `beliefs` for compatibility with existing merge code.
  * evidence         — dict id -> evidence node. Belief/decision nodes hold
                       evidence_ids instead of embedding evidence records.
  * relations        — typed edges between nodes. Valid types are:
                       depends_on, supplements, contradicts.
  * merges           — audit records of applied node merges.
  * sessions         — per-run summaries appended at trajectory end.

Ids are allocated monotonically as nodes are created, but relations are no
longer direction-constrained by chronological order.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .confidence import normalize_confidence_config, relation_propagation_config


VALID_RELATION_TYPES = {"depends_on", "supplements", "contradicts"}


class BeliefGraph:
    def __init__(
        self,
        confidence_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Keep the historical attribute name so merge.py and callers that expect
        # graph.beliefs continue to work. Values can now be belief or decision nodes.
        self.beliefs: Dict[int, Dict[str, Any]] = {}
        self.evidence: Dict[int, Dict[str, Any]] = {}
        self.relations: List[Dict[str, Any]] = []
        self.merges: List[Dict[str, Any]] = []
        self.sessions: List[Dict[str, Any]] = []
        self._next_id = 0
        self._next_evidence_id = 0
        self._next_relation_id = 0
        self.confidence_config = normalize_confidence_config(confidence_config)

    # -- ids / nodes -------------------------------------------------------
    @property
    def next_id(self) -> int:
        return self._next_id

    def allocate_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def add_belief(self, belief: Dict[str, Any]) -> None:
        bid = belief.get("id")
        if not isinstance(bid, int):
            raise ValueError("node must carry an int 'id' (use allocate_id())")
        belief.setdefault("node_type", "belief")
        belief.setdefault("evidence_ids", [])
        # New schema: evidence records live in graph.evidence, not in the node.
        belief.pop("evidence", None)
        self.beliefs[bid] = belief

    def remove_belief(self, bid: int) -> Optional[Dict[str, Any]]:
        return self.beliefs.pop(bid, None)

    # -- evidence ------------------------------------------------------
    def allocate_evidence_id(self) -> int:
        i = self._next_evidence_id
        self._next_evidence_id += 1
        return i

    def add_evidence(self, evidence: Dict[str, Any]) -> int:
        eid = evidence.get("id")
        if not isinstance(eid, int):
            eid = self.allocate_evidence_id()
        rest = dict(evidence)
        rest.pop("id", None)
        # Keep id as the first JSON field for readability / schema stability.
        ev = {"id": eid}
        ev.update(rest)
        ev.setdefault("node_type", "evidence")
        self.evidence[eid] = ev
        if eid >= self._next_evidence_id:
            self._next_evidence_id = eid + 1
        return eid

    def get_evidence(self, eid: int) -> Optional[Dict[str, Any]]:
        return self.evidence.get(eid)

    def evidence_records(self, evidence_ids: List[int]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for raw in evidence_ids or []:
            try:
                eid = int(raw)
            except (TypeError, ValueError):
                continue
            ev = self.evidence.get(eid)
            if ev is not None:
                out.append(ev)
        return out

    def active(self) -> List[Dict[str, Any]]:
        return [self.beliefs[i] for i in sorted(self.beliefs.keys())]

    def ids(self) -> List[int]:
        return sorted(self.beliefs.keys())

    # -- relations ---------------------------------------------------------
    def allocate_relation_id(self) -> int:
        i = self._next_relation_id
        self._next_relation_id += 1
        return i

    @staticmethod
    def _rel_key(r: Dict[str, Any]) -> Tuple[int, int, str]:
        return (r.get("from_id"), r.get("to_id"), r.get("type"))

    def _relation_propagation_config(self) -> Dict[str, Any]:
        return relation_propagation_config(self.confidence_config)

    def _clean_relation_weight(
        self,
        relation: Dict[str, Any],
        relation_type: str,
    ) -> Optional[float]:
        if relation_type == "supplements":
            return None
        default = float(self._relation_propagation_config()["default_relation_weight"])
        try:
            weight = float(relation.get("weight"))
        except (TypeError, ValueError):
            weight = default
        if not math.isfinite(weight):
            weight = default
        return max(0.0, weight)

    def _clean_relation_condition(
        self,
        relation: Dict[str, Any],
        relation_type: str,
    ) -> Optional[Dict[str, float]]:
        if relation_type == "supplements":
            return None
        cfg = self._relation_propagation_config()
        raw_condition = relation.get("activated_condition")
        if not isinstance(raw_condition, dict):
            raw_condition = {}
        try:
            threshold = float(raw_condition.get("input_conf_threshold"))
        except (TypeError, ValueError):
            threshold = float(cfg["input_confidence_threshold"])
        threshold = min(1.0, max(0.0, threshold))
        return {"input_conf_threshold": threshold}

    def add_relations(self, rels: List[Dict[str, Any]]) -> int:
        seen = {self._rel_key(r) for r in self.relations}
        active_ids = set(self.beliefs)
        n = 0
        for r in rels:
            try:
                fid = int(r.get("from_id"))
                tid = int(r.get("to_id"))
            except (TypeError, ValueError):
                continue
            rtype = r.get("type")
            if rtype not in VALID_RELATION_TYPES:
                continue
            if fid == tid or fid not in active_ids or tid not in active_ids:
                continue
            note = r.get("note", "") or ""
            if not isinstance(note, str):
                note = str(note)
            rid = r.get("id")
            if not isinstance(rid, int):
                rid = self.allocate_relation_id()
            # Keep id as the first JSON field.
            clean = {
                "id": rid,
                "from_id": fid,
                "to_id": tid,
                "type": rtype,
                "note": note.strip(),
                "weight": self._clean_relation_weight(r, rtype),
                "activated_condition": self._clean_relation_condition(r, rtype),
            }
            k = self._rel_key(clean)
            if k in seen:
                continue
            seen.add(k)
            self.relations.append(clean)
            if rid >= self._next_relation_id:
                self._next_relation_id = rid + 1
            n += 1
        return n

    def remap_relations(self, mapping: Dict[int, int]) -> Dict[str, Any]:
        """
        Rewrite relation endpoints after a merge. Every absorbed id is replaced
        by its canonical id. Relations that become self-loops or duplicates are
        dropped. There is no chronological direction constraint anymore.
        """
        report = {"rewritten": 0, "dropped_self": 0, "dropped_duplicate": 0}
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for r in self.relations:
            fid = mapping.get(r["from_id"], r["from_id"])
            tid = mapping.get(r["to_id"], r["to_id"])
            changed = (fid != r["from_id"]) or (tid != r["to_id"])
            if fid == tid:
                report["dropped_self"] += 1
                continue
            key = (fid, tid, r.get("type"))
            if key in seen:
                report["dropped_duplicate"] += 1
                continue
            seen.add(key)
            if changed:
                r = dict(r)
                r["from_id"], r["to_id"] = fid, tid
                report["rewritten"] += 1
            out.append(r)
        self.relations = out
        return report

    # -- snapshots ---------------------------------------------------------
    def snapshot(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        nodes = self.active()
        d: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_nodes": len(self.beliefs),
            "n_beliefs": sum(1 for n in nodes if n.get("node_type", "belief") == "belief"),
            "n_decisions": sum(1 for n in nodes if n.get("node_type") == "decision"),
            "nodes": nodes,
            "beliefs": [n for n in nodes if n.get("node_type", "belief") == "belief"],
            "decisions": [n for n in nodes if n.get("node_type") == "decision"],
            "evidence": [self.evidence[i] for i in sorted(self.evidence.keys())],
            "relations": list(self.relations),
            "merges": list(self.merges),
            "sessions": list(self.sessions),
        }
        if extra:
            d.update(extra)
        return d

    def save_snapshot(self, path: Any, extra: Optional[Dict[str, Any]] = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(extra), f, ensure_ascii=False, indent=2)
