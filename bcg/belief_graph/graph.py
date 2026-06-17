"""
graph.py
========
The in-memory belief graph mutated by the streaming engine.

  * beliefs            — dict id → belief (ACTIVE nodes only; absorbed
                         duplicates are removed and archived in `merges`)
  * forward_relations  — informs edges (from_id < to_id)
  * backward_relations — confirms / contradicts / extends (from_id > to_id)
  * merges             — audit records of every applied merge (with full
                         snapshots of absorbed beliefs)
  * sessions           — per-session summaries appended at session end

Ids are allocated monotonically as beliefs are created, so the streaming
order guarantees "smaller id == earlier" — the chronological renumbering
hack of the old batch pipeline is no longer needed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class BeliefGraph:
    def __init__(self) -> None:
        self.beliefs: Dict[int, Dict[str, Any]] = {}
        self.forward_relations: List[Dict[str, Any]] = []
        self.backward_relations: List[Dict[str, Any]] = []
        self.merges: List[Dict[str, Any]] = []
        self.sessions: List[Dict[str, Any]] = []
        self._next_id = 0

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
            raise ValueError("belief must carry an int 'id' (use allocate_id())")
        self.beliefs[bid] = belief

    def remove_belief(self, bid: int) -> Optional[Dict[str, Any]]:
        return self.beliefs.pop(bid, None)

    def active(self) -> List[Dict[str, Any]]:
        return [self.beliefs[i] for i in sorted(self.beliefs.keys())]

    def ids(self) -> List[int]:
        return sorted(self.beliefs.keys())

    # -- relations -----------------------------------------------------------
    @staticmethod
    def _rel_key(r: Dict[str, Any]) -> Tuple[int, int, str]:
        return (r.get("from_id"), r.get("to_id"), r.get("type"))

    def add_forward(self, rels: List[Dict[str, Any]]) -> int:
        seen = {self._rel_key(r) for r in self.forward_relations}
        n = 0
        for r in rels:
            k = self._rel_key(r)
            if k in seen:
                continue
            seen.add(k)
            self.forward_relations.append(r)
            n += 1
        return n

    def add_backward(self, rels: List[Dict[str, Any]]) -> int:
        seen = {self._rel_key(r) for r in self.backward_relations}
        n = 0
        for r in rels:
            k = self._rel_key(r)
            if k in seen:
                continue
            seen.add(k)
            self.backward_relations.append(r)
            n += 1
        return n

    def remap_relations(self, mapping: Dict[int, int]) -> Dict[str, Any]:
        """
        Rewrite relation endpoints after a merge: every absorbed id is replaced
        by its canonical id. Edges that become self-loops, duplicates, or
        direction-invalid (forward needs from<to, backward needs from>to) are
        DROPPED and reported.
        """
        report = {"rewritten": 0, "dropped_self": 0, "dropped_direction": [],
                  "dropped_duplicate": 0}

        def _remap(rels: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            seen: set = set()
            for r in rels:
                fid = mapping.get(r["from_id"], r["from_id"])
                tid = mapping.get(r["to_id"], r["to_id"])
                changed = (fid != r["from_id"]) or (tid != r["to_id"])
                if fid == tid:
                    report["dropped_self"] += 1
                    continue
                if kind == "forward" and not fid < tid:
                    report["dropped_direction"].append(
                        {"kind": kind, "original": dict(r),
                         "remapped": {"from_id": fid, "to_id": tid}})
                    continue
                if kind == "backward" and not fid > tid:
                    report["dropped_direction"].append(
                        {"kind": kind, "original": dict(r),
                         "remapped": {"from_id": fid, "to_id": tid}})
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
            return out

        self.forward_relations = _remap(self.forward_relations, "forward")
        self.backward_relations = _remap(self.backward_relations, "backward")
        return report

    # -- snapshots -----------------------------------------------------------
    def snapshot(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_beliefs": len(self.beliefs),
            "beliefs": self.active(),
            "forward_relations": list(self.forward_relations),
            "backward_relations": list(self.backward_relations),
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
