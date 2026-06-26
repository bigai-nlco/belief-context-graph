"""
graph.py
========
The in-memory belief graph mutated by the streaming engine.

Current schema:
  * nodes / beliefs  — dict id -> active node. A node has node_type="belief" or
                       node_type="decision". The public attribute is still named
                       `beliefs` for compatibility with existing merge code.
  * relations        — typed edges between nodes. Valid types are:
                       causal, depends_on, supplements, contradicts.
  * merges           — audit records of applied node merges.
  * sessions         — per-run summaries appended at trajectory end.

Ids are allocated monotonically as nodes are created, but relations are no
longer direction-constrained by chronological order.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VALID_RELATION_TYPES = {"causal", "depends_on", "supplements", "contradicts"}


class BeliefGraph:
    def __init__(self) -> None:
        # Keep the historical attribute name so merge.py and callers that expect
        # graph.beliefs continue to work. Values can now be belief or decision nodes.
        self.beliefs: dict[int, dict[str, Any]] = {}
        self.relations: list[dict[str, Any]] = []
        self.merges: list[dict[str, Any]] = []
        self.sessions: list[dict[str, Any]] = []
        self._next_id = 0

    # -- ids / nodes -------------------------------------------------------
    @property
    def next_id(self) -> int:
        return self._next_id

    def allocate_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def add_belief(self, belief: dict[str, Any]) -> None:
        bid = belief.get("id")
        if not isinstance(bid, int):
            raise ValueError("node must carry an int 'id' (use allocate_id())")
        belief.setdefault("node_type", "belief")
        self.beliefs[bid] = belief

    def remove_belief(self, bid: int) -> dict[str, Any] | None:
        return self.beliefs.pop(bid, None)

    def active(self) -> list[dict[str, Any]]:
        return [self.beliefs[i] for i in sorted(self.beliefs.keys())]

    def ids(self) -> list[int]:
        return sorted(self.beliefs.keys())

    # -- relations ---------------------------------------------------------
    @staticmethod
    def _rel_key(r: dict[str, Any]) -> tuple[int, int, str]:
        return (r.get("from_id"), r.get("to_id"), r.get("type"))

    def add_relations(self, rels: list[dict[str, Any]]) -> int:
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
            clean = {"from_id": fid, "to_id": tid, "type": rtype, "note": note.strip()}
            k = self._rel_key(clean)
            if k in seen:
                continue
            seen.add(k)
            self.relations.append(clean)
            n += 1
        return n

    def remap_relations(self, mapping: dict[int, int]) -> dict[str, Any]:
        """
        Rewrite relation endpoints after a merge. Every absorbed id is replaced
        by its canonical id. Relations that become self-loops or duplicates are
        dropped. There is no chronological direction constraint anymore.
        """
        report = {"rewritten": 0, "dropped_self": 0, "dropped_duplicate": 0}
        out: list[dict[str, Any]] = []
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
    def snapshot(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        nodes = self.active()
        d: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "n_nodes": len(self.beliefs),
            "n_beliefs": sum(
                1 for n in nodes if n.get("node_type", "belief") == "belief"
            ),
            "n_decisions": sum(1 for n in nodes if n.get("node_type") == "decision"),
            "nodes": nodes,
            "beliefs": [n for n in nodes if n.get("node_type", "belief") == "belief"],
            "decisions": [n for n in nodes if n.get("node_type") == "decision"],
            "relations": list(self.relations),
            "merges": list(self.merges),
            "sessions": list(self.sessions),
        }
        if extra:
            d.update(extra)
        return d

    def save_snapshot(self, path: Any, extra: dict[str, Any] | None = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(extra), f, ensure_ascii=False, indent=2)
