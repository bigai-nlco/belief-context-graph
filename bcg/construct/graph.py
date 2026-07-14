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
  * factors          — dict id -> factor template node. Relations activate
                       factor templates; relation endpoints provide the concrete
                       input/output binding for confidence propagation.
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
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .factor import (
    DEFAULT_FACTOR_INPUT_CONFIDENCE_THRESHOLD,
    DEFAULT_FACTOR_SIMILARITY_THRESHOLD,
    factor_spec_from_relation,
    relation_factor_endpoint_ids,
)


VALID_RELATION_TYPES = {"depends_on", "supplements", "contradicts"}


class BeliefGraph:
    def __init__(self) -> None:
        # Keep the historical attribute name so merge.py and callers that expect
        # graph.beliefs continue to work. Values can now be belief or decision nodes.
        self.beliefs: Dict[int, Dict[str, Any]] = {}
        self.evidence: Dict[int, Dict[str, Any]] = {}
        self.factors: Dict[int, Dict[str, Any]] = {}
        self.relations: List[Dict[str, Any]] = []
        self.merges: List[Dict[str, Any]] = []
        self.sessions: List[Dict[str, Any]] = []
        self._next_id = 0
        self._next_evidence_id = 0
        self._next_factor_id = 0
        self._next_relation_id = 0
        self.factor_embedder = None
        self.factor_note_generator = None
        self._factor_note_cache: Dict[int, str] = {}
        self.factor_similarity_threshold = DEFAULT_FACTOR_SIMILARITY_THRESHOLD
        self.factor_input_confidence_threshold = DEFAULT_FACTOR_INPUT_CONFIDENCE_THRESHOLD

    def configure_factor_reuse(
        self,
        *,
        embedder=None,
        note_generator=None,
        similarity_threshold: Optional[float] = None,
        input_confidence_threshold: Optional[float] = None,
    ) -> None:
        """Configure semantic factor reuse and activation gates.

        Factor reuse requires embeddings. A new relation's
        activation_condition["note"] must be embedded before the factor can be
        created or matched, and every existing candidate factor must also carry
        a valid embedding. Missing embeddings are treated as configuration/data
        errors instead of falling back to exact note matching.
        """
        self.factor_embedder = embedder
        self.factor_note_generator = note_generator
        if similarity_threshold is not None:
            self.factor_similarity_threshold = float(similarity_threshold)
        if input_confidence_threshold is not None:
            self.factor_input_confidence_threshold = float(input_confidence_threshold)

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
        belief.setdefault("factor_ids", [])
        # New schema: evidence records live in graph.evidence, not in the node.
        belief.pop("evidence", None)
        self.beliefs[bid] = belief

    def remove_belief(self, bid: int) -> Optional[Dict[str, Any]]:
        return self.beliefs.pop(bid, None)

    # -- evidence / factor nodes ------------------------------------------
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

    def allocate_factor_id(self) -> int:
        i = self._next_factor_id
        self._next_factor_id += 1
        return i

    def add_factor(self, factor: Dict[str, Any]) -> int:
        fid = factor.get("id")
        if not isinstance(fid, int):
            fid = self.allocate_factor_id()

        rest = dict(factor)
        rest.pop("id", None)
        # Factor templates no longer carry a separate display/name field.
        # Drop it here as a guard against legacy specs or old snapshots being
        # passed through add_factor().
        rest.pop("name", None)

        fac = {"id": fid}
        fac.update(rest)
        fac.setdefault("node_type", "factor")
        fac.setdefault("activation_condition", {})
        fac.setdefault("input_variables", [])
        fac.setdefault("output_variables", [])
        fac.setdefault("activated_relation_ids", [])
        fac["embedding"] = self._require_factor_embedding(
            fac, context=f"factor {fid}"
        )
        self.factors[fid] = fac
        if fid >= self._next_factor_id:
            self._next_factor_id = fid + 1
        return fid

    @staticmethod
    def _coerce_embedding(value: Any) -> List[float]:
        if value is None or isinstance(value, (str, bytes)):
            return []
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return []

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return -1.0
        return dot / (na * nb)

    def _require_factor_embedding(
        self,
        factor: Dict[str, Any],
        *,
        context: str,
    ) -> List[float]:
        """Return a valid factor embedding or raise a clear error.

        Factor reuse is intentionally embedding-only. Missing or malformed
        embeddings should fail fast so the caller can fix configuration or
        legacy data instead of silently reusing by note text.
        """
        embedding = self._coerce_embedding(factor.get("embedding"))
        if not embedding:
            note = self._factor_note_text(factor)
            suffix = f" note={note!r}" if note else ""
            raise ValueError(f"factor embedding is required for {context}.{suffix}")
        return embedding

    def _embed_factor_spec(self, spec: Dict[str, Any]) -> None:
        """Attach a required embedding for activation_condition["note"]."""
        if self._coerce_embedding(spec.get("embedding")):
            return

        if self.factor_embedder is None:
            raise RuntimeError(
                "factor embedding is required for factor reuse, but no "
                "factor_embedder is configured"
            )

        condition = spec.get("activation_condition") or {}
        if not isinstance(condition, dict):
            raise ValueError("factor activation_condition must be a dict")

        text = condition.get("note")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "factor activation_condition['note'] is required to create "
                "a factor embedding"
            )

        vectors = self.factor_embedder.embed(
            [text.strip()], purpose="factor:activation_condition"
        )
        if not vectors or not vectors[0]:
            raise ValueError(
                "factor_embedder returned no embedding for "
                "activation_condition['note']"
            )
        spec["embedding"] = [float(x) for x in vectors[0]]
        self._require_factor_embedding(spec, context="new factor spec")

    @staticmethod
    def _factor_note_text(factor: Dict[str, Any]) -> str:
        condition = factor.get("activation_condition") or {}
        if not isinstance(condition, dict):
            return ""
        note = condition.get("note")
        return note.strip() if isinstance(note, str) else ""

    @staticmethod
    def _lexical_tokens(text: str) -> set[str]:
        stop = {
            "the", "a", "an", "of", "to", "for", "and", "or", "as", "by",
            "with", "that", "this", "these", "those", "is", "are", "was",
            "were", "be", "being", "been", "on", "in", "at", "from",
            "into", "it", "its", "their", "his", "her", "one", "another",
            "node", "belief", "decision", "claim", "relation", "factor",
            "supports", "support", "supported", "refutes", "refute", "refuted",
            "depends", "dependent", "contradicts", "contradict", "evidence",
            "source", "input", "output", "statement", "assertion",
        }
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]*", str(text or "").lower())
        return {w for w in words if len(w) > 2 and w not in stop and not w.isdigit()}

    @classmethod
    def _passes_factor_lexical_guard(cls, a: str, b: str) -> bool:
        """Prevent broad embedding matches from collapsing unrelated factors.

        The guard is intentionally lightweight: embedding similarity remains the
        main signal, but at least a small content-word overlap is required.
        """
        ta = cls._lexical_tokens(a)
        tb = cls._lexical_tokens(b)
        if not ta or not tb:
            return False
        overlap = len(ta & tb)
        if overlap >= 2:
            return True
        return overlap >= 1 and (overlap / max(1, min(len(ta), len(tb)))) >= 0.25

    def _find_factor_by_template(self, spec: Dict[str, Any]) -> Optional[int]:
        """Find an existing reusable factor template matching ``spec``.

        Reuse rule: among factors with the same ``factor_type``, reuse the top
        embedding match when similarity >= factor_similarity_threshold AND a
        lightweight lexical guard passes over activation_condition["note"].

        Weight is deliberately ignored for reuse. Missing embeddings are errors;
        there is no exact-note fallback.
        """
        factor_type = spec.get("factor_type")
        spec_embedding = self._require_factor_embedding(spec, context="new factor spec")
        spec_note = self._factor_note_text(spec)

        best: Optional[Tuple[float, int]] = None
        for fid, fac in self.factors.items():
            if fac.get("factor_type") != factor_type:
                continue
            fac_embedding = self._require_factor_embedding(
                fac, context=f"existing factor {fid}"
            )
            fac_note = self._factor_note_text(fac)
            if not self._passes_factor_lexical_guard(spec_note, fac_note):
                continue
            sim = self._cosine(spec_embedding, fac_embedding)
            if best is None or sim > best[0]:
                best = (sim, fid)

        if best is not None and best[0] >= self.factor_similarity_threshold:
            return best[1]
        return None

    def upsert_factor_template(self, spec: Dict[str, Any]) -> int:
        """Return an existing semantically matching factor id, or create one.

        Factors are reusable computation templates. Multiple semantic edges can
        activate the same factor id via ``relation.activated_factor_ids`` when
        they have the same factor_type and the activation-condition semantic text
        is embedding-similar enough.
        """
        self._embed_factor_spec(spec)
        existing = self._find_factor_by_template(spec)
        if existing is not None:
            return existing
        return self.add_factor(spec)

    @staticmethod
    def _append_unique_int(values: List[int], raw: Any) -> None:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return
        if value not in values:
            values.append(value)

    @staticmethod
    def _append_int(values: List[int], raw: Any) -> None:
        try:
            values.append(int(raw))
        except (TypeError, ValueError):
            return

    def _factor_note_for_relation(self, relation: Dict[str, Any]) -> Optional[str]:
        """Return cached/generated mechanism-level factor note for one relation."""
        endpoints = relation_factor_endpoint_ids(relation)
        if endpoints is None:
            return None
        rid = relation.get("id")
        try:
            rid_int = int(rid)
        except (TypeError, ValueError):
            rid_int = -1
        if rid_int >= 0 and rid_int in self._factor_note_cache:
            return self._factor_note_cache[rid_int]

        input_id, output_id = endpoints
        input_node = self.beliefs.get(input_id)
        output_node = self.beliefs.get(output_id)
        note: Optional[str] = None
        if callable(self.factor_note_generator):
            try:
                note = self.factor_note_generator(relation, input_node, output_node)
            except Exception:
                note = None
        if isinstance(note, str):
            note = re.sub(r"\s+", " ", note).strip()
        else:
            note = None
        if rid_int >= 0 and note:
            self._factor_note_cache[rid_int] = note
        return note

    def _input_confidence_for_relation(self, relation: Dict[str, Any]) -> Optional[float]:
        endpoints = relation_factor_endpoint_ids(relation)
        if endpoints is None:
            return None
        input_id, _ = endpoints
        node = self.beliefs.get(input_id)
        if node is None:
            return None
        try:
            return float(node.get("confidence", 0.5))
        except (TypeError, ValueError):
            return 0.5

    def _activate_relation_factor(self, relation: Dict[str, Any]) -> List[int]:
        """Attach factor ids activated by one semantic relation.

        ``supplements`` deliberately activates no factor. ``depends_on`` and
        ``contradicts`` activate reusable support/refute factor templates only
        when the input node's confidence is above the configured gate.
        """
        input_confidence = self._input_confidence_for_relation(relation)
        if (input_confidence is None
                or input_confidence <= self.factor_input_confidence_threshold):
            relation["activated_factor_ids"] = []
            return []

        spec = factor_spec_from_relation(
            relation,
            self.beliefs,
            input_confidence_threshold=self.factor_input_confidence_threshold,
            factor_note=self._factor_note_for_relation(relation),
        )
        if spec is None:
            relation["activated_factor_ids"] = []
            return []
        factor_id = self.upsert_factor_template(spec)
        relation["activated_factor_ids"] = [factor_id]
        return [factor_id]

    def sync_factors_from_relations(self) -> None:
        """Synchronise factor/node bookkeeping from active relations.

        The relation remains the concrete semantic edge. The factor stores the
        reusable template plus pair-aligned aggregate bookkeeping over all active
        activations: ``input_variables[i]`` and ``output_variables[i]`` describe
        the same relation activation, so repeated node ids are intentional. Each
        affected node's ``factor_ids`` records inbound factors that influence its
        confidence.
        """
        active_ids = set(self.beliefs)

        # Ensure every factor-capable relation has an activated factor id.
        for relation in self.relations:
            if (relation.get("from_id") in active_ids
                    and relation.get("to_id") in active_ids):
                self._activate_relation_factor(relation)

        for fac in self.factors.values():
            fac["input_variables"] = []
            fac["output_variables"] = []
            fac["activated_relation_ids"] = []
        for node in self.beliefs.values():
            node.setdefault("factor_ids", [])
            node["factor_ids"] = []

        for relation in self.relations:
            endpoints = relation_factor_endpoint_ids(relation)
            if endpoints is None:
                continue
            input_id, output_id = endpoints
            if input_id not in active_ids or output_id not in active_ids:
                continue
            for raw_fid in relation.get("activated_factor_ids") or []:
                try:
                    fid = int(raw_fid)
                except (TypeError, ValueError):
                    continue
                fac = self.factors.get(fid)
                if fac is None:
                    continue
                self._append_int(fac.setdefault("input_variables", []), input_id)
                self._append_int(fac.setdefault("output_variables", []), output_id)
                self._append_unique_int(
                    fac.setdefault("activated_relation_ids", []), relation.get("id")
                )
                self._append_unique_int(
                    self.beliefs[output_id].setdefault("factor_ids", []), fid
                )

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
            activated = r.get("activated_factor_ids")
            if not isinstance(activated, list):
                activated = []
            # Keep id as the first JSON field.
            clean = {
                "id": rid,
                "from_id": fid,
                "to_id": tid,
                "type": rtype,
                "note": note.strip(),
                "activated_factor_ids": list(activated),
            }
            k = self._rel_key(clean)
            if k in seen:
                continue
            seen.add(k)
            self.relations.append(clean)
            if rid >= self._next_relation_id:
                self._next_relation_id = rid + 1
            n += 1
        if n:
            self.sync_factors_from_relations()
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
        self.sync_factors_from_relations()
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
            "factors": [self.factors[i] for i in sorted(self.factors.keys())],
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
