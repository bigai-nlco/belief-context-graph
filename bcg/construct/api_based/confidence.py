"""
confidence.py
=============
Confidence assignment and posterior recomputation for belief / decision nodes.

The confidence module follows the current confidence design:

    L_posterior = logit(P_prior) + sum(w_i * evidence_i)
    P_posterior = sigmoid(L_posterior)

Code fields:
    initial_confidence  == P_prior
    evidence_confidence == sum(w_i * evidence_i) for ADDITIONAL evidence only
    confidence          == P_posterior

Important implementation detail:
    Evidence created together with a new belief/decision is provenance for that
    node, but it is NOT added to evidence_confidence. When another duplicate
    node is later merged into the canonical node, that absorbed node's evidence
    becomes ADDITIONAL evidence for the canonical node and is included in the
    posterior recomputation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional


# =============================================================
# Stage A — initial confidence rules
# =============================================================

BASE_CONFIDENCE: Dict[tuple, float] = {
    # role          stance         confidence
    ("user",       "asserted"):    0.88,
    ("user",       "recalled"):    0.78,
    ("user",       "judged"):      0.72,
    ("user",       "speculated"):  0.58,

    ("tool",       "asserted"):    0.88,    # what the tool returned is treated as fact
    ("tool",       "recalled"):    0.78,
    ("tool",       "judged"):      0.74,
    ("tool",       "speculated"):  0.62,

    ("assistant",  "asserted"):    0.78,    # final answers / committed claims
    ("assistant",  "recalled"):    0.68,
    ("assistant",  "judged"):      0.65,
    ("assistant",  "speculated"):  0.45,
}

SOURCE_RELIABILITY = {
    "user":       0.85,
    "tool":       0.80,
    "assistant":  0.65,
}

STANCE_QUALITY = {
    "asserted": 0.75,
    "recalled": 0.65,
    "judged": 0.55,
    "speculated": 0.35,
}

VALID_STANCES = {"asserted", "recalled", "judged", "speculated"}

# Evidence aggregation is deliberately simple at this stage.  The source
# component is a role-based reliability; the stance component reuses the same
# hard rule table so no model-supplied confidence number is accepted.
SOURCE_RELIABILITY: Dict[str, float] = dict(SOURCE_RELIABILITY)


def _norm_role(role: Any) -> str:
    role_s = str(role or "").strip().lower()
    return "tool" if role_s == "function" else role_s


def _norm_stance(stance: Any) -> str:
    stance_s = str(stance or "asserted").strip().lower()
    return stance_s if stance_s in VALID_STANCES else "asserted"


def initial_confidence(role: str, stance: str) -> float:
    role = _norm_role(role)
    stance = _norm_stance(stance)
    key = (role, stance)
    if key in BASE_CONFIDENCE:
        return BASE_CONFIDENCE[key]
    return SOURCE_RELIABILITY.get(role, 0.55)


def source_reliability(role: str) -> float:
    return SOURCE_RELIABILITY.get(_norm_role(role), 0.55)


def stance_quality(stance: str) -> float:
    """Return the hard-rule evidence quality for a stance.
    If a role is available, use the same (role, stance) table as priors.  When
    the role is unknown, fall back to a role-neutral stance ordering.
    """
    return STANCE_QUALITY.get(_norm_stance(stance), 0.35)


def _role_from_record(record: Dict[str, Any]) -> str:
    """Read role from the new flat field, with legacy source fallback."""
    src = record.get("source") or {}
    return str(record.get("role") or src.get("role") or src.get("type") or "")


def evidence_contribution(evidence: Dict[str, Any]) -> float:
    """Compute one additional evidence term: w_i * evidence_i."""
    role = _role_from_record(evidence)
    stance = evidence.get("stance") or "asserted"
    return source_reliability(role) * stance_quality(stance)


# =============================================================
# Posterior helpers
# =============================================================

CONF_FLOOR = 0.001
CONF_CEIL = 0.999


def _clamp_probability(value: float) -> float:
    return max(CONF_FLOOR, min(CONF_CEIL, float(value)))


def logit(p: float) -> float:
    p = _clamp_probability(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def posterior_confidence(
    initial: float,
    evidence_score: float = 0.0,
) -> float:
    return round(sigmoid(logit(initial) + evidence_score), 3)


def recompute_node_confidence(node: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute confidence from current scalar components in-place."""
    initial = float(node.get("initial_confidence") or 0.55)
    evidence_score = float(node.get("evidence_confidence") or 0.0)
    node["confidence"] = posterior_confidence(initial, evidence_score)
    return node


def init_belief_confidence(belief: Dict[str, Any]) -> Dict[str, Any]:
    """Initialise flat confidence fields on a freshly created belief/decision."""
    role = _role_from_record(belief)
    stance = belief.get("stance", "asserted")
    conf = round(initial_confidence(role, stance), 3)

    belief["initial_confidence"] = conf
    belief["evidence_confidence"] = 0.0
    belief["confidence"] = conf
    belief["confidence_history"] = [{
        "step": "initial",
        "value": conf,
    }]
    return belief


def sum_evidence_contributions(evidence_records: Iterable[Dict[str, Any]]) -> float:
    return sum(evidence_contribution(ev) for ev in evidence_records if isinstance(ev, dict))


def additional_evidence_from_node(
    node: Dict[str, Any],
    evidence_by_id: Dict[int, Dict[str, Any]],
) -> tuple[List[int], List[Dict[str, Any]]]:
    """Return additional evidence for a node, excluding evidence_ids[0].

    By design, evidence_ids[0] is the provenance evidence that created the
    canonical belief/decision itself. It is not scored again. Every later
    evidence id is treated as additional evidence and contributes to
    evidence_confidence.
    """
    evidence_ids = node.get("evidence_ids") or []
    if len(evidence_ids) <= 1:
        return [], []

    out_ids: List[int] = []
    out_records: List[Dict[str, Any]] = []
    for raw_eid in evidence_ids[1:]:
        try:
            eid = int(raw_eid)
        except (TypeError, ValueError):
            continue
        ev = evidence_by_id.get(eid)
        if isinstance(ev, dict):
            out_ids.append(eid)
            out_records.append(ev)
    return out_ids, out_records


def recompute_evidence_confidence_from_node(
    node: Dict[str, Any],
    evidence_by_id: Dict[int, Dict[str, Any]],
    *,
    record_history: bool = False,
    step: str = "evidence_recompute",
) -> Dict[str, Any]:
    """Synchronise evidence_confidence and confidence from evidence_ids.

    This is intentionally a full recomputation, not an incremental addition.
    It prevents stale zero values after evidence_ids grow through initial
    multi-evidence extraction or later merges, and avoids double counting when a
    node is merged more than once.
    """
    old_conf = float(node.get("confidence") or 0.0)
    old_ev_score = float(node.get("evidence_confidence") or 0.0)
    scored_ids, records = additional_evidence_from_node(node, evidence_by_id)
    new_ev_score = round(sum_evidence_contributions(records), 6)
    node["evidence_confidence"] = new_ev_score
    recompute_node_confidence(node)

    if record_history and scored_ids:
        node.setdefault("confidence_history", []).append({
            "step": step,
            "value": node["confidence"],
            "delta": round(float(node.get("confidence") or 0.0) - old_conf, 3),
            "evidence_confidence": new_ev_score,
            "evidence_delta": round(new_ev_score - old_ev_score, 6),
            "scored_evidence_ids": list(scored_ids),
        })
    return node


def record_evidence_merge_confidence(
    canonical: Dict[str, Any],
    *,
    added_evidence_ids: List[int],
    added_evidence_records: List[Dict[str, Any]],
    absorbed_ids: List[int],
    newest_id: int,
    evidence_by_id: Optional[Dict[int, Dict[str, Any]]] = None,
) -> None:
    """Record posterior update caused by additional evidence during a merge.

    The canonical node keeps its own initial_confidence as P_prior. After a
    merge, evidence_confidence is recomputed from canonical.evidence_ids[1:] so
    the first provenance evidence is excluded and all additional evidence is
    included exactly once.
    """
    old = float(canonical.get("confidence") or 0.0)
    old_ev_score = float(canonical.get("evidence_confidence") or 0.0)

    if evidence_by_id is not None:
        scored_evidence_ids, scored_records = additional_evidence_from_node(
            canonical, evidence_by_id
        )
        new_ev_score = round(sum_evidence_contributions(scored_records), 6)
    else:
        # Backward-compatible fallback for older callers. New merge code passes
        # evidence_by_id so this branch should rarely be used.
        scored_evidence_ids = list(added_evidence_ids)
        new_ev_score = round(old_ev_score + sum_evidence_contributions(
            added_evidence_records
        ), 6)

    canonical["evidence_confidence"] = new_ev_score
    recompute_node_confidence(canonical)

    evidence_delta = round(new_ev_score - old_ev_score, 6)
    canonical.setdefault("confidence_history", []).append({
        "step": "merge_evidence",
        "value": canonical["confidence"],
        "delta": round(canonical["confidence"] - old, 3),
        "evidence_confidence": new_ev_score,
        "evidence_delta": evidence_delta,
        "scored_evidence_ids": list(scored_evidence_ids),
        "added_evidence_ids": list(added_evidence_ids),
        "from_belief_id": newest_id,
        "absorbed_belief_ids": list(absorbed_ids),
    })


