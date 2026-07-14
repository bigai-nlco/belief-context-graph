"""
confidence.py
=============
Confidence assignment and posterior recomputation for belief / decision nodes.

The confidence module follows the current confidence design:

    L_posterior = logit(P_prior) + sum(w_i * evidence_i) + sum(factor_i)
    P_posterior = sigmoid(L_posterior)

Code fields:
    initial_confidence  == P_prior
    evidence_confidence == sum(w_i * evidence_i) for ADDITIONAL evidence only
    factor_confidence   == sum(factor_i) from factor templates activated by relations
    confidence          == P_posterior

A Factor is reusable; a Relation supplies each concrete input/output binding.
A node's ``factor_ids`` records inbound/affecting factors for that node.

Important implementation detail:
    Evidence created together with a new belief/decision is provenance for that
    node, but it is NOT added to evidence_confidence. When another duplicate
    node is later merged into the canonical node, that absorbed node's evidence
    becomes ADDITIONAL evidence for the canonical node and is included in the
    posterior recomputation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .factor import relation_factor_endpoint_ids


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
    factor_score: float = 0.0,
) -> float:
    return round(sigmoid(logit(initial) + evidence_score + factor_score), 3)


def recompute_node_confidence(node: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute confidence from current scalar components in-place."""
    initial = float(node.get("initial_confidence") or 0.55)
    evidence_score = float(node.get("evidence_confidence") or 0.0)
    factor_score = float(node.get("factor_confidence") or 0.0)
    node["confidence"] = posterior_confidence(initial, evidence_score, factor_score)
    return node


def init_belief_confidence(belief: Dict[str, Any]) -> Dict[str, Any]:
    """Initialise flat confidence fields on a freshly created belief/decision."""
    role = _role_from_record(belief)
    stance = belief.get("stance", "asserted")
    conf = round(initial_confidence(role, stance), 3)

    belief["initial_confidence"] = conf
    belief["evidence_confidence"] = 0.0
    belief["factor_confidence"] = 0.0
    belief.setdefault("factor_ids", [])
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
    node.setdefault("factor_confidence", 0.0)
    node.setdefault("factor_ids", [])
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
    included exactly once. Any existing factor_confidence is preserved here; the
    caller can subsequently resync relations and run factor propagation if merge
    rewired endpoints.
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
    canonical.setdefault("factor_confidence", 0.0)
    canonical.setdefault("factor_ids", [])
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


# =============================================================
# Factor contributions / forward propagation
# =============================================================


def _as_probability(value: Any, default: float = 0.5) -> float:
    try:
        return _clamp_probability(float(value))
    except (TypeError, ValueError):
        return _clamp_probability(default)


def _factor_direction(factor_type: Any) -> int:
    return -1 if str(factor_type or "").strip().lower() == "refute" else 1


def _condition_threshold(condition: Dict[str, Any], key: str) -> Optional[float]:
    raw = condition.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def factor_contributions_for_node(
    node_id: int,
    nodes_by_id: Dict[int, Dict[str, Any]],
    relations: Iterable[Dict[str, Any]],
    factors_by_id: Dict[int, Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute all active factor contributions targeting one node.

    The semantic edge supplies the concrete input/output binding; the factor id
    supplies type, name, weight and activation condition. ``supplements`` has no
    endpoints according to ``relation_factor_endpoint_ids`` and is ignored.
    """
    total = 0.0
    details: List[Dict[str, Any]] = []

    for relation in relations or []:
        endpoints = relation_factor_endpoint_ids(relation)
        if endpoints is None:
            continue
        input_id, output_id = endpoints
        if output_id != node_id:
            continue
        input_node = nodes_by_id.get(input_id)
        output_node = nodes_by_id.get(output_id)
        if input_node is None or output_node is None:
            continue

        input_conf = _as_probability(input_node.get("confidence"), default=0.5)

        for raw_fid in relation.get("activated_factor_ids") or []:
            try:
                factor_id = int(raw_fid)
            except (TypeError, ValueError):
                continue
            factor = factors_by_id.get(factor_id)
            if not isinstance(factor, dict):
                continue

            condition = factor.get("activation_condition") or {}
            if not isinstance(condition, dict):
                condition = {}
            input_threshold = _condition_threshold(
                condition, "valid_only_if_input_confidence")
            if input_threshold is not None and input_conf <= input_threshold:
                continue

            try:
                weight = float(factor.get("weight", 0.5))
            except (TypeError, ValueError):
                weight = 0.5
            direction = _factor_direction(factor.get("factor_type"))
            contribution = direction * weight * input_conf
            contribution = round(contribution, 6)
            total += contribution
            details.append({
                "factor_id": factor_id,
                "relation_id": relation.get("id"),
                "relation_type": relation.get("type"),
                "input_id": input_id,
                "output_id": output_id,
                "input_confidence": round(input_conf, 6),
                "factor_type": factor.get("factor_type"),
                "weight": weight,
                "contribution": contribution,
            })

    return round(total, 6), details


def recompute_factor_confidence_from_relations(
    node: Dict[str, Any],
    nodes_by_id: Dict[int, Dict[str, Any]],
    relations: Iterable[Dict[str, Any]],
    factors_by_id: Dict[int, Dict[str, Any]],
    *,
    record_history: bool = False,
    step: str = "factor_propagation",
    min_delta: float = 0.001,
) -> Dict[str, Any]:
    old_conf = float(node.get("confidence") or 0.0)
    old_factor_score = float(node.get("factor_confidence") or 0.0)
    node_id = int(node.get("id"))

    factor_score, details = factor_contributions_for_node(
        node_id, nodes_by_id, relations, factors_by_id)
    node["factor_confidence"] = factor_score
    node["factor_ids"] = sorted({
        int(d["factor_id"]) for d in details if isinstance(d.get("factor_id"), int)
    })
    recompute_node_confidence(node)

    new_conf = float(node.get("confidence") or 0.0)
    factor_delta = round(factor_score - old_factor_score, 6)
    conf_delta = round(new_conf - old_conf, 3)
    if record_history and (abs(factor_delta) >= min_delta or abs(conf_delta) >= min_delta):
        node.setdefault("confidence_history", []).append({
            "step": step,
            "value": node["confidence"],
            "delta": conf_delta,
            "factor_confidence": factor_score,
            "factor_delta": factor_delta,
            "activated_factor_ids": sorted({d["factor_id"] for d in details}),
            "factor_details": details,
        })
    return node


def _relation_has_active_factor(
    relation: Dict[str, Any],
    factors_by_id: Dict[int, Dict[str, Any]],
) -> bool:
    for raw_fid in relation.get("activated_factor_ids") or []:
        try:
            factor_id = int(raw_fid)
        except (TypeError, ValueError):
            continue
        if isinstance(factors_by_id.get(factor_id), dict):
            return True
    return False


def _factor_relation_indexes(
    nodes_by_id: Dict[int, Dict[str, Any]],
    relations: Iterable[Dict[str, Any]],
    factors_by_id: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, List[int]], List[int]]:
    """Index active factor relations by affected output and input node.

    Only relations with valid endpoints and at least one activated factor id are
    indexed. ``relations_by_output`` lets recomputation scan only inbound
    relations for the current node. ``outputs_by_input`` lets propagation move
    one hop downstream from nodes whose confidence changed.
    """
    relations_by_output: Dict[int, List[Dict[str, Any]]] = {}
    outputs_by_input_sets: Dict[int, set[int]] = {}
    affected_output_ids: set[int] = set()

    for relation in relations or []:
        endpoints = relation_factor_endpoint_ids(relation)
        if endpoints is None:
            continue
        input_id, output_id = endpoints
        if input_id not in nodes_by_id or output_id not in nodes_by_id:
            continue
        if not _relation_has_active_factor(relation, factors_by_id):
            continue

        relations_by_output.setdefault(output_id, []).append(relation)
        affected_output_ids.add(output_id)
        outputs_by_input_sets.setdefault(input_id, set()).add(output_id)

    outputs_by_input = {
        input_id: sorted(output_ids)
        for input_id, output_ids in outputs_by_input_sets.items()
    }
    return relations_by_output, outputs_by_input, sorted(affected_output_ids)


def propagate_factor_confidences(
    nodes_by_id: Dict[int, Dict[str, Any]],
    relations: Iterable[Dict[str, Any]],
    factors_by_id: Dict[int, Dict[str, Any]],
    *,
    max_iters: int = 3,
    min_delta: float = 0.001,
    record_history: bool = True,
    step: str = "factor_propagation",
    seed_node_ids: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Propagate factor effects through relation outputs.

    The computation is still a full recomputation of ``sum(factor_i)`` for each
    visited node, not an additive delta. The optimization is that each round only
    visits output nodes that can be affected by active factor relations. Round 1
    starts from ``seed_node_ids`` when supplied, otherwise from all active factor
    output nodes. Every later round moves one hop downstream through relations
    whose input node changed by at least ``min_delta`` in the previous round.

    ``max_iters`` is therefore the maximum number of outward propagation rounds
    through this input -> output relation graph. The default 3 means direct
    affected outputs plus at most two further downstream rounds.
    """
    relations_list = list(relations or [])
    relations_by_output, outputs_by_input, affected_output_ids = (
        _factor_relation_indexes(nodes_by_id, relations_list, factors_by_id)
    )

    if seed_node_ids is None:
        frontier = list(affected_output_ids)
    else:
        frontier = []
        seen_seed: set[int] = set()
        for raw_id in seed_node_ids:
            try:
                node_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if node_id in seen_seed:
                continue
            seen_seed.add(node_id)
            if node_id in relations_by_output:
                frontier.append(node_id)
        frontier.sort()

    changed_nodes: List[int] = []
    iterations = 0
    iteration_reports: List[Dict[str, Any]] = []
    max_rounds = max(1, int(max_iters))

    for iteration in range(max_rounds):
        frontier = sorted({node_id for node_id in frontier if node_id in nodes_by_id})
        if not frontier:
            break

        iterations = iteration + 1
        iteration_changed: List[int] = []
        before = {
            node_id: float(nodes_by_id[node_id].get("confidence") or 0.0)
            for node_id in frontier
        }

        for node_id in frontier:
            node = nodes_by_id.get(node_id)
            if node is None:
                continue
            recompute_factor_confidence_from_relations(
                node, nodes_by_id, relations_by_output.get(node_id, []), factors_by_id,
                record_history=record_history,
                step=step,
                min_delta=min_delta,
            )
            after = float(node.get("confidence") or 0.0)
            if abs(after - before.get(node_id, 0.0)) >= min_delta:
                iteration_changed.append(node_id)

        for node_id in iteration_changed:
            if node_id not in changed_nodes:
                changed_nodes.append(node_id)

        iteration_reports.append({
            "round": iterations,
            "visited_node_ids": list(frontier),
            "changed_node_ids": list(iteration_changed),
        })

        if not iteration_changed:
            break

        next_frontier: set[int] = set()
        for node_id in iteration_changed:
            next_frontier.update(outputs_by_input.get(node_id, []))
        frontier = sorted(next_frontier)

    return {
        "iterations": iterations,
        "changed_node_ids": changed_nodes,
        "n_changed": len(changed_nodes),
        "affected_output_node_ids": affected_output_ids,
        "iteration_reports": iteration_reports,
        "propagation_mode": "relation_output_frontier",
        "max_iters": max_rounds,
        "min_delta": min_delta,
    }
