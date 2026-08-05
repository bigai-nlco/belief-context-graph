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
    factor_confidence   == sum(direction * relation_weight * input_confidence)
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
from collections.abc import Iterable
from typing import Any

from bcg.core.confidence import (  # noqa: F401 - compat re-exports for legacy imports
    BASE_CONFIDENCE,
    CONF_CEIL,
    CONF_FLOOR,
    DEFAULT_CONFIDENCE_CONFIG,
    DEFAULT_INPUT_CONFIDENCE_THRESHOLD,
    DEFAULT_PROPAGATION_MIN_CONFIDENCE_DELTA,
    DEFAULT_RELATION_WEIGHT,
    MAX_PROPAGATION_ITERATIONS,
    SOURCE_RELIABILITY,
    STANCE_QUALITY,
    VALID_STANCES,
    _as_float,
    _norm_role,
    _norm_stance,
    _role_from_record,
    init_belief_confidence,
    initial_confidence,
    logit,
    posterior_confidence,
    sigmoid,
    source_reliability,
    stance_quality,
)

# =============================================================
# Stage A — initial confidence rules
# =============================================================


# Evidence aggregation is deliberately simple at this stage.  The source
# component is a role-based reliability; the stance component reuses the same
# hard rule table so no model-supplied confidence number is accepted.
SOURCE_RELIABILITY: dict[str, float] = dict(SOURCE_RELIABILITY)




def normalize_confidence_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized confidence settings used by relation propagation."""
    rel_defaults = DEFAULT_CONFIDENCE_CONFIG["relation_propagation"]
    raw = config or {}
    if not isinstance(raw, dict):
        raw = {}
    rel_raw = raw.get("relation_propagation") or {}
    if not isinstance(rel_raw, dict):
        rel_raw = {}
    return {
        "relation_propagation": {
            "default_relation_weight": max(
                0.0,
                _as_float(
                    rel_raw.get("default_relation_weight"),
                    rel_defaults["default_relation_weight"],
                ),
            ),
            "input_confidence_threshold": min(
                1.0,
                max(
                    0.0,
                    _as_float(
                        rel_raw.get("input_confidence_threshold"),
                        rel_defaults["input_confidence_threshold"],
                    ),
                ),
            ),
            "min_confidence_delta": max(
                0.0,
                _as_float(
                    rel_raw.get("min_confidence_delta"),
                    rel_defaults["min_confidence_delta"],
                ),
            ),
            "max_iterations": max(
                0,
                int(_as_float(rel_raw.get("max_iterations"), rel_defaults["max_iterations"])),
            ),
        }
    }


def relation_propagation_config(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return normalized relation-propagation settings."""
    return dict(normalize_confidence_config(config).get("relation_propagation") or {})


def evidence_contribution(evidence: dict[str, Any]) -> float:
    """Compute one additional evidence term: w_i * evidence_i."""
    role = _role_from_record(evidence)
    stance = evidence.get("stance") or "asserted"
    return source_reliability(role) * stance_quality(stance)


# =============================================================
# Posterior helpers
# =============================================================


def recompute_node_confidence(node: dict[str, Any]) -> dict[str, Any]:
    """Recompute confidence from current scalar components in-place."""
    initial = float(node.get("initial_confidence") or 0.55)
    evidence_score = float(node.get("evidence_confidence") or 0.0)
    factor_score = float(node.get("factor_confidence") or 0.0)
    node["confidence"] = posterior_confidence(initial, evidence_score, factor_score)
    return node




def sum_evidence_contributions(evidence_records: Iterable[dict[str, Any]]) -> float:
    return sum(evidence_contribution(ev) for ev in evidence_records if isinstance(ev, dict))


def additional_evidence_from_node(
    node: dict[str, Any],
    evidence_by_id: dict[int, dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Return additional evidence for a node, excluding evidence_ids[0].

    By design, evidence_ids[0] is the provenance evidence that created the
    canonical belief/decision itself. It is not scored again. Every later
    evidence id is treated as additional evidence and contributes to
    evidence_confidence.
    """
    evidence_ids = node.get("evidence_ids") or []
    if len(evidence_ids) <= 1:
        return [], []

    out_ids: list[int] = []
    out_records: list[dict[str, Any]] = []
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
    node: dict[str, Any],
    evidence_by_id: dict[int, dict[str, Any]],
    *,
    record_history: bool = False,
    step: str = "evidence_recompute",
) -> dict[str, Any]:
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
    canonical: dict[str, Any],
    *,
    added_evidence_ids: list[int],
    added_evidence_records: list[dict[str, Any]],
    absorbed_ids: list[int],
    newest_id: int,
    evidence_by_id: dict[int, dict[str, Any]] | None = None,
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


def _relation_signal(relation: dict[str, Any]) -> tuple[int, int, float] | None:
    """Return ``(output_id, input_id, direction)`` for propagating relations."""
    relation_type = relation.get("type")
    try:
        from_id = int(relation.get("from_id"))
        to_id = int(relation.get("to_id"))
    except (TypeError, ValueError):
        return None
    if relation_type == "depends_on":
        return from_id, to_id, 1.0
    if relation_type == "contradicts":
        return to_id, from_id, -1.0
    return None


def relation_output_node_id(relation: dict[str, Any]) -> int | None:
    signal = _relation_signal(relation)
    return None if signal is None else signal[0]


def _relation_input_threshold(
    relation: dict[str, Any],
    config: dict[str, Any],
) -> float:
    condition = relation.get("activated_condition")
    if not isinstance(condition, dict):
        condition = {}
    return min(
        1.0,
        max(
            0.0,
            _as_float(
                condition.get("input_conf_threshold"),
                config["input_confidence_threshold"],
            ),
        ),
    )


def _relation_weight(relation: dict[str, Any], config: dict[str, Any]) -> float:
    weight = relation.get("weight")
    try:
        value = float(weight)
    except (TypeError, ValueError):
        value = float(config["default_relation_weight"])
    if not math.isfinite(value):
        value = float(config["default_relation_weight"])
    return max(0.0, value)


def relation_factor_contribution(
    relation: dict[str, Any],
    nodes_by_id: dict[int, dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    signal = _relation_signal(relation)
    if signal is None:
        return None
    output_id, input_id, direction = signal
    output = nodes_by_id.get(output_id)
    input_node = nodes_by_id.get(input_id)
    if output is None or input_node is None:
        return None

    prop_cfg = relation_propagation_config(config)
    input_confidence = float(input_node.get("confidence") or 0.0)
    threshold = _relation_input_threshold(relation, prop_cfg)
    if input_confidence <= threshold:
        return None

    weight = _relation_weight(relation, prop_cfg)
    contribution = direction * weight * input_confidence
    return {
        "relation_id": relation.get("id"),
        "relation_type": relation.get("type"),
        "input_id": input_id,
        "output_id": output_id,
        "direction": direction,
        "weight": weight,
        "input_confidence": round(input_confidence, 6),
        "input_conf_threshold": threshold,
        "contribution": round(contribution, 6),
    }


def compute_factor_confidence(
    output_node_id: int,
    nodes_by_id: dict[int, dict[str, Any]],
    relations: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    contributions: list[dict[str, Any]] = []
    total = 0.0
    for relation in relations:
        signal = _relation_signal(relation)
        if signal is None or signal[0] != output_node_id:
            continue
        contribution = relation_factor_contribution(
            relation,
            nodes_by_id,
            config=config,
        )
        if contribution is None:
            continue
        contributions.append(contribution)
        total += float(contribution["contribution"])
    return round(total, 6), contributions


def propagate_relation_confidences(
    nodes_by_id: dict[int, dict[str, Any]],
    relations: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    seed_output_node_ids: Iterable[int] | None = None,
    record_history: bool = True,
    step: str = "relation_propagation",
) -> dict[str, Any]:
    prop_cfg = relation_propagation_config(config)
    max_iterations = int(prop_cfg["max_iterations"])
    min_delta = float(prop_cfg["min_confidence_delta"])
    relation_list = [relation for relation in relations if isinstance(relation, dict)]

    if seed_output_node_ids is None:
        frontier = set(nodes_by_id)
    else:
        frontier = {
            int(node_id)
            for node_id in seed_output_node_ids
            if isinstance(node_id, (int, float)) or str(node_id).lstrip("-").isdigit()
        } & set(nodes_by_id)

    report: dict[str, Any] = {
        "iterations": 0,
        "updated_node_ids": [],
        "min_confidence_delta": min_delta,
        "max_iterations": max_iterations,
    }
    updated_ids: set[int] = set()
    if max_iterations <= 0 or not frontier:
        return report

    for iteration in range(1, max_iterations + 1):
        if not frontier:
            break
        current = sorted(frontier & set(nodes_by_id))
        frontier = set()
        changed_this_round: list[int] = []
        for node_id in current:
            node = nodes_by_id.get(node_id)
            if node is None:
                continue
            old_confidence = float(node.get("confidence") or 0.0)
            old_factor = float(node.get("factor_confidence") or 0.0)
            factor_score, contributions = compute_factor_confidence(
                node_id,
                nodes_by_id,
                relation_list,
                config=config,
            )
            node["factor_confidence"] = factor_score
            recompute_node_confidence(node)
            new_confidence = float(node.get("confidence") or 0.0)
            confidence_delta = new_confidence - old_confidence
            factor_delta = factor_score - old_factor
            if abs(confidence_delta) <= min_delta and abs(factor_delta) <= min_delta:
                continue
            updated_ids.add(node_id)
            changed_this_round.append(node_id)
            if record_history:
                node.setdefault("confidence_history", []).append({
                    "step": step,
                    "value": node["confidence"],
                    "delta": round(confidence_delta, 3),
                    "factor_confidence": factor_score,
                    "factor_delta": round(factor_delta, 6),
                    "iteration": iteration,
                    "relation_contributions": contributions,
                })
            if abs(confidence_delta) > min_delta:
                for relation in relation_list:
                    output_id = relation_output_node_id(relation)
                    if output_id is None or output_id not in nodes_by_id:
                        continue
                    signal = _relation_signal(relation)
                    if signal is not None and signal[1] == node_id:
                        frontier.add(output_id)
        report["iterations"] = iteration
        if not changed_this_round:
            break
    report["updated_node_ids"] = sorted(updated_ids)
    return report

