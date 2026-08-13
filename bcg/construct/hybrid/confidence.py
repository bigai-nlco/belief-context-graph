"""
confidence.py
=============
Confidence assignment and posterior recomputation for belief / decision nodes.

A node's initial confidence is computed from configurable source reliability and
stance quality. Live nodes and evidence receive one of four model-inferred
stances: ``asserted``, ``recalled``, ``judged``, or ``speculated``.

Posterior form::

    L_posterior = logit(P_prior) + sum(w_i * evidence_i)
    P_posterior = sigmoid(L_posterior)

Code fields:
    initial_confidence  == P_prior
    evidence_confidence == sum(w_i * evidence_i) for ADDITIONAL evidence only
    factor_confidence   == sum(direction * relation_weight * input_confidence)
    confidence          == P_posterior

Evidence created together with a new belief/decision is provenance for that
node, but it is NOT added to evidence_confidence. When another duplicate node
is later merged into the canonical node, that absorbed node's evidence becomes
ADDITIONAL evidence for the canonical node and is included in the posterior
recomputation.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from typing import Any

from bcg.core.confidence import (
    clamp_confidence as _clamp_probability,
)
from bcg.core.confidence import (
    posterior_confidence,
    select_independent_evidence,
)

from .constants import VALID_STANCES

DEFAULT_CONFIDENCE_CONFIG: dict[str, Any] = {
    "initial_method": "weighted_average",
    "evidence_method": "product",
    "source_weight": 0.5,
    "stance_weight": 0.5,
    "default_source_reliability": 0.55,
    "default_stance_quality": 0.90,
    "source_reliability": {
        "user": 0.85,
        "assistant": 0.65,
        "tool": 0.80,
        "function": 0.80,
        "user_input": 0.85,
        "assistant_other": 0.65,
        "tool_call": 0.72,
        "tool_result": 0.90,
        "llm_reasoning": 0.58,
        "historical_retrieval": 0.70,
        "manual": 0.86,
        "unknown": 0.55,
    },
    "stance_quality": {
        "asserted": 0.90,
        "recalled": 0.72,
        "judged": 0.65,
        "speculated": 0.40,
    },
    "relation_propagation": {
        "default_relation_weight": 0.5,
        "input_confidence_threshold": 0.8,
        "min_confidence_delta": 0.001,
        "max_iterations": 3,
    },
}

DEFAULT_RELATION_WEIGHT = 0.5
DEFAULT_INPUT_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_PROPAGATION_MIN_CONFIDENCE_DELTA = 0.001
MAX_PROPAGATION_ITERATIONS = 3


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _norm_role(role: Any) -> str:
    role_s = str(role or "").strip().lower()
    return "tool" if role_s == "function" else role_s


def _norm_stance(stance: Any) -> str:
    value = str(stance or "").strip().lower()
    return value if value in VALID_STANCES else ""


def _require_stance(stance: Any, *, record_type: str) -> str:
    value = _norm_stance(stance)
    if not value:
        raise ValueError(
            f"{record_type} requires one model-inferred stance from "
            f"{sorted(VALID_STANCES)}; got {stance!r}"
        )
    return value


def normalize_confidence_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a fully populated confidence config with user overrides applied."""
    cfg = _deep_merge(DEFAULT_CONFIDENCE_CONFIG, config or {})
    cfg["default_source_reliability"] = _clamp_probability(
        _as_float(cfg.get("default_source_reliability"), 0.55)
    )
    cfg["default_stance_quality"] = _clamp_probability(
        _as_float(cfg.get("default_stance_quality"), 0.90)
    )
    cfg["source_reliability"] = {
        _norm_role(k): _clamp_probability(
            _as_float(v, cfg["default_source_reliability"])
        )
        for k, v in dict(cfg.get("source_reliability") or {}).items()
    }
    raw_stance_quality = dict(cfg.get("stance_quality") or {})
    default_stance_quality = dict(DEFAULT_CONFIDENCE_CONFIG["stance_quality"])
    cfg["stance_quality"] = {
        stance: _clamp_probability(
            _as_float(
                raw_stance_quality.get(stance),
                default_stance_quality.get(stance, cfg["default_stance_quality"]),
            )
        )
        for stance in sorted(VALID_STANCES)
    }
    cfg["source_weight"] = max(0.0, _as_float(cfg.get("source_weight"), 0.5))
    cfg["stance_weight"] = max(0.0, _as_float(cfg.get("stance_weight"), 0.5))

    rel_defaults = DEFAULT_CONFIDENCE_CONFIG["relation_propagation"]
    rel_raw = cfg.get("relation_propagation") or {}
    if not isinstance(rel_raw, dict):
        rel_raw = {}
    cfg["relation_propagation"] = {
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
            int(
                _as_float(rel_raw.get("max_iterations"), rel_defaults["max_iterations"])
            ),
        ),
    }
    return cfg


def relation_propagation_config(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return normalized relation-propagation settings."""
    return dict(normalize_confidence_config(config).get("relation_propagation") or {})


def source_reliability(role: str, config: dict[str, Any] | None = None) -> float:
    cfg = normalize_confidence_config(config)
    table = cfg.get("source_reliability") or {}
    return float(
        table.get(_norm_role(role), cfg.get("default_source_reliability", 0.55))
    )


def stance_quality(stance: str, config: dict[str, Any] | None = None) -> float:
    cfg = normalize_confidence_config(config)
    normalized = _require_stance(stance, record_type="confidence input")
    table = cfg.get("stance_quality") or {}
    return float(table.get(normalized, cfg.get("default_stance_quality", 0.90)))


def _combine_source_and_stance(
    source_score: float,
    stance_score: float,
    config: dict[str, Any],
) -> float:
    method = str(config.get("initial_method") or "weighted_average").strip().lower()
    if method == "product":
        return _clamp_probability(source_score * stance_score)
    if method == "min":
        return _clamp_probability(min(source_score, stance_score))
    if method == "max":
        return _clamp_probability(max(source_score, stance_score))
    sw = max(0.0, _as_float(config.get("source_weight"), 0.5))
    tw = max(0.0, _as_float(config.get("stance_weight"), 0.5))
    denom = sw + tw
    if denom <= 0:
        sw = tw = 0.5
        denom = 1.0
    return _clamp_probability((sw * source_score + tw * stance_score) / denom)


def initial_confidence(
    role: str,
    stance: str,
    config: dict[str, Any] | None = None,
) -> float:
    cfg = normalize_confidence_config(config)
    return _combine_source_and_stance(
        source_reliability(role, cfg),
        stance_quality(stance, cfg),
        cfg,
    )


def _role_from_record(record: dict[str, Any]) -> str:
    """Read role from the flat field, with source fallback."""
    src = record.get("source") or {}
    return str(record.get("role") or src.get("role") or src.get("type") or "")


def evidence_contribution(
    evidence: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> float:
    """Compute one additional evidence term according to the configured policy."""
    cfg = normalize_confidence_config(config)
    role = _role_from_record(evidence)
    stance = _require_stance(evidence.get("stance"), record_type="evidence")
    source_score = source_reliability(role, cfg)
    stance_score = stance_quality(stance, cfg)
    method = str(cfg.get("evidence_method") or "product").strip().lower()
    if method == "weighted_average":
        return _combine_source_and_stance(source_score, stance_score, cfg)
    if method == "min":
        return min(source_score, stance_score)
    if method == "max":
        return max(source_score, stance_score)
    return source_score * stance_score


def recompute_node_confidence(node: dict[str, Any]) -> dict[str, Any]:
    """Recompute confidence from current scalar components in place."""
    initial = float(node.get("initial_confidence") or 0.55)
    evidence_score = float(node.get("evidence_confidence") or 0.0)
    factor_score = float(node.get("factor_confidence") or 0.0)
    node["confidence"] = posterior_confidence(initial, evidence_score, factor_score)
    return node


def init_belief_confidence(
    belief: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Initialise flat confidence fields on a freshly created node."""
    role = _role_from_record(belief)
    stance = _require_stance(belief.get("stance"), record_type="node")
    conf = round(initial_confidence(role, stance, config), 3)

    belief["initial_confidence"] = conf
    belief["evidence_confidence"] = 0.0
    belief["factor_confidence"] = 0.0
    belief["confidence"] = conf
    belief["confidence_history"] = [
        {
            "step": "initial",
            "value": conf,
            "evidence_confidence": 0.0,
            "factor_confidence": 0.0,
            "confidence_config": {
                "source": _norm_role(role),
                "stance": _norm_stance(stance),
            },
        }
    ]
    return belief


def sum_evidence_contributions(
    evidence_records: Iterable[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> float:
    return sum(
        evidence_contribution(ev, config)
        for ev in evidence_records
        if isinstance(ev, dict)
    )


def additional_evidence_from_node(
    node: dict[str, Any],
    evidence_by_id: dict[int, dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Return evidence added after the node was originally created."""
    evidence_ids = node.get("evidence_ids") or []
    try:
        initial_count = int(node.get("initial_evidence_count", 1))
    except (TypeError, ValueError):
        initial_count = 1
    initial_count = max(0, min(initial_count, len(evidence_ids)))
    if len(evidence_ids) <= initial_count:
        return [], []

    out_ids: list[int] = []
    out_records: list[dict[str, Any]] = []
    for raw_eid in evidence_ids[initial_count:]:
        try:
            eid = int(raw_eid)
        except (TypeError, ValueError):
            continue
        ev = evidence_by_id.get(eid)
        if isinstance(ev, dict):
            out_ids.append(eid)
            out_records.append(ev)
    return select_independent_evidence(
        zip(out_ids, out_records, strict=False),
        contribution=lambda evidence: evidence_contribution(evidence, config),
    )


def recompute_evidence_confidence_from_node(
    node: dict[str, Any],
    evidence_by_id: dict[int, dict[str, Any]],
    *,
    record_history: bool = False,
    step: str = "evidence_recompute",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synchronise evidence_confidence and confidence from evidence_ids."""
    old_conf = float(node.get("confidence") or 0.0)
    old_ev_score = float(node.get("evidence_confidence") or 0.0)
    scored_ids, records = additional_evidence_from_node(node, evidence_by_id, config)
    new_ev_score = round(sum_evidence_contributions(records, config), 6)
    node["evidence_confidence"] = new_ev_score
    recompute_node_confidence(node)

    if record_history and scored_ids:
        node.setdefault("confidence_history", []).append(
            {
                "step": step,
                "value": node["confidence"],
                "delta": round(float(node.get("confidence") or 0.0) - old_conf, 3),
                "evidence_confidence": new_ev_score,
                "evidence_delta": round(new_ev_score - old_ev_score, 6),
                "scored_evidence_ids": list(scored_ids),
            }
        )
    return node


def record_evidence_merge_confidence(
    canonical: dict[str, Any],
    *,
    added_evidence_ids: list[int],
    added_evidence_records: list[dict[str, Any]],
    absorbed_ids: list[int],
    newest_id: int,
    evidence_by_id: dict[int, dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Record the posterior update caused by additional merge evidence."""
    old = float(canonical.get("confidence") or 0.0)
    old_ev_score = float(canonical.get("evidence_confidence") or 0.0)

    if evidence_by_id is not None:
        scored_evidence_ids, scored_records = additional_evidence_from_node(
            canonical, evidence_by_id, config
        )
        new_ev_score = round(sum_evidence_contributions(scored_records, config), 6)
    else:
        scored_evidence_ids, scored_records = select_independent_evidence(
            zip(added_evidence_ids, added_evidence_records, strict=False),
            contribution=lambda evidence: evidence_contribution(evidence, config),
        )
        new_ev_score = round(
            old_ev_score + sum_evidence_contributions(scored_records, config),
            6,
        )

    canonical["evidence_confidence"] = new_ev_score
    recompute_node_confidence(canonical)

    evidence_delta = round(new_ev_score - old_ev_score, 6)
    canonical.setdefault("confidence_history", []).append(
        {
            "step": "merge_evidence",
            "value": canonical["confidence"],
            "delta": round(canonical["confidence"] - old, 3),
            "evidence_confidence": new_ev_score,
            "evidence_delta": evidence_delta,
            "scored_evidence_ids": list(scored_evidence_ids),
            "added_evidence_ids": list(added_evidence_ids),
            "from_belief_id": newest_id,
            "absorbed_belief_ids": list(absorbed_ids),
        }
    )


def _relation_signal(relation: dict[str, Any]) -> tuple[int, int, float] | None:
    """Return ``(output_id, input_id, direction)`` for propagating relations."""
    relation_type = relation.get("type")
    try:
        from_id = int(relation.get("from_id"))
        to_id = int(relation.get("to_id"))
    except (TypeError, ValueError):
        return None
    if relation_type == "depends_on":
        # A depends_on B => B supports A.
        return from_id, to_id, 1.0
    if relation_type == "contradicts":
        # A contradicts B => A lowers B.
        return to_id, from_id, -1.0
    return None


def relation_output_node_id(relation: dict[str, Any]) -> int | None:
    """Return the node whose factor_confidence may change for one relation."""
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
    """Compute one active relation contribution, or ``None`` if inactive."""
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
    """Return relation-derived factor score for one output node."""
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
    """Propagate relation confidence through depends_on / contradicts edges.

    The update is a bounded synchronous fixed-point iteration.  Each visited
    node's factor_confidence is recomputed from all currently active incoming
    propagating relations, then confidence is recomputed from
    initial_confidence + evidence_confidence + factor_confidence.
    """
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
                node.setdefault("confidence_history", []).append(
                    {
                        "step": step,
                        "value": node["confidence"],
                        "delta": round(confidence_delta, 3),
                        "factor_confidence": factor_score,
                        "factor_delta": round(factor_delta, 6),
                        "iteration": iteration,
                        "relation_contributions": contributions,
                    }
                )
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
