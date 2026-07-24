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
from typing import Any, Dict, Iterable, List, Optional

from .constants import VALID_STANCES

DEFAULT_CONFIDENCE_CONFIG: Dict[str, Any] = {
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
}

CONF_FLOOR = 0.001
CONF_CEIL = 0.999


def _clamp_probability(value: float) -> float:
    return max(CONF_FLOOR, min(CONF_CEIL, float(value)))


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
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


def normalize_confidence_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    return cfg


def source_reliability(role: str, config: Optional[Dict[str, Any]] = None) -> float:
    cfg = normalize_confidence_config(config)
    table = cfg.get("source_reliability") or {}
    return float(
        table.get(_norm_role(role), cfg.get("default_source_reliability", 0.55))
    )


def stance_quality(stance: str, config: Optional[Dict[str, Any]] = None) -> float:
    cfg = normalize_confidence_config(config)
    normalized = _require_stance(stance, record_type="confidence input")
    table = cfg.get("stance_quality") or {}
    return float(table.get(normalized, cfg.get("default_stance_quality", 0.90)))


def _combine_source_and_stance(
    source_score: float,
    stance_score: float,
    config: Dict[str, Any],
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
    config: Optional[Dict[str, Any]] = None,
) -> float:
    cfg = normalize_confidence_config(config)
    return _combine_source_and_stance(
        source_reliability(role, cfg),
        stance_quality(stance, cfg),
        cfg,
    )


def _role_from_record(record: Dict[str, Any]) -> str:
    """Read role from the flat field, with source fallback."""
    src = record.get("source") or {}
    return str(record.get("role") or src.get("role") or src.get("type") or "")


def evidence_contribution(
    evidence: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
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


def logit(p: float) -> float:
    p = _clamp_probability(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def posterior_confidence(initial: float, evidence_score: float = 0.0) -> float:
    return round(sigmoid(logit(initial) + evidence_score), 3)


def recompute_node_confidence(node: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute confidence from current scalar components in place."""
    initial = float(node.get("initial_confidence") or 0.55)
    evidence_score = float(node.get("evidence_confidence") or 0.0)
    node["confidence"] = posterior_confidence(initial, evidence_score)
    return node


def init_belief_confidence(
    belief: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Initialise flat confidence fields on a freshly created node."""
    role = _role_from_record(belief)
    stance = _require_stance(belief.get("stance"), record_type="node")
    conf = round(initial_confidence(role, stance, config), 3)

    belief["initial_confidence"] = conf
    belief["evidence_confidence"] = 0.0
    belief["confidence"] = conf
    belief["confidence_history"] = [{
        "step": "initial",
        "value": conf,
        "confidence_config": {
            "source": _norm_role(role),
            "stance": _norm_stance(stance),
        },
    }]
    return belief


def sum_evidence_contributions(
    evidence_records: Iterable[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> float:
    return sum(
        evidence_contribution(ev, config)
        for ev in evidence_records
        if isinstance(ev, dict)
    )


def additional_evidence_from_node(
    node: Dict[str, Any],
    evidence_by_id: Dict[int, Dict[str, Any]],
) -> tuple[List[int], List[Dict[str, Any]]]:
    """Return evidence added after the node was originally created."""
    evidence_ids = node.get("evidence_ids") or []
    try:
        initial_count = int(node.get("initial_evidence_count", 1))
    except (TypeError, ValueError):
        initial_count = 1
    initial_count = max(0, min(initial_count, len(evidence_ids)))
    if len(evidence_ids) <= initial_count:
        return [], []

    out_ids: List[int] = []
    out_records: List[Dict[str, Any]] = []
    for raw_eid in evidence_ids[initial_count:]:
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
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Synchronise evidence_confidence and confidence from evidence_ids."""
    old_conf = float(node.get("confidence") or 0.0)
    old_ev_score = float(node.get("evidence_confidence") or 0.0)
    scored_ids, records = additional_evidence_from_node(node, evidence_by_id)
    new_ev_score = round(sum_evidence_contributions(records, config), 6)
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
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Record the posterior update caused by additional merge evidence."""
    old = float(canonical.get("confidence") or 0.0)
    old_ev_score = float(canonical.get("evidence_confidence") or 0.0)

    if evidence_by_id is not None:
        scored_evidence_ids, scored_records = additional_evidence_from_node(
            canonical, evidence_by_id
        )
        new_ev_score = round(sum_evidence_contributions(scored_records, config), 6)
    else:
        scored_evidence_ids = list(added_evidence_ids)
        new_ev_score = round(
            old_ev_score + sum_evidence_contributions(added_evidence_records, config),
            6,
        )

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
