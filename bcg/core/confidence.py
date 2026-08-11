"""Deterministic confidence math shared by the graph model and both construct backends.

Only pure functions live here: no I/O, no LLM calls, no backend policy. The
backend-specific initial-confidence policies remain in
``bcg.construct.{unified,hybrid}.confidence``.
"""

from __future__ import annotations

import math
from typing import Any

CONF_FLOOR = 0.001
CONF_CEIL = 0.999


def clamp_confidence(value: float) -> float:
    """Clamp a confidence value into the representable range."""
    return max(CONF_FLOOR, min(CONF_CEIL, float(value)))


def logit(p: float) -> float:
    """Log-odds transform with clamped input."""
    p = clamp_confidence(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
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
    """Posterior from initial prior plus evidence and relation factor terms."""
    return round(sigmoid(logit(initial) + evidence_score + factor_score), 3)


# ---------------------------------------------------------------------------
# unified initial-confidence policy, shared with the SDK memory facade.
# Moved verbatim from ``bcg.construct.unified.confidence`` so ``BCGMemory``
# no longer depends on a concrete construct backend. The hybrid backend keeps
# its configurable policy; this hard-rule table is the legacy unified
# behavior (no model-supplied confidence is accepted).
# ---------------------------------------------------------------------------

BASE_CONFIDENCE: dict[tuple, float] = {
    # role          stance         confidence
    ("user", "asserted"): 0.88,
    ("user", "recalled"): 0.78,
    ("user", "judged"): 0.72,
    ("user", "speculated"): 0.58,
    ("tool", "asserted"): 0.88,  # what the tool returned is treated as fact
    ("tool", "recalled"): 0.78,
    ("tool", "judged"): 0.74,
    ("tool", "speculated"): 0.62,
    ("assistant", "asserted"): 0.78,  # final answers / committed claims
    ("assistant", "recalled"): 0.68,
    ("assistant", "judged"): 0.65,
    ("assistant", "speculated"): 0.45,
}

SOURCE_RELIABILITY = {
    "user": 0.85,
    "tool": 0.80,
    "assistant": 0.65,
}

STANCE_QUALITY = {
    "asserted": 0.75,
    "recalled": 0.65,
    "judged": 0.55,
    "speculated": 0.35,
}

VALID_STANCES = {"asserted", "recalled", "judged", "speculated"}

DEFAULT_RELATION_WEIGHT = 0.5
DEFAULT_INPUT_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_PROPAGATION_MIN_CONFIDENCE_DELTA = 0.001
MAX_PROPAGATION_ITERATIONS = 3

DEFAULT_CONFIDENCE_CONFIG: dict[str, Any] = {
    "relation_propagation": {
        "default_relation_weight": DEFAULT_RELATION_WEIGHT,
        "input_confidence_threshold": DEFAULT_INPUT_CONFIDENCE_THRESHOLD,
        "min_confidence_delta": DEFAULT_PROPAGATION_MIN_CONFIDENCE_DELTA,
        "max_iterations": MAX_PROPAGATION_ITERATIONS,
    }
}


def _norm_role(role: Any) -> str:
    role_s = str(role or "").strip().lower()
    return "tool" if role_s == "function" else role_s


def _norm_stance(stance: Any) -> str:
    stance_s = str(stance or "asserted").strip().lower()
    return stance_s if stance_s in VALID_STANCES else "asserted"


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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


def _role_from_record(record: dict[str, Any]) -> str:
    """Read role from the new flat field, with legacy source fallback."""
    src = record.get("source") or {}
    return str(record.get("role") or src.get("role") or src.get("type") or "")


def init_belief_confidence(belief: dict[str, Any]) -> dict[str, Any]:
    """Initialise flat confidence fields on a freshly created belief/decision."""
    role = _role_from_record(belief)
    stance = belief.get("stance", "asserted")
    conf = round(initial_confidence(role, stance), 3)

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
        }
    ]
    return belief


__all__ = [
    "BASE_CONFIDENCE",
    "CONF_CEIL",
    "CONF_FLOOR",
    "DEFAULT_CONFIDENCE_CONFIG",
    "DEFAULT_INPUT_CONFIDENCE_THRESHOLD",
    "DEFAULT_PROPAGATION_MIN_CONFIDENCE_DELTA",
    "DEFAULT_RELATION_WEIGHT",
    "MAX_PROPAGATION_ITERATIONS",
    "SOURCE_RELIABILITY",
    "STANCE_QUALITY",
    "VALID_STANCES",
    "clamp_confidence",
    "init_belief_confidence",
    "initial_confidence",
    "logit",
    "posterior_confidence",
    "sigmoid",
    "source_reliability",
    "stance_quality",
]
