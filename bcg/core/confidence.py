"""Deterministic confidence math shared by the graph model and both construct backends.

Only pure functions live here: no I/O, no LLM calls, no backend policy. The
backend-specific initial-confidence policies remain in
``bcg.construct.{api_based,light}.confidence``.
"""

from __future__ import annotations

import math

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


__all__ = [
    "CONF_CEIL",
    "CONF_FLOOR",
    "clamp_confidence",
    "logit",
    "posterior_confidence",
    "sigmoid",
]
