"""Dimension-based confidence assignment and relation-driven updates."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from bcg.belief_graph.constants import (
    CONFIDENCE_METHOD,
    DEFAULT_EVIDENCE_DIRECTNESS,
    DEFAULT_RELATION_DIRECTNESS,
    DEFAULT_SOURCE_RELIABILITY,
    DEFAULT_STANCE_CERTAINTY,
    VALID_BACKWARD_TYPES,
)
from bcg.graph import CONFIDENCE_DIMENSION_KEYS


@dataclass(frozen=True, slots=True)
class ConfidenceConfig:
    """Configurable fallback values for deterministic confidence assessment."""

    default: float = 0.55
    source_reliability: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SOURCE_RELIABILITY)
    )
    stance_certainty: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_STANCE_CERTAINTY)
    )
    evidence_directness: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_EVIDENCE_DIRECTNESS)
    )
    relation_directness: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RELATION_DIRECTNESS)
    )
    floor: float = 0.05
    ceiling: float = 0.97


@dataclass(frozen=True, slots=True)
class ConfidenceAssessment:
    """Auditable confidence assessment whose value is derived from dimensions."""

    dimensions: dict[str, float]
    reason: str
    fallback_used: bool = False

    @property
    def value(self) -> float:
        return confidence_from_dimensions(self.dimensions)

    def to_history(
        self,
        *,
        step: str,
        delta: float | None = None,
        from_belief_id: int | None = None,
    ) -> dict[str, Any]:
        output: dict[str, Any] = {
            "step": step,
            "value": self.value,
            "method": CONFIDENCE_METHOD,
            "dimensions": normalize_dimensions(self.dimensions),
            "formula": _formula(),
            "fallback_used": self.fallback_used,
            "reason": self.reason,
        }
        if delta is not None:
            output["delta"] = round(delta, 3)
        if from_belief_id is not None:
            output["from_belief_id"] = from_belief_id
        return output


def initial_confidence(
    source_type: str,
    stance: str,
    *,
    config: ConfidenceConfig | None = None,
) -> float:
    """Return confidence computed from fallback source and stance dimensions."""

    config = config or ConfidenceConfig()
    dimensions = normalize_dimensions(
        {
            "source_reliability": _configured(
                config.source_reliability,
                source_type,
                config.default,
                config=config,
            ),
            "evidence_directness": config.default,
            "claim_specificity": config.default,
            "linguistic_certainty": _configured(
                config.stance_certainty,
                stance,
                config.default,
                config=config,
            ),
        },
        config=config,
    )
    return confidence_from_dimensions(dimensions)


def init_belief_confidence(
    belief: dict[str, Any],
    *,
    config: ConfidenceConfig | None = None,
) -> dict[str, Any]:
    """Attach deterministic dimension-based confidence to a fresh belief."""

    assessment = assess_initial_confidence(belief, config=config)
    belief["confidence_dimensions"] = assessment.dimensions
    belief["confidence"] = assessment.value
    belief["initial_confidence"] = assessment.value
    belief["confidence_history"] = [assessment.to_history(step="initial")]
    return belief


def assess_initial_confidence(
    belief: dict[str, Any],
    *,
    config: ConfidenceConfig | None = None,
) -> ConfidenceAssessment:
    """Score a new belief from source, evidence, claim shape, and stance."""

    config = config or ConfidenceConfig()
    source = belief.get("source") or {}
    source_type = str(source.get("type") or "")
    stance = str(belief.get("stance") or "asserted")
    source_known = source_type in config.source_reliability
    stance_known = stance in config.stance_certainty
    dimensions = normalize_dimensions(
        {
            "source_reliability": _configured(
                config.source_reliability,
                source_type,
                config.default,
                config=config,
            ),
            "evidence_directness": _evidence_directness(belief, config),
            "claim_specificity": _claim_specificity(str(belief.get("belief") or "")),
            "linguistic_certainty": _linguistic_certainty(
                str(belief.get("belief") or ""),
                stance,
                config,
            ),
        },
        config=config,
    )
    reason = (
        "computed as the average of source reliability, evidence directness, "
        "claim specificity, and linguistic certainty"
    )
    return ConfidenceAssessment(
        dimensions=dimensions,
        reason=reason,
        fallback_used=not (source_known and stance_known),
    )


def apply_relations(
    beliefs: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    *,
    config: ConfidenceConfig | None = None,
) -> list[dict[str, Any]]:
    """Apply backward relation confidence updates to copied belief dicts."""

    output = [
        _ensure_confidence_state(copy.deepcopy(belief), config=config)
        for belief in beliefs
    ]
    by_id = {belief["id"]: belief for belief in output if "id" in belief}
    for relation in relations:
        _apply_relation_update(by_id, relation, config=config)
    return output


def apply_relations_incremental(
    beliefs_by_id: dict[int, dict[str, Any]],
    relations: list[dict[str, Any]],
    *,
    config: ConfidenceConfig | None = None,
) -> list[dict[str, Any]]:
    """Apply freshly discovered backward relations to live belief dicts."""

    updates: list[dict[str, Any]] = []
    for relation in relations:
        update = _apply_relation_update(beliefs_by_id, relation, config=config)
        if update is not None:
            updates.append(update)
    return updates


def record_merge_confidence(
    canonical: dict[str, Any],
    adopted_value: float,
    newest_id: int,
    absorbed_ids: list[int],
    reason: str = "",
    adopted_dimensions: dict[str, float] | None = None,
) -> None:
    """Record confidence adoption when duplicates merge."""

    old_confidence = float(canonical.get("confidence") or 0.0)
    dimensions = normalize_dimensions(
        adopted_dimensions or _dimensions_from_value(adopted_value)
    )
    new_confidence = confidence_from_dimensions(dimensions)
    canonical["confidence_dimensions"] = dimensions
    canonical["confidence"] = new_confidence
    canonical.setdefault("confidence_history", []).append(
        {
            "step": "merge",
            "value": new_confidence,
            "method": CONFIDENCE_METHOD,
            "dimensions": dimensions,
            "formula": _formula(),
            "delta": round(new_confidence - old_confidence, 3),
            "from_belief_id": newest_id,
            "reason": (
                f"absorbed duplicate belief(s) {absorbed_ids}; adopted "
                f"confidence dimensions of newest member #{newest_id}."
                + (f" {reason}" if reason else "")
            ),
        }
    )


def confidence_from_dimensions(dimensions: dict[str, float]) -> float:
    """Compute confidence as the average of the four canonical dimensions."""

    normalized = normalize_dimensions(dimensions)
    return round(sum(normalized.values()) / len(CONFIDENCE_DIMENSION_KEYS), 3)


def normalize_dimensions(
    dimensions: dict[str, float],
    *,
    config: ConfidenceConfig | None = None,
) -> dict[str, float]:
    """Clamp and complete the canonical confidence dimensions."""

    floor = config.floor if config is not None else 0.05
    ceiling = config.ceiling if config is not None else 0.97
    return {
        key: round(_clamp(float(dimensions.get(key, 0.0)), floor, ceiling), 3)
        for key in CONFIDENCE_DIMENSION_KEYS
    }


def _apply_relation_update(
    beliefs_by_id: dict[int, dict[str, Any]],
    relation: dict[str, Any],
    *,
    config: ConfidenceConfig | None = None,
) -> dict[str, Any] | None:
    config = config or ConfidenceConfig()
    try:
        from_id = int(relation["from_id"])
        to_id = int(relation["to_id"])
    except (KeyError, TypeError, ValueError):
        return None
    relation_type = relation.get("type")
    if relation_type not in VALID_BACKWARD_TYPES:
        return None
    if from_id == to_id or from_id not in beliefs_by_id or to_id not in beliefs_by_id:
        return None

    source = _ensure_confidence_state(beliefs_by_id[from_id], config=config)
    target = _ensure_confidence_state(beliefs_by_id[to_id], config=config)
    old_confidence = float(target.get("confidence") or 0.0)
    old_dimensions = _latest_dimensions(target, config=config)
    source_dimensions = _latest_dimensions(source, config=config)
    dimensions = _relation_dimensions(
        old_dimensions,
        source_dimensions,
        str(relation_type),
        config,
    )
    new_confidence = confidence_from_dimensions(dimensions)
    if abs(new_confidence - old_confidence) < 1e-9:
        return None

    target["confidence_dimensions"] = dimensions
    target["confidence"] = new_confidence
    assessment = ConfidenceAssessment(
        dimensions=dimensions,
        reason=str(relation.get("note") or ""),
    )
    target.setdefault("confidence_history", []).append(
        assessment.to_history(
            step=str(relation_type),
            delta=new_confidence - old_confidence,
            from_belief_id=from_id,
        )
    )
    return {
        "to_id": to_id,
        "from_id": from_id,
        "type": relation_type,
        "old": round(old_confidence, 3),
        "new": new_confidence,
    }


def _relation_dimensions(
    target: dict[str, float],
    source: dict[str, float],
    relation_type: str,
    config: ConfidenceConfig,
) -> dict[str, float]:
    if relation_type == "confirms":
        directness = _configured(
            config.relation_directness,
            "confirms",
            config.default,
            config=config,
        )
        return normalize_dimensions(
            {
                "source_reliability": max(
                    target["source_reliability"],
                    _blend(
                        target["source_reliability"], source["source_reliability"], 0.5
                    ),
                ),
                "evidence_directness": max(
                    target["evidence_directness"],
                    _blend(target["evidence_directness"], directness, 0.6),
                ),
                "claim_specificity": target["claim_specificity"],
                "linguistic_certainty": max(
                    target["linguistic_certainty"],
                    _blend(
                        target["linguistic_certainty"],
                        source["linguistic_certainty"],
                        0.4,
                    ),
                ),
            },
            config=config,
        )
    if relation_type == "contradicts":
        source_confidence = confidence_from_dimensions(source)
        drop = 0.25 * source_confidence
        return normalize_dimensions(
            {
                "source_reliability": target["source_reliability"],
                "evidence_directness": target["evidence_directness"] - drop,
                "claim_specificity": target["claim_specificity"],
                "linguistic_certainty": target["linguistic_certainty"] - drop,
            },
            config=config,
        )
    if relation_type == "extends":
        directness = _configured(
            config.relation_directness,
            "extends",
            config.default,
            config=config,
        )
        return normalize_dimensions(
            {
                "source_reliability": max(
                    target["source_reliability"],
                    _blend(
                        target["source_reliability"], source["source_reliability"], 0.25
                    ),
                ),
                "evidence_directness": max(
                    target["evidence_directness"],
                    _blend(target["evidence_directness"], directness, 0.35),
                ),
                "claim_specificity": min(1.0, target["claim_specificity"] + 0.08),
                "linguistic_certainty": target["linguistic_certainty"],
            },
            config=config,
        )
    return target


def _ensure_confidence_state(
    belief: dict[str, Any],
    *,
    config: ConfidenceConfig | None = None,
) -> dict[str, Any]:
    config = config or ConfidenceConfig()
    if belief.get("confidence_dimensions"):
        belief["confidence_dimensions"] = normalize_dimensions(
            belief["confidence_dimensions"],
            config=config,
        )
        belief["confidence"] = confidence_from_dimensions(
            belief["confidence_dimensions"]
        )
    elif "confidence" in belief:
        belief["confidence_dimensions"] = _dimensions_from_value(
            float(belief.get("confidence") or config.default),
            config=config,
        )
    else:
        init_belief_confidence(belief, config=config)
    belief.setdefault("initial_confidence", belief.get("confidence", config.default))
    if not belief.get("confidence_history"):
        belief["confidence_history"] = [
            ConfidenceAssessment(
                dimensions=belief["confidence_dimensions"],
                reason="legacy confidence value mapped equally across dimensions",
                fallback_used=True,
            ).to_history(step="initial")
        ]
    return belief


def _latest_dimensions(
    belief: dict[str, Any],
    *,
    config: ConfidenceConfig,
) -> dict[str, float]:
    if belief.get("confidence_dimensions"):
        return normalize_dimensions(belief["confidence_dimensions"], config=config)
    for entry in reversed(belief.get("confidence_history") or []):
        if isinstance(entry, dict) and entry.get("dimensions"):
            return normalize_dimensions(entry["dimensions"], config=config)
    return _dimensions_from_value(
        float(belief.get("confidence") or config.default), config=config
    )


def _evidence_directness(
    belief: dict[str, Any],
    config: ConfidenceConfig,
) -> float:
    evidence_items = belief.get("evidence") or []
    values: list[float] = []
    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            continue
        match = str(evidence.get("match") or "")
        via = str(evidence.get("via") or "")
        if match in config.evidence_directness:
            values.append(config.evidence_directness[match])
        elif via == "manual":
            values.append(config.evidence_directness.get("manual", config.default))
    if values:
        return max(values)
    if belief.get("supporting_excerpts"):
        return config.evidence_directness.get("manual", config.default)
    return config.default


def _claim_specificity(text: str) -> float:
    words = re.findall(r"\w+", text)
    if not words:
        return 0.2
    n_words = len(words)
    if n_words <= 3:
        value = 0.45
    elif n_words <= 8:
        value = 0.68
    elif n_words <= 20:
        value = 0.78
    else:
        value = 0.72
    if re.search(r"\b[A-Z][a-z]+|\d", text):
        value += 0.07
    if re.search(r"\b(something|someone|stuff|things|it|this|that)\b", text, re.I):
        value -= 0.08
    return _clamp(value)


def _linguistic_certainty(
    text: str,
    stance: str,
    config: ConfidenceConfig,
) -> float:
    value = _configured(config.stance_certainty, stance, config.default, config=config)
    if re.search(r"\b(may|might|could|possibly|probably|seems|appears)\b", text, re.I):
        value = min(value, 0.48)
    return value


def _dimensions_from_value(
    value: float,
    *,
    config: ConfidenceConfig | None = None,
) -> dict[str, float]:
    floor = config.floor if config is not None else 0.05
    ceiling = config.ceiling if config is not None else 0.97
    clipped = _clamp(value, floor, ceiling)
    return {key: round(clipped, 3) for key in CONFIDENCE_DIMENSION_KEYS}


def _configured(
    source: dict[str, float],
    key: str,
    fallback: float,
    *,
    config: ConfidenceConfig | None = None,
) -> float:
    floor = config.floor if config is not None else 0.05
    ceiling = config.ceiling if config is not None else 0.97
    return _clamp(float(source.get(key, fallback)), floor, ceiling)


def _blend(old: float, new: float, weight: float) -> float:
    return old + weight * (new - old)


def _clamp(value: float, floor: float = 0.05, ceiling: float = 0.97) -> float:
    return min(ceiling, max(floor, value))


def _formula() -> str:
    return (
        "average(source_reliability, evidence_directness, "
        "claim_specificity, linguistic_certainty)"
    )
