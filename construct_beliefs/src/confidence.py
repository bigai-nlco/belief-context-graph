"""
confidence.py
=============
Two-stage confidence assignment.

Stage A: INITIAL confidence
    Pure rules — a lookup of (source_type, stance) → base score.
    Rationale:
      - user_input and tool_response are authoritative sources, so their
        asserted claims start very high.
      - llm_reasoning (<think>) is the model's own thinking, which carries
        less weight than user / tool sources unless externally corroborated.
      - For any source, hedged stances are penalised relative to asserted.

Stage B: DYNAMIC update from belief relations
    Walk each relation discovered by the linker:
      - confirms: the target gets a confidence boost toward the source's level.
      - contradicts: the target gets a penalty proportional to the source's
                     confidence.
      - extends: only weak corroboration, smaller bump.
    Multiple updates compound, but final confidence is clamped to [0.05, 0.97].
    Each belief carries a `confidence_history` list recording the path, so the
    visualizer (or human reviewer) can see exactly why the score moved.
"""

from __future__ import annotations

from typing import Any, Dict, List


# =============================================================
# Stage A — initial confidence rules
# =============================================================
#
# Layout: (source_type, stance) → base confidence.
# A missing key falls back to the source-type default with a stance penalty.

BASE_CONFIDENCE: Dict[tuple, float] = {
    # source_type        stance         confidence
    ("user_input",       "asserted"):    0.95,
    ("user_input",       "recalled"):    0.85,
    ("user_input",       "judged"):      0.80,
    ("user_input",       "speculated"):  0.65,

    ("tool_call",        "asserted"):    0.90,
    ("tool_call",        "judged"):      0.70,
    ("tool_call",        "speculated"):  0.55,

    ("assistant_other",  "asserted"):    0.88,    # final-answer commitment
    ("assistant_other",  "recalled"):    0.78,
    ("assistant_other",  "judged"):      0.75,
    ("assistant_other",  "speculated"):  0.55,

    ("llm_reasoning",    "asserted"):    0.78,
    ("llm_reasoning",    "recalled"):    0.68,
    ("llm_reasoning",    "judged"):      0.62,
    ("llm_reasoning",    "speculated"):  0.40,

    ("tool_result",      "asserted"):    0.92,    # what the tool returned is treated as fact
    ("tool_result",      "recalled"):    0.82,
    ("tool_result",      "judged"):      0.78,
    ("tool_result",      "speculated"):  0.70,

    ("historical_retrieval", "asserted"):   0.80,
    ("historical_retrieval", "recalled"):   0.72,
    ("historical_retrieval", "judged"):     0.68,
    ("historical_retrieval", "speculated"): 0.50,
}

SOURCE_DEFAULT = {
    "user_input":            0.88,
    "tool_call":             0.78,
    "assistant_other":       0.80,
    "llm_reasoning":         0.62,
    "tool_result":           0.85,
    "historical_retrieval":  0.68,
}


def initial_confidence(source_type: str, stance: str) -> float:
    key = (source_type, stance)
    if key in BASE_CONFIDENCE:
        return BASE_CONFIDENCE[key]
    return SOURCE_DEFAULT.get(source_type, 0.55)


# =============================================================
# Stage B — relation-driven dynamic update
# =============================================================
#
# Each relation is { from_id (later), to_id (earlier), type, note }.
# We only update the EARLIER belief (to_id), using the LATER one (from_id)
# as evidence. The size of the move is bounded so it can't run away.

# Bounded global clamp.
CONF_FLOOR = 0.05
CONF_CEIL  = 0.97

# Per-relation move magnitudes (capped — see code).
CONFIRM_MAX_BOOST     = 0.20    # how much one confirm can lift a belief
CONFIRM_GAP_FRACTION  = 0.50    # fraction of the (src - tgt) gap to consume
CONTRADICT_MAX_DROP   = 0.25
CONTRADICT_GAP_FRACTION = 0.60
EXTEND_BOOST          = 0.05    # extends only gives a small bump


def _apply_one(
    target_conf: float, source_conf: float, rel_type: str,
) -> float:
    if rel_type == "confirms":
        if source_conf <= target_conf:
            return target_conf  # no upward push if evidence is weaker
        boost = min(CONFIRM_MAX_BOOST,
                    CONFIRM_GAP_FRACTION * (source_conf - target_conf))
        return min(CONF_CEIL, target_conf + boost)
    if rel_type == "contradicts":
        drop = min(CONTRADICT_MAX_DROP,
                   CONTRADICT_GAP_FRACTION * source_conf)
        return max(CONF_FLOOR, target_conf - drop)
    if rel_type == "extends":
        boost = EXTEND_BOOST if source_conf > target_conf else EXTEND_BOOST / 2
        return min(CONF_CEIL, target_conf + boost)
    return target_conf


def apply_relations(
    beliefs: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Apply relation-driven updates to `beliefs` (modifies a copy, returns it).
    Expects each belief to already have:
        id, confidence, source.type, stance
    Stores a `confidence_history` list on every belief, even those untouched.
    """
    out = [dict(b) for b in beliefs]
    for b in out:
        b["initial_confidence"] = b.get("confidence")
        b["confidence_history"] = [{
            "step": "initial",
            "value": b.get("confidence"),
            "reason": f"base rule for ({b.get('source', {}).get('type')}, {b.get('stance')})",
        }]

    by_id = {b["id"]: b for b in out}

    for rel in relations:
        try:
            fid = int(rel["from_id"])
            tid = int(rel["to_id"])
        except (KeyError, TypeError, ValueError):
            continue
        rtype = rel.get("type")
        if rtype not in {"confirms", "contradicts", "extends"}:
            continue
        if fid == tid:
            continue
        if fid not in by_id or tid not in by_id:
            continue

        target = by_id[tid]
        source = by_id[fid]
        old = target["confidence"]
        new = _apply_one(old, source["confidence"], rtype)
        if abs(new - old) < 1e-9:
            continue
        target["confidence"] = round(new, 3)
        target["confidence_history"].append({
            "step": rtype,
            "value": target["confidence"],
            "delta": round(new - old, 3),
            "from_belief_id": fid,
            "reason": rel.get("note", ""),
        })

    return out
