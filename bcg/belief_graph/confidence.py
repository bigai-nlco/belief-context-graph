"""
confidence.py
=============
Two-stage confidence assignment, adapted for the streaming engine.

Stage A: INITIAL confidence — assigned the moment a belief is CREATED.
    Pure rules — a lookup of (role, stance) → base score.
    Rationale:
      - user and tool messages are authoritative sources, so their asserted
        claims start very high.
      - assistant messages are the model's own output (reasoning + answers),
        which carries less weight than user / tool sources unless externally
        corroborated.
      - For any source, hedged stances are penalised relative to asserted.

Stage B: DYNAMIC update — applied INCREMENTALLY at each session end, using
    only the backward relations discovered for that session:
      - confirms: the target gets a confidence boost toward the source's level.
      - contradicts: the target gets a penalty proportional to the source's
                     confidence.
      - extends: only weak corroboration, smaller bump.
    Updates across sessions compound, but confidence is clamped to
    [0.05, 0.97]. Each belief carries a `confidence_history` recording every
    move (initial value, each relation update, and merge adoptions), so the
    visualizer / reviewer can see exactly why the score moved.
"""

from __future__ import annotations

from typing import Any, Dict, List


# =============================================================
# Stage A — initial confidence rules
# =============================================================

BASE_CONFIDENCE: Dict[tuple, float] = {
    # role          stance         confidence
    ("user",       "asserted"):    0.95,
    ("user",       "recalled"):    0.85,
    ("user",       "judged"):      0.80,
    ("user",       "speculated"):  0.65,

    ("tool",       "asserted"):    0.92,    # what the tool returned is treated as fact
    ("tool",       "recalled"):    0.82,
    ("tool",       "judged"):      0.78,
    ("tool",       "speculated"):  0.70,

    ("assistant",  "asserted"):    0.85,    # final answers / committed claims
    ("assistant",  "recalled"):    0.75,
    ("assistant",  "judged"):      0.72,
    ("assistant",  "speculated"):  0.50,
}

SOURCE_DEFAULT = {
    "user":       0.88,
    "tool":       0.85,
    "assistant":  0.72,
}


def initial_confidence(role: str, stance: str) -> float:
    key = (role, stance)
    if key in BASE_CONFIDENCE:
        return BASE_CONFIDENCE[key]
    return SOURCE_DEFAULT.get(role, 0.55)


def init_belief_confidence(belief: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set confidence / initial_confidence / confidence_history on a freshly
    created belief (expects `source.type` (the role) and `stance` to be present).
    """
    conf = initial_confidence((belief.get("source") or {}).get("type", ""),
                              belief.get("stance", "asserted"))
    belief["confidence"] = conf
    belief["initial_confidence"] = conf
    belief["confidence_history"] = [{
        "step": "initial",
        "value": conf,
        "reason": f"base rule for ({(belief.get('source') or {}).get('type')}, {belief.get('stance')})",
    }]
    return belief


# =============================================================
# Stage B — relation-driven dynamic update (incremental)
# =============================================================
#
# Each relation is { from_id (later), to_id (earlier), type, note }.
# We only update the EARLIER belief (to_id), using the LATER one (from_id)
# as evidence. The size of the move is bounded so it can't run away.

CONF_FLOOR = 0.05
CONF_CEIL  = 0.97

CONFIRM_MAX_BOOST       = 0.20    # how much one confirm can lift a belief
CONFIRM_GAP_FRACTION    = 0.50    # fraction of the (src - tgt) gap to consume
CONTRADICT_MAX_DROP     = 0.25
CONTRADICT_GAP_FRACTION = 0.60
EXTEND_BOOST            = 0.05    # extends only gives a small bump


def _apply_one(target_conf: float, source_conf: float, rel_type: str) -> float:
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


def apply_relations_incremental(
    beliefs_by_id: Dict[int, Dict[str, Any]],
    relations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Apply ONE batch of freshly discovered backward relations to the live
    graph beliefs (mutated in place). Beliefs are expected to already carry
    confidence / confidence_history (set at creation by
    init_belief_confidence). Returns the list of applied updates.
    """
    updates: List[Dict[str, Any]] = []
    for rel in relations:
        try:
            fid = int(rel["from_id"])
            tid = int(rel["to_id"])
        except (KeyError, TypeError, ValueError):
            continue
        rtype = rel.get("type")
        if rtype not in {"confirms", "contradicts", "extends"}:
            continue
        if fid == tid or fid not in beliefs_by_id or tid not in beliefs_by_id:
            continue

        target = beliefs_by_id[tid]
        source = beliefs_by_id[fid]
        old = float(target.get("confidence") or 0.0)
        new = _apply_one(old, float(source.get("confidence") or 0.0), rtype)
        if abs(new - old) < 1e-9:
            continue
        target["confidence"] = round(new, 3)
        target.setdefault("confidence_history", []).append({
            "step": rtype,
            "value": target["confidence"],
            "delta": round(new - old, 3),
            "from_belief_id": fid,
            "reason": rel.get("note", ""),
        })
        updates.append({
            "to_id": tid, "from_id": fid, "type": rtype,
            "old": round(old, 3), "new": target["confidence"],
        })
    return updates


def record_merge_confidence(
    canonical: Dict[str, Any],
    adopted_value: float,
    newest_id: int,
    absorbed_ids: List[int],
    reason: str = "",
) -> None:
    """Adopt the newest member's confidence on the canonical belief (merge policy)."""
    old = float(canonical.get("confidence") or 0.0)
    canonical["confidence"] = round(float(adopted_value), 3)
    canonical.setdefault("confidence_history", []).append({
        "step": "merge",
        "value": canonical["confidence"],
        "delta": round(canonical["confidence"] - old, 3),
        "from_belief_id": newest_id,
        "reason": (f"absorbed duplicate belief(s) {absorbed_ids}; adopted confidence of "
                   f"newest member #{newest_id}." + (f" {reason}" if reason else "")),
    })
