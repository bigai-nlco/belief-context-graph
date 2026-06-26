"""Belief relation linking and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from bcg.belief_graph.constants import VALID_BACKWARD_TYPES, VALID_FORWARD_TYPES
from bcg.belief_graph.extraction import TextGenerator
from bcg.belief_graph.prompts import (
    EARLIER_BELIEFS_PLACEHOLDER,
    EXISTING_BELIEFS_PLACEHOLDER,
    NEW_BELIEFS_PLACEHOLDER,
    PROMPT_LINK_BACKWARD_INC,
    PROMPT_LINK_FORWARD_INC,
    SESSION_BELIEFS_PLACEHOLDER,
)
from bcg.llm import parse_json_response

LinkDirection = Literal["forward", "backward"]


@dataclass(slots=True)
class LinkResult:
    relations: list[dict[str, Any]]
    prompt: str | None = None
    raw_output: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


def validate_relations(
    raw_relations: list[Any],
    beliefs: list[dict[str, Any]],
    *,
    valid_types: set[str],
    direction: LinkDirection,
) -> list[dict[str, Any]]:
    """Validate ids, relation type, direction, and duplicate edges."""

    known_ids = {belief.get("id") for belief in beliefs}
    output: list[dict[str, Any]] = []
    for relation in raw_relations:
        if not isinstance(relation, dict):
            continue
        try:
            from_id = int(relation.get("from_id"))
            to_id = int(relation.get("to_id"))
        except (TypeError, ValueError):
            continue
        relation_type = relation.get("type")
        if relation_type not in valid_types:
            continue
        if from_id not in known_ids or to_id not in known_ids or from_id == to_id:
            continue
        if direction == "forward" and from_id >= to_id:
            continue
        if direction == "backward" and from_id < to_id:
            continue
        note = relation.get("note", "")
        output.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "type": relation_type,
                "note": note.strip() if isinstance(note, str) else str(note),
            }
        )

    seen: set[tuple[int, int, str]] = set()
    deduped: list[dict[str, Any]] = []
    for relation in output:
        key = (relation["from_id"], relation["to_id"], relation["type"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(relation)
    return deduped


def validate_incremental_relations(
    raw_relations: list[Any],
    *,
    valid_types: set[str],
    from_ids: set[int],
    to_ids: set[int],
    direction: LinkDirection,
) -> list[dict[str, Any]]:
    """Validate relations against explicit allowed endpoint sets."""

    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for relation in raw_relations:
        if not isinstance(relation, dict):
            continue
        try:
            from_id = int(relation.get("from_id"))
            to_id = int(relation.get("to_id"))
        except (TypeError, ValueError):
            continue
        relation_type = relation.get("type")
        if relation_type not in valid_types:
            continue
        if from_id not in from_ids or to_id not in to_ids or from_id == to_id:
            continue
        if direction == "forward" and not from_id < to_id:
            continue
        if direction == "backward" and not from_id > to_id:
            continue
        key = (from_id, to_id, relation_type)
        if key in seen:
            continue
        seen.add(key)
        note = relation.get("note", "")
        output.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "type": relation_type,
                "note": note.strip() if isinstance(note, str) else str(note),
            }
        )
    return output


async def link_forward_incremental(
    generator: TextGenerator,
    existing: list[dict[str, Any]],
    new_beliefs: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_chars_per_block: int = 14000,
    label: str = "link:forward",
) -> LinkResult:
    """Link derivation edges into beliefs created by the current turn."""

    if not new_beliefs:
        return LinkResult(
            relations=[],
            skipped=True,
            skip_reason="no new beliefs",
        )
    if not existing and len(new_beliefs) < 2:
        return LinkResult(
            relations=[],
            skipped=True,
            skip_reason="nothing to link against",
        )
    prompt = PROMPT_LINK_FORWARD_INC.replace(
        EXISTING_BELIEFS_PLACEHOLDER,
        _build_belief_blob(existing, max_chars=max_chars_per_block),
    ).replace(
        NEW_BELIEFS_PLACEHOLDER,
        _build_belief_blob(new_beliefs, max_chars=max_chars_per_block),
    )
    try:
        raw = await generator.generate_text(
            prompt,
            label=label,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        return LinkResult(
            relations=[],
            prompt=prompt,
            raw_output=f"[ERROR] {exc}",
            skipped=True,
            skip_reason=str(exc),
        )
    parsed = parse_json_response(raw)
    new_ids = {int(belief["id"]) for belief in new_beliefs if "id" in belief}
    all_ids = {
        int(belief["id"]) for belief in [*existing, *new_beliefs] if "id" in belief
    }
    relations = validate_incremental_relations(
        parsed.get("forward_relations", []) if isinstance(parsed, dict) else [],
        valid_types=VALID_FORWARD_TYPES,
        from_ids=all_ids,
        to_ids=new_ids,
        direction="forward",
    )
    return LinkResult(relations=relations, prompt=prompt, raw_output=raw)


async def link_backward_session(
    generator: TextGenerator,
    earlier: list[dict[str, Any]],
    session_beliefs: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_chars_per_block: int = 14000,
    label: str = "link:backward",
) -> LinkResult:
    """Link evaluation edges from this session's beliefs to earlier beliefs."""

    if not session_beliefs:
        return LinkResult(
            relations=[],
            skipped=True,
            skip_reason="no session beliefs",
        )
    if not earlier and len(session_beliefs) < 2:
        return LinkResult(
            relations=[],
            skipped=True,
            skip_reason="nothing to evaluate against",
        )
    prompt = PROMPT_LINK_BACKWARD_INC.replace(
        EARLIER_BELIEFS_PLACEHOLDER,
        _build_belief_blob(earlier, max_chars=max_chars_per_block),
    ).replace(
        SESSION_BELIEFS_PLACEHOLDER,
        _build_belief_blob(session_beliefs, max_chars=max_chars_per_block),
    )
    try:
        raw = await generator.generate_text(
            prompt,
            label=label,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        return LinkResult(
            relations=[],
            prompt=prompt,
            raw_output=f"[ERROR] {exc}",
            skipped=True,
            skip_reason=str(exc),
        )
    parsed = parse_json_response(raw)
    session_ids = {int(belief["id"]) for belief in session_beliefs if "id" in belief}
    all_ids = {
        int(belief["id"]) for belief in [*earlier, *session_beliefs] if "id" in belief
    }
    relations = validate_incremental_relations(
        parsed.get("relations", []) if isinstance(parsed, dict) else [],
        valid_types=VALID_BACKWARD_TYPES,
        from_ids=session_ids,
        to_ids=all_ids,
        direction="backward",
    )
    return LinkResult(relations=relations, prompt=prompt, raw_output=raw)


def _build_belief_blob(beliefs: list[dict[str, Any]], *, max_chars: int) -> str:
    compact = [_truncate_belief(_compact_belief(belief)) for belief in beliefs]
    blob = json.dumps(compact, ensure_ascii=False, indent=2)
    if len(blob) <= max_chars:
        return blob

    for start in range(1, len(compact) + 1):
        candidate = [
            {"omitted_earlier_beliefs": start},
            *compact[start:],
        ]
        blob = json.dumps(candidate, ensure_ascii=False, indent=2)
        if len(blob) <= max_chars:
            return blob

    marker = [{"omitted_earlier_beliefs": len(compact)}]
    blob = json.dumps(marker, ensure_ascii=False, indent=2)
    return blob if len(blob) <= max_chars else "[]"


def _truncate_belief(
    belief: dict[str, Any], *, max_text_chars: int = 220
) -> dict[str, Any]:
    item = dict(belief)
    text = item.get("belief")
    if isinstance(text, str) and len(text) > max_text_chars:
        item["belief"] = text[: max_text_chars - 4] + " ..."
    return item


def _compact_belief(belief: dict[str, Any]) -> dict[str, Any]:
    source = belief.get("source") or {}
    return {
        "id": belief.get("id"),
        "layer": belief.get("layer"),
        "source": source.get("type"),
        "traj": source.get("trajectory_index"),
        "stance": belief.get("stance"),
        "conf": belief.get("confidence"),
        "belief": belief.get("belief"),
    }
