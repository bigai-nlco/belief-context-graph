"""
Evidence provenance for semantic-chunk node generation.

Each generated belief/decision starts with one exact, contiguous chunk evidence
record. ``BeliefGraph.add_evidence`` allocates the global evidence id stored in
the node's ``evidence_ids``; ``chunk_index`` preserves the turn-local order.
"""

from __future__ import annotations

from typing import Any

from .constants import VALID_STANCES


def clean_stance(stance: Any) -> str:
    value = str(stance or "").strip().lower()
    if value not in VALID_STANCES:
        raise ValueError(f"evidence requires a model-inferred stance; got {value!r}")
    return value


def source_descriptor(
    *,
    role: str,
    item_id: str,
    turn_index: int,
    flat_turn_index: int,
    date: str | None = None,
    has_answer: bool | None = None,
) -> dict[str, Any]:
    """Compact location descriptor shared by node.source/evidence.source."""
    del role, flat_turn_index  # role is stored directly on the node/evidence.
    descriptor: dict[str, Any] = {
        "turn_id": turn_index,
        "item_id": item_id,
    }
    if date is not None:
        descriptor["date"] = date
    if has_answer is not None:
        descriptor["has_answer"] = bool(has_answer)
    return descriptor


def evidence_from_chunk(
    chunk_start: int,
    chunk_end: int,
    turn_content: str,
    source: dict[str, Any],
    *,
    chunk_index: int,
    stance: str,
    sentence_indices: list[int] | None = None,
    stance_confidence: float | None = None,
    stance_scores: dict[str, float] | None = None,
    stance_model: str | None = None,
    role: str = "unknown",
) -> dict[str, Any]:
    """Build one exact evidence record for a contiguous semantic chunk."""
    start = max(0, int(chunk_start))
    end = max(start, min(len(turn_content), int(chunk_end)))
    record: dict[str, Any] = {
        "node_type": "evidence",
        "text": turn_content[start:end],
        "start": start,
        "end": end,
        "match": "exact",
        "via": "semantic_chunk",
        "chunk_index": int(chunk_index),
        "sentence_indices": list(sentence_indices or []),
        "stance": clean_stance(stance),
        "role": role,
        "source": dict(source),
    }
    if stance_confidence is not None:
        record["stance_confidence"] = float(stance_confidence)
    if isinstance(stance_scores, dict):
        record["stance_scores"] = {
            str(key): float(value)
            for key, value in stance_scores.items()
            if isinstance(value, (int, float))
        }
    if stance_model:
        record["stance_model"] = str(stance_model)
    return record


def evidence_key(evidence: dict[str, Any]) -> tuple:
    """Stable deduplication key used when evidence is unioned during merges."""
    source = evidence.get("source") or {}
    return (
        source.get("turn_id"),
        evidence.get("start"),
        evidence.get("end"),
        evidence.get("text"),
        evidence.get("stance"),
        evidence.get("role"),
    )
