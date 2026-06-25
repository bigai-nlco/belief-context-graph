"""Evidence provenance helpers for belief graph extraction."""

from __future__ import annotations

import difflib
import re
from typing import Any

from bcg.belief_graph.constants import (
    FUZZY_MAX_SPAN_FACTOR,
    FUZZY_MAX_SPAN_SLACK,
    FUZZY_MIN_RATIO,
)
from bcg.belief_graph.segment import Segment
from bcg.belief_graph.utils import trim_span


def locate_excerpt(excerpt: str, content: str) -> tuple[int | None, int | None, str]:
    """Locate an excerpt in content with exact, normalized, then fuzzy matching."""

    if not excerpt or not content:
        return None, None, "not_found"
    excerpt = excerpt.strip()
    if len(excerpt) < 2:
        return None, None, "not_found"

    index = content.find(excerpt)
    if index >= 0:
        return index, index + len(excerpt), "exact"

    tokens = excerpt.split()
    if tokens:
        pattern = r"\s+".join(re.escape(token) for token in tokens)
        match = re.search(pattern, content, flags=re.IGNORECASE)
        if match is not None:
            start, end = trim_span(content, match.start(), match.end())
            if end > start:
                return start, end, "normalized"

    matcher = difflib.SequenceMatcher(None, content, excerpt, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
    if blocks:
        matched = sum(block.size for block in blocks)
        start = blocks[0].a
        end = blocks[-1].a + blocks[-1].size
        span = end - start
        max_span = max(
            len(excerpt) * FUZZY_MAX_SPAN_FACTOR,
            len(excerpt) + FUZZY_MAX_SPAN_SLACK,
        )
        if matched >= FUZZY_MIN_RATIO * len(excerpt) and span <= max_span:
            start, end = trim_span(content, start, end)
            if end > start:
                return start, end, "fuzzy"

    return None, None, "not_found"


def source_descriptor(
    *,
    source_type: str,
    scenario: str,
    item_id: str,
    session_id: str | None,
    session_index: int,
    session_date: str | None,
    turn_index: int,
    trajectory_index: int,
    role: str,
    segment: Segment,
    has_answer: bool | None = None,
) -> dict[str, Any]:
    """Build source metadata shared by beliefs and evidence records."""

    descriptor: dict[str, Any] = {
        "type": source_type,
        "role": role,
        "scenario": scenario,
        "item_id": item_id,
        "session_id": session_id,
        "session_index": session_index,
        "session_date": session_date,
        "turn_index": turn_index,
        "trajectory_index": trajectory_index,
        "segment_index": segment.seg_idx,
        "segment_type": segment.type,
        "segment_start": segment.start,
        "segment_end": segment.end,
    }
    if has_answer is not None:
        descriptor["has_answer"] = bool(has_answer)
    return descriptor


def evidence_from_excerpt(
    excerpt: str,
    segment: Segment,
    turn_content: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Convert an LLM excerpt into an offset-aware evidence record."""

    start_local, end_local, match_kind = locate_excerpt(excerpt, segment.content)
    if match_kind == "not_found" or start_local is None or end_local is None:
        return {
            "text": excerpt,
            "start": None,
            "end": None,
            "match": "not_found",
            "via": "llm_excerpt",
            "source": dict(source),
        }
    start = segment.start + start_local
    end = segment.start + end_local
    return {
        "text": turn_content[start:end],
        "start": start,
        "end": end,
        "match": match_kind,
        "via": "llm_excerpt",
        "source": dict(source),
    }


def evidence_from_sentence(
    sentence_text: str,
    sentence_start: int,
    sentence_end: int,
    segment: Segment,
    turn_content: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Create exact evidence from an already offset-tracked sentence."""

    del sentence_text
    start = segment.start + sentence_start
    end = segment.start + sentence_end
    return {
        "text": turn_content[start:end],
        "start": start,
        "end": end,
        "match": "exact",
        "via": "split_sentence",
        "source": dict(source),
    }


def evidence_key(evidence: dict[str, Any]) -> tuple[Any, ...]:
    """Return a stable deduplication key for evidence union during merges."""

    source = evidence.get("source") or {}
    return (
        source.get("session_id"),
        source.get("trajectory_index"),
        source.get("segment_index"),
        evidence.get("start"),
        evidence.get("end"),
        evidence.get("text"),
    )
