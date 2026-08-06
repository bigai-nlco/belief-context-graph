"""
evidence.py
============
Evidence provenance. Belief/decision nodes now carry ``evidence_ids`` instead
of embedded evidence records. The actual evidence records are stored on the
BeliefGraph as evidence nodes:

    {
      "id": 0,
      "node_type": "evidence",
      "text":  "<exact slice content[start:end] when located, else excerpt>",
      "start": 118,
      "end":   178,
      "match": "exact" | "normalized" | "fuzzy" | "not_found",
      "via":   "llm_excerpt" | "split_sentence",
      "stance": "asserted" | "recalled" | "speculated" | "judged",
      "role": "user" | "assistant" | "tool",
      "source": {"turn_id": 0, ...}
    }

Evidence created together with a new belief is provenance only. It becomes
additional evidence for another node only when that node is merged into a
canonical duplicate; confidence.py performs that posterior recomputation.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from .._shared.spans import trim_span

# Fuzzy-match acceptance knobs.
FUZZY_MIN_RATIO = 0.6  # matched chars must cover >= 60% of the excerpt
FUZZY_MAX_SPAN_FACTOR = 2.0  # located span must stay near the excerpt length
FUZZY_MAX_SPAN_SLACK = 80

VALID_STANCES = {"asserted", "recalled", "speculated", "judged"}


def clean_stance(stance: Any) -> str:
    s = str(stance or "asserted").strip().lower()
    return s if s in VALID_STANCES else "asserted"


def locate_excerpt(excerpt: str, content: str) -> tuple[int | None, int | None, str]:
    """
    Locate `excerpt` inside `content` and return (start, end, match_kind).
    match_kind in {"exact", "normalized", "fuzzy", "not_found"};
    start/end are None when not found.
    """
    if not excerpt or not content:
        return None, None, "not_found"
    excerpt = excerpt.strip()
    if len(excerpt) < 2:
        return None, None, "not_found"

    # 1) exact
    idx = content.find(excerpt)
    if idx >= 0:
        return idx, idx + len(excerpt), "exact"

    # 2) whitespace-normalized (case-insensitive, any whitespace run matches)
    tokens = excerpt.split()
    if tokens:
        pattern = r"\s+".join(re.escape(t) for t in tokens)
        try:
            m = re.search(pattern, content, flags=re.IGNORECASE)
        except re.error:
            m = None
        if m:
            s, e = trim_span(content, m.start(), m.end())
            if e > s:
                return s, e, "normalized"

    # 3) fuzzy — best-aligned region via difflib matching blocks.
    sm = difflib.SequenceMatcher(None, content, excerpt, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if blocks:
        matched = sum(b.size for b in blocks)
        s = blocks[0].a
        e = blocks[-1].a + blocks[-1].size
        span = e - s
        max_span = max(
            len(excerpt) * FUZZY_MAX_SPAN_FACTOR, len(excerpt) + FUZZY_MAX_SPAN_SLACK
        )
        if matched >= FUZZY_MIN_RATIO * len(excerpt) and span <= max_span:
            s, e = trim_span(content, s, e)
            if e > s:
                return s, e, "fuzzy"

    return None, None, "not_found"


def source_descriptor(
    *,
    role: str,  # kept for caller compatibility; stored outside source
    item_id: str,
    turn_index: int,
    flat_turn_index: int,  # same value as turn_index in this stream pipeline
    date: str | None = None,
    has_answer: bool | None = None,
) -> dict[str, Any]:
    """Location descriptor shared by belief.source and evidence.source.

    New compact schema keeps role outside source and stores only turn_id for the
    turn coordinate. Legacy turn_index / trajectory_index are intentionally no
    longer emitted because they denote the same stream position here.
    """
    d: dict[str, Any] = {
        "turn_id": turn_index,
        "item_id": item_id,
    }
    if date is not None:
        d["date"] = date
    if has_answer is not None:
        d["has_answer"] = bool(has_answer)
    return d


def evidence_from_excerpt(
    excerpt: str,
    turn_content: str,
    source: dict[str, Any],
    *,
    stance: str = "asserted",
    role: str = "unknown",
) -> dict[str, Any]:
    """Locate one LLM excerpt inside the turn content and build the record.
    When located, `text` is replaced by the exact slice so
    turn_content[start:end] == text always holds."""
    clean = clean_stance(stance)
    s, e, kind = locate_excerpt(excerpt, turn_content)
    if kind == "not_found" or s is None or e is None:
        return {
            "node_type": "evidence",
            "text": excerpt,
            "start": None,
            "end": None,
            "match": "not_found",
            "via": "llm_excerpt",
            "stance": clean,
            "role": role,
            "source": dict(source),
        }
    return {
        "node_type": "evidence",
        "text": turn_content[s:e],
        "start": s,
        "end": e,
        "match": kind,
        "via": "llm_excerpt",
        "stance": clean,
        "role": role,
        "source": dict(source),
    }


def evidence_from_sentence(
    sentence_start: int,
    sentence_end: int,
    turn_content: str,
    source: dict[str, Any],
    *,
    stance: str = "asserted",
    role: str = "unknown",
) -> dict[str, Any]:
    """Evidence for a whole sentence (offsets are exact by construction)."""
    return {
        "node_type": "evidence",
        "text": turn_content[sentence_start:sentence_end],
        "start": sentence_start,
        "end": sentence_end,
        "match": "exact",
        "via": "split_sentence",
        "stance": clean_stance(stance),
        "role": role,
        "source": dict(source),
    }


def evidence_key(ev: dict[str, Any]) -> tuple:
    """Dedup key for evidence union during merges."""
    src = ev.get("source") or {}
    return (
        src.get("turn_id"),
        ev.get("start"),
        ev.get("end"),
        ev.get("text"),
        ev.get("stance"),
        ev.get("role"),
    )
