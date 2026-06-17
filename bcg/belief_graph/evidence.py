"""
evidence.py  (v3)
=================
Evidence provenance. Every belief carries an `evidence` list; each entry points
back into the ORIGINAL turn content by exact character offsets:

    {
      "text":  "<exact slice content[start:end] when located, else the excerpt verbatim>",
      "start": 118,            # char offset in the original turn content
      "end":   178,            # exclusive
      "match": "exact" | "normalized" | "fuzzy" | "not_found",
      "via":   "llm_excerpt" | "split_sentence",
      "source": { ...location descriptor, see source_descriptor()... }
    }

Two producers:
  * sentence mode — the content was split into COMPLETE sentences with exact
    offsets, so evidence is built directly (match="exact", via="split_sentence").
    This is how "evidence must be a whole sentence" is guaranteed.
  * excerpt mode  — the model returns verbatim excerpts; each is located inside
    the turn content by a three-stage matcher (exact → whitespace-normalized →
    difflib fuzzy). Evidence may be a sentence fragment in this mode.

There is no segmentation in v3: the whole turn is one unit, so all offsets are
already relative to the original turn content.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, Optional, Tuple

# Fuzzy-match acceptance knobs.
FUZZY_MIN_RATIO = 0.6          # matched chars must cover >= 60% of the excerpt
FUZZY_MAX_SPAN_FACTOR = 2.0    # located span must stay near the excerpt length
FUZZY_MAX_SPAN_SLACK = 80


def _trim_span(text: str, s: int, e: int) -> Tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def locate_excerpt(excerpt: str, content: str) -> Tuple[Optional[int], Optional[int], str]:
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
            s, e = _trim_span(content, m.start(), m.end())
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
        max_span = max(len(excerpt) * FUZZY_MAX_SPAN_FACTOR,
                       len(excerpt) + FUZZY_MAX_SPAN_SLACK)
        if matched >= FUZZY_MIN_RATIO * len(excerpt) and span <= max_span:
            s, e = _trim_span(content, s, e)
            if e > s:
                return s, e, "fuzzy"

    return None, None, "not_found"


def source_descriptor(
    *,
    role: str,                       # "user" | "assistant" | "tool" (== source.type)
    item_id: str,
    turn_index: int,
    flat_turn_index: int,
    date: Optional[str] = None,
    has_answer: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Location descriptor shared by belief.source and each evidence.source.
    `type` IS the role (user/assistant/tool). `trajectory_index` is the global
    flat turn index across the trajectory (processing order) — what the
    visualizer uses to index into result.json's flat `trajectory` list.
    """
    d: Dict[str, Any] = {
        "type": role,
        "role": role,
        "item_id": item_id,
        "turn_index": turn_index,
        "trajectory_index": flat_turn_index,
    }
    if date is not None:
        d["date"] = date
    if has_answer is not None:
        d["has_answer"] = bool(has_answer)
    return d


def evidence_from_excerpt(
    excerpt: str,
    turn_content: str,
    source: Dict[str, Any],
) -> Dict[str, Any]:
    """Locate one LLM excerpt inside the turn content and build the record.
    When located, `text` is replaced by the exact slice so
    turn_content[start:end] == text always holds."""
    s, e, kind = locate_excerpt(excerpt, turn_content)
    if kind == "not_found" or s is None or e is None:
        return {"text": excerpt, "start": None, "end": None,
                "match": "not_found", "via": "llm_excerpt", "source": dict(source)}
    return {"text": turn_content[s:e], "start": s, "end": e,
            "match": kind, "via": "llm_excerpt", "source": dict(source)}


def evidence_from_sentence(
    sentence_start: int,
    sentence_end: int,
    turn_content: str,
    source: Dict[str, Any],
) -> Dict[str, Any]:
    """Evidence for a whole sentence (offsets are exact by construction)."""
    return {"text": turn_content[sentence_start:sentence_end],
            "start": sentence_start, "end": sentence_end,
            "match": "exact", "via": "split_sentence", "source": dict(source)}


def evidence_key(ev: Dict[str, Any]) -> tuple:
    """Dedup key for evidence union during merges."""
    src = ev.get("source") or {}
    return (
        src.get("trajectory_index"),
        ev.get("start"),
        ev.get("end"),
        ev.get("text"),
    )
