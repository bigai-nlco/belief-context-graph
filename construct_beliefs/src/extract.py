"""
extract.py
==========
Per-segment belief extraction. Stage 1 (I/O) and Stage 2 (reasoning) call
into this; Stage 3 (linking) lives in link.py.

Public API:
    extract_one_segment(client, model, segment, io_context=None) -> list[belief]
    format_io_context(io_beliefs) -> str        # used to feed stage 2
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional

from .llm import call_model, parse_json_response
from .prompts import (
    CONTENT_PLACEHOLDER,
    IO_CONTEXT_PLACEHOLDER,
    EXTRACTION_PROMPTS,
)
from .segment import Segment


VALID_STANCES = {"asserted", "recalled", "speculated", "judged"}


# Segment.type → "source.type" stored on each belief.
# Note: assistant_other beliefs are tagged as their own source so the visualizer
# and confidence rules can treat the final-answer layer distinctly.
SEGTYPE_TO_SOURCETYPE = {
    "user_input":         "user_input",
    "tool_call":          "tool_call",
    "assistant_other":    "assistant_other",
    "think":              "llm_reasoning",
    "tool_response":      "tool_result",
}


def _clean_belief(raw: Any) -> Optional[Dict[str, Any]]:
    """Validate / coerce one belief object coming back from the model."""
    if not isinstance(raw, dict):
        return None
    text = raw.get("belief")
    if not isinstance(text, str) or not text.strip():
        return None
    stance = (raw.get("stance") or "").strip().lower()
    if stance not in VALID_STANCES:
        stance = "asserted"
    excerpts_in = raw.get("supporting_excerpts") or []
    excerpts = [e for e in excerpts_in if isinstance(e, str) and e.strip()]
    if not excerpts:
        return None
    return {
        "belief": text.strip(),
        "stance": stance,
        "supporting_excerpts": excerpts,
    }


def format_io_context(io_beliefs: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """
    Compact representation of the I/O belief list used as context for stage 2.
    Stays under `max_chars` to keep the prompt manageable.
    """
    items = []
    total = 0
    for b in io_beliefs:
        src = (b.get("source") or {}).get("type", "?")
        traj_idx = (b.get("source") or {}).get("trajectory_index")
        line = {
            "id":     b.get("id"),
            "source": src,
            "traj":   traj_idx,
            "stance": b.get("stance"),
            "belief": b.get("belief"),
        }
        s = json.dumps(line, ensure_ascii=False)
        if total + len(s) + 2 > max_chars:
            items.append(f'  (... {len(io_beliefs) - len(items)} more belief(s) omitted for length ...)')
            break
        items.append("  " + s)
        total += len(s) + 2
    return "[\n" + ",\n".join(items) + "\n]"


def extract_one_segment(
    client,
    model: str,
    seg: Segment,
    io_context: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the appropriate prompt on a single segment and return:
        { "beliefs": [cleaned beliefs without source], "raw_output": str, "skipped": bool }
    Source attachment is done by the pipeline (knows the segment's traj/seg index).
    """
    if seg.type not in EXTRACTION_PROMPTS:
        return {"beliefs": [], "raw_output": None, "skipped": True,
                "skip_reason": f"unknown segment type {seg.type!r}"}

    template, needs_io = EXTRACTION_PROMPTS[seg.type]
    prompt = template.replace(CONTENT_PLACEHOLDER, seg.content)
    if needs_io:
        ctx = format_io_context(io_context or [])
        prompt = prompt.replace(IO_CONTEXT_PLACEHOLDER, ctx)

    try:
        raw = call_model(client, model, prompt, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        return {"beliefs": [], "prompt": prompt, "raw_output": f"[ERROR] {e}", "skipped": True,
                "skip_reason": str(e)}

    parsed = parse_json_response(raw)
    out_beliefs: List[Dict[str, Any]] = []
    for b in (parsed.get("beliefs", []) if isinstance(parsed, dict) else []) or []:
        cb = _clean_belief(b)
        if cb is not None:
            out_beliefs.append(cb)
    return {"beliefs": out_beliefs, "prompt": prompt, "raw_output": raw, "skipped": False}


def attach_source_and_id(
    beliefs: List[Dict[str, Any]],
    seg: Segment,
    next_id: int,
) -> List[Dict[str, Any]]:
    """
    Tag each belief with a global id and the source / segment provenance.
    Returns the updated belief list; mutates the input copies.
    """
    out: List[Dict[str, Any]] = []
    layer = "io" if seg.type in {"user_input", "tool_call", "assistant_other"} else "reasoning"
    for b in beliefs:
        b2 = dict(b)
        b2["id"] = next_id
        next_id += 1
        b2["layer"] = layer
        b2["source"] = {
            "type": SEGTYPE_TO_SOURCETYPE.get(seg.type, seg.type),
            "role": seg.role,
            "trajectory_index": seg.traj_idx,
            "segment_index": seg.seg_idx,
            "segment_type": seg.type,
        }
        out.append(b2)
    return out
