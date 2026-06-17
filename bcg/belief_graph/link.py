"""
link.py  (v3)
=============
Backward (evaluation) linking, run ONCE at trajectory end over the full graph.

Forward ("informs") linking is no longer here — it is produced together with
the new nodes by `extract.update_graph` in the single per-turn call.

    link_backward_all(...)  — one LLM call over the whole belief graph →
        confirms / contradicts / extends. from_id is the LATER belief (the
        evidence), to_id any strictly smaller (earlier) id. Drives the
        confidence update in confidence.py.

LLM calls go through `llm.call_model` via the module reference so tests can
monkeypatch `construct_beliefs.llm.call_model`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from . import llm
from .prompts import ALL_BELIEFS_PLACEHOLDER, PROMPT_LINK_BACKWARD_ALL

VALID_BACKWARD_TYPES = {"confirms", "contradicts", "extends"}


def _compact_belief(b: Dict[str, Any]) -> Dict[str, Any]:
    src = b.get("source") or {}
    c: Dict[str, Any] = {
        "id":     b.get("id"),
        "role":   src.get("type"),
        "turn":   src.get("turn_index"),
        "stance": b.get("stance"),
        "conf":   b.get("confidence"),
        "belief": b.get("belief"),
    }
    if b.get("event_time"):
        c["time"] = b.get("event_time")
    return c


def _build_blob(beliefs: List[Dict[str, Any]], max_chars: int = 28000) -> str:
    if not beliefs:
        return "[]"
    compact = [_compact_belief(b) for b in beliefs]
    blob = json.dumps(compact, ensure_ascii=False, indent=2)
    if len(blob) > max_chars:
        trimmed = []
        for c in compact:
            cc = dict(c)
            if isinstance(cc.get("belief"), str) and len(cc["belief"]) > 200:
                cc["belief"] = cc["belief"][:180] + " …"
            trimmed.append(cc)
        blob = json.dumps(trimmed, ensure_ascii=False, indent=2)
    return blob


def _validate(
    rels_in: Any,
    *,
    valid_types: Set[str],
    all_ids: Set[int],
) -> List[Dict[str, Any]]:
    """Backward edges: from_id (later) > to_id (earlier), both active."""
    out: List[Dict[str, Any]] = []
    seen: Set[tuple] = set()
    for r in rels_in or []:
        if not isinstance(r, dict):
            continue
        try:
            fid = int(r.get("from_id"))
            tid = int(r.get("to_id"))
        except (TypeError, ValueError):
            continue
        rtype = r.get("type")
        if rtype not in valid_types:
            continue
        if fid not in all_ids or tid not in all_ids:
            continue
        if not fid > tid:
            continue
        note = r.get("note", "") or ""
        if not isinstance(note, str):
            note = str(note)
        key = (fid, tid, rtype)
        if key in seen:
            continue
        seen.add(key)
        out.append({"from_id": fid, "to_id": tid, "type": rtype, "note": note.strip()})
    return out


def link_backward_all(
    client, model: str,
    beliefs: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    max_chars: int = 28000,
) -> Dict[str, Any]:
    """Evaluation pass over the whole graph: confirms/contradicts/extends."""
    if len(beliefs) < 2:
        return {"relations": [], "raw_output": None, "skipped": True,
                "skip_reason": "fewer than 2 beliefs"}

    prompt = PROMPT_LINK_BACKWARD_ALL.replace(
        ALL_BELIEFS_PLACEHOLDER, _build_blob(beliefs, max_chars))
    try:
        raw = llm.call_model(client, model, prompt,
                             temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        return {"relations": [], "raw_output": f"[ERROR] {e}", "skipped": True,
                "skip_reason": str(e)}

    parsed = llm.parse_json_response(raw)
    rels_in = parsed.get("relations", []) if isinstance(parsed, dict) else []
    all_ids = {b["id"] for b in beliefs}
    rels = _validate(rels_in, valid_types=VALID_BACKWARD_TYPES, all_ids=all_ids)
    return {"relations": rels, "raw_output": raw, "skipped": False}
