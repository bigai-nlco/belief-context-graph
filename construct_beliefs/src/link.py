"""
link.py
=======
Two single-pass linkers over the complete belief list:

    link_forward(...)   — derivation flow.  Edge from earlier to later
                          belief, type = "informs".  Used to build the
                          belief evolution path / reasoning trail.
                          Does NOT affect confidence.

    link_backward(...)  — evaluation.  Edge from later evidence to earlier
                          target, types = confirms / contradicts / extends.
                          Drives the dynamic confidence update in
                          confidence.py.

Both share the same plumbing: a compact JSON of the belief list goes to the
model, the model returns a JSON list of edges, and we validate ids /
direction / dedup before handing back.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from .llm import call_model, parse_json_response
from .prompts import (
    BELIEFS_LIST_PLACEHOLDER,
    PROMPT_LINK_BACKWARD,
    PROMPT_LINK_FORWARD,
)


VALID_BACKWARD_TYPES = {"confirms", "contradicts", "extends"}
VALID_FORWARD_TYPES  = {"informs"}


def _compact_belief(b: Dict[str, Any]) -> Dict[str, Any]:
    """Trim a belief to the fields the linker actually needs to see."""
    src = b.get("source") or {}
    return {
        "id":     b.get("id"),
        "layer":  b.get("layer"),
        "source": src.get("type"),
        "traj":   src.get("trajectory_index"),
        "stance": b.get("stance"),
        "conf":   b.get("confidence"),
        "belief": b.get("belief"),
    }


def _build_blob(beliefs: List[Dict[str, Any]], max_chars: int) -> str:
    compact = [_compact_belief(b) for b in beliefs]
    blob = json.dumps(compact, ensure_ascii=False, indent=2)
    if len(blob) > max_chars:
        trimmed = []
        for c in compact:
            cc = dict(c)
            if isinstance(cc.get("belief"), str) and len(cc["belief"]) > 220:
                cc["belief"] = cc["belief"][:200] + " …"
            trimmed.append(cc)
        blob = json.dumps(trimmed, ensure_ascii=False, indent=2)
    return blob


def _run_link(
    client,
    model: str,
    beliefs: List[Dict[str, Any]],
    *,
    prompt_template: str,
    response_field: str,
    valid_types: Set[str],
    direction: str,                  # "forward" or "backward"
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    max_chars_per_pass: int = 14000,
) -> Dict[str, Any]:
    """Shared runner. Returns {<response_field>: [...], raw_output, skipped}."""
    if len(beliefs) < 2:
        return {response_field: [], "raw_output": None, "skipped": True,
                "skip_reason": "fewer than 2 beliefs"}

    blob = _build_blob(beliefs, max_chars_per_pass)
    prompt = prompt_template.replace(BELIEFS_LIST_PLACEHOLDER, blob)
    try:
        raw = call_model(client, model, prompt,
                         temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        return {response_field: [], "prompt": prompt, "raw_output": f"[ERROR] {e}", "skipped": True,
                "skip_reason": str(e)}

    parsed = parse_json_response(raw)
    rels_in = parsed.get(response_field, []) if isinstance(parsed, dict) else []

    known_ids = {b.get("id") for b in beliefs}
    out: List[Dict[str, Any]] = []
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
        if fid not in known_ids or tid not in known_ids:
            continue
        if fid == tid:
            continue
        # Direction enforcement:
        #   backward: from_id (evidence) >= to_id (target). fid < tid is dropped.
        #   forward:  from_id (informant) <  to_id (derived). fid >= tid is dropped.
        if direction == "backward" and fid < tid:
            continue
        if direction == "forward"  and fid >= tid:
            continue
        note = r.get("note", "") or ""
        if not isinstance(note, str):
            note = str(note)
        out.append({
            "from_id": fid,
            "to_id":   tid,
            "type":    rtype,
            "note":    note.strip(),
        })

    # Dedup
    seen: Set[tuple] = set()
    dedup: List[Dict[str, Any]] = []
    for r in out:
        key = (r["from_id"], r["to_id"], r["type"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)

    return {response_field: dedup, "prompt": prompt, "raw_output": raw, "skipped": False}


def link_backward(
    client, model: str, beliefs: List[Dict[str, Any]],
    temperature: float = 0.0, max_tokens: Optional[int] = None,
    max_chars_per_pass: int = 14000,
) -> Dict[str, Any]:
    """Evaluation pass: confirms / contradicts / extends, later -> earlier."""
    return _run_link(
        client, model, beliefs,
        prompt_template=PROMPT_LINK_BACKWARD,
        response_field="relations",
        valid_types=VALID_BACKWARD_TYPES,
        direction="backward",
        temperature=temperature, max_tokens=max_tokens,
        max_chars_per_pass=max_chars_per_pass,
    )


def link_forward(
    client, model: str, beliefs: List[Dict[str, Any]],
    temperature: float = 0.0, max_tokens: Optional[int] = None,
    max_chars_per_pass: int = 14000,
) -> Dict[str, Any]:
    """Derivation pass: informs, earlier -> later."""
    return _run_link(
        client, model, beliefs,
        prompt_template=PROMPT_LINK_FORWARD,
        response_field="forward_relations",
        valid_types=VALID_FORWARD_TYPES,
        direction="forward",
        temperature=temperature, max_tokens=max_tokens,
        max_chars_per_pass=max_chars_per_pass,
    )
