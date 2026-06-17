"""
extract.py  (v3)
================
Single-call incremental graph update for the streaming engine.

`update_graph(...)` runs ONE LLM call per turn that returns BOTH:
  * the NEW belief nodes (each with a temporary id "nK"), and
  * the NEW forward ("informs") edges (endpoints are an existing integer id or a
    temporary "nK").

The existing graph (nodes + forward edges) is passed as read-only context; the
prompt instructs the model not to re-emit anything already present (see
prompts._GRAPH_CONTEXT_BLOCK). Id allocation, tmp-id resolution, evidence
attachment and edge validation are done by the streaming engine (stream.py).

Two evidence modes:
  * sentences — content arrives as an indexed sentence list; the model returns
    supporting_sentence_indices, so evidence is always a WHOLE sentence. When
    clustering is on, the same single call shows the sentences grouped by topic
    (still one call).
  * excerpt   — the whole content goes in; the model returns verbatim excerpts.

LLM calls go through `llm.call_model` via the module reference so tests can
monkeypatch `construct_beliefs.llm.call_model`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import llm
from .prompts import (
    build_update_prompt,
    format_clustered_sentences_for_prompt,
    format_sentences_for_prompt,
)

VALID_STANCES = {"asserted", "recalled", "speculated", "judged"}


def _clean_str(v: Any) -> Optional[str]:
    if isinstance(v, str) and v.strip() and v.strip().lower() not in ("null", "none", "n/a"):
        return v.strip()
    return None


def _clean_belief(raw: Any, mode: str, n_sentences: int, ordinal: int) -> Optional[Dict[str, Any]]:
    """Validate / coerce one belief object coming back from the model."""
    if not isinstance(raw, dict):
        return None
    text = raw.get("belief")
    if not isinstance(text, str) or not text.strip():
        return None
    stance = (raw.get("stance") or "").strip().lower()
    if stance not in VALID_STANCES:
        stance = "asserted"

    tmp = raw.get("tmp_id")
    if not isinstance(tmp, str) or not tmp.strip():
        tmp = f"n{ordinal}"
    tmp = tmp.strip()

    out: Dict[str, Any] = {
        "tmp_id": tmp,
        "belief": text.strip(),
        "stance": stance,
        "event_time": _clean_str(raw.get("event_time")),
        "time_text": _clean_str(raw.get("time_text")),
    }

    if mode != "excerpt":
        idx_in = raw.get("supporting_sentence_indices")
        indices: Optional[List[int]] = None
        if isinstance(idx_in, list):
            cleaned = sorted({int(i) for i in idx_in
                              if isinstance(i, (int, float)) and 0 <= int(i) < n_sentences})
            if cleaned:
                indices = cleaned
        # None means "could not validate" → caller falls back to ALL sentences
        out["supporting_sentence_indices"] = indices
    else:
        excerpts_in = raw.get("supporting_excerpts") or []
        excerpts = [e.strip() for e in excerpts_in if isinstance(e, str) and e.strip()]
        if not excerpts:
            return None
        out["supporting_excerpts"] = excerpts
    return out


def _clean_forward(raw: Any) -> List[Dict[str, Any]]:
    """Keep raw (unresolved) forward edges. 'from' may be int or 'nK'; 'to' is 'nK'."""
    out: List[Dict[str, Any]] = []
    for r in raw or []:
        if not isinstance(r, dict):
            continue
        if (r.get("type") or "informs") != "informs":
            continue
        frm = r.get("from", r.get("from_id"))
        to = r.get("to", r.get("to_id"))
        if frm is None or to is None:
            continue
        note = r.get("note", "") or ""
        if not isinstance(note, str):
            note = str(note)
        out.append({"from": frm, "to": to, "type": "informs", "note": note.strip()})
    return out


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _node_line(b: Dict[str, Any]) -> str:
    src = b.get("source") or {}
    line = {
        "id":     b.get("id"),
        "role":   src.get("type", "?"),
        "turn":   src.get("turn_index"),
        "stance": b.get("stance"),
        "conf":   b.get("confidence"),
    }
    if b.get("event_time"):
        line["time"] = b.get("event_time")
    belief = b.get("belief") or ""
    line["belief"] = belief if len(belief) <= 240 else belief[:220] + " …"
    return json.dumps(line, ensure_ascii=False)


def format_graph_nodes(beliefs: List[Dict[str, Any]], char_budget: int = 9000) -> str:
    """Compact JSON view of the existing belief nodes. Over budget → keep the
    MOST RECENT (highest id) nodes and prepend an omission note."""
    if not beliefs:
        return "[]"
    ordered = sorted(beliefs, key=lambda b: b.get("id", 0))
    lines = [_node_line(b) for b in ordered]
    total = sum(len(s) + 4 for s in lines)
    omitted = 0
    while lines and total > char_budget:
        total -= len(lines[0]) + 4
        lines.pop(0)
        omitted += 1
    items = [f"  (... {omitted} earlier node(s) omitted for length ...)"] if omitted else []
    items += ["  " + s for s in lines]
    return "[\n" + ",\n".join(items) + "\n]"


def format_graph_edges(forward_relations: List[Dict[str, Any]],
                       keep_ids: Optional[set] = None,
                       max_edges: int = 400) -> str:
    """Compact view of existing forward edges so the model won't duplicate them.
    If keep_ids is given, only edges whose endpoints are both still present are shown."""
    if not forward_relations:
        return "[]"
    rels = forward_relations
    if keep_ids is not None:
        rels = [r for r in rels if r.get("from_id") in keep_ids and r.get("to_id") in keep_ids]
    if not rels:
        return "[]"
    rels = sorted(rels, key=lambda r: (r.get("to_id", 0), r.get("from_id", 0)))[-max_edges:]
    lines = ["  " + json.dumps({"from": r.get("from_id"), "to": r.get("to_id"),
                                "type": r.get("type", "informs")}, ensure_ascii=False)
             for r in rels]
    return "[\n" + ",\n".join(lines) + "\n]"


# ---------------------------------------------------------------------------
# Update entry point
# ---------------------------------------------------------------------------

def update_graph(
    client,
    model: str,
    *,
    role: str,
    mode: str = "sentences",                     # "sentences" | "excerpt"
    content: Optional[str] = None,               # excerpt mode
    sentences: Optional[List[str]] = None,       # sentence texts (sentences mode)
    clusters: Optional[List[List[int]]] = None,  # global-index groups (clustering on)
    graph_nodes_str: str = "[]",
    graph_edges_str: str = "[]",
    current_date: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    One LLM call → new nodes + new forward edges (both unresolved). Returns:
        { "beliefs": [cleaned beliefs w/ tmp_id], "forward_relations": [raw edges],
          "raw_output": str|None, "skipped": bool, "skip_reason"?: str }
    """
    n_sentences = len(sentences or [])
    if mode != "excerpt":
        if clusters:
            sentences_block = format_clustered_sentences_for_prompt(sentences or [], clusters)
        else:
            sentences_block = format_sentences_for_prompt(sentences or [])
        prompt = build_update_prompt(
            role, mode="sentences", sentences_block=sentences_block,
            graph_nodes=graph_nodes_str, graph_edges=graph_edges_str,
            current_date=current_date)
    else:
        prompt = build_update_prompt(
            role, mode="excerpt", content=content or "",
            graph_nodes=graph_nodes_str, graph_edges=graph_edges_str,
            current_date=current_date)

    if prompt is None:
        return {"beliefs": [], "forward_relations": [], "raw_output": None,
                "skipped": True, "skip_reason": f"unknown role {role!r}"}

    try:
        raw = llm.call_model(client, model, prompt,
                             temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        return {"beliefs": [], "forward_relations": [], "raw_output": f"[ERROR] {e}",
                "skipped": True, "skip_reason": str(e)}

    parsed = llm.parse_json_response(raw)
    out_beliefs: List[Dict[str, Any]] = []
    seen_tmp: set = set()
    raw_beliefs = (parsed.get("beliefs", []) if isinstance(parsed, dict) else []) or []
    for ordinal, b in enumerate(raw_beliefs):
        cb = _clean_belief(b, mode, n_sentences, ordinal)
        if cb is None:
            continue
        # ensure unique tmp ids
        if cb["tmp_id"] in seen_tmp:
            cb["tmp_id"] = f"n{ordinal}"
        seen_tmp.add(cb["tmp_id"])
        out_beliefs.append(cb)

    raw_fwd = (parsed.get("forward_relations", []) if isinstance(parsed, dict) else []) or []
    forward = _clean_forward(raw_fwd)
    return {"beliefs": out_beliefs, "forward_relations": forward,
            "raw_output": raw, "skipped": False}
