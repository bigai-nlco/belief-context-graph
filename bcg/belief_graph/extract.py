"""
extract.py
==========
Single-call incremental graph update for the streaming engine.

`update_graph(...)` runs one LLM call per turn. The model returns:
  * new belief nodes,
  * new decision nodes, and
  * typed relations among new nodes and existing nodes.

Valid relation types: causal, depends_on, supplements, contradicts.
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
VALID_RELATION_TYPES = {"causal", "depends_on", "supplements", "contradicts"}


def _clean_str(v: Any) -> Optional[str]:
    if isinstance(v, str) and v.strip() and v.strip().lower() not in ("null", "none", "n/a"):
        return v.strip()
    return None


def _clean_entities(v: Any) -> List[str]:
    if not isinstance(v, list):
        return []
    out: List[str] = []
    seen = set()
    for x in v:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _clean_node(
    raw: Any,
    mode: str,
    n_sentences: int,
    ordinal: int,
    *,
    node_type: str,
) -> Optional[Dict[str, Any]]:
    """Validate / coerce one belief or decision object from the model."""
    if not isinstance(raw, dict):
        return None

    text_key = "decision" if node_type == "decision" else "belief"
    text = raw.get(text_key)
    if not isinstance(text, str) or not text.strip():
        # Allow decision objects that used the historical `belief` key by mistake.
        if node_type == "decision" and isinstance(raw.get("belief"), str):
            text = raw.get("belief")
        else:
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
        "node_type": node_type,
        "belief": text.strip(),
        "stance": stance,
        "entities": _clean_entities(raw.get("entities")),
        "event_time": _clean_str(raw.get("event_time")),
        "time_text": _clean_str(raw.get("time_text")),
    }

    if node_type == "decision":
        out["decision"] = text.strip()

    if mode != "excerpt":
        idx_in = raw.get("supporting_sentence_indices")
        indices: Optional[List[int]] = None
        if isinstance(idx_in, list):
            cleaned = sorted({int(i) for i in idx_in
                              if isinstance(i, (int, float)) and 0 <= int(i) < n_sentences})
            if cleaned:
                indices = cleaned
        out["supporting_sentence_indices"] = indices
    else:
        excerpts_in = raw.get("supporting_excerpts") or []
        excerpts = [e.strip() for e in excerpts_in if isinstance(e, str) and e.strip()]
        if not excerpts:
            return None
        out["supporting_excerpts"] = excerpts
    return out


def _clean_relations(raw: Any) -> List[Dict[str, Any]]:
    """Keep unresolved typed relations. Endpoints may be existing ids or tmp ids."""
    out: List[Dict[str, Any]] = []
    for r in raw or []:
        if not isinstance(r, dict):
            continue
        rtype = (r.get("type") or "").strip()
        if rtype not in VALID_RELATION_TYPES:
            continue
        frm = r.get("from", r.get("from_id"))
        to = r.get("to", r.get("to_id"))
        if frm is None or to is None:
            continue
        note = r.get("note", "") or ""
        if not isinstance(note, str):
            note = str(note)
        out.append({"from": frm, "to": to, "type": rtype, "note": note.strip()})
    return out


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _node_line(b: Dict[str, Any]) -> str:
    src = b.get("source") or {}
    line = {
        "id": b.get("id"),
        "node_type": b.get("node_type", "belief"),
        "role": src.get("type", "?"),
        "turn": src.get("turn_index"),
        "stance": b.get("stance"),
        "conf": b.get("confidence"),
        "entities": b.get("entities") or [],
    }
    if b.get("event_time"):
        line["time"] = b.get("event_time")
    belief = b.get("belief") or b.get("decision") or ""
    line["content"] = belief if len(belief) <= 240 else belief[:220] + " …"
    return json.dumps(line, ensure_ascii=False)


def format_graph_nodes(nodes: List[Dict[str, Any]], char_budget: int = 9000) -> str:
    """Compact JSON view of existing graph nodes. Over budget keeps recent nodes."""
    if not nodes:
        return "[]"
    ordered = sorted(nodes, key=lambda b: b.get("id", 0))
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


def format_graph_edges(relations: List[Dict[str, Any]],
                       keep_ids: Optional[set] = None,
                       max_edges: int = 400) -> str:
    """Compact view of existing typed relations so the model won't duplicate them."""
    if not relations:
        return "[]"
    rels = relations
    if keep_ids is not None:
        rels = [r for r in rels if r.get("from_id") in keep_ids and r.get("to_id") in keep_ids]
    if not rels:
        return "[]"
    rels = sorted(rels, key=lambda r: (r.get("from_id", 0), r.get("to_id", 0)))[-max_edges:]
    lines = ["  " + json.dumps({"from": r.get("from_id"), "to": r.get("to_id"),
                                "type": r.get("type")}, ensure_ascii=False)
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
    mode: str = "sentences",
    content: Optional[str] = None,
    sentences: Optional[List[str]] = None,
    clusters: Optional[List[List[int]]] = None,
    graph_nodes_str: str = "[]",
    graph_edges_str: str = "[]",
    current_date: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    One LLM call. Returns cleaned unresolved nodes and typed relations:
        {
          "nodes": [...], "beliefs": [...], "decisions": [...],
          "relations": [...], "raw_output": str|None,
          "skipped": bool, "skip_reason"?: str
        }
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
        return {"nodes": [], "beliefs": [], "decisions": [], "relations": [],
                "raw_output": None, "skipped": True, "skip_reason": f"unknown role {role!r}"}

    try:
        raw = llm.call_model(client, model, prompt,
                             temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        return {"nodes": [], "beliefs": [], "decisions": [], "relations": [],
                "raw_output": f"[ERROR] {e}", "skipped": True, "skip_reason": str(e)}

    parsed = llm.parse_json_response(raw)
    parsed = parsed if isinstance(parsed, dict) else {}

    out_nodes: List[Dict[str, Any]] = []
    out_beliefs: List[Dict[str, Any]] = []
    out_decisions: List[Dict[str, Any]] = []
    seen_tmp: set = set()
    ordinal = 0

    for b in parsed.get("beliefs", []) or []:
        cb = _clean_node(b, mode, n_sentences, ordinal, node_type="belief")
        if cb is None:
            continue
        if cb["tmp_id"] in seen_tmp:
            cb["tmp_id"] = f"n{ordinal}"
        seen_tmp.add(cb["tmp_id"])
        out_nodes.append(cb)
        out_beliefs.append(cb)
        ordinal += 1

    for d in parsed.get("decisions", []) or []:
        cd = _clean_node(d, mode, n_sentences, ordinal, node_type="decision")
        if cd is None:
            continue
        if cd["tmp_id"] in seen_tmp:
            cd["tmp_id"] = f"n{ordinal}"
        seen_tmp.add(cd["tmp_id"])
        out_nodes.append(cd)
        out_decisions.append(cd)
        ordinal += 1

    relations = _clean_relations(parsed.get("relations", []) or [])
    return {"nodes": out_nodes, "beliefs": out_beliefs, "decisions": out_decisions,
            "relations": relations, "raw_output": raw, "skipped": False}
