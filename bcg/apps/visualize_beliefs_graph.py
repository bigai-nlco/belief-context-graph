#!/usr/bin/env python3
"""
visualize_beliefs_graph.py
==========================

Interactive HTML visualizer for construct_beliefs result.json, final_graph.json,
or belief_graph.jsonl.

This version is aligned with the current graph schema and exposes the newer
runtime artifacts:

  - belief / decision nodes, with raw code IDs displayed from 0;
  - relations, with raw relation IDs displayed from 0 in the inspector only;
  - factors, including factor id, factor type, weight, inputs/outputs, and
    activation_condition in the inspector only;
  - relation.activated_factor_ids, shown in the relation inspector only;
  - timing records for graph-building sub-steps:
      node_generation, merging, llm_check, edge_generation, turn_total,
      plus final_merge when present;
  - concurrency/runtime notes, especially the fact that llm_check timing is
    wall-clock when merge verification groups are run concurrently;
  - decision history / confidence history for decision nodes.

Usage
-----
    python -m bcg.visualize_beliefs_graph path/to/result.json -o graph.html
    python -m bcg.visualize_beliefs_graph path/to/final_graph.json -o graph.html
    python -m bcg.visualize_beliefs_graph path/to/belief_graph.jsonl -o graph.html
"""

from __future__ import annotations

import html as html_lib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from bcg.apps.cli_help import RichArgumentParser

ROLE_LABEL = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
}

REL_COLOR_TYPES = ("causal", "depends_on", "supplements", "contradicts", "supports")


def escape(s: Any) -> str:
    return html_lib.escape("" if s is None else str(s))


def safe_class(s: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(s or "unknown").lower()).strip("-") or "unknown"


def node_text(n: dict[str, Any]) -> str:
    return str(n.get("belief") or n.get("decision") or n.get("content") or "")


def node_role(n: dict[str, Any]) -> str:
    src = n.get("source") or {}
    return str(n.get("role") or src.get("role") or src.get("type") or "unknown")


def int_or_none(v: Any) -> int | None:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def float_or_none(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def as_list(v: Any) -> list[Any]:
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if v is None:
        return []
    return [v]


def coerce_int_list(v: Any) -> list[int]:
    out: list[int] = []
    seen = set()
    for x in as_list(v):
        ix = int_or_none(x)
        if ix is None or ix in seen:
            continue
        seen.add(ix)
        out.append(ix)
    return out


def load_graph_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        last: dict[str, Any] | None = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                last = obj
        if last is None:
            raise ValueError(f"no JSON object found in {path}")
        return last
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return obj


def normalize_trajectory(data: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory = data.get("trajectory")
    if isinstance(trajectory, list):
        return [m if isinstance(m, dict) else {"role": "unknown", "content": str(m)}
                for m in trajectory]
    messages = data.get("messages")
    if isinstance(messages, list):
        return [m if isinstance(m, dict) else {"role": "unknown", "content": str(m)}
                for m in messages]
    return []


def normalize_nodes(data: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = data.get("all_nodes")
    if not isinstance(nodes, list):
        nodes = data.get("nodes")

    if not isinstance(nodes, list):
        nodes = []
        seen = set()
        for key in ("all_beliefs", "beliefs", "all_decisions", "decisions"):
            vals = data.get(key) or []
            if not isinstance(vals, list):
                continue
            for n in vals:
                if not isinstance(n, dict):
                    continue
                nid = int_or_none(n.get("id"))
                if nid is None or nid in seen:
                    continue
                seen.add(nid)
                nodes.append(n)

    clean_nodes: list[dict[str, Any]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = int_or_none(n.get("id"))
        if nid is None:
            continue
        if not node_text(n):
            continue
        nn = dict(n)
        nn["id"] = nid
        nn.setdefault("node_type", "decision" if "decision" in nn and "belief" not in nn else "belief")
        nn.setdefault("source", nn.get("source") or {})
        clean_nodes.append(nn)

    clean_nodes.sort(key=lambda x: x.get("id", 0))
    return clean_nodes


def normalize_factors(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("factors")
    if raw is None:
        raw = data.get("all_factors")
    items: list[Any] = []

    if isinstance(raw, dict):
        for key, val in raw.items():
            if isinstance(val, dict):
                fac = dict(val)
                fac.setdefault("id", int_or_none(key))
                items.append(fac)
    elif isinstance(raw, list):
        items = raw

    clean: list[dict[str, Any]] = []
    next_id = 0
    used = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        fac = dict(item)
        fid = int_or_none(fac.get("id"))
        if fid is None:
            while next_id in used:
                next_id += 1
            fid = next_id
        used.add(fid)
        # New factor schema intentionally has no "name" field. Drop legacy names
        # so the visualizer mirrors current output and does not encourage review
        # based on stale names.
        fac.pop("name", None)
        fac["id"] = fid
        fac.setdefault("node_type", "factor")
        fac.setdefault("factor_type", fac.get("type") or "factor")
        fac.setdefault("activation_condition", fac.get("activation_condition") or {})
        fac.setdefault("input_variables", coerce_int_list(fac.get("input_variables")))
        fac.setdefault("output_variables", coerce_int_list(fac.get("output_variables")))
        clean.append(fac)

    clean.sort(key=lambda x: x.get("id", 0))
    return clean


def normalize_relations(
    data: dict[str, Any],
    node_ids: Iterable[int],
) -> list[dict[str, Any]]:
    rels = data.get("relations") or []
    if not rels:
        rels = []
        for key in ("forward_relations", "backward_relations"):
            vals = data.get(key) or []
            if isinstance(vals, list):
                rels.extend(vals)

    node_id_set = set(node_ids)
    clean_rels: list[dict[str, Any]] = []
    seen_rel = set()
    next_rel_id = 0

    for r in rels:
        if not isinstance(r, dict):
            continue
        fid = int_or_none(r.get("from_id", r.get("from")))
        tid = int_or_none(r.get("to_id", r.get("to")))
        if fid is None or tid is None:
            continue
        if fid == tid or fid not in node_id_set or tid not in node_id_set:
            continue

        rtype = str(r.get("type") or "depends_on")
        if rtype == "informs":
            rtype = "depends_on"
        elif rtype == "extends":
            rtype = "supplements"
        elif rtype == "confirms":
            rtype = "supports"

        rid = int_or_none(r.get("id"))
        if rid is None:
            rid = next_rel_id
        next_rel_id = max(next_rel_id, rid + 1)

        activated = r.get("activated_factor_ids")
        if not isinstance(activated, list):
            activated = r.get("factor_ids")
        if not isinstance(activated, list):
            activated = r.get("activated_factors")
        activated_factor_ids = coerce_int_list(activated)

        # If a legacy edge had one factor_id field, surface it too.
        single_fid = int_or_none(r.get("factor_id"))
        if single_fid is not None and single_fid not in activated_factor_ids:
            activated_factor_ids.append(single_fid)

        key = (fid, tid, rtype, rid)
        if key in seen_rel:
            continue
        seen_rel.add(key)

        clean_rels.append({
            "id": rid,
            "from_id": fid,
            "to_id": tid,
            "type": rtype,
            "note": str(r.get("note") or ""),
            "activated_factor_ids": activated_factor_ids,
            "raw": r,
        })

    clean_rels.sort(key=lambda x: x.get("id", 0))
    return clean_rels


def normalize_timing(data: dict[str, Any]) -> dict[str, Any]:
    timing = data.get("timing")
    if not isinstance(timing, dict):
        return {}
    # Keep the original nested structure. The HTML/JS renderer knows how to
    # display per_turn, by_step, final_merge, duration_seconds, start/end.
    return timing


def normalize_decision_history(data: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = (
        data.get("decision_history"),
        data.get("decision_histories"),
        data.get("final_decision_history"),
    )
    out: list[dict[str, Any]] = []
    for cand in candidates:
        if isinstance(cand, list):
            for item in cand:
                if isinstance(item, dict):
                    out.append(item)
                else:
                    out.append({"value": item})
        elif isinstance(cand, dict):
            for key, val in cand.items():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            rec = dict(item)
                            rec.setdefault("node_id", int_or_none(key))
                            out.append(rec)
                        else:
                            out.append({"node_id": int_or_none(key), "value": item})
                elif isinstance(val, dict):
                    rec = dict(val)
                    rec.setdefault("node_id", int_or_none(key))
                    out.append(rec)
    return out


def normalize_input(data: dict[str, Any]) -> dict[str, Any]:
    trajectory = normalize_trajectory(data)
    nodes = normalize_nodes(data)
    factors = normalize_factors(data)
    rels = normalize_relations(data, [n["id"] for n in nodes])

    evidence = data.get("evidence") if isinstance(data.get("evidence"), list) else []
    evidence_clean = [e for e in evidence if isinstance(e, dict)]

    return {
        "trajectory": trajectory,
        "nodes": nodes,
        "relations": rels,
        "factors": factors,
        "evidence": evidence_clean,
        "timing": normalize_timing(data),
        "options": data.get("options") if isinstance(data.get("options"), dict) else {},
        "token_usage": data.get("token_usage") if isinstance(data.get("token_usage"), dict) else {},
        "decision_history": normalize_decision_history(data),
        "raw_counts": data.get("counts") if isinstance(data.get("counts"), dict) else {},
    }


def evidence_records_for_node(
    node: dict[str, Any],
    evidence_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen = set()

    for ev in node.get("evidence") or []:
        if isinstance(ev, dict):
            key = ev.get("id") if ev.get("id") is not None else id(ev)
            if key in seen:
                continue
            seen.add(key)
            records.append(ev)

    for raw_eid in node.get("evidence_ids") or []:
        eid = int_or_none(raw_eid)
        if eid is None or eid in seen:
            continue
        ev = evidence_by_id.get(eid)
        if ev is not None:
            seen.add(eid)
            records.append(ev)
    return records


def slice_with_marks(original: str, intervals: list[tuple[int, int, int]]) -> list[dict[str, Any]]:
    if not intervals:
        return [{"text": original, "nodes": []}]

    points = {0, len(original)}
    for s, e, _ in intervals:
        if 0 <= s < e <= len(original):
            points.add(s)
            points.add(e)

    pieces: list[dict[str, Any]] = []
    sorted_points = sorted(points)
    for i in range(len(sorted_points) - 1):
        a, b = sorted_points[i], sorted_points[i + 1]
        if a >= b:
            continue
        active = [nid for s, e, nid in intervals if s <= a and b <= e]
        pieces.append({"text": original[a:b], "nodes": sorted(set(active))})
    return pieces


def compute_layout(nodes: list[dict[str, Any]], width: int = 1040, height: int = 380):
    if not nodes:
        return {}, width, height, [], 150, 70

    by_turn: dict[int, list[dict[str, Any]]] = {}
    for n in nodes:
        src = n.get("source") or {}
        ti = src.get("turn_id", src.get("turn_index", src.get("trajectory_index", -1)))
        if not isinstance(ti, int):
            ti = -1
        by_turn.setdefault(ti, []).append(n)

    turns = sorted(by_turn)
    col_w, pad_x, pad_y = 155, 74, 42
    max_per_col = max(len(v) for v in by_turn.values())
    final_w = max(width, pad_x * 2 + len(turns) * col_w)
    final_h = max(height, pad_y + max_per_col * 62 + 48)

    pos: dict[int, tuple[float, float]] = {}
    for ci, ti in enumerate(turns):
        x = pad_x + col_w / 2 + ci * col_w
        for j, n in enumerate(sorted(by_turn[ti], key=lambda x: x["id"])):
            pos[n["id"]] = (x, pad_y + 20 + j * 62)
    return pos, final_w, final_h, turns, col_w, pad_x


CSS = r"""
:root {
  --bg:#faf7f2; --panel:#fff; --ink:#1a1a1a; --soft:#555; --faint:#8a8276; --rule:#e8e2d8;
  --accent:#b8442f; --evidence:#ffe9a3; --evidence-active:#ffd23f;
  --user-fg:#1e5a9c; --user-bg:#e0ecfa; --assistant-fg:#8b2c5b; --assistant-bg:#fbe3ee;
  --tool-fg:#2e6e3a; --tool-bg:#e1f0e4; --unknown-fg:#666; --unknown-bg:#eee;
  --decision-fg:#7c3aed; --decision-bg:#efe6fd; --factor-fg:#7a4d00; --factor-bg:#fff0c2;
  --causal:#c84a3e; --depends:#6b89b8; --supp:#b3500e; --contra:#1a1a1a; --supports:#1f6a35;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;overflow:hidden;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
body{display:flex;flex-direction:column}
header.page-head{padding:18px 30px 12px;border-bottom:1px solid var(--rule);background:var(--panel);display:flex;justify-content:space-between;gap:16px;align-items:baseline}
h1{font-size:24px;margin:0;font-family:Georgia,serif}
.meta{font-size:13px;color:var(--soft)}
.stats{padding:9px 30px;border-bottom:1px solid var(--rule);background:var(--panel);display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--soft)}
.stats b{color:var(--ink)}
.layout{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,.90fr) minmax(0,1.10fr)}
.left{overflow:auto;background:var(--panel);border-right:1px solid var(--rule);padding:18px 26px 70px}
.right{display:flex;flex-direction:column;min-width:0;overflow:hidden}
.graph-wrap{position:relative;flex:0 1 58%;min-height:250px;overflow:auto;border-bottom:1px solid var(--rule);padding:50px 24px 20px}
.detail-wrap{flex:1;overflow:auto;padding:18px 26px 70px}
.section-title{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--soft);margin:0 0 12px}
.msg{border:1px solid var(--rule);border-radius:7px;margin-bottom:14px;background:#fff}
.msg-user{background:#fbf5ec}.msg-tool,.msg-function{background:#f1f6ec}.msg-system{background:#f5f1ea}
.msg.flash{box-shadow:0 0 0 2px var(--accent)}
.msg-head{display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--rule);padding:7px 12px;font-size:12px;color:var(--soft)}
.msg-role{font-family:monospace;font-weight:700;text-transform:uppercase;color:white;border-radius:3px;padding:2px 7px}
.role-user{background:var(--user-fg)}.role-assistant{background:#111}.role-tool,.role-function{background:var(--tool-fg)}.role-system{background:#999}.role-unknown{background:#777}
.msg-count{margin-left:auto;font-family:monospace;font-size:11px}
.msg-body{white-space:pre-wrap;word-break:break-word;margin:0;padding:12px;font:12px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:380px;overflow:auto}
.ev{background:var(--evidence);border-bottom:1px solid #d4b75c;cursor:pointer}.ev.active{background:var(--evidence-active);outline:2px solid #c89800}
svg{display:block;min-width:100%}
.col-label{font:10px monospace;fill:var(--faint);text-anchor:middle}
.node{cursor:pointer}.node rect{rx:8;ry:8;stroke-width:1.6}.node:hover rect{stroke-width:2.6}.node.active rect{stroke-width:3;filter:drop-shadow(0 0 4px var(--accent))}.node.dimmed,.edge.dimmed,.edge-hit.dimmed{opacity:.22}
.node .id{font:9px monospace;text-anchor:middle}.node .conf{font:bold 11px monospace;text-anchor:middle}
.node.source-user rect{fill:var(--user-bg);stroke:var(--user-fg)}.node.source-user text{fill:var(--user-fg)}
.node.source-assistant rect{fill:var(--assistant-bg);stroke:var(--assistant-fg)}.node.source-assistant text{fill:var(--assistant-fg)}
.node.source-tool rect,.node.source-function rect{fill:var(--tool-bg);stroke:var(--tool-fg)}.node.source-tool text,.node.source-function text{fill:var(--tool-fg)}
.node.source-unknown rect{fill:var(--unknown-bg);stroke:var(--unknown-fg)}.node.source-unknown text{fill:var(--unknown-fg)}
.node.node-decision rect{stroke-dasharray:5 3;fill:var(--decision-bg)}
.edge{fill:none;stroke-width:1.8;cursor:pointer;pointer-events:stroke}.edge:hover,.edge.active{stroke-width:3}.edge-hit{fill:none;stroke:transparent;stroke-width:13;cursor:pointer;pointer-events:stroke}
.edge.type-causal{stroke:var(--causal)}.edge.type-depends_on{stroke:var(--depends)}.edge.type-supplements{stroke:var(--supp);stroke-dasharray:5 3}.edge.type-contradicts{stroke:var(--contra);stroke-dasharray:2 3}.edge.type-supports{stroke:var(--supports)}
.legend{position:absolute;right:24px;top:12px;display:flex;gap:12px;background:rgba(255,255,255,.94);border:1px solid var(--rule);border-radius:5px;padding:5px 9px;font:10px monospace;color:var(--soft)}
.sw{display:inline-block;width:18px;border-top:2px solid;vertical-align:middle;margin-right:4px}.sw.causal{border-color:var(--causal)}.sw.depends_on{border-color:var(--depends)}.sw.supplements{border-color:var(--supp);border-top-style:dashed}.sw.contradicts{border-color:var(--contra);border-top-style:dotted}
.detail-empty{font-size:13px;color:var(--faint);font-style:italic}
.card{background:#fff;border:1px solid var(--rule);border-radius:7px;padding:15px 17px;margin-bottom:14px}
.card h3{margin:0 0 9px;font-family:Georgia,serif}.card h4{margin:15px 0 7px;font-size:12px;color:var(--soft);text-transform:uppercase;letter-spacing:.08em}
.node-text{font-family:Georgia,serif;font-size:15px;line-height:1.5;margin:10px 0}
.badges{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}.badge{font-size:10px;text-transform:uppercase;letter-spacing:.08em;font-weight:700;border-radius:3px;padding:2px 7px;background:#eee;color:#333}
.badge.role-user{color:var(--user-fg);background:var(--user-bg)}.badge.role-assistant{color:var(--assistant-fg);background:var(--assistant-bg)}.badge.role-tool,.badge.role-function{color:var(--tool-fg);background:var(--tool-bg)}.badge.type-decision{color:var(--decision-fg);background:var(--decision-bg)}.badge.factor{color:var(--factor-fg);background:var(--factor-bg);cursor:pointer}.badge.rel-causal{color:var(--causal);background:#fbe9e6}.badge.rel-depends_on{color:var(--depends);background:#e7eef7}.badge.rel-supplements{color:var(--supp);background:#fde6cf}.badge.rel-contradicts{color:#111;background:#e9e9e9}
.kv{font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--soft);margin-top:8px;word-break:break-word}
.entities{font:12px monospace;color:var(--soft)}
.excerpts{margin:5px 0 0;padding:0;list-style:none}.excerpts li{font:11px/1.55 monospace;color:var(--soft);margin:4px 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
.confidence-grid,.overview-grid{display:grid;grid-template-columns:repeat(2,minmax(150px,1fr));gap:8px;margin:8px 0 6px}
.conf-cell,.metric-cell{background:var(--bg);border:1px solid var(--rule);border-radius:5px;padding:7px}
.conf-label,.metric-label{font:10px monospace;color:var(--faint);text-transform:uppercase}.conf-value,.metric-value{font:bold 13px monospace;color:var(--ink);margin-top:2px}
.raw-list{margin:6px 0 0;padding-left:18px}.raw-list li{font:11px/1.55 monospace;color:var(--soft);margin:3px 0}
.side{background:var(--bg);border:1px solid var(--rule);border-radius:5px;padding:10px}.side-label{font:10px monospace;color:var(--faint);text-transform:uppercase}.side-text{font-family:Georgia,serif;font-size:13.5px;line-height:1.45}
.table-wrap{overflow:auto;border:1px solid var(--rule);border-radius:6px;background:#fff;margin:8px 0 12px;max-height:360px}
table{width:100%;border-collapse:collapse;font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
th,td{text-align:left;vertical-align:top;padding:6px 8px;border-bottom:1px solid var(--rule)}
th{position:sticky;top:0;background:#f8f4ee;color:var(--soft);z-index:1}
tr:last-child td{border-bottom:0}
button.linkish{border:0;background:transparent;color:var(--accent);font:inherit;padding:0;cursor:pointer;text-decoration:underline;text-underline-offset:2px}
.json-pre{white-space:pre-wrap;background:#fbf8f3;border:1px solid var(--rule);border-radius:6px;padding:10px;font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#444;max-height:320px;overflow:auto}
.hint{padding:9px 30px;border-top:1px solid var(--rule);background:var(--panel);font-size:12px;color:var(--faint)}
.note{font-size:12px;line-height:1.55;color:var(--soft);background:#fbf8f3;border:1px solid var(--rule);border-radius:6px;padding:10px;margin-top:8px}
"""


JS = r"""
let current = null;
const D = window.GRAPH_DATA;

function esc(s){
  return (s==null?'':String(s))
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
function textOf(n){return n.belief || n.decision || n.content || ''}
function fmt(v,digits=3){return (typeof v==='number' && Number.isFinite(v)) ? v.toFixed(digits) : '—'}
function fmtSec(v){return (typeof v==='number' && Number.isFinite(v)) ? v.toFixed(6) : '0.000000'}
function rawJson(v){return esc(JSON.stringify(v, null, 2))}
function listValue(v){return Array.isArray(v)?(v.length?esc(v.join(', ')):'[]'):esc(v==null?'—':v)}
function relationKey(r){return r.from_id+'->'+r.to_id+':'+r.type+':'+r.id}
function factorById(id){return (D.factors||[]).find(f=>Number(f.id)===Number(id))}
function nodeById(id){return (D.nodes||[]).find(n=>Number(n.id)===Number(id))}
function relById(id){return (D.relations||[]).find(r=>Number(r.id)===Number(id))}
function evidenceById(id){return (D.evidence||[]).find(e=>Number(e.id)===Number(id))}
function factorBadge(id){
  return `<span class="badge factor" onclick="event.stopPropagation();selectFactor(${Number(id)})">factor #${esc(id)}</span>`;
}
function relationButton(r){
  return `<button class="linkish" onclick="selectEdge(${r.from_id},${r.to_id},'${esc(r.type)}',${r.id})">R#${esc(r.id)}</button>`;
}
function nodeButton(id){
  return `<button class="linkish" onclick="selectNode(${Number(id)})">#${esc(id)}</button>`;
}

function activatedByFactor(fid){
  return (D.relations||[]).filter(r => Array.isArray(r.activated_factor_ids) && r.activated_factor_ids.map(Number).includes(Number(fid)));
}
function factorsForRelation(r){
  if(!r || !Array.isArray(r.activated_factor_ids)) return [];
  return r.activated_factor_ids.map(factorById).filter(Boolean);
}
function evidenceForNode(n){
  const out=[], seen=new Set();
  (Array.isArray(n.evidence)?n.evidence:[]).forEach(e=>{
    if(e&&typeof e==='object'){
      const k=e.id!=null?'id:'+e.id:'obj:'+out.length;
      if(!seen.has(k)){seen.add(k);out.push(e)}
    }
  });
  (Array.isArray(n.evidence_ids)?n.evidence_ids:[]).forEach(raw=>{
    const e=evidenceById(raw);
    if(e&&!seen.has('id:'+e.id)){seen.add('id:'+e.id);out.push(e)}
  });
  return out;
}

function resetVisuals(){
  document.querySelectorAll('.active,.dimmed,.flash').forEach(e=>e.classList.remove('active','dimmed','flash'));
}
function clearSelection(){current=null;resetVisuals();overviewDetail()}

function selectNode(id){
  if(current&&current.kind==='node'&&current.id===id){clearSelection();return}
  clearSelection();
  current={kind:'node',id};
  let connected=new Set([id]);
  D.relations.forEach(r=>{if(r.from_id===id)connected.add(r.to_id);if(r.to_id===id)connected.add(r.from_id)});
  document.querySelectorAll('.node').forEach(el=>{
    let nid=+el.dataset.id;
    if(nid===id)el.classList.add('active'); else if(!connected.has(nid))el.classList.add('dimmed');
  });
  document.querySelectorAll('.edge,.edge-hit').forEach(el=>{
    let f=+el.dataset.fromId,t=+el.dataset.toId;
    if(f===id||t===id)el.classList.add('active'); else el.classList.add('dimmed');
  });
  let first=null;
  document.querySelectorAll('.ev').forEach(s=>{
    let ids=(s.dataset.nodes||'').split(' ').filter(Boolean);
    if(ids.includes(String(id))){s.classList.add('active');if(!first)first=s}
  });
  let n=nodeById(id);
  if(n&&n.source&&Number.isInteger(n.source.turn_id)){
    let p=document.getElementById('msg-'+n.source.turn_id);
    if(p)p.classList.add('flash');
    if(first)first.scrollIntoView({behavior:'smooth',block:'center'}); else if(p)p.scrollIntoView({behavior:'smooth',block:'start'});
  }
  nodeDetail(id);
}

function selectEdge(f,t,type,rid=null){
  const r = rid==null ? D.relations.find(x=>x.from_id===f&&x.to_id===t&&x.type===type) : relById(rid);
  if(!r){return}
  if(current&&current.kind==='edge'&&current.id===r.id){clearSelection();return}
  clearSelection();
  current={kind:'edge',id:r.id,from:r.from_id,to:r.to_id,type:r.type};
  let key=relationKey(r);
  document.querySelectorAll('.edge,.edge-hit').forEach(el=>{
    if(el.dataset.key===key)el.classList.add('active'); else el.classList.add('dimmed');
  });
  document.querySelectorAll('.node').forEach(el=>{
    let id=+el.dataset.id;
    if(id===r.from_id||id===r.to_id)el.classList.add('active'); else el.classList.add('dimmed');
  });
  edgeDetail(r.id);
}

function selectFactor(id){
  if(current&&current.kind==='factor'&&current.id===id){clearSelection();return}
  clearSelection();
  current={kind:'factor',id};
  const rels=activatedByFactor(id);
  const touched=new Set();
  rels.forEach(r=>{touched.add(r.from_id); touched.add(r.to_id)});
  document.querySelectorAll('.edge,.edge-hit').forEach(el=>{
    const rid=Number(el.dataset.relId);
    if(rels.some(r=>Number(r.id)===rid)) el.classList.add('active'); else el.classList.add('dimmed');
  });
  document.querySelectorAll('.node').forEach(el=>{
    const nid=Number(el.dataset.id);
    if(touched.has(nid)) el.classList.add('active'); else el.classList.add('dimmed');
  });
  factorDetail(id);
}

function confidenceBlock(n){
  const factorIds=Array.isArray(n.factor_ids)?n.factor_ids:[];
  return `<h4>Confidence</h4>
  <div class="confidence-grid">
    <div class="conf-cell"><div class="conf-label">confidence</div><div class="conf-value">${fmt(n.confidence)}</div></div>
    <div class="conf-cell"><div class="conf-label">initial_confidence</div><div class="conf-value">${fmt(n.initial_confidence)}</div></div>
    <div class="conf-cell"><div class="conf-label">evidence_confidence</div><div class="conf-value">${fmt(n.evidence_confidence,6)}</div></div>
    <div class="conf-cell"><div class="conf-label">factor_confidence</div><div class="conf-value">${fmt(n.factor_confidence,6)}</div></div>
  </div>
  <div class="kv">factor_ids=${factorIds.length?factorIds.map(factorBadge).join(' '):'[]'} · evidence_ids=${listValue(n.evidence_ids||[])}</div>`;
}
function evidenceBlock(n){
  const evs=evidenceForNode(n);
  if(!evs.length) return '';
  return '<h4>Evidence nodes</h4><ul class="raw-list">'+evs.map(e=>{
    const src=e.source||{};
    return `<li>#${esc(e.id)} · stance=${esc(e.stance||'')} · role=${esc(e.role||src.role||src.type||'')} · turn_id=${esc(src.turn_id ?? src.turn_index ?? '')} · span=${esc(e.start)}-${esc(e.end)}<br>${esc(e.text||'')}</li>`;
  }).join('')+'</ul>';
}
function confidenceHistoryBlock(n){
  const h=Array.isArray(n.confidence_history)?n.confidence_history:[];
  if(!h.length) return '';
  return '<h4>Confidence history</h4><ul class="raw-list">'+h.map((x,i)=>{
    if(x && typeof x === 'object'){
      return `<li>${i}. ${esc(x.step||x.stage||'')} · value=${esc(x.value ?? x.confidence ?? '')}${x.delta!=null?' · delta='+esc(x.delta):''}${x.reason?' · '+esc(x.reason):''}</li>`;
    }
    return `<li>${i}. ${esc(x)}</li>`;
  }).join('')+'</ul>';
}
function decisionHistoryForNode(n){
  const out=[];
  const fields=['decision_history','decision_confidence_history','answer_history','history'];
  fields.forEach(k=>{
    if(Array.isArray(n[k])) n[k].forEach(x=>out.push({source:k, record:x}));
  });
  if(Array.isArray(D.decision_history)){
    D.decision_history.forEach(x=>{
      if(!x || typeof x!=='object') return;
      const nid = x.node_id ?? x.decision_node_id ?? x.decision_id ?? x.id;
      if(Number(nid)===Number(n.id)) out.push({source:'result.decision_history', record:x});
    });
  }
  return out;
}
function decisionHistoryBlock(n){
  if((n.node_type||'belief')!=='decision') return '';
  const h=decisionHistoryForNode(n);
  if(!h.length) return '<h4>Decision history</h4><div class="note">No explicit decision_history field was found for this decision node. Confidence history is shown separately if present.</div>';
  return '<h4>Decision history</h4><div class="table-wrap"><table><thead><tr><th>#</th><th>source</th><th>record</th></tr></thead><tbody>'+
    h.map((x,i)=>`<tr><td>${i}</td><td>${esc(x.source)}</td><td><pre class="json-pre">${rawJson(x.record)}</pre></td></tr>`).join('')+
    '</tbody></table></div>';
}
function turnTimingBlock(n){
  const src=n.source||{};
  const tid=src.turn_id ?? src.turn_index ?? src.trajectory_index;
  if(!Array.isArray(D.timing?.per_turn)) return '';
  const rows=D.timing.per_turn.filter(t=>Number(t.turn_index)===Number(tid));
  if(!rows.length) return '';
  return '<h4>Turn timing</h4>'+timingTable(rows);
}
function nodeRelationsBlock(id){
  const rows=D.relations.filter(r=>r.from_id===id||r.to_id===id);
  if(!rows.length) return '';
  return '<h4>Incident relations</h4><div class="table-wrap"><table><thead><tr><th>rel</th><th>type</th><th>from</th><th>to</th><th>activated factors</th></tr></thead><tbody>'+
    rows.map(r=>`<tr><td>${relationButton(r)}</td><td>${esc(r.type)}</td><td>${nodeButton(r.from_id)}</td><td>${nodeButton(r.to_id)}</td><td>${r.activated_factor_ids.length?r.activated_factor_ids.map(factorBadge).join(' '):'—'}</td></tr>`).join('')+
    '</tbody></table></div>';
}

function nodeDetail(id){
  let n=nodeById(id);if(!n){overviewDetail();return}
  let src=n.source||{}, role=n.role||src.role||src.type||'unknown', nt=n.node_type||'belief';
  let conf=(typeof n.confidence==='number')?n.confidence.toFixed(2):'—';
  let entities=Array.isArray(n.entities)&&n.entities.length?`<h4>Entities</h4><div class="entities">${n.entities.map(esc).join(', ')}</div>`:'';
  let excerpts=Array.isArray(n.supporting_excerpts)?n.supporting_excerpts:[];
  let exc=excerpts.length?'<h4>Supporting excerpts</h4><ul class="excerpts">'+excerpts.map(e=>`<li>✓ ${esc(e)}</li>`).join('')+'</ul>':'';
  let temporal=(n.time_text||n.event_time)?`<h4>Time</h4><div class="kv">time_text=${esc(n.time_text||'null')} · event_time=${esc(n.event_time||'null')}</div>`:'';
  document.getElementById('detail').innerHTML=`<div class="card">
    <h3>${nt==='decision'?'Decision':'Belief'} #${esc(n.id)} · ${conf}</h3>
    <div class="badges"><span class="badge role-${esc(role)}">${esc(role)}</span><span class="badge type-${esc(nt)}">${esc(nt)}</span><span class="badge">${esc(n.stance||'')}</span></div>
    <div class="node-text">${esc(textOf(n))}</div>
    ${confidenceBlock(n)}
    <h4>Source</h4><div class="kv">turn_id=${esc(src.turn_id ?? src.turn_index ?? src.trajectory_index ?? 'null')} · role=${esc(role)}${src.date?' · '+esc(src.date):''}</div>
    ${entities}${temporal}${evidenceBlock(n)}${exc}${confidenceHistoryBlock(n)}${decisionHistoryBlock(n)}${turnTimingBlock(n)}${nodeRelationsBlock(id)}
  </div>`;
}

function edgeDetail(rid){
  const r=relById(rid); if(!r){overviewDetail();return}
  const a=nodeById(r.from_id), b=nodeById(r.to_id);
  if(!a||!b){overviewDetail();return}
  const fids=Array.isArray(r.activated_factor_ids)?r.activated_factor_ids:[];
  const factorCards=factorsForRelation(r).map(f=>factorMiniBlock(f)).join('');
  document.getElementById('detail').innerHTML=`<div class="card">
    <h3>Relation #${esc(r.id)}</h3>
    <div class="badges"><span class="badge rel-${esc(r.type)}">${esc(r.type)}</span><span class="badge">#${esc(r.from_id)} → #${esc(r.to_id)}</span>${fids.map(factorBadge).join(' ')}</div>
    ${r.note?`<h4>Note</h4><div class="kv">${esc(r.note)}</div>`:''}
    <h4>Activated factor ids</h4><div class="kv">${fids.length?fids.map(factorBadge).join(' '):'[]'}</div>
    <div class="pair">
      <div class="side"><div class="side-label">from · #${esc(r.from_id)}</div><div class="side-text">${esc(textOf(a))}</div></div>
      <div class="side"><div class="side-label">to · #${esc(r.to_id)}</div><div class="side-text">${esc(textOf(b))}</div></div>
    </div>
    ${factorCards?'<h4>Activated factors</h4>'+factorCards:''}
    <h4>Raw relation</h4><pre class="json-pre">${rawJson(r.raw || r)}</pre>
  </div>`;
}

function factorMiniBlock(f){
  const ac=f.activation_condition||{};
  return `<div class="side" style="margin-bottom:8px">
    <div class="side-label">factor #${esc(f.id)} · ${esc(f.factor_type||'factor')}</div>
    <div class="kv">weight=${esc(f.weight ?? '—')} · input_variables=${listValue(f.input_variables||[])} · output_variables=${listValue(f.output_variables||[])}</div>
    <div class="side-text">${esc(ac.note || ac.text || ac.condition || '')}</div>
  </div>`;
}
function factorDetail(id){
  const f=factorById(id); if(!f){overviewDetail();return}
  const rels=activatedByFactor(id);
  document.getElementById('detail').innerHTML=`<div class="card">
    <h3>Factor #${esc(f.id)}</h3>
    <div class="badges"><span class="badge factor">factor</span><span class="badge">${esc(f.factor_type||'factor')}</span></div>
    <div class="confidence-grid">
      <div class="conf-cell"><div class="conf-label">weight</div><div class="conf-value">${esc(f.weight ?? '—')}</div></div>
      <div class="conf-cell"><div class="conf-label">activated by relations</div><div class="conf-value">${rels.length}</div></div>
      <div class="conf-cell"><div class="conf-label">input variables</div><div class="conf-value">${listValue(f.input_variables||[])}</div></div>
      <div class="conf-cell"><div class="conf-label">output variables</div><div class="conf-value">${listValue(f.output_variables||[])}</div></div>
    </div>
    <h4>Activation condition</h4><pre class="json-pre">${rawJson(f.activation_condition||{})}</pre>
    ${rels.length?'<h4>Relations activating this factor</h4><div class="table-wrap"><table><thead><tr><th>rel</th><th>type</th><th>from</th><th>to</th></tr></thead><tbody>'+rels.map(r=>`<tr><td>${relationButton(r)}</td><td>${esc(r.type)}</td><td>${nodeButton(r.from_id)}</td><td>${nodeButton(r.to_id)}</td></tr>`).join('')+'</tbody></table></div>':'<div class="note">No relation in this snapshot lists this factor in activated_factor_ids.</div>'}
    <h4>Raw factor</h4><pre class="json-pre">${rawJson(f)}</pre>
  </div>`;
}

function timingTable(rows){
  if(!Array.isArray(rows)||!rows.length) return '<div class="note">No per-turn timing rows.</div>';
  return '<div class="table-wrap"><table><thead><tr><th>turn</th><th>role</th><th>node_generation</th><th>merging</th><th>llm_check</th><th>edge_generation</th><th>turn_total</th></tr></thead><tbody>'+
    rows.map(t=>`<tr><td>${esc(t.turn_index ?? '')}</td><td>${esc(t.role ?? '')}</td><td>${fmtSec(t.node_generation)}</td><td>${fmtSec(t.merging)}</td><td>${fmtSec(t.llm_check)}</td><td>${fmtSec(t.edge_generation)}</td><td>${fmtSec(t.turn_total)}</td></tr>`).join('')+
    '</tbody></table></div>';
}
function byStepTable(byStep){
  if(!byStep || typeof byStep !== 'object' || !Object.keys(byStep).length) return '';
  return '<h4>Timing by step</h4><div class="table-wrap"><table><thead><tr><th>step</th><th>total seconds</th><th>n_turns</th></tr></thead><tbody>'+
    Object.entries(byStep).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${fmtSec(v?.total_seconds)}</td><td>${esc(v?.n_turns ?? '')}</td></tr>`).join('')+
    '</tbody></table></div>';
}
function finalMergeBlock(fm){
  if(!fm || typeof fm !== 'object') return '';
  return `<h4>Final merge timing</h4><div class="confidence-grid">
    <div class="conf-cell"><div class="conf-label">merging</div><div class="conf-value">${fmtSec(fm.merging)}</div></div>
    <div class="conf-cell"><div class="conf-label">llm_check</div><div class="conf-value">${fmtSec(fm.llm_check)}</div></div>
    <div class="conf-cell"><div class="conf-label">total</div><div class="conf-value">${fmtSec(fm.total)}</div></div>
  </div>`;
}
function factorsOverview(){
  if(!Array.isArray(D.factors)||!D.factors.length) return '<div class="note">No factors field found in this graph snapshot.</div>';
  return '<div class="table-wrap"><table><thead><tr><th>factor</th><th>type</th><th>weight</th><th>inputs</th><th>outputs</th><th>activation note</th><th>activated by</th></tr></thead><tbody>'+
    D.factors.map(f=>{
      const rels=activatedByFactor(f.id);
      const ac=f.activation_condition||{};
      return `<tr><td>${factorBadge(f.id)}</td><td>${esc(f.factor_type||'')}</td><td>${esc(f.weight ?? '')}</td><td>${listValue(f.input_variables||[])}</td><td>${listValue(f.output_variables||[])}</td><td>${esc(ac.note||ac.text||ac.condition||'')}</td><td>${rels.length?rels.map(relationButton).join(' '):'—'}</td></tr>`;
    }).join('')+
    '</tbody></table></div>';
}
function relationsOverview(){
  if(!Array.isArray(D.relations)||!D.relations.length) return '<div class="note">No relations.</div>';
  return '<div class="table-wrap"><table><thead><tr><th>rel</th><th>type</th><th>from</th><th>to</th><th>activated factor ids</th></tr></thead><tbody>'+
    D.relations.map(r=>`<tr><td>${relationButton(r)}</td><td>${esc(r.type)}</td><td>${nodeButton(r.from_id)}</td><td>${nodeButton(r.to_id)}</td><td>${r.activated_factor_ids.length?r.activated_factor_ids.map(factorBadge).join(' '):'[]'}</td></tr>`).join('')+
    '</tbody></table></div>';
}
function concurrencyBlock(){
  const opt=D.options||{};
  const verify=opt.verify_merge;
  const inc=opt.incremental_merge;
  const rows=[
    ['incremental_merge', inc],
    ['verify_merge', verify],
    ['merge_strategy', opt.merge_strategy],
    ['incremental_merge_threshold', opt.incremental_merge_threshold],
    ['context_chars', opt.context_chars]
  ];
  return `<h4>Concurrency / runtime notes</h4>
  <div class="note">
    Online runs are expected to process the same problem_id serially and different problem_ids concurrently.
    In timing, <b>llm_check</b> is wall-clock time for merge verification; when multiple candidate groups are verified concurrently,
    it is not the sum of every individual LLM call. The final merge timing follows the same convention when applicable.
  </div>
  <div class="table-wrap"><table><thead><tr><th>option</th><th>value</th></tr></thead><tbody>${
    rows.map(([k,v])=>`<tr><td>${esc(k)}</td><td>${esc(v ?? '—')}</td></tr>`).join('')
  }</tbody></table></div>`;
}
function overviewDetail(){
  const timing=D.timing||{};
  const perTurn=Array.isArray(timing.per_turn)?timing.per_turn:[];
  const activatedRefs=(D.relations||[]).reduce((acc,r)=>acc+(Array.isArray(r.activated_factor_ids)?r.activated_factor_ids.length:0),0);
  document.getElementById('detail').innerHTML=`<div class="card">
    <h3>Graph overview</h3>
    <div class="overview-grid">
      <div class="metric-cell"><div class="metric-label">nodes</div><div class="metric-value">${D.nodes.length}</div></div>
      <div class="metric-cell"><div class="metric-label">relations</div><div class="metric-value">${D.relations.length}</div></div>
      <div class="metric-cell"><div class="metric-label">factors</div><div class="metric-value">${D.factors.length}</div></div>
      <div class="metric-cell"><div class="metric-label">activated factor refs</div><div class="metric-value">${activatedRefs}</div></div>
    </div>
    <h4>Relations and activated factor ids</h4>${relationsOverview()}
    <h4>Factors</h4>${factorsOverview()}
    <h4>Per-turn timing</h4>${perTurn.length?timingTable(perTurn):'<div class="note">No timing.per_turn records were found. Use a result.json produced by the timing-enabled pipeline to see per-step timing.</div>'}
    ${byStepTable(timing.by_step)}${finalMergeBlock(timing.final_merge)}
    ${concurrencyBlock()}
  </div>`;
}

document.addEventListener('click',e=>{
  let ev=e.target.closest('.ev');
  if(ev){
    let ids=(ev.dataset.nodes||'').split(' ').filter(Boolean);
    if(ids.length)selectNode(+ids[0]);
  }
});
document.addEventListener('keydown',e=>{if(e.key==='Escape')clearSelection()});
document.addEventListener('DOMContentLoaded',overviewDetail);
"""


def render_message_panel(
    i: int,
    msg: dict[str, Any],
    nodes_for_msg: list[tuple[int, dict[str, Any], list[tuple[int, int]]]],
) -> str:
    role = msg.get("role") or "?"
    content = msg.get("content", "") or ""
    role_class = safe_class(ROLE_LABEL.get(role, role))
    intervals: list[tuple[int, int, int]] = []
    for nid, _n, ranges in nodes_for_msg:
        for s, e in ranges:
            intervals.append((s, e, nid))

    pieces = slice_with_marks(content, intervals)
    body: list[str] = []
    for p in pieces:
        txt = escape(p["text"])
        if p["nodes"]:
            ids = " ".join(str(x) for x in p["nodes"])
            body.append(f'<span class="ev" data-nodes="{ids}">{txt}</span>')
        else:
            body.append(txt)

    count = len(nodes_for_msg)
    return (
        f'<article class="msg msg-{role_class}" id="msg-{i}">'
        f'<header class="msg-head"><span class="msg-role role-{role_class}">{escape(ROLE_LABEL.get(role, role))}</span>'
        f'<span>turn_id <b>{i}</b></span><span class="msg-count">{count} node{"s" if count != 1 else ""}</span></header>'
        f'<pre class="msg-body">{"".join(body)}</pre></article>'
    )


def render_graph_svg(
    nodes: list[dict[str, Any]],
    rels: list[dict[str, Any]],
    pos: dict[int, tuple[float, float]],
    width: int,
    height: int,
    turns: list[int],
    col_w: int,
    pad_x: int,
) -> str:
    parts: list[str] = []

    for ci, ti in enumerate(turns):
        cx = pad_x + col_w / 2 + ci * col_w
        parts.append(f'<text class="col-label" x="{cx}" y="{height-12}">turn[{ti}]</text>')

    marker_defs = []
    marker_colors = {
        "causal": "#c84a3e",
        "depends_on": "#6b89b8",
        "supplements": "#b3500e",
        "contradicts": "#1a1a1a",
        "supports": "#1f6a35",
    }
    for t, c in marker_colors.items():
        marker_defs.append(
            f'<marker id="arr-{t}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        )
    parts.append("<defs>" + "".join(marker_defs) + "</defs>")

    def edge_path(fid: int, tid: int) -> str | None:
        if fid not in pos or tid not in pos:
            return None
        x1, y1 = pos[fid]
        x2, y2 = pos[tid]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        px, py = -dy / length, dx / length
        offset = max(24, min(78, abs(dx) * 0.22 + 18))
        cx = mx + px * offset
        cy = my + py * offset
        return f"M {x1:.1f},{y1:.1f} Q {cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}"

    for r in rels:
        fid, tid, typ = r["from_id"], r["to_id"], safe_class(r.get("type"))
        p = edge_path(fid, tid)
        if not p:
            continue
        marker = typ if typ in marker_colors else "depends_on"
        key = f"{fid}->{tid}:{typ}:{r.get('id')}"
        rid = int(r.get("id", 0))
        parts.append(
            f'<path class="edge type-{typ}" d="{p}" marker-end="url(#arr-{marker})" '
            f'data-rel-id="{rid}" data-from-id="{fid}" data-to-id="{tid}" '
            f'data-type="{typ}" data-key="{escape(key)}" '
            f'onclick="selectEdge({fid},{tid},\'{typ}\',{rid})"/>'
        )
        # Invisible wider hit target: keeps edges easy to click without drawing
        # relation IDs or activated factor IDs on the graph canvas. Those IDs
        # are shown only in the Inspector after the edge is selected.
        parts.append(
            f'<path class="edge-hit" d="{p}" '
            f'data-rel-id="{rid}" data-from-id="{fid}" data-to-id="{tid}" '
            f'data-type="{typ}" data-key="{escape(key)}" '
            f'onclick="selectEdge({fid},{tid},\'{typ}\',{rid})"/>'
        )

    for n in nodes:
        nid = n["id"]
        if nid not in pos:
            continue
        x, y = pos[nid]
        role = safe_class(node_role(n))
        nt = safe_class(n.get("node_type", "belief"))
        conf = n.get("confidence")
        conf_text = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "—"
        label = "D" if nt == "decision" else "B"
        # IDs intentionally use the raw code ID, starting from 0. No +1 offset.
        parts.append(
            f'<g class="node source-{role} node-{nt}" data-id="{nid}" '
            f'transform="translate({x:.1f},{y:.1f})" onclick="selectNode({nid})">'
            f'<rect x="-42" y="-19" width="84" height="38"/>'
            f'<text class="id" x="0" y="-6">{label}#{nid}</text>'
            f'<text class="conf" x="0" y="9">{escape(conf_text)}</text>'
            f'</g>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">' + "".join(parts) + "</svg>"
    )


def render_html(data: dict[str, Any], src_path: str) -> str:
    norm = normalize_input(data)
    trajectory = norm["trajectory"]
    nodes = norm["nodes"]
    rels = norm["relations"]
    factors = norm["factors"]
    evidence = norm["evidence"]

    evidence_by_id: dict[int, dict[str, Any]] = {}
    for ev in evidence:
        eid = int_or_none(ev.get("id"))
        if eid is not None:
            evidence_by_id[eid] = ev

    by_msg: dict[int, list[tuple[int, dict[str, Any], list[tuple[int, int]]]]] = {}
    for n in nodes:
        src = n.get("source") or {}
        ti = src.get("turn_id", src.get("turn_index", src.get("trajectory_index")))
        if not isinstance(ti, int) or ti < 0 or ti >= len(trajectory):
            continue
        host = trajectory[ti].get("content", "") or ""
        ranges: list[tuple[int, int]] = []
        for ev in evidence_records_for_node(n, evidence_by_id):
            s, e = ev.get("start"), ev.get("end")
            if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(host):
                ranges.append((s, e))
        by_msg.setdefault(ti, []).append((n["id"], n, ranges))

    msg_html = (
        "".join(render_message_panel(i, m, by_msg.get(i, [])) for i, m in enumerate(trajectory))
        or "<p>No trajectory.</p>"
    )

    pos, gw, gh, turns, col_w, pad_x = compute_layout(nodes)
    graph_svg = render_graph_svg(nodes, rels, pos, gw, gh, turns, col_w, pad_x)

    role_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    rel_counts: dict[str, int] = {}
    activated_refs = 0
    for n in nodes:
        role_counts[node_role(n)] = role_counts.get(node_role(n), 0) + 1
        nt = n.get("node_type", "belief")
        type_counts[nt] = type_counts.get(nt, 0) + 1
    for r in rels:
        rel_counts[r.get("type", "?")] = rel_counts.get(r.get("type", "?"), 0) + 1
        activated_refs += len(r.get("activated_factor_ids") or [])

    role_stats = "".join(f"<span><b>{v}</b> {escape(k)}</span>" for k, v in sorted(role_counts.items()))
    type_stats = "".join(f"<span><b>{v}</b> {escape(k)}</span>" for k, v in sorted(type_counts.items()))
    rel_stats = ", ".join(f"{v} {escape(k)}" for k, v in sorted(rel_counts.items()))

    meta_model = data.get("model", "") or ""
    prompt_name = data.get("prompt_name", "construct_beliefs") or "construct_beliefs"

    graph_data = {
        "nodes": nodes,
        "relations": rels,
        "evidence": evidence,
        "factors": factors,
        "timing": norm["timing"],
        "options": norm["options"],
        "token_usage": norm["token_usage"],
        "decision_history": norm["decision_history"],
        "raw_counts": norm["raw_counts"],
    }
    graph_json = json.dumps(graph_data, ensure_ascii=False).replace("</", "<\\/")

    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>Belief Graph · {escape(prompt_name)}</title><style>{CSS}</style></head><body>"
        f'<header class="page-head"><h1>Belief Graph · {escape(prompt_name)}</h1>'
        f'<div class="meta">source · <b>{escape(Path(src_path).name)}</b> &nbsp; model · <b>{escape(meta_model)}</b></div></header>'
        f'<div class="stats"><span><b>{len(nodes)}</b> nodes</span>{type_stats}{role_stats}'
        f'<span><b>{len(rels)}</b> relations</span><span>{escape(rel_stats)}</span>'
        f'<span><b>{len(factors)}</b> factors</span><span><b>{activated_refs}</b> activated factor refs</span></div>'
        '<div class="layout"><div class="left"><h2 class="section-title">Conversation trajectory</h2>'
        f'{msg_html}</div><div class="right"><div class="graph-wrap"><div class="legend">'
        '<span><span class="sw causal"></span>causal</span>'
        '<span><span class="sw depends_on"></span>depends_on</span>'
        '<span><span class="sw supplements"></span>supplements</span>'
        '<span><span class="sw contradicts"></span>contradicts</span>'
        f'</div>{graph_svg}</div><div class="detail-wrap"><h2 class="section-title">Inspector</h2><div id="detail"></div></div></div></div>'
        '<div class="hint">Raw IDs match code IDs and start from 0. Click a node, edge path, or factor badge. Esc clears selection.</div>'
        f'<script>window.GRAPH_DATA = {graph_json};{JS}</script>'
        "</body></html>"
    )


def main(argv: list[str] | None = None) -> None:
    parser = RichArgumentParser(
        prog="bcg construct visualize",
        description="Graph visualizer for construct_beliefs result.json / final_graph.json / belief_graph.jsonl"
    )
    parser.add_argument("input", help="Path to result.json, final_graph.json, or belief_graph.jsonl")
    parser.add_argument("--output", "-o", default=None, help="Output HTML path")
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[error] file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = load_graph_file(in_path)
    except Exception as e:
        print(f"[error] failed to read {in_path}: {e}", file=sys.stderr)
        sys.exit(1)

    html = render_html(data, str(in_path))
    out_path = Path(args.output) if args.output else in_path.parent / f"graph_{in_path.stem}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()


__all__ = [
    "ROLE_LABEL",
    "REL_COLOR_TYPES",
    "escape",
    "safe_class",
    "node_text",
    "node_role",
    "int_or_none",
    "float_or_none",
    "as_list",
    "coerce_int_list",
    "load_graph_file",
    "normalize_trajectory",
    "normalize_nodes",
    "normalize_factors",
    "normalize_relations",
    "normalize_timing",
    "normalize_decision_history",
    "normalize_input",
    "evidence_records_for_node",
    "slice_with_marks",
    "compute_layout",
    "CSS",
    "JS",
    "render_message_panel",
    "render_graph_svg",
    "render_html",
    "main",
]
