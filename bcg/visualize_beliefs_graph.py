#!/usr/bin/env python3
"""
visualize_beliefs_graph.py  (v6 schema compatible)
====================================================
Graph-style visualizer for construct_beliefs result.json, final_graph.json, or
belief_graph.jsonl.

Compatible with the newer schema:
  - nodes live in all_nodes / nodes, with node_type="belief" or "decision";
  - legacy all_beliefs is still supported as a fallback;
  - edges live in a unified relations list;
  - relation types are causal, depends_on, supplements, contradicts;
  - no forward/backward split is required.

Usage
-----
    python scripts/visualize_beliefs_graph.py path/to/result.json -o graph.html
    python scripts/visualize_beliefs_graph.py path/to/belief_graph.jsonl -o graph.html
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROLE_LABEL = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
}

SOURCE_LABEL = {
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
    "unknown": "unknown",
}

REL_TYPES = ("causal", "depends_on", "supplements", "contradicts")


def escape(s: Any) -> str:
    return html_lib.escape("" if s is None else str(s))


def safe_class(s: Any) -> str:
    return (
        re.sub(r"[^a-zA-Z0-9_-]+", "-", str(s or "unknown").lower()).strip("-")
        or "unknown"
    )


def node_text(n: dict[str, Any]) -> str:
    return str(n.get("belief") or n.get("decision") or n.get("content") or "")


def node_role(n: dict[str, Any]) -> str:
    src = n.get("source") or {}
    return str(src.get("role") or src.get("type") or "unknown")


def normalize_input(
    data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    trajectory = data.get("trajectory") or []

    nodes = data.get("all_nodes")
    if not isinstance(nodes, list):
        nodes = data.get("nodes")
    if not isinstance(nodes, list):
        nodes = []
        seen = set()
        for key in ("all_beliefs", "beliefs", "all_decisions", "decisions"):
            vals = data.get(key) or []
            if isinstance(vals, list):
                for n in vals:
                    if not isinstance(n, dict):
                        continue
                    nid = n.get("id")
                    if nid in seen:
                        continue
                    seen.add(nid)
                    nodes.append(n)

    clean_nodes: list[dict[str, Any]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if not isinstance(n.get("id"), int):
            continue
        if not node_text(n):
            continue
        n = dict(n)
        n.setdefault(
            "node_type",
            "decision" if "decision" in n and "belief" not in n else "belief",
        )
        clean_nodes.append(n)
    clean_nodes.sort(key=lambda x: x.get("id", 0))

    rels = data.get("relations") or []
    # Legacy fallback: merge old split edge fields if present.
    if not rels:
        rels = []
        for key in ("forward_relations", "backward_relations"):
            vals = data.get(key) or []
            if isinstance(vals, list):
                rels.extend(vals)
    clean_rels: list[dict[str, Any]] = []
    node_ids = {n["id"] for n in clean_nodes}
    seen_rel = set()
    for r in rels:
        if not isinstance(r, dict):
            continue
        try:
            fid = int(r.get("from_id", r.get("from")))
            tid = int(r.get("to_id", r.get("to")))
        except (TypeError, ValueError):
            continue
        if fid == tid or fid not in node_ids or tid not in node_ids:
            continue
        rtype = str(r.get("type") or "depends_on")
        # Preserve old names visually if old output is opened, but map informs/extends.
        if rtype == "informs":
            rtype = "depends_on"
        elif rtype == "extends":
            rtype = "supplements"
        elif rtype == "confirms":
            rtype = "supports"
        key = (fid, tid, rtype)
        if key in seen_rel:
            continue
        seen_rel.add(key)
        clean_rels.append(
            {
                "from_id": fid,
                "to_id": tid,
                "type": rtype,
                "note": str(r.get("note") or ""),
            }
        )

    return trajectory, clean_nodes, clean_rels


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
    return json.loads(text)


def slice_with_marks(
    original: str, intervals: list[tuple[int, int, int]]
) -> list[dict[str, Any]]:
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


def compute_layout(nodes: list[dict[str, Any]], width: int = 980, height: int = 360):
    if not nodes:
        return {}, width, height, [], 150, 70
    by_traj: dict[int, list[dict[str, Any]]] = {}
    for n in nodes:
        ti = (n.get("source") or {}).get("trajectory_index", -1)
        if not isinstance(ti, int):
            ti = -1
        by_traj.setdefault(ti, []).append(n)
    trajs = sorted(by_traj)
    col_w, pad_x, pad_y = 150, 70, 38
    max_per_col = max(len(v) for v in by_traj.values())
    final_w = max(width, pad_x * 2 + len(trajs) * col_w)
    final_h = max(height, pad_y + max_per_col * 58 + 42)
    pos: dict[int, tuple[float, float]] = {}
    for ci, ti in enumerate(trajs):
        x = pad_x + col_w / 2 + ci * col_w
        for j, n in enumerate(sorted(by_traj[ti], key=lambda x: x["id"])):
            pos[n["id"]] = (x, pad_y + 18 + j * 58)
    return pos, final_w, final_h, trajs, col_w, pad_x


CSS = r"""
:root {
  --bg:#faf7f2; --panel:#fff; --ink:#1a1a1a; --soft:#555; --faint:#8a8276; --rule:#e8e2d8;
  --accent:#b8442f; --evidence:#ffe9a3; --evidence-active:#ffd23f;
  --user-fg:#1e5a9c; --user-bg:#e0ecfa; --assistant-fg:#8b2c5b; --assistant-bg:#fbe3ee;
  --tool-fg:#2e6e3a; --tool-bg:#e1f0e4; --unknown-fg:#666; --unknown-bg:#eee;
  --decision-fg:#7c3aed; --decision-bg:#efe6fd;
  --causal:#c84a3e; --depends:#6b89b8; --supp:#b3500e; --contra:#1a1a1a; --supports:#1f6a35;
}
*{box-sizing:border-box} html,body{margin:0;height:100%;overflow:hidden;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
body{display:flex;flex-direction:column} header{padding:18px 30px 12px;border-bottom:1px solid var(--rule);background:var(--panel);display:flex;justify-content:space-between;gap:16px;align-items:baseline} h1{font-size:24px;margin:0;font-family:Georgia,serif}.meta{font-size:13px;color:var(--soft)}
.stats{padding:9px 30px;border-bottom:1px solid var(--rule);background:var(--panel);display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--soft)}.stats b{color:var(--ink)}
.layout{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,.92fr) minmax(0,1.08fr)}.left{overflow:auto;background:var(--panel);border-right:1px solid var(--rule);padding:18px 26px 70px}.right{display:flex;flex-direction:column;min-width:0;overflow:hidden}.graph-wrap{position:relative;flex:0 1 60%;min-height:230px;overflow:auto;border-bottom:1px solid var(--rule);padding:48px 24px 20px}.detail-wrap{flex:1;overflow:auto;padding:18px 26px 70px}.section-title{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--soft);margin:0 0 12px}
.msg{border:1px solid var(--rule);border-radius:7px;margin-bottom:14px;background:#fff}.msg-user{background:#fbf5ec}.msg-tool,.msg-function{background:#f1f6ec}.msg-system{background:#f5f1ea}.msg.flash{box-shadow:0 0 0 2px var(--accent)}.msg-head{display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--rule);padding:7px 12px;font-size:12px;color:var(--soft)}.msg-role{font-family:monospace;font-weight:700;text-transform:uppercase;color:white;border-radius:3px;padding:2px 7px}.role-user{background:var(--user-fg)}.role-assistant{background:#111}.role-tool,.role-function{background:var(--tool-fg)}.role-system{background:#999}.msg-count{margin-left:auto;font-family:monospace;font-size:11px}.msg-body{white-space:pre-wrap;word-break:break-word;margin:0;padding:12px;font:12px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:380px;overflow:auto}.ev{background:var(--evidence);border-bottom:1px solid #d4b75c;cursor:pointer}.ev.active{background:var(--evidence-active);outline:2px solid #c89800}
svg{display:block;min-width:100%}.col-label{font:10px monospace;fill:var(--faint);text-anchor:middle}.node{cursor:pointer}.node rect{rx:8;ry:8;stroke-width:1.6}.node:hover rect{stroke-width:2.6}.node.active rect{stroke-width:3;filter:drop-shadow(0 0 4px var(--accent))}.node.dimmed,.edge.dimmed{opacity:.22}.node .id{font:9px monospace;text-anchor:middle}.node .conf{font:bold 11px monospace;text-anchor:middle}.node.source-user rect{fill:var(--user-bg);stroke:var(--user-fg)}.node.source-user text{fill:var(--user-fg)}.node.source-assistant rect{fill:var(--assistant-bg);stroke:var(--assistant-fg)}.node.source-assistant text{fill:var(--assistant-fg)}.node.source-tool rect,.node.source-function rect{fill:var(--tool-bg);stroke:var(--tool-fg)}.node.source-tool text,.node.source-function text{fill:var(--tool-fg)}.node.source-unknown rect{fill:var(--unknown-bg);stroke:var(--unknown-fg)}.node.source-unknown text{fill:var(--unknown-fg)}.node.node-decision rect{stroke-dasharray:5 3}.node-type-label{font:8px monospace;text-anchor:middle;opacity:.75}
.edge{fill:none;stroke-width:1.8;cursor:pointer}.edge:hover,.edge.active{stroke-width:3}.edge.type-causal{stroke:var(--causal)}.edge.type-depends_on{stroke:var(--depends)}.edge.type-supplements{stroke:var(--supp);stroke-dasharray:5 3}.edge.type-contradicts{stroke:var(--contra);stroke-dasharray:2 3}.edge.type-supports{stroke:var(--supports)}
.legend{position:absolute;right:24px;top:12px;display:flex;gap:12px;background:rgba(255,255,255,.94);border:1px solid var(--rule);border-radius:5px;padding:5px 9px;font:10px monospace;color:var(--soft)}.sw{display:inline-block;width:18px;border-top:2px solid;vertical-align:middle;margin-right:4px}.sw.causal{border-color:var(--causal)}.sw.depends_on{border-color:var(--depends)}.sw.supplements{border-color:var(--supp);border-top-style:dashed}.sw.contradicts{border-color:var(--contra);border-top-style:dotted}
.detail-empty{font-size:13px;color:var(--faint);font-style:italic}.card{background:#fff;border:1px solid var(--rule);border-radius:7px;padding:15px 17px}.card h3{margin:0 0 9px;font-family:Georgia,serif}.node-text{font-family:Georgia,serif;font-size:15px;line-height:1.5;margin:10px 0}.badges{display:flex;gap:6px;flex-wrap:wrap}.badge{font-size:10px;text-transform:uppercase;letter-spacing:.08em;font-weight:700;border-radius:3px;padding:2px 7px;background:#eee}.badge.role-user{color:var(--user-fg);background:var(--user-bg)}.badge.role-assistant{color:var(--assistant-fg);background:var(--assistant-bg)}.badge.role-tool,.badge.role-function{color:var(--tool-fg);background:var(--tool-bg)}.badge.type-decision{color:var(--decision-fg);background:var(--decision-bg)}.badge.rel-causal{color:var(--causal);background:#fbe9e6}.badge.rel-depends_on{color:var(--depends);background:#e7eef7}.badge.rel-supplements{color:var(--supp);background:#fde6cf}.badge.rel-contradicts{color:#111;background:#e9e9e9}.kv{font:12px monospace;color:var(--soft);margin-top:8px}.entities{font:12px monospace;color:var(--soft)}.excerpts{margin:5px 0 0;padding:0;list-style:none}.excerpts li{font:11px/1.55 monospace;color:var(--soft);margin:4px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}.side{background:var(--bg);border:1px solid var(--rule);border-radius:5px;padding:10px}.side-label{font:10px monospace;color:var(--faint);text-transform:uppercase}.side-text{font-family:Georgia,serif;font-size:13.5px;line-height:1.45}.hint{padding:9px 30px;border-top:1px solid var(--rule);background:var(--panel);font-size:12px;color:var(--faint)}
"""

JS = r"""
let current = null; const D = window.GRAPH_DATA;
function esc(s){return (s==null?'':String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;')}
function textOf(n){return n.belief || n.decision || n.content || ''}
function clearSelection(){current=null;document.querySelectorAll('.active,.dimmed,.flash').forEach(e=>e.classList.remove('active','dimmed','flash'));detailEmpty()}
function selectNode(id){if(current&&current.kind==='node'&&current.id===id){clearSelection();return}clearSelection();current={kind:'node',id};let connected=new Set([id]);D.relations.forEach(r=>{if(r.from_id===id)connected.add(r.to_id);if(r.to_id===id)connected.add(r.from_id)});document.querySelectorAll('.node').forEach(el=>{let nid=+el.dataset.id;if(nid===id)el.classList.add('active');else if(!connected.has(nid))el.classList.add('dimmed')});document.querySelectorAll('.edge').forEach(el=>{let f=+el.dataset.fromId,t=+el.dataset.toId;if(f===id||t===id)el.classList.add('active');else el.classList.add('dimmed')});let first=null;document.querySelectorAll('.ev').forEach(s=>{let ids=(s.dataset.nodes||'').split(' ').filter(Boolean);if(ids.includes(String(id))){s.classList.add('active');if(!first)first=s}});let n=D.nodes.find(x=>x.id===id);if(n&&n.source&&Number.isInteger(n.source.trajectory_index)){let p=document.getElementById('msg-'+n.source.trajectory_index);if(p)p.classList.add('flash');if(first)first.scrollIntoView({behavior:'smooth',block:'center'});else if(p)p.scrollIntoView({behavior:'smooth',block:'start'})}nodeDetail(id)}
function selectEdge(f,t,type){if(current&&current.kind==='edge'&&current.from===f&&current.to===t&&current.type===type){clearSelection();return}clearSelection();current={kind:'edge',from:f,to:t,type};let key=f+'->'+t+':'+type;document.querySelectorAll('.edge').forEach(el=>{if(el.dataset.key===key)el.classList.add('active');else el.classList.add('dimmed')});document.querySelectorAll('.node').forEach(el=>{let id=+el.dataset.id;if(id===f||id===t)el.classList.add('active');else el.classList.add('dimmed')});edgeDetail(f,t,type)}
function detailEmpty(){document.getElementById('detail').innerHTML='<div class="detail-empty">Click a node to inspect a belief/decision, or an edge to inspect a relation.</div>'}
function nodeDetail(id){let n=D.nodes.find(x=>x.id===id);if(!n){detailEmpty();return}let src=n.source||{}, role=src.role||src.type||'unknown', nt=n.node_type||'belief';let conf=(typeof n.confidence==='number')?n.confidence.toFixed(2):'—';let entities=Array.isArray(n.entities)&&n.entities.length?`<h4>Entities</h4><div class="entities">${n.entities.map(esc).join(', ')}</div>`:'';let exc='';let evs=Array.isArray(n.supporting_excerpts)?n.supporting_excerpts:[];if(evs.length){exc='<h4>Supporting excerpts</h4><ul class="excerpts">'+evs.map(e=>`<li>✓ ${esc(e)}</li>`).join('')+'</ul>'}let temporal=(n.time_text||n.event_time)?`<h4>Time</h4><div class="kv">time_text=${esc(n.time_text||'null')} · event_time=${esc(n.event_time||'null')}</div>`:'';document.getElementById('detail').innerHTML=`<div class="card"><h3>${nt==='decision'?'Decision':'Belief'} #${String(n.id+1).padStart(2,'0')} · ${conf}</h3><div class="badges"><span class="badge role-${esc(role)}">${esc(role)}</span><span class="badge type-${esc(nt)}">${esc(nt)}</span><span class="badge">${esc(n.stance||'')}</span></div><div class="node-text">${esc(textOf(n))}</div><h4>Source</h4><div class="kv">traj[${esc(src.trajectory_index)}] · turn=${esc(src.turn_index)}${src.date?' · '+esc(src.date):''}</div>${entities}${temporal}${exc}</div>`}
function edgeDetail(f,t,type){let a=D.nodes.find(x=>x.id===f), b=D.nodes.find(x=>x.id===t), r=D.relations.find(x=>x.from_id===f&&x.to_id===t&&x.type===type);if(!a||!b){detailEmpty();return}document.getElementById('detail').innerHTML=`<div class="card"><h3>Relation</h3><div class="badges"><span class="badge rel-${esc(type)}">${esc(type)}</span><span class="badge">#${String(f+1).padStart(2,'0')} → #${String(t+1).padStart(2,'0')}</span></div>${r&&r.note?`<h4>Note</h4><div class="kv">${esc(r.note)}</div>`:''}<div class="pair"><div class="side"><div class="side-label">from · #${String(f+1).padStart(2,'0')}</div><div class="side-text">${esc(textOf(a))}</div></div><div class="side"><div class="side-label">to · #${String(t+1).padStart(2,'0')}</div><div class="side-text">${esc(textOf(b))}</div></div></div></div>`}
document.addEventListener('click',e=>{let ev=e.target.closest('.ev');if(ev){let ids=(ev.dataset.nodes||'').split(' ').filter(Boolean);if(ids.length)selectNode(+ids[0])}});document.addEventListener('keydown',e=>{if(e.key==='Escape')clearSelection()});document.addEventListener('DOMContentLoaded',detailEmpty);
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
        f'<span>trajectory_index <b>{i}</b></span><span class="msg-count">{count} node{"s" if count != 1 else ""}</span></header>'
        f'<pre class="msg-body">{"".join(body)}</pre></article>'
    )


def render_graph_svg(
    nodes: list[dict[str, Any]],
    rels: list[dict[str, Any]],
    pos: dict[int, tuple[float, float]],
    width: int,
    height: int,
    trajs: list[int],
    col_w: int,
    pad_x: int,
) -> str:
    parts: list[str] = []
    for ci, ti in enumerate(trajs):
        cx = pad_x + col_w / 2 + ci * col_w
        parts.append(
            f'<text class="col-label" x="{cx}" y="{height - 12}">traj[{ti}]</text>'
        )

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
            f'<marker id="arr-{t}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        )
    parts.append("<defs>" + "".join(marker_defs) + "</defs>")

    def path(fid: int, tid: int) -> str | None:
        if fid not in pos or tid not in pos:
            return None
        x1, y1 = pos[fid]
        x2, y2 = pos[tid]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        px, py = -dy / length, dx / length
        offset = max(24, min(74, abs(dx) * 0.22 + 18))
        return f"M {x1:.1f},{y1:.1f} Q {mx + px * offset:.1f},{my + py * offset:.1f} {x2:.1f},{y2:.1f}"

    for r in rels:
        fid, tid, typ = r["from_id"], r["to_id"], safe_class(r.get("type"))
        p = path(fid, tid)
        if not p:
            continue
        marker = typ if typ in marker_colors else "depends_on"
        key = f"{fid}->{tid}:{typ}"
        parts.append(
            f'<path class="edge type-{typ}" d="{p}" marker-end="url(#arr-{marker})" '
            f'data-from-id="{fid}" data-to-id="{tid}" data-type="{typ}" data-key="{key}" '
            f"onclick=\"selectEdge({fid},{tid},'{typ}')\"/>"
        )

    for n in nodes:
        nid = n["id"]
        if nid not in pos:
            continue
        x, y = pos[nid]
        role = safe_class(node_role(n))
        nt = safe_class(n.get("node_type", "belief"))
        conf = n.get("confidence")
        conf_text = f"{float(conf):.2f}" if isinstance(conf, int | float) else "—"
        label = "D" if nt == "decision" else "B"
        parts.append(
            f'<g class="node source-{role} node-{nt}" data-id="{nid}" transform="translate({x:.1f},{y:.1f})" onclick="selectNode({nid})">'
            f'<rect x="-40" y="-18" width="80" height="36"/>'
            f'<text class="id" x="0" y="-6">{label}#{nid + 1:02d}</text>'
            f'<text class="conf" x="0" y="8">{escape(conf_text)}</text>'
            f"</g>"
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        + "".join(parts)
        + "</svg>"
    )


def render_html(data: dict[str, Any], src_path: str) -> str:
    trajectory, nodes, rels = normalize_input(data)

    by_msg: dict[int, list[tuple[int, dict[str, Any], list[tuple[int, int]]]]] = {}
    for n in nodes:
        src = n.get("source") or {}
        ti = src.get("trajectory_index")
        if not isinstance(ti, int) or ti < 0 or ti >= len(trajectory):
            continue
        host = trajectory[ti].get("content", "") or ""
        ranges: list[tuple[int, int]] = []
        for ev in n.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            s, e = ev.get("start"), ev.get("end")
            if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(host):
                ranges.append((s, e))
        by_msg.setdefault(ti, []).append((n["id"], n, ranges))

    msg_html = (
        "".join(
            render_message_panel(i, m, by_msg.get(i, []))
            for i, m in enumerate(trajectory)
        )
        or "<p>No trajectory.</p>"
    )
    pos, gw, gh, trajs, col_w, pad_x = compute_layout(nodes)
    graph_svg = render_graph_svg(nodes, rels, pos, gw, gh, trajs, col_w, pad_x)

    role_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    rel_counts: dict[str, int] = {}
    for n in nodes:
        role_counts[node_role(n)] = role_counts.get(node_role(n), 0) + 1
        nt = n.get("node_type", "belief")
        type_counts[nt] = type_counts.get(nt, 0) + 1
    for r in rels:
        rel_counts[r.get("type", "?")] = rel_counts.get(r.get("type", "?"), 0) + 1
    role_stats = "".join(
        f"<span><b>{v}</b> {escape(k)}</span>" for k, v in sorted(role_counts.items())
    )
    type_stats = "".join(
        f"<span><b>{v}</b> {escape(k)}</span>" for k, v in sorted(type_counts.items())
    )
    rel_stats = ", ".join(f"{v} {escape(k)}" for k, v in sorted(rel_counts.items()))

    meta_model = data.get("model", "") or ""
    prompt_name = data.get("prompt_name", "construct_beliefs") or "construct_beliefs"
    graph_data = {"nodes": nodes, "relations": rels}

    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>Belief Graph · {escape(prompt_name)}</title><style>{CSS}</style></head><body>"
        f'<header><h1>Belief Graph · {escape(prompt_name)}</h1><div class="meta">source · <b>{escape(Path(src_path).name)}</b> &nbsp; model · <b>{escape(meta_model)}</b></div></header>'
        f'<div class="stats"><span><b>{len(nodes)}</b> nodes</span>{type_stats}{role_stats}<span><b>{len(rels)}</b> relations</span><span>{rel_stats}</span></div>'
        '<div class="layout"><div class="left"><h2 class="section-title">Conversation trajectory</h2>'
        f'{msg_html}</div><div class="right"><div class="graph-wrap"><div class="legend">'
        '<span><span class="sw causal"></span>causal</span><span><span class="sw depends_on"></span>depends_on</span><span><span class="sw supplements"></span>supplements</span><span><span class="sw contradicts"></span>contradicts</span>'
        f'</div>{graph_svg}</div><div class="detail-wrap"><h2 class="section-title">Inspector</h2><div id="detail"></div></div></div></div>'
        '<div class="hint">Click a node to inspect a belief/decision and highlight evidence. Click an edge to inspect a typed relation. Esc clears.</div>'
        f"<script>window.GRAPH_DATA = {json.dumps(graph_data, ensure_ascii=False)};{JS}</script>"
        "</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graph visualizer for construct_beliefs result.json / final_graph.json / belief_graph.jsonl"
    )
    parser.add_argument(
        "input", help="Path to result.json, final_graph.json, or belief_graph.jsonl"
    )
    parser.add_argument("--output", "-o", default=None, help="Output HTML path")
    args = parser.parse_args()

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
    out_path = (
        Path(args.output)
        if args.output
        else in_path.parent / f"graph_{in_path.stem}.html"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
