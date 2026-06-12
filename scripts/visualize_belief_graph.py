#!/usr/bin/env python3
"""Temporary HTML visualizer for BCG belief graph run outputs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

CSS = """
:root { --bg:#faf7f2; --panel:#fff; --ink:#1a1a1a; --muted:#746d64;
  --rule:#e8e2d8; --accent:#b8442f; --inform:#6b89b8; --confirm:#1f6a35;
  --contradict:#c84a3e; --extend:#b3500e; --ev:#ffe9a3; }
* { box-sizing: border-box; }
body { margin:0; height:100vh; overflow:hidden; background:var(--bg);
  color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
header { height:58px; display:flex; align-items:center; justify-content:space-between;
  padding:0 28px; border-bottom:1px solid var(--rule); background:var(--panel); }
h1 { margin:0; font-size:20px; font-weight:650; }
.meta { color:var(--muted); font-size:12px; }
.layout { height:calc(100vh - 58px); display:grid; grid-template-columns: minmax(320px,.95fr) minmax(420px,1.05fr); }
.left { overflow:auto; border-right:1px solid var(--rule); background:var(--panel); padding:18px 22px 72px; }
.right { min-width:0; display:grid; grid-template-rows:minmax(220px,54%) minmax(180px,46%); }
.graph { position:relative; overflow:auto; border-bottom:1px solid var(--rule); padding:46px 20px 20px; }
.detail { overflow:auto; padding:18px 22px 72px; }
.msg { border:1px solid var(--rule); border-radius:6px; margin-bottom:14px; background:#fff; }
.msg header { height:auto; padding:8px 12px; border-bottom:1px solid var(--rule); display:flex; gap:10px; justify-content:flex-start; }
.role { font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:white; background:#222; border-radius:3px; padding:3px 7px; }
.idx { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; color:var(--muted); }
pre { margin:0; padding:12px; white-space:pre-wrap; word-break:break-word; font:12px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace; max-height:360px; overflow:auto; }
.ev { background:var(--ev); border-bottom:1px solid #caa642; cursor:pointer; }
.ev.active { outline:2px solid #d6aa00; }
.toolbar { position:absolute; top:12px; left:20px; display:flex; gap:8px; }
button { border:1px solid var(--rule); background:white; border-radius:4px; padding:5px 9px; cursor:pointer; color:var(--muted); }
button.active { background:#222; color:white; border-color:#222; }
svg { display:block; min-width:100%; }
.edge { fill:none; stroke-width:1.7; cursor:pointer; }
.edge.forward { stroke:var(--inform); }
.edge.backward { display:none; }
.show-backward .edge.backward { display:inline; }
.edge.confirms { stroke:var(--confirm); }
.edge.contradicts { stroke:var(--contradict); }
.edge.extends { stroke:var(--extend); stroke-dasharray:4 3; }
.node rect { fill:white; stroke:#333; stroke-width:1.3; cursor:pointer; }
.node text { pointer-events:none; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; text-anchor:middle; }
.node.active rect { stroke:var(--accent); stroke-width:3; }
.dim { opacity:.18; }
.card { border:1px solid var(--rule); border-radius:6px; background:var(--panel); padding:16px; }
.badges { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 12px; }
.badge { font-size:10px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); background:var(--bg); border:1px solid var(--rule); padding:3px 7px; border-radius:3px; }
.dims { display:grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap:4px 10px; margin:6px 0 12px; }
.dimitem { font-size:11px; color:var(--muted); font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.belief { font-size:15px; line-height:1.45; }
.muted { color:var(--muted); font-size:12px; }
"""

JS = """
const D = window.GRAPH_DATA;
const DIM_KEYS = ['source_reliability','evidence_directness','claim_specificity','linguistic_certainty'];
function esc(s){return String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function dimsHtml(h){
  const dims = h.dimensions || {};
  if(!DIM_KEYS.some(k => dims[k] != null)) return '';
  return `<div class="dims">${DIM_KEYS.map(k=>`<div class="dimitem">${esc(k)} · ${Number(dims[k] ?? 0).toFixed(3)}</div>`).join('')}</div>`;
}
function clearAll(){
  document.querySelectorAll('.active').forEach(e=>e.classList.remove('active'));
  document.querySelectorAll('.dim').forEach(e=>e.classList.remove('dim'));
  document.getElementById('detail').innerHTML='<p class="muted">Click a belief node or relation edge.</p>';
}
function selectNode(id){
  clearAll();
  const connected = new Set([id]);
  D.edges.forEach(e=>{ if(e.from_id===id) connected.add(e.to_id); if(e.to_id===id) connected.add(e.from_id); });
  document.querySelectorAll('.node').forEach(n=>{ const nid=+n.dataset.id; if(nid===id)n.classList.add('active'); else if(!connected.has(nid))n.classList.add('dim'); });
  document.querySelectorAll('.edge').forEach(e=>{ const f=+e.dataset.from, t=+e.dataset.to; if(f!==id && t!==id)e.classList.add('dim'); });
  document.querySelectorAll('.ev').forEach(e=>{ if((e.dataset.beliefs||'').split(' ').includes(String(id))) e.classList.add('active'); });
  const b = D.beliefs.find(x=>x.id===id);
  if(!b) return;
  const src = b.source || {};
  const evidence = (b.evidence || []).length ? b.evidence : (b.supporting_excerpts||[]).map(x=>({text:x}));
  const merged = (b.merged_from || []).length ? `<h3>Merged From</h3><p class="muted">${esc(b.merged_from.join(', '))}</p>` : '';
  const time = b.event_time || b.time_text ? `<h3>Time</h3><p class="muted">${esc(b.event_time || '')} ${esc(b.time_text || '')}</p>` : '';
  document.getElementById('detail').innerHTML = `<div class="card">
    <h2>Belief #${String(id+1).padStart(2,'0')} · ${(b.confidence ?? 0).toFixed(2)}</h2>
    <div class="badges"><span class="badge">${esc(src.type)}</span><span class="badge">${esc(b.stance)}</span><span class="badge">${esc(b.layer)}</span></div>
    <div class="belief">${esc(b.belief)}</div>
    ${time}
    <h3>Source</h3><p class="muted">traj[${esc(src.trajectory_index)}] · session[${esc(src.session_index)}] · turn[${esc(src.turn_index)}] · ${esc(src.role)} · ${esc(src.segment_type)}[${esc(src.segment_index)}]</p>
    <h3>Confidence</h3>${(b.confidence_history||[]).map(h=>`<p class="muted">${esc(h.step)} · ${(h.value??0).toFixed(3)} ${h.delta ? '('+h.delta.toFixed(3)+')' : ''} · ${esc(h.method||'')} ${esc(h.reason||'')}</p>${dimsHtml(h)}`).join('')}
    ${merged}
    <h3>Evidence</h3>${evidence.map(x=>`<p class="muted">${esc(x.text)} ${x.match ? '· '+esc(x.match) : ''} ${x.start != null ? '· '+esc(x.start)+'-'+esc(x.end) : ''}</p>`).join('')}
  </div>`;
}
function selectEdge(f,t,type,dir){
  clearAll();
  document.querySelectorAll('.node').forEach(n=>{ const id=+n.dataset.id; if(id===f||id===t)n.classList.add('active'); else n.classList.add('dim'); });
  document.querySelectorAll('.edge').forEach(e=>{ if(+e.dataset.from===f && +e.dataset.to===t && e.dataset.type===type && e.dataset.dir===dir)e.classList.add('active'); else e.classList.add('dim'); });
  const rel = D.edges.find(e=>e.from_id===f && e.to_id===t && e.type===type && e._dir===dir) || {};
  const from = D.beliefs.find(b=>b.id===f) || {};
  const to = D.beliefs.find(b=>b.id===t) || {};
  let impact = '';
  if(dir === 'backward' && Array.isArray(to.confidence_history)){
    const hit = to.confidence_history.find(h => h.step === type && h.from_belief_id === f);
    if(hit) impact = `<h3>Confidence Impact</h3><p class="muted">${hit.delta >= 0 ? '+' : ''}${(hit.delta ?? 0).toFixed(2)} -> ${(hit.value ?? 0).toFixed(2)}</p>`;
  } else if(dir === 'forward') {
    impact = `<h3>Confidence Impact</h3><p class="muted">Forward derivation relations do not modify confidence.</p>`;
  }
  document.getElementById('detail').innerHTML = `<div class="card">
    <h2>${esc(dir)} relation · ${esc(type)}</h2>
    <p class="muted">#${f+1} -> #${t+1}</p>
    <p>${esc(rel.note||'')}</p>
    ${impact}
    <h3>From</h3><p>${esc(from.belief||'')}</p>
    <h3>To</h3><p>${esc(to.belief||'')}</p>
  </div>`;
}
function toggleBackward(){
  const svg=document.querySelector('svg');
  const btn=document.querySelector('button');
  const active=svg.classList.toggle('show-backward');
  btn.classList.toggle('active', active);
  btn.textContent = active ? 'hide backward' : 'show backward';
}
document.addEventListener('click', e=>{ const ev=e.target.closest('.ev'); if(ev){ const id=(ev.dataset.beliefs||'').split(' ')[0]; if(id) selectNode(+id); }});
document.addEventListener('keydown', e=>{ if(e.key==='Escape') clearAll(); });
document.addEventListener('DOMContentLoaded', clearAll);
"""


def load_run(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") == "bcg.memory.v1":
        return data

    sibling_memory = path.parent / "memory.json"
    memory = {}
    if sibling_memory.exists():
        memory = json.loads(sibling_memory.read_text(encoding="utf-8"))
    if "nodes" in data:
        beliefs = [
            node["belief"]
            for node in data.get("nodes", [])
            if isinstance(node, dict) and isinstance(node.get("belief"), dict)
        ]
        relations = [
            edge["relation"]
            for edge in data.get("edges", [])
            if isinstance(edge, dict) and isinstance(edge.get("relation"), dict)
        ]
        return {
            "schema": "bcg.memory.v1",
            "run_id": data.get("metadata", {}).get("run_id", path.parent.name),
            "trajectory": memory.get("trajectory", []),
            "beliefs": beliefs,
            "forward_relations": [r for r in relations if r.get("type") == "informs"],
            "backward_relations": [
                r
                for r in relations
                if r.get("type") in {"confirms", "contradicts", "extends"}
            ],
            "sessions": data.get("sessions", memory.get("sessions", [])),
            "merges": data.get("merges", memory.get("merges", [])),
            "counts": memory.get("counts", {}),
        }
    return data


def render_html(data: dict[str, Any], src_path: Path) -> str:
    beliefs = sorted(data.get("beliefs") or [], key=lambda b: b.get("id", -1))
    forward = data.get("forward_relations") or []
    backward = data.get("backward_relations") or []
    edges = [{**r, "_dir": "forward"} for r in forward] + [
        {**r, "_dir": "backward"} for r in backward
    ]
    trajectory = data.get("trajectory") or []
    positions, width, height = _layout(beliefs)
    messages_html = _render_messages(trajectory, beliefs)
    graph_svg = _render_graph(beliefs, forward, backward, positions, width, height)
    graph_data = json.dumps(
        {
            "beliefs": beliefs,
            "edges": edges,
            "sessions": data.get("sessions") or [],
            "merges": data.get("merges") or [],
        },
        ensure_ascii=False,
    )
    stats = _render_stats(data, beliefs, forward, backward)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>BCG Belief Graph · {html.escape(data.get('run_id', ''))}</title>"
        f"<style>{CSS}</style></head><body>"
        f"<header><h1>BCG Belief Graph</h1><div class='meta'>{html.escape(src_path.name)} · "
        f"{len(beliefs)} beliefs · {len(forward)} forward · {len(backward)} backward</div></header>"
        "<main class='layout'>"
        f"<section class='left'>{stats}{messages_html}</section>"
        "<section class='right'>"
        f"<div class='graph'><div class='toolbar'><button onclick='toggleBackward()'>show backward</button></div>{graph_svg}</div>"
        "<div class='detail'><div id='detail'></div></div>"
        "</section></main>"
        f"<script>window.GRAPH_DATA={graph_data};{JS}</script>"
        "</body></html>"
    )


def _render_messages(
    trajectory: list[dict[str, Any]], beliefs: list[dict[str, Any]]
) -> str:
    if not trajectory:
        return "<p class='muted'>No trajectory found. Open memory.json for full evidence context.</p>"
    by_message: dict[int, list[dict[str, Any]]] = {}
    for belief in beliefs:
        source = belief.get("source") or {}
        index = source.get("trajectory_index")
        if isinstance(index, int) and 0 <= index < len(trajectory):
            by_message.setdefault(index, []).append(belief)

    panels: list[str] = []
    for index, message in enumerate(trajectory):
        content = str(message.get("content") or "")
        marked = _mark_excerpts(content, by_message.get(index, []))
        panels.append(
            "<article class='msg'>"
            f"<header><span class='role'>{html.escape(str(message.get('role') or '?'))}</span>"
            f"<span class='idx'>trajectory_index <b>{index}</b></span></header>"
            f"<pre>{marked}</pre></article>"
        )
    return "\n".join(panels)


def _render_stats(
    data: dict[str, Any],
    beliefs: list[dict[str, Any]],
    forward: list[dict[str, Any]],
    backward: list[dict[str, Any]],
) -> str:
    source_counts: dict[str, int] = {}
    for belief in beliefs:
        source_type = str((belief.get("source") or {}).get("type") or "unknown")
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
    source_bits = " · ".join(
        f"{html.escape(source)} {count}"
        for source, count in sorted(source_counts.items())
    )
    merge_count = len(data.get("merges") or [])
    session_count = len(data.get("sessions") or [])
    return (
        "<div class='card' style='margin-bottom:14px'>"
        f"<div class='muted'>{html.escape(str(data.get('scenario') or ''))} · "
        f"{session_count} sessions · {merge_count} merges</div>"
        f"<div class='muted'>{source_bits}</div>"
        f"<div class='muted'>{len(forward)} informs · {len(backward)} evaluation relations</div>"
        "</div>"
    )


def _mark_excerpts(content: str, beliefs: list[dict[str, Any]]) -> str:
    ranges: list[tuple[int, int, int]] = []
    for belief in beliefs:
        evidence_items = belief.get("evidence") or []
        for evidence in evidence_items:
            if not isinstance(evidence, dict):
                continue
            start = evidence.get("start")
            end = evidence.get("end")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end:
                ranges.append((start, min(end, len(content)), int(belief["id"])))
        if evidence_items:
            continue
        source = belief.get("source") or {}
        start_bound = source.get("segment_start")
        end_bound = source.get("segment_end")
        if not isinstance(start_bound, int) or not isinstance(end_bound, int):
            start_bound, end_bound = 0, len(content)
        for excerpt in belief.get("supporting_excerpts") or []:
            if not isinstance(excerpt, str) or not excerpt:
                continue
            start = content.find(excerpt, start_bound, end_bound)
            if start >= 0:
                ranges.append((start, start + len(excerpt), int(belief["id"])))
    if not ranges:
        return html.escape(content)
    points = sorted(
        {0, len(content), *[p for start, end, _ in ranges for p in (start, end)]}
    )
    parts: list[str] = []
    for start, end in zip(points, points[1:], strict=False):
        active = [
            bid for r_start, r_end, bid in ranges if r_start <= start and end <= r_end
        ]
        text = html.escape(content[start:end])
        if active:
            ids = " ".join(str(bid) for bid in active)
            parts.append(f"<span class='ev' data-beliefs='{ids}'>{text}</span>")
        else:
            parts.append(text)
    return "".join(parts)


def _layout(
    beliefs: list[dict[str, Any]],
) -> tuple[dict[int, tuple[float, float]], int, int]:
    by_traj: dict[int, list[dict[str, Any]]] = {}
    for belief in beliefs:
        source = belief.get("source") or {}
        by_traj.setdefault(int(source.get("trajectory_index", -1)), []).append(belief)
    columns = sorted(by_traj)
    width = max(720, 120 + len(columns) * 130)
    height = max(260, 70 + max((len(v) for v in by_traj.values()), default=1) * 54)
    positions: dict[int, tuple[float, float]] = {}
    for col, traj_index in enumerate(columns):
        x = 70 + col * 130
        for row, belief in enumerate(
            sorted(by_traj[traj_index], key=lambda b: b["id"])
        ):
            positions[int(belief["id"])] = (x, 50 + row * 54)
    return positions, width, height


def _render_graph(
    beliefs: list[dict[str, Any]],
    forward: list[dict[str, Any]],
    backward: list[dict[str, Any]],
    positions: dict[int, tuple[float, float]],
    width: int,
    height: int,
) -> str:
    parts = [
        "<defs><marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#555'/></marker></defs>"
    ]
    for direction, relations in (("forward", forward), ("backward", backward)):
        for relation in relations:
            from_id = int(relation.get("from_id", -1))
            to_id = int(relation.get("to_id", -1))
            if from_id not in positions or to_id not in positions:
                continue
            x1, y1 = positions[from_id]
            x2, y2 = positions[to_id]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2 - 35
            relation_type = relation.get("type", "informs")
            parts.append(
                f"<path class='edge {direction} {html.escape(relation_type)}' "
                f"d='M{x1},{y1} Q{cx},{cy} {x2},{y2}' marker-end='url(#arrow)' "
                f"data-from='{from_id}' data-to='{to_id}' data-type='{html.escape(relation_type)}' data-dir='{direction}' "
                f"onclick=\"selectEdge({from_id},{to_id},'{html.escape(relation_type)}','{direction}')\"/>"
            )
    for belief in beliefs:
        belief_id = int(belief["id"])
        if belief_id not in positions:
            continue
        x, y = positions[belief_id]
        confidence = float(belief.get("confidence") or 0)
        parts.append(
            f"<g class='node' data-id='{belief_id}' transform='translate({x},{y})' onclick='selectNode({belief_id})'>"
            "<rect x='-36' y='-15' width='72' height='32' rx='6'/>"
            f"<text x='0' y='-3' font-size='10'>#{belief_id + 1:02d}</text>"
            f"<text x='0' y='10' font-size='12' font-weight='700'>{confidence:.2f}</text></g>"
        )
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}'>"
        + "".join(parts)
        + "</svg>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize BCG graph.json or memory.json"
    )
    parser.add_argument("input", help="Path to graph.json or memory.json")
    parser.add_argument("--output", "-o", default=None, help="Output HTML path")
    args = parser.parse_args()

    input_path = Path(args.input)
    data = load_run(input_path)
    output_path = (
        Path(args.output) if args.output else input_path.parent / "belief_graph.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(data, input_path), encoding="utf-8")
    print(f"[ok] wrote {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
