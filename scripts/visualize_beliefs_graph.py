#!/usr/bin/env python3
"""
visualize_beliefs_graph.py
==========================
Graph-style visualizer for the result.json produced by `construct_beliefs`.

Layout
------
+-----------------------+-----------------------------------+
|                       |  Belief graph (SVG)               |
|                       |    nodes  = beliefs (colored by   |
|  Conversation         |             role: user/assistant/ |
|  trajectory           |             tool)                 |
|  (with evidence       |    edges  = informs (forward) +   |
|   highlighting from    +    confirms/contradicts/extends   |
|   exact offsets)      |  Detail panel                     |
|                       |    (filled when a node or edge    |
|                       |     is clicked)                   |
+-----------------------+-----------------------------------+

v3 notes
--------
* Beliefs are colored by ROLE (user / assistant / tool); there is no `layer`.
* Evidence carries EXACT character offsets (start/end) into the original turn
  content, so highlighting is direct — no segment rederivation, no excerpt
  re-search. Nodes are positioned by (trajectory_index); within a column they
  stack in id (chronological) order.

Usage
-----
    python visualize_beliefs_graph.py path/to/result.json
    python visualize_beliefs_graph.py result.json -o graph.html
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
from pathlib import Path
from typing import Any

# =============================================================
# Segment recomputation (matches construct_beliefs/segment.py)
# =============================================================
# We rederive segment boundaries from the raw trajectory so that excerpt
# matching can be constrained to the originating segment's byte range.
# Keeping this here means the visualizer has no runtime dependency on the
# `construct_beliefs` package — `result.json` is enough.

# v3: evidence offsets are exact (start/end into the original turn content),
# so the visualizer highlights directly from them — no segment rederivation
# or excerpt re-search. slice_with_marks() turns the per-belief (start,end)
# ranges into rendered, click-linked spans.


def slice_with_marks(
    original: str, intervals: list[tuple[int, int, int]]
) -> list[dict[str, Any]]:
    if not intervals:
        return [{"text": original, "beliefs": []}]
    points = {0, len(original)}
    for s, e, _ in intervals:
        points.add(s)
        points.add(e)
    sorted_points = sorted(points)
    pieces: list[dict[str, Any]] = []
    for i in range(len(sorted_points) - 1):
        seg_start = sorted_points[i]
        seg_end = sorted_points[i + 1]
        if seg_start >= seg_end:
            continue
        text = original[seg_start:seg_end]
        active = set()
        for s, e, bid in intervals:
            if s <= seg_start and seg_end <= e:
                active.add(bid)
        pieces.append({"text": text, "beliefs": sorted(active)})
    return pieces


# =============================================================
# Static palette (kept in sync with visualize_beliefs_v3.py)
# =============================================================

ALLOWED_SOURCES = {"user", "assistant", "tool"}

SOURCE_LABEL = {
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}

ROLE_LABEL = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
}


def confidence_band(c: float) -> tuple[str, int]:
    if c >= 0.90:
        return ("certain", 5)
    if c >= 0.75:
        return ("high", 4)
    if c >= 0.50:
        return ("medium", 3)
    if c >= 0.25:
        return ("low", 2)
    return ("verylow", 1)


# =============================================================
# Layout: position each belief as a graph node
# =============================================================


def compute_layout(
    beliefs: list[dict[str, Any]],
    width: int = 900,
    height: int = 280,
):
    """
    Flat structured layout: one column per trajectory_index; within a column,
    nodes stack vertically in id order (which is chronological after the
    pipeline's chrono-renumber step). No layer separation.
    Returns (positions, final_width, final_height, sorted_traj, col_w, pad_x).
    """
    if not beliefs:
        return {}, width, height, [], 130, 60

    by_traj: dict[int, list[dict[str, Any]]] = {}
    for b in beliefs:
        ti = (b.get("source") or {}).get("trajectory_index", -1)
        by_traj.setdefault(ti, []).append(b)

    sorted_traj = sorted(by_traj.keys())
    n_cols = len(sorted_traj)
    col_w = 130
    pad_x = 60
    pad_y_top = 32
    pad_y_bot = 36  # extra room for column labels at bottom

    max_per_col = max((len(by_traj[ti]) for ti in sorted_traj), default=1)
    slot_h = 52
    final_height = max(height, pad_y_top + max_per_col * slot_h + pad_y_bot)
    final_width = max(width, pad_x * 2 + n_cols * col_w)

    positions: dict[int, tuple[float, float]] = {}
    for ci, ti in enumerate(sorted_traj):
        x_center = pad_x + col_w / 2 + ci * col_w
        beliefs_in_col = sorted(by_traj[ti], key=lambda b: b["id"])
        for i, b in enumerate(beliefs_in_col):
            positions[b["id"]] = (x_center, pad_y_top + 16 + i * slot_h)

    return positions, final_width, final_height, sorted_traj, col_w, pad_x


# =============================================================
# Embedded CSS
# =============================================================

CSS = r"""
:root {
  --bg: #faf7f2;
  --panel: #ffffff;
  --ink: #1a1a1a;
  --ink-soft: #4a4a4a;
  --ink-faint: #8a8276;
  --rule: #e8e2d8;
  --accent: #b8442f;
  --evidence: #ffe9a3;
  --evidence-active: #ffd23f;
  --evidence-active-edge: #c89800;

  --src-user-fg:   #1e5a9c; --src-user-bg:   #e0ecfa;
  --src-llm-fg:    #555;    --src-llm-bg:    #ececec;
  --src-tool-fg:   #2e6e3a; --src-tool-bg:   #e1f0e4;
  --src-hist-fg:   #6d28d9; --src-hist-bg:   #ede4fb;
  --src-call-fg:   #b3500e; --src-call-bg:   #fde6cf;
  --src-final-fg:  #8b2c5b; --src-final-bg:  #fbe3ee;

  --stance-asserted-fg: #c84a3e; --stance-asserted-bg: #fbe9e6;
  --stance-recalled-fg: #2563a8; --stance-recalled-bg: #e3edf7;
  --stance-speculated-fg: #6b6b6b; --stance-speculated-bg: #ececec;
  --stance-judged-fg:   #7c3aed; --stance-judged-bg:   #efe6fd;

  --rel-confirms-fg:    #1f6a35;
  --rel-contradicts-fg: #c84a3e;
  --rel-extends-fg:     #b3500e;
  --rel-informs-fg:     #6b89b8;

  --role-user-bg:      #fbf5ec;
  --role-assistant-bg: #ffffff;
  --role-system-bg:    #f5f1ea;
  --role-tool-bg:      #f1f6ec;
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; overflow: hidden;
             background: var(--bg); color: var(--ink);
             font-family: 'Inter Tight', -apple-system, sans-serif;
             font-feature-settings: 'ss01','cv11'; -webkit-font-smoothing: antialiased; }
body { display: flex; flex-direction: column; }

header.app-header {
  padding: 22px 36px 14px; border-bottom: 1px solid var(--rule);
  display: flex; align-items: baseline; justify-content: space-between;
  flex-wrap: wrap; gap: 16px;
}
header.app-header h1 {
  font-family: 'Fraunces', Georgia, serif; font-weight: 600;
  font-size: 26px; letter-spacing: -0.02em; margin: 0;
  font-variation-settings: 'opsz' 48, 'SOFT' 50;
}
header.app-header h1 .sep { color: var(--accent); margin: 0 8px; }
header.app-header h1 .tag { font-weight: 400; font-style: italic; font-variation-settings: 'opsz' 36; }
.file-meta { font-size: 13px; color: var(--ink-soft); display: flex; gap: 18px; flex-wrap: wrap; }
.file-meta b { color: var(--ink); font-weight: 500; }

.stats {
  padding: 10px 36px; font-size: 12.5px; color: var(--ink-soft);
  display: flex; gap: 20px; flex-wrap: wrap; border-bottom: 1px solid var(--rule);
  background: var(--panel);
}
.stats b { color: var(--ink); font-weight: 600; }
.stats .stat-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.stat-dot.user      { background: var(--src-user-fg); }
.stat-dot.assistant { background: var(--src-final-fg); }
.stat-dot.tool      { background: var(--src-tool-fg); }

header.app-header, .stats, .hint { flex: 0 0 auto; }

.layout {
  flex: 1 1 auto;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 0.92fr) minmax(0, 1.08fr);
  overflow: hidden;
}
.left-panel {
  background: var(--panel);
  border-right: 1px solid var(--rule);
  overflow-y: auto;
  padding: 18px 28px 80px;
  min-width: 0;
}
.right-panel {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bg);
  min-width: 0;
}
.graph-wrap {
  flex: 0 1 auto;
  max-height: 60%;
  min-height: 200px;
  border-bottom: 1px solid var(--rule);
  overflow: auto;
  background: var(--bg);
  position: relative;
  padding: 50px 24px 22px;     /* extra top padding leaves space for the toolbar */
}
.detail-wrap {
  flex: 1 1 0;
  min-height: 0;
  overflow-y: auto;
  padding: 18px 28px 60px;
  background: var(--bg);
}

.section-title {
  font-family: 'Fraunces', Georgia, serif; font-weight: 600;
  font-size: 12.5px; text-transform: uppercase; letter-spacing: 0.12em;
  color: var(--ink-soft); margin: 0 0 12px;
  font-variation-settings: 'opsz' 12;
}

/* Left panel — message blocks */
.msg {
  border: 1px solid var(--rule); border-radius: 6px; margin-bottom: 16px;
  background: var(--panel); scroll-margin-top: 12px;
  transition: box-shadow 200ms ease, border-color 200ms ease;
}
.msg.flash { box-shadow: 0 0 0 2px var(--accent); border-color: var(--accent); }
.msg-user      { background: var(--role-user-bg); }
.msg-system    { background: var(--role-system-bg); }
.msg-tool      { background: var(--role-tool-bg); }
.msg-assistant { background: var(--role-assistant-bg); }
.msg-head {
  display:flex; align-items:center; gap:12px; padding:8px 14px;
  border-bottom: 1px solid var(--rule); font-size: 12px; color: var(--ink-soft);
}
.msg-role {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;
  padding: 3px 8px; border-radius: 3px; color: white;
}
.role-user { background: var(--src-user-fg); }
.role-assistant { background: #1a1a1a; }
.role-system { background: var(--ink-faint); }
.role-tool { background: var(--src-tool-fg); }
.msg-idx { font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.msg-idx b { color: var(--ink); font-weight: 600; }
.msg-belief-count {
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
  text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-soft);
}
.msg-belief-count.empty { color: var(--ink-faint); font-style: italic; }
.msg-body {
  font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace;
  font-size: 12px; line-height: 1.7; white-space: pre-wrap; word-wrap: break-word;
  margin: 0; padding: 13px; color: var(--ink); max-height: 380px; overflow-y: auto;
}

.ev {
  background: var(--evidence); border-bottom: 1px solid #d4b75c; border-radius: 2px;
  padding: 0 1px; cursor: pointer;
}
.ev:hover { background: #ffd87a; }
.ev.active { background: var(--evidence-active); outline: 2px solid var(--evidence-active-edge); outline-offset: 1px; }

/* Graph SVG */
svg.belief-graph { display: block; min-width: 100%; }
.col-tick { stroke: var(--rule); stroke-width: 1; }
.col-label {
  font-family: 'JetBrains Mono', monospace; font-size: 10px;
  fill: var(--ink-faint); text-anchor: middle;
}

.node-pill { rx: 7; ry: 7; stroke-width: 1.5; cursor: pointer; transition: stroke-width 120ms; }
.node:hover .node-pill { stroke-width: 2.5; }
.node.active .node-pill { stroke-width: 3; filter: drop-shadow(0 0 4px var(--accent)); }
.node.dimmed { opacity: 0.25; }

.node-id  { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
            text-anchor: middle; pointer-events: none; }
.node-conf { font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
             font-weight: 600; text-anchor: middle; pointer-events: none; }

.node.source-user        .node-pill { fill: var(--src-user-bg);   stroke: var(--src-user-fg); }
.node.source-user        .node-conf { fill: var(--src-user-fg); }
.node.source-user        .node-id   { fill: var(--src-user-fg); opacity: 0.8; }
.node.source-assistant   .node-pill { fill: var(--src-final-bg);  stroke: var(--src-final-fg); }
.node.source-assistant   .node-conf { fill: var(--src-final-fg); }
.node.source-assistant   .node-id   { fill: var(--src-final-fg); opacity: 0.8; }
.node.source-tool        .node-pill { fill: var(--src-tool-bg);   stroke: var(--src-tool-fg); }
.node.source-tool        .node-conf { fill: var(--src-tool-fg); }
.node.source-tool        .node-id   { fill: var(--src-tool-fg); opacity: 0.8; }

.edge { fill: none; stroke-width: 1.5; cursor: pointer; transition: stroke-width 120ms, opacity 120ms; }
.edge:hover { stroke-width: 2.8; }
.edge.active { stroke-width: 3; }
.edge.dimmed { opacity: 0.18; }

.edge.type-confirms    { stroke: var(--rel-confirms-fg); }
.edge.type-contradicts { stroke: var(--rel-contradicts-fg); }
.edge.type-extends     { stroke: var(--rel-extends-fg); stroke-dasharray: 4 3; }
.edge.type-informs     { stroke: var(--rel-informs-fg); stroke-width: 1.6; opacity: 0.9; }
.edge.type-informs:hover { opacity: 1; stroke-width: 2.5; }
.edge.type-informs.active { opacity: 1; stroke-width: 2.8; }

/* Backward edges hidden by default — toggle adds .show-backward on the SVG */
.belief-graph .edge.edge-backward { display: none; }
.belief-graph.show-backward .edge.edge-backward { display: inline; }
/* When backward is shown, dim forward edges so the colored ones pop */
.belief-graph.show-backward .edge.type-informs { opacity: 0.45; }
.belief-graph.show-backward .edge.type-informs:hover,
.belief-graph.show-backward .edge.type-informs.active { opacity: 1; }

.graph-toolbar {
  position: absolute; left: 24px; top: 12px;
  display: flex; gap: 10px; align-items: center; z-index: 3;
}
.toggle-bwd {
  font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
  background: rgba(255,255,255,0.92); border: 1px solid var(--rule);
  color: var(--ink-soft); padding: 5px 11px; border-radius: 4px;
  cursor: pointer; transition: all 120ms;
  letter-spacing: 0.02em;
}
.toggle-bwd:hover { background: white; color: var(--ink); border-color: var(--ink-soft); }
.toggle-bwd.active {
  background: var(--ink); color: white; border-color: var(--ink);
}

.legend {
  position: absolute; right: 24px; top: 12px;
  display: flex; gap: 14px; font-family: 'JetBrains Mono', monospace; font-size: 10px;
  color: var(--ink-soft); background: rgba(255,255,255,0.92);
  padding: 5px 10px; border: 1px solid var(--rule); border-radius: 4px;
  z-index: 3;
}
.legend .swatch { display:inline-block; width:18px; height:0; border-top:2px solid; vertical-align:middle; margin-right:4px; }
.legend .sw-informs     { border-color: var(--rel-informs-fg); }
.legend .sw-confirms    { border-color: var(--rel-confirms-fg); }
.legend .sw-contradicts { border-color: var(--rel-contradicts-fg); }
.legend .sw-extends     { border-color: var(--rel-extends-fg); border-top-style: dashed; }

.rel-pill.type-informs     { color: var(--rel-informs-fg);     background: #e7eef7; }
.rel-pill.type-confirms    { color: var(--rel-confirms-fg);    background: #e0f0e4; }
.rel-pill.type-contradicts { color: var(--rel-contradicts-fg); background: var(--stance-asserted-bg); }
.rel-pill.type-extends     { color: var(--rel-extends-fg);     background: var(--src-call-bg); }

/* Detail panel */
.detail-empty {
  font-family: 'Inter Tight', sans-serif; font-size: 13px;
  color: var(--ink-faint); font-style: italic;
}
.detail-card { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px;
               padding: 16px 18px; }
.detail-card h3 {
  margin: 0 0 8px; font-family: 'Fraunces', Georgia, serif; font-weight: 600;
  font-size: 17px; color: var(--ink); font-variation-settings: 'opsz' 16;
}
.detail-card h4 {
  margin: 14px 0 6px; font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
  text-transform: uppercase; letter-spacing: 0.1em; color: var(--ink-faint); font-weight: 500;
}
.detail-belief-text {
  font-family: 'Fraunces', Georgia, serif; font-size: 15px; line-height: 1.5;
  color: var(--ink); margin: 0 0 12px;
}
.detail-meta {
  display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 8px;
}
.detail-meta .badge {
  font-family: 'Inter Tight', sans-serif; font-size: 9.5px;
  text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
  padding: 2px 7px; border-radius: 3px;
}
.badge.source-user      { color: var(--src-user-fg); background: var(--src-user-bg); }
.badge.source-assistant { color: var(--src-final-fg); background: var(--src-final-bg); }
.badge.source-tool      { color: var(--src-tool-fg); background: var(--src-tool-bg); }
.badge.stance-asserted   { color: var(--stance-asserted-fg);   background: var(--stance-asserted-bg); }
.badge.stance-recalled   { color: var(--stance-recalled-fg);   background: var(--stance-recalled-bg); }
.badge.stance-speculated { color: var(--stance-speculated-fg); background: var(--stance-speculated-bg); }
.badge.stance-judged     { color: var(--stance-judged-fg);     background: var(--stance-judged-bg); }
.badge.layer { color: var(--ink-soft); background: var(--bg); border: 1px solid var(--rule); }

.conf-history {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  color: var(--ink-soft); margin: 0 0 4px; padding: 0; list-style: none;
}
.conf-history li {
  padding: 4px 0; border-top: 1px dotted var(--rule);
  display: flex; gap: 10px; align-items: baseline;
}
.conf-history li:first-child { border-top: none; }
.conf-history .ch-step { width: 90px; font-weight: 600; }
.conf-history .ch-val { font-weight: 700; min-width: 45px; }
.conf-history .ch-delta { color: var(--ink-faint); min-width: 60px; }
.conf-history .ch-reason { color: var(--ink-soft); font-style: italic; }

.detail-excerpts {
  list-style: none; margin: 4px 0 0; padding: 0; display: flex; flex-direction: column; gap: 5px;
}
.detail-excerpts li {
  font-family: 'JetBrains Mono', monospace; font-size: 10.5px; line-height: 1.6;
  color: var(--ink-soft); padding-left: 18px; position: relative;
}
.detail-excerpts .mark { position: absolute; left: 0; top: 0; font-weight: 700; }
.detail-excerpts .found   .mark { color: #2e8540; }
.detail-excerpts .missing .mark { color: var(--accent); }
.detail-excerpts .missing { color: var(--accent); font-style: italic; }

.edge-detail-pair {
  display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px;
}
.edge-detail-pair .side { background: var(--bg); border: 1px solid var(--rule);
                          border-radius: 4px; padding: 10px 12px; }
.edge-detail-pair .side-label {
  font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
  text-transform: uppercase; letter-spacing: 0.1em; color: var(--ink-faint); margin-bottom: 4px;
}
.edge-detail-pair .side-text {
  font-family: 'Fraunces', Georgia, serif; font-size: 13.5px; line-height: 1.45; color: var(--ink);
}
.edge-detail-rel {
  display: flex; gap: 8px; align-items: center; margin: 2px 0 8px;
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
}
.edge-detail-rel .arrow { color: var(--ink-faint); }
.edge-detail-rel .rel-pill {
  font-family: 'Inter Tight', sans-serif; font-size: 9.5px;
  text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
  padding: 2px 8px; border-radius: 3px;
}

.hint {
  padding: 9px 36px; border-top: 1px solid var(--rule);
  font-size: 11.5px; color: var(--ink-faint); background: var(--panel);
}
.hint kbd { font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
            background: var(--bg); padding: 2px 6px; border: 1px solid var(--rule);
            border-radius: 3px; color: var(--ink-soft); }
"""


# =============================================================
# Embedded JS
# =============================================================

JS = r"""
let currentSelection = null;  // {kind: 'node', id} | {kind: 'edge', from, to, type} | null
const D = window.GRAPH_DATA;

function clearSelection() {
  currentSelection = null;
  document.querySelectorAll('.node.active').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.edge.active').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.node.dimmed, .edge.dimmed').forEach(el => el.classList.remove('dimmed'));
  document.querySelectorAll('.ev.active').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.msg.flash').forEach(el => el.classList.remove('flash'));
  renderDetailEmpty();
}

function selectNode(beliefId) {
  if (currentSelection && currentSelection.kind === 'node' && currentSelection.id === beliefId) {
    clearSelection(); return;
  }
  clearSelection();
  currentSelection = { kind: 'node', id: beliefId };

  // Highlight node + dim other unconnected nodes/edges
  const connected = new Set([beliefId]);
  D.edges.forEach(e => {
    if (e.from_id === beliefId) connected.add(e.to_id);
    if (e.to_id   === beliefId) connected.add(e.from_id);
  });

  document.querySelectorAll('.node').forEach(el => {
    const nid = parseInt(el.dataset.id, 10);
    if (nid === beliefId) el.classList.add('active');
    else if (!connected.has(nid)) el.classList.add('dimmed');
  });
  document.querySelectorAll('.edge').forEach(el => {
    const f = parseInt(el.dataset.fromId, 10);
    const t = parseInt(el.dataset.toId,   10);
    if (f === beliefId || t === beliefId) el.classList.add('active');
    else el.classList.add('dimmed');
  });

  // Light up evidence in the trajectory + flash the host message panel.
  let firstEv = null;
  document.querySelectorAll('.ev').forEach(s => {
    const ids = (s.dataset.beliefs || '').split(' ').filter(Boolean);
    if (ids.includes(String(beliefId))) {
      s.classList.add('active');
      if (!firstEv) firstEv = s;
    }
  });
  const belief = D.beliefs.find(b => b.id === beliefId);
  if (belief) {
    const ti = belief.source && belief.source.trajectory_index;
    if (typeof ti === 'number' && ti >= 0) {
      const panel = document.getElementById('msg-' + ti);
      if (panel) panel.classList.add('flash');
      if (firstEv) firstEv.scrollIntoView({ behavior: 'smooth', block: 'center' });
      else if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  renderNodeDetail(beliefId);
}

function selectEdge(fromId, toId, type, direction) {
  if (currentSelection && currentSelection.kind === 'edge'
      && currentSelection.from === fromId && currentSelection.to === toId
      && currentSelection.type === type && currentSelection.direction === direction) {
    clearSelection(); return;
  }
  clearSelection();
  currentSelection = { kind: 'edge', from: fromId, to: toId, type, direction };

  // Active edge + active nodes at both ends.
  const edgeKey = fromId + '->' + toId + ':' + type;
  document.querySelectorAll('.edge').forEach(el => {
    if (el.dataset.key === edgeKey && el.dataset.dir === direction) el.classList.add('active');
    else el.classList.add('dimmed');
  });
  document.querySelectorAll('.node').forEach(el => {
    const nid = parseInt(el.dataset.id, 10);
    if (nid === fromId || nid === toId) el.classList.add('active');
    else el.classList.add('dimmed');
  });

  renderEdgeDetail(fromId, toId, type, direction);
}

// ---- Detail panel renderers ----
function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                   .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function renderDetailEmpty() {
  document.getElementById('detail').innerHTML =
    '<div class="detail-empty">Click a node to inspect a belief, or an edge to inspect a relation.</div>';
}

function renderNodeDetail(beliefId) {
  const b = D.beliefs.find(b => b.id === beliefId);
  if (!b) { renderDetailEmpty(); return; }
  const src = b.source || {};
  const meta = [
    `<span class="badge source-${src.type}">${escapeHtml(D.source_label[src.type] || src.type || 'unknown')}</span>`,
    `<span class="badge stance-${b.stance}">${escapeHtml(b.stance)}</span>`,
    `<span class="badge layer">${escapeHtml(b.layer || '')}</span>`,
  ].join('');

  let hist = '';
  if (Array.isArray(b.confidence_history) && b.confidence_history.length) {
    hist = '<h4>Confidence trail</h4><ul class="conf-history">' +
      b.confidence_history.map(h => {
        const delta = (h.delta != null) ? ((h.delta >= 0 ? '+' : '') + h.delta.toFixed(2)) : '';
        const fromIdStr = (h.from_belief_id != null) ? ` ← from #${(h.from_belief_id+1).toString().padStart(2,'0')}` : '';
        return `<li>
          <span class="ch-step">${escapeHtml(h.step)}${escapeHtml(fromIdStr)}</span>
          <span class="ch-val">${(h.value != null) ? h.value.toFixed(2) : '—'}</span>
          <span class="ch-delta">${escapeHtml(delta)}</span>
          <span class="ch-reason">${escapeHtml(h.reason || '')}</span>
        </li>`;
      }).join('') + '</ul>';
  }

  let exc = '';
  if (Array.isArray(b.supporting_excerpts) && b.supporting_excerpts.length) {
    exc = '<h4>Supporting excerpts</h4><ul class="detail-excerpts">' +
      b.supporting_excerpts.map(e => {
        const found = (b._evidence_status || {})[e] !== false;  // default to found
        const cls = found ? 'found' : 'missing';
        const mark = found ? '✓' : '✗';
        return `<li class="${cls}"><span class="mark">${mark}</span>${escapeHtml(e)}</li>`;
      }).join('') + '</ul>';
  }

  // Source breadcrumb
  const breadcrumb = (typeof src.trajectory_index === 'number' && src.trajectory_index >= 0)
    ? `traj[${src.trajectory_index}] · ${escapeHtml(src.role || src.type || '?')}`
        + (src.date ? ` · ${escapeHtml(src.date)}` : '')
    : 'external (no trajectory)';

  document.getElementById('detail').innerHTML = `
    <div class="detail-card">
      <h3>Belief #${(b.id+1).toString().padStart(2,'0')}  ·  ${b.confidence.toFixed(2)}</h3>
      <div class="detail-meta">${meta}</div>
      <div class="detail-belief-text">${escapeHtml(b.belief)}</div>
      <h4>Source</h4>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--ink-soft);">${breadcrumb}</div>
      ${hist}
      ${exc}
    </div>`;
}

function renderEdgeDetail(fromId, toId, type, direction) {
  const from = D.beliefs.find(b => b.id === fromId);
  const to   = D.beliefs.find(b => b.id === toId);
  if (!from || !to) { renderDetailEmpty(); return; }
  const pool = (direction === 'forward') ? D.forward_edges : D.backward_edges;
  const rel = pool.find(e => e.from_id === fromId && e.to_id === toId && e.type === type);
  const note = rel ? rel.note : '';

  // Confidence impact: only for backward relations (forward edges don't move confidence).
  let impact = '';
  if (direction === 'backward' && Array.isArray(to.confidence_history)) {
    const hit = to.confidence_history.find(h => h.step === type && h.from_belief_id === fromId);
    if (hit) {
      const delta = (hit.delta >= 0 ? '+' : '') + hit.delta.toFixed(2);
      impact = `<h4>Confidence impact on #${(to.id+1).toString().padStart(2,'0')}</h4>
        <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--ink-soft);">
          confidence ${delta} → ${hit.value.toFixed(2)}
        </div>`;
    }
  } else if (direction === 'forward') {
    impact = `<h4>Confidence impact</h4>
      <div style="font-family:'Inter Tight',sans-serif;font-size:12px;color:var(--ink-faint);font-style:italic;">
        Forward (derivation) relations do not modify confidence.
      </div>`;
  }

  const dirLabel = (direction === 'forward')
    ? 'forward · earlier → later'
    : 'backward · later → earlier';

  document.getElementById('detail').innerHTML = `
    <div class="detail-card">
      <h3>Relation <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--ink-faint);font-weight:400;text-transform:uppercase;letter-spacing:0.08em;margin-left:6px;">${escapeHtml(dirLabel)}</span></h3>
      <div class="edge-detail-rel">
        <span>#${(from.id+1).toString().padStart(2,'0')}</span>
        <span class="arrow">→</span>
        <span class="rel-pill type-${type}">${escapeHtml(type)}</span>
        <span class="arrow">→</span>
        <span>#${(to.id+1).toString().padStart(2,'0')}</span>
      </div>
      ${note ? `<h4>Note</h4><div style="font-family:'Inter Tight',sans-serif;font-size:13px;color:var(--ink-soft);font-style:italic;">${escapeHtml(note)}</div>` : ''}
      ${impact}
      <h4>Beliefs</h4>
      <div class="edge-detail-pair">
        <div class="side">
          <div class="side-label">from · #${(from.id+1).toString().padStart(2,'0')} · ${from.confidence.toFixed(2)} · ${escapeHtml(from.stance)}</div>
          <div class="side-text">${escapeHtml(from.belief)}</div>
        </div>
        <div class="side">
          <div class="side-label">to · #${(to.id+1).toString().padStart(2,'0')} · ${to.confidence.toFixed(2)} · ${escapeHtml(to.stance)}</div>
          <div class="side-text">${escapeHtml(to.belief)}</div>
        </div>
      </div>
    </div>`;
}

// ---- Global click delegations ----
document.addEventListener('click', (e) => {
  const ev = e.target.closest('.ev');
  if (ev) {
    const ids = (ev.dataset.beliefs || '').split(' ').filter(Boolean);
    if (ids.length) selectNode(parseInt(ids[0], 10));
    return;
  }
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') clearSelection(); });

function toggleBackward() {
  const svg = document.querySelector('.belief-graph');
  const btn = document.querySelector('.toggle-bwd');
  if (!svg || !btn) return;
  const active = svg.classList.toggle('show-backward');
  btn.classList.toggle('active', active);
  btn.textContent = active ? '✓ backward edges' : '+ backward edges';
  // If we just hid backward and the current selection is a backward edge,
  // clear it so the inspector doesn't reference an invisible edge.
  if (!active && currentSelection && currentSelection.kind === 'edge'
      && currentSelection.direction === 'backward') {
    clearSelection();
  }
}

document.addEventListener('DOMContentLoaded', renderDetailEmpty);
"""


# =============================================================
# HTML rendering
# =============================================================


def render_message_panel(
    traj_idx: int,
    msg: dict[str, Any],
    beliefs_for_msg: list[tuple[int, dict[str, Any], list[tuple[int, int]]]],
    # beliefs_for_msg = list of (belief_global_id, belief, list_of_(start,end)_intervals_within_msg)
) -> str:
    role = msg.get("role") or "?"
    content = msg.get("content", "") or ""
    role_label = ROLE_LABEL.get(role, role)
    role_class = re.sub(r"[^a-z]", "", role.lower()) or "unknown"

    intervals: list[tuple[int, int, int]] = []
    for gid, _b, ranges in beliefs_for_msg:
        for s, e in ranges:
            intervals.append((s, e, gid))

    pieces = slice_with_marks(content, intervals)
    body_parts: list[str] = []
    for p in pieces:
        text = html_lib.escape(p["text"])
        if not p["beliefs"]:
            body_parts.append(text)
        else:
            ids = " ".join(str(b) for b in p["beliefs"])
            body_parts.append(f'<span class="ev" data-beliefs="{ids}">{text}</span>')
    body_html = "".join(body_parts)

    summary_cls = "msg-belief-count" + ("" if beliefs_for_msg else " empty")
    summary_text = (
        (f"{len(beliefs_for_msg)} belief" + ("s" if len(beliefs_for_msg) != 1 else ""))
        if beliefs_for_msg
        else "no beliefs"
    )

    return (
        f'<article class="msg msg-{role_class}" id="msg-{traj_idx}">'
        f'<header class="msg-head">'
        f'<span class="msg-role role-{role_class}">{html_lib.escape(role_label)}</span>'
        f'<span class="msg-idx">trajectory_index <b>{traj_idx}</b></span>'
        f'<span class="{summary_cls}">{summary_text}</span>'
        f"</header>"
        f'<pre class="msg-body">{body_html}</pre>'
        f"</article>"
    )


def render_graph_svg(
    beliefs: list[dict[str, Any]],
    forward_relations: list[dict[str, Any]],
    backward_relations: list[dict[str, Any]],
    positions: dict[int, tuple[float, float]],
    width: int,
    height: int,
    sorted_traj: list[int],
    col_w: int,
    pad_x: int,
) -> str:
    parts: list[str] = []

    # Column labels at the bottom (single tick per traj message).
    for ci, ti in enumerate(sorted_traj):
        cx = pad_x + col_w / 2 + ci * col_w
        parts.append(
            f'<text class="col-label" x="{cx}" y="{height - 12}">traj[{ti}]</text>'
        )

    # Arrow markers (one per edge type so color follows)
    parts.append(
        "<defs>"
        '<marker id="arr-confirms" viewBox="0 0 10 10" refX="9" refY="5" '
        ' markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '  <path d="M0,0 L10,5 L0,10 z" fill="#1f6a35"/>'
        "</marker>"
        '<marker id="arr-contradicts" viewBox="0 0 10 10" refX="9" refY="5" '
        ' markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '  <path d="M0,0 L10,5 L0,10 z" fill="#c84a3e"/>'
        "</marker>"
        '<marker id="arr-extends" viewBox="0 0 10 10" refX="9" refY="5" '
        ' markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '  <path d="M0,0 L10,5 L0,10 z" fill="#b3500e"/>'
        "</marker>"
        '<marker id="arr-informs" viewBox="0 0 10 10" refX="9" refY="5" '
        ' markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        '  <path d="M0,0 L10,5 L0,10 z" fill="#6b89b8"/>'
        "</marker>"
        "</defs>"
    )

    def _edge_path(fid: int, tid: int) -> str | None:
        if fid not in positions or tid not in positions:
            return None
        x1, y1 = positions[fid]
        x2, y2 = positions[tid]
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        px = -dy / length
        py = dx / length
        offset = max(30, min(80, abs(dx) * 0.25 + 20))
        cx = mx + px * offset
        cy = my + py * offset
        return f"M {x1:.1f},{y1:.1f} Q {cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}"

    # Forward edges first (so backward edges sit on top — they're the louder
    # ones epistemically and should be more visible).
    for r in forward_relations:
        fid = r["from_id"]
        tid = r["to_id"]
        path = _edge_path(fid, tid)
        if not path:
            continue
        rtype = r.get("type", "informs")
        edge_key = f"{fid}->{tid}:{rtype}"
        parts.append(
            f'<path class="edge edge-forward type-{rtype}" d="{path}" '
            f'marker-end="url(#arr-{rtype})" '
            f'data-from-id="{fid}" data-to-id="{tid}" data-type="{rtype}" data-dir="forward" '
            f'data-key="{edge_key}" '
            f"onclick=\"selectEdge({fid},{tid},'{rtype}','forward')\"/>"
        )

    # Backward edges
    for r in backward_relations:
        fid = r["from_id"]
        tid = r["to_id"]
        path = _edge_path(fid, tid)
        if not path:
            continue
        rtype = r.get("type", "extends")
        edge_key = f"{fid}->{tid}:{rtype}"
        parts.append(
            f'<path class="edge edge-backward type-{rtype}" d="{path}" '
            f'marker-end="url(#arr-{rtype})" '
            f'data-from-id="{fid}" data-to-id="{tid}" data-type="{rtype}" data-dir="backward" '
            f'data-key="{edge_key}" '
            f"onclick=\"selectEdge({fid},{tid},'{rtype}','backward')\"/>"
        )

    # Nodes
    for b in beliefs:
        bid = b["id"]
        if bid not in positions:
            continue
        x, y = positions[bid]
        src_type = (b.get("source") or {}).get("type", "unknown")
        conf = float(b.get("confidence", 0.0))
        parts.append(
            f'<g class="node source-{src_type}" data-id="{bid}" transform="translate({x:.1f},{y:.1f})" '
            f' onclick="selectNode({bid})">'
            f'<rect class="node-pill" x="-36" y="-15" width="72" height="30"/>'
            f'<text class="node-id" x="0" y="-3">#{bid + 1:02d}</text>'
            f'<text class="node-conf" x="0" y="9">{conf:.2f}</text>'
            f"</g>"
        )

    return (
        f'<svg class="belief-graph" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        + "".join(parts)
        + "</svg>"
    )


def render_html(data: dict[str, Any], src_path: str) -> str:
    trajectory = data.get("trajectory") or []
    all_beliefs_raw = data.get("all_beliefs") or []
    # Read both link layers. Fall back to legacy `relations` field for files
    # produced before the forward/backward split.
    forward_relations = data.get("forward_relations") or []
    backward_relations = data.get("backward_relations") or data.get("relations") or []
    model = data.get("model", "") or ""
    prompt_name = data.get("prompt_name", "construct_beliefs") or "construct_beliefs"

    all_beliefs: list[dict[str, Any]] = [
        b
        for b in all_beliefs_raw
        if isinstance(b, dict) and isinstance(b.get("belief"), str)
    ]

    # For each belief, take highlight ranges DIRECTLY from evidence offsets
    # (v3 evidence carries exact start/end into the original turn content — no
    # segment rederivation or excerpt re-search needed). Group by trajectory_index.
    by_msg: dict[int, list[tuple[int, dict[str, Any], list[tuple[int, int]]]]] = {}
    for b in all_beliefs:
        bid = b["id"]
        src = b.get("source") or {}
        ti = src.get("trajectory_index")
        if not isinstance(ti, int) or not (0 <= ti < len(trajectory)):
            continue
        host_content = trajectory[ti].get("content", "") or ""

        excerpt_status: dict[str, bool] = {}
        all_ranges: list[tuple[int, int]] = []
        for ev in b.get("evidence") or []:
            txt = ev.get("text")
            s, e = ev.get("start"), ev.get("end")
            located = (
                isinstance(s, int)
                and isinstance(e, int)
                and 0 <= s < e <= len(host_content)
            )
            if isinstance(txt, str):
                excerpt_status[txt] = located or excerpt_status.get(txt, False)
            if located:
                all_ranges.append((s, e))
        b["_evidence_status"] = excerpt_status
        by_msg.setdefault(ti, []).append((bid, b, all_ranges))

    # Render left panel (trajectory with evidence highlights).
    msg_panels = []
    for i, m in enumerate(trajectory):
        msg_panels.append(render_message_panel(i, m, by_msg.get(i, [])))
    msgs_html = (
        "\n".join(msg_panels) if msg_panels else '<p class="muted">No trajectory.</p>'
    )

    # Compute graph layout.
    positions, gw, gh, sorted_traj, col_w, pad_x = compute_layout(all_beliefs)

    graph_svg = render_graph_svg(
        all_beliefs,
        forward_relations,
        backward_relations,
        positions,
        gw,
        gh,
        sorted_traj,
        col_w,
        pad_x,
    )

    # Stats counters
    SRC_ORDER = ("user", "assistant", "tool")
    src_counts = {k: 0 for k in SRC_ORDER}
    for b in all_beliefs:
        t = (b.get("source") or {}).get("type", "")
        if t in src_counts:
            src_counts[t] += 1
    total = len(all_beliefs)

    fwd_count = len(forward_relations)
    bwd_counts: dict[str, int] = {}
    for r in backward_relations:
        t = r.get("type", "?")
        bwd_counts[t] = bwd_counts.get(t, 0) + 1
    bwd_pieces = ", ".join(f"{v} {html_lib.escape(k)}" for k, v in bwd_counts.items())

    stats_html = (
        f"<span><b>{total}</b> belief{'s' if total != 1 else ''}</span>"
        + "".join(
            f'<span><span class="stat-dot {k}"></span><b>{src_counts[k]}</b> {SOURCE_LABEL[k]}</span>'
            for k in SRC_ORDER
            if src_counts[k] > 0
        )
        + f'<span><b>{fwd_count}</b> forward <span style="color:var(--ink-faint)">(informs)</span></span>'
        + f"<span><b>{len(backward_relations)}</b> backward"
        + (
            f' <span style="color:var(--ink-faint)">({bwd_pieces})</span>'
            if bwd_pieces
            else ""
        )
        + "</span>"
        + f"<span><b>{len(trajectory)}</b> messages</span>"
    )

    # Embed graph data for JS.
    graph_data = {
        "beliefs": all_beliefs,
        "forward_edges": forward_relations,
        "backward_edges": backward_relations,
        # Unified edges list for click handlers (each tagged with direction).
        "edges": (
            [{**r, "_dir": "forward"} for r in forward_relations]
            + [{**r, "_dir": "backward"} for r in backward_relations]
        ),
        "source_label": SOURCE_LABEL,
    }
    graph_data_json = json.dumps(graph_data, ensure_ascii=False)

    return (
        '<!doctype html>\n<html lang="en"><head>'
        '<meta charset="utf-8">'
        f"<title>Belief Graph · {html_lib.escape(prompt_name)}</title>"
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'
        f"<style>{CSS}</style>"
        "</head><body>"
        '<header class="app-header">'
        f'<h1>Belief Graph <span class="sep">·</span> <span class="tag">{html_lib.escape(prompt_name)}</span></h1>'
        '<div class="file-meta">'
        f"<span>source · <b>{html_lib.escape(Path(src_path).name)}</b></span>"
        f"<span>model · <b>{html_lib.escape(model)}</b></span>"
        "</div></header>"
        f'<div class="stats">{stats_html}</div>'
        '<div class="layout">'
        f'<div class="left-panel"><h2 class="section-title">Conversation trajectory</h2>{msgs_html}</div>'
        '<div class="right-panel">'
        f'<div class="graph-wrap">'
        f'  <div class="graph-toolbar">'
        f'    <button class="toggle-bwd" onclick="toggleBackward()" title="Show / hide backward (evaluation) edges">+ backward edges</button>'
        f"  </div>"
        f'  <div class="legend">'
        f'    <span><span class="swatch sw-informs"></span>informs</span>'
        f'    <span><span class="swatch sw-confirms"></span>confirms</span>'
        f'    <span><span class="swatch sw-contradicts"></span>contradicts</span>'
        f'    <span><span class="swatch sw-extends"></span>extends</span>'
        f"  </div>"
        f"  {graph_svg}"
        f"</div>"
        f'<div class="detail-wrap"><h2 class="section-title">Inspector</h2><div id="detail"></div></div>'
        "</div></div>"
        '<div class="hint">'
        "Click a node to inspect a belief (highlights its evidence and host message). "
        "Click an edge to inspect a relation. Click an evidence span to jump to its node. "
        "<kbd>Esc</kbd> clears."
        "</div>"
        "<script>"
        f"window.GRAPH_DATA = {graph_data_json};"
        f"{JS}"
        "</script>"
        "</body></html>"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Graph visualizer for construct_beliefs result.json"
    )
    parser.add_argument("input", help="Path to result.json")
    parser.add_argument("--output", "-o", default=None, help="Output HTML path")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[error] file not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)

    html = render_html(data, str(in_path))
    out_path = (
        Path(args.output)
        if args.output
        else in_path.parent / f"graph_{in_path.stem}.html"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[ok] wrote {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
