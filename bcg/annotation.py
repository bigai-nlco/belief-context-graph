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

v4 notes (interactive annotation frontend)
------------------------------------------
* This file now emits an INTERACTIVE ANNOTATION frontend, not a static view.
  The complete original result.json is embedded as `window.RAW_DATA`; all
  rendering (trajectory, SVG graph, stats, inspector) is performed in the
  browser by the JS in `JS`, against a mutable working copy (STATE).
* Editing: add/delete belief nodes, add/delete/retype/flip relation edges,
  and add/delete evidence by drag-selecting source text in the left panel.
  Relations use six unified types (temporal/causal/dependency/coreference/
  elaboration/contradiction); legacy types still render. Direction is the
  pool the edge lives in (forward_relations vs backward_relations).
* Export writes the COMPLETE original document back out, overwriting only
  all_beliefs / forward_relations / backward_relations from STATE (internal
  `_`-prefixed flags stripped); every other field is passed through verbatim,
  so the exported file re-ingests cleanly into this same generator.
* The module-level Python helpers below (compute_layout, render_graph_svg,
  render_message_panel, slice_with_marks, confidence_band) are SUPERSEDED by
  their JS equivalents and are retained only for reference; render_html no
  longer calls them.

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
from typing import Any, Dict, List, Optional, Tuple


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
    original: str, intervals: List[Tuple[int, int, int]]
) -> List[Dict[str, Any]]:
    if not intervals:
        return [{'text': original, 'beliefs': []}]
    points = {0, len(original)}
    for s, e, _ in intervals:
        points.add(s)
        points.add(e)
    sorted_points = sorted(points)
    pieces: List[Dict[str, Any]] = []
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
        pieces.append({'text': text, 'beliefs': sorted(active)})
    return pieces


# =============================================================
# Static palette (kept in sync with visualize_beliefs_v3.py)
# =============================================================

ALLOWED_SOURCES = {'user', 'assistant', 'tool'}

SOURCE_LABEL = {
    'user':      'user',
    'assistant': 'assistant',
    'tool':      'tool',
}

ROLE_LABEL = {
    'system': 'system', 'user': 'user', 'assistant': 'assistant',
    'tool': 'tool', 'function': 'tool',
}


def confidence_band(c: float) -> Tuple[str, int]:
    if c >= 0.90: return ('certain',  5)
    if c >= 0.75: return ('high',     4)
    if c >= 0.50: return ('medium',   3)
    if c >= 0.25: return ('low',      2)
    return ('verylow', 1)


# =============================================================
# Layout: position each belief as a graph node
# =============================================================

def compute_layout(
    beliefs: List[Dict[str, Any]], width: int = 900, height: int = 280,
):
    """
    Flat structured layout: one column per trajectory_index; within a column,
    nodes stack vertically in id order (which is chronological after the
    pipeline's chrono-renumber step). No layer separation.
    Returns (positions, final_width, final_height, sorted_traj, col_w, pad_x).
    """
    if not beliefs:
        return {}, width, height, [], 130, 60

    by_traj: Dict[int, List[Dict[str, Any]]] = {}
    for b in beliefs:
        ti = (b.get('source') or {}).get('trajectory_index', -1)
        by_traj.setdefault(ti, []).append(b)

    sorted_traj = sorted(by_traj.keys())
    n_cols = len(sorted_traj)
    col_w = 130
    pad_x = 60
    pad_y_top = 32
    pad_y_bot = 36          # extra room for column labels at bottom

    max_per_col = max((len(by_traj[ti]) for ti in sorted_traj), default=1)
    slot_h = 52
    final_height = max(height, pad_y_top + max_per_col * slot_h + pad_y_bot)
    final_width  = max(width,  pad_x * 2 + n_cols * col_w)

    positions: Dict[int, Tuple[float, float]] = {}
    for ci, ti in enumerate(sorted_traj):
        x_center = pad_x + col_w / 2 + ci * col_w
        beliefs_in_col = sorted(by_traj[ti], key=lambda b: b['id'])
        for i, b in enumerate(beliefs_in_col):
            positions[b['id']] = (x_center, pad_y_top + 16 + i * slot_h)

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

/* ===== annotation frontend additions ===== */
:root {
  --rel-temporal-fg:      #5b7fa6;
  --rel-causal-fg:        #d97a0f;
  --rel-dependency-fg:    #7c4dcf;
  --rel-coreference-fg:   #0f9b8e;
  --rel-elaboration-fg:   #9a6b3f;
  --rel-contradiction-fg: #cf3a2c;
}
.mode-badge {
  font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.12em; color: #fff; background: var(--accent);
  padding: 2px 8px; border-radius: 4px; vertical-align: middle; margin-left: 6px;
}

/* the six frozen relation colors */
.edge { stroke: var(--ink-faint); }
.edge.type-temporal      { stroke: var(--rel-temporal-fg); }
.edge.type-causal        { stroke: var(--rel-causal-fg); }
.edge.type-dependency    { stroke: var(--rel-dependency-fg); }
.edge.type-coreference   { stroke: var(--rel-coreference-fg); }
.edge.type-elaboration   { stroke: var(--rel-elaboration-fg); stroke-dasharray: 5 3; }
.edge.type-contradiction { stroke: var(--rel-contradiction-fg); }
/* keep legacy informs visible (default state shows backward edges) */
.belief-graph.show-backward .edge.type-informs { opacity: 0.9; }

/* manual + pending node markers */
.node.manual .node-pill  { stroke-dasharray: 4 2; }
.node.pending .node-pill { stroke: var(--accent); stroke-width: 3; filter: drop-shadow(0 0 4px var(--accent)); }

/* legend */
.legend { flex-wrap: wrap; max-width: 64%; }
.legend .sw-temporal      { border-color: var(--rel-temporal-fg); }
.legend .sw-causal        { border-color: var(--rel-causal-fg); }
.legend .sw-dependency    { border-color: var(--rel-dependency-fg); }
.legend .sw-coreference   { border-color: var(--rel-coreference-fg); }
.legend .sw-elaboration   { border-color: var(--rel-elaboration-fg); border-top-style: dashed; }
.legend .sw-contradiction { border-color: var(--rel-contradiction-fg); }

/* inspector rel-pill colors for the six types */
.rel-pill.type-temporal      { color: var(--rel-temporal-fg);      background: #e6edf4; }
.rel-pill.type-causal        { color: var(--rel-causal-fg);        background: #fbedd8; }
.rel-pill.type-dependency    { color: var(--rel-dependency-fg);    background: #efe7fb; }
.rel-pill.type-coreference   { color: var(--rel-coreference-fg);   background: #d9f2ee; }
.rel-pill.type-elaboration   { color: var(--rel-elaboration-fg);   background: #f0e7dc; }
.rel-pill.type-contradiction { color: var(--rel-contradiction-fg); background: #fbe3e0; }

/* annotation toolbar */
.anno-toolbar {
  display: flex; align-items: center; gap: 10px; flex: 0 0 auto;
  padding: 10px 36px; border-bottom: 1px solid var(--rule); background: var(--panel);
}
.anno-spacer { flex: 1 1 auto; }
.ct {
  font-family: 'Inter Tight', sans-serif; font-size: 12.5px; font-weight: 500;
  border: 1px solid var(--rule); background: #fff; color: var(--ink);
  padding: 6px 13px; border-radius: 5px; cursor: pointer; transition: all 120ms;
}
.ct:hover { border-color: var(--ink-soft); }
.ct-connect.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.ct-export { background: var(--ink); border-color: var(--ink); color: #fff; }
.ct-export:hover { background: #000; }
.connect-banner {
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: var(--accent);
  opacity: 0; transition: opacity 120ms;
}
.connect-banner.show { opacity: 1; }
.sel-hint {
  font-family: 'Inter Tight', sans-serif; font-weight: 400; font-size: 11px;
  color: var(--ink-faint); text-transform: none; letter-spacing: 0; font-style: italic;
}
.msg-body { user-select: text; }

/* inspector edit controls */
.mono-soft { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: var(--ink-soft); }
.dir-tag {
  font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--ink-faint);
  font-weight: 400; text-transform: uppercase; letter-spacing: 0.06em; margin-left: 6px;
}
.badge.manual-badge { color: var(--accent); background: #fbe7e2; }
.edit-select, .edit-text {
  width: 100%; font-family: 'Inter Tight', sans-serif; font-size: 13px;
  border: 1px solid var(--rule); border-radius: 5px; padding: 7px 9px; color: var(--ink);
  background: #fff; margin-bottom: 4px;
}
.edit-text { font-family: 'JetBrains Mono', monospace; font-size: 12px; resize: vertical; }
.btn-ghost.sm { font-size: 11.5px; padding: 5px 10px; }
.ev-list { list-style: none; margin: 4px 0 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
.ev-list li { display: flex; align-items: baseline; gap: 8px; font-size: 12px; }
.ev-snip { font-family: 'Fraunces', Georgia, serif; color: var(--ink); flex: 1 1 auto; }
.ev-pos { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--ink-faint); white-space: nowrap; }
.ev-tag { color: var(--accent); }
.mini-del {
  border: none; background: none; color: var(--ink-faint); cursor: pointer; font-size: 12px;
  padding: 0 2px; flex: 0 0 auto;
}
.mini-del:hover { color: var(--accent); }
.danger-row {
  margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--rule);
  display: flex; align-items: center; gap: 10px;
}
.danger-note { font-size: 11px; color: var(--ink-faint); }
.btn-danger {
  font-family: 'Inter Tight', sans-serif; font-size: 12.5px; font-weight: 500;
  border: 1px solid #e0b4ad; background: #fbece9; color: var(--accent);
  padding: 6px 13px; border-radius: 5px; cursor: pointer;
}
.btn-danger:hover { background: var(--accent); color: #fff; border-color: var(--accent); }

/* modal */
.btn-primary {
  font-family: 'Inter Tight', sans-serif; font-size: 13px; font-weight: 600;
  border: 1px solid var(--ink); background: var(--ink); color: #fff;
  padding: 8px 18px; border-radius: 6px; cursor: pointer;
}
.btn-primary:hover { background: #000; }
.btn-ghost {
  font-family: 'Inter Tight', sans-serif; font-size: 13px;
  border: 1px solid var(--rule); background: #fff; color: var(--ink-soft);
  padding: 8px 16px; border-radius: 6px; cursor: pointer;
}
.btn-ghost:hover { border-color: var(--ink-soft); color: var(--ink); }
.modal-overlay {
  position: fixed; inset: 0; background: rgba(26,22,18,0.42);
  display: none; align-items: center; justify-content: center; z-index: 50;
}
.modal-overlay.open { display: flex; }
.modal {
  background: var(--panel); border: 1px solid var(--rule); border-radius: 10px;
  width: min(540px, calc(100vw - 48px)); max-height: calc(100vh - 60px); overflow-y: auto;
  box-shadow: 0 24px 60px rgba(0,0,0,0.22);
}
.modal-head { display: flex; align-items: center; justify-content: space-between; padding: 16px 20px 8px; }
.modal-head h3 { margin: 0; font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 18px; }
.modal-x { border: none; background: none; font-size: 16px; color: var(--ink-faint); cursor: pointer; }
.modal-x:hover { color: var(--ink); }
.modal-body { padding: 8px 20px 4px; display: flex; flex-direction: column; gap: 12px; }
.modal-foot { display: flex; justify-content: flex-end; gap: 10px; padding: 12px 20px 18px; }
.fld {
  display: flex; flex-direction: column; gap: 5px;
  font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
  text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-faint);
}
.fld textarea, .fld input, .fld select {
  font-family: 'Inter Tight', sans-serif; font-size: 13.5px; text-transform: none; letter-spacing: 0;
  color: var(--ink); border: 1px solid var(--rule); border-radius: 6px; padding: 8px 10px; background: #fff;
}
.fld textarea { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; resize: vertical; }
.form-row { display: flex; gap: 12px; }
.form-row .fld { flex: 1 1 0; min-width: 0; }
.form-note {
  font-family: 'Inter Tight', sans-serif; font-size: 12px; color: var(--ink-soft);
  background: var(--bg); border: 1px solid var(--rule); border-radius: 6px; padding: 8px 10px;
}
.sel-preview {
  font-family: 'Fraunces', Georgia, serif; font-size: 14px; line-height: 1.5; color: var(--ink);
  background: var(--evidence); border-radius: 6px; padding: 10px 12px;
}
.edge-preview { font-family: 'JetBrains Mono', monospace; font-size: 14px; color: var(--ink); text-align: center; }
.form-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.mini-belief {
  font-family: 'Fraunces', Georgia, serif; font-size: 12.5px; line-height: 1.4; color: var(--ink-soft);
  background: var(--bg); border: 1px solid var(--rule); border-radius: 6px; padding: 8px 10px;
}

/* toast */
#toast {
  position: fixed; left: 50%; bottom: 26px; transform: translateX(-50%) translateY(12px);
  background: var(--ink); color: #fff; font-family: 'Inter Tight', sans-serif; font-size: 13px;
  padding: 9px 16px; border-radius: 7px; opacity: 0; pointer-events: none;
  transition: opacity 160ms, transform 160ms; z-index: 60; box-shadow: 0 8px 24px rgba(0,0,0,0.2);
}
#toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
"""


# =============================================================
# Embedded JS
# =============================================================

JS = r"""
/* ============================================================
   Belief-graph annotation frontend
   - STATE is a mutable working copy of the embedded RAW_DATA.
   - Every edit mutates STATE; rerender() then rebuilds the
     trajectory, graph and stats from scratch.
   - Export writes the COMPLETE original document back out, with
     only all_beliefs / forward_relations / backward_relations
     replaced by the edited STATE.
   ============================================================ */

const RAW = window.RAW_DATA || {};
const TRAJ = RAW.trajectory || [];

const SOURCE_LABEL = { user: 'user', assistant: 'assistant', tool: 'tool' };
const ROLE_LABEL = { system: 'system', user: 'user', assistant: 'assistant', tool: 'tool', function: 'tool' };

/* The six frozen relation types. Forward and backward share one table;
   direction is carried by which pool the edge lives in, not by its type. */
const REL_TYPES = ['temporal', 'causal', 'dependency', 'coreference', 'elaboration', 'contradiction'];
const REL_LABEL = {
  temporal: '时间关系', causal: '因果关系', dependency: '依赖关系',
  coreference: '共同实体关系', elaboration: '细化/补充关系', contradiction: '矛盾关系',
  informs: 'informs (legacy)', confirms: 'confirms (legacy)',
  contradicts: 'contradicts (legacy)', extends: 'extends (legacy)'
};
const REL_SHORT = {
  temporal: '时间', causal: '因果', dependency: '依赖',
  coreference: '共同实体', elaboration: '细化', contradiction: '矛盾'
};
const MARKER_COLORS = {
  temporal: '#5b7fa6', causal: '#d97a0f', dependency: '#7c4dcf',
  coreference: '#0f9b8e', elaboration: '#9a6b3f', contradiction: '#cf3a2c',
  informs: '#6b89b8', confirms: '#1f6a35', contradicts: '#c84a3e', extends: '#b3500e',
  default: '#8a8276'
};
const STANCES = ['asserted', 'recalled', 'speculated', 'judged'];

let STATE = initState();
let currentSelection = null;     // {kind:'node', id} | {kind:'edge', from, to, type, direction}
let connectMode = false;
let pendingSource = null;
let showBackward = true;

/* ---------- small helpers ---------- */
function byId(id) { return document.getElementById(id); }
function val(id) { const el = byId(id); return el ? el.value : ''; }
function deepCopy(x) { return JSON.parse(JSON.stringify(x)); }
function beliefById(id) { return STATE.beliefs.find(b => b.id === id); }
function maxId() { return STATE.beliefs.reduce((m, b) => Math.max(m, b.id), -1); }
function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function truncate(s, n) { s = String(s == null ? '' : s); return s.length > n ? s.slice(0, n - 1) + '…' : s; }

function initState() {
  const beliefs = (RAW.all_beliefs || [])
    .filter(b => b && typeof b === 'object' && typeof b.belief === 'string')
    .map(b => deepCopy(b));
  const fwd = deepCopy(RAW.forward_relations || []);
  const bwd = deepCopy(RAW.backward_relations || RAW.relations || []);
  return { beliefs: beliefs, forward_edges: fwd, backward_edges: bwd };
}
function unifiedEdges() {
  return STATE.forward_edges.map(e => Object.assign({}, e, { _dir: 'forward' }))
    .concat(STATE.backward_edges.map(e => Object.assign({}, e, { _dir: 'backward' })));
}

/* ---------- layout (one column per trajectory_index) ---------- */
function computeLayout(beliefs) {
  const minW = 900, minH = 280, colW = 130, padX = 60, padYTop = 32, padYBot = 36, slotH = 52;
  if (!beliefs.length) return { positions: {}, width: minW, height: minH, sortedTraj: [], colW: colW, padX: padX };
  const byTraj = {};
  beliefs.forEach(b => {
    const ti = (b.source || {}).trajectory_index;
    const k = (ti == null ? -1 : ti);
    (byTraj[k] = byTraj[k] || []).push(b);
  });
  const sortedTraj = Object.keys(byTraj).map(Number).sort((a, b) => a - b);
  let maxPer = 1;
  sortedTraj.forEach(ti => { maxPer = Math.max(maxPer, byTraj[ti].length); });
  const height = Math.max(minH, padYTop + maxPer * slotH + padYBot);
  const width = Math.max(minW, padX * 2 + sortedTraj.length * colW);
  const positions = {};
  sortedTraj.forEach((ti, ci) => {
    const xc = padX + colW / 2 + ci * colW;
    byTraj[ti].slice().sort((a, b) => a.id - b.id).forEach((b, i) => {
      positions[b.id] = [xc, padYTop + 16 + i * slotH];
    });
  });
  return { positions: positions, width: width, height: height, sortedTraj: sortedTraj, colW: colW, padX: padX };
}

/* ---------- left panel ---------- */
function sliceWithMarks(original, intervals) {
  if (!intervals.length) return [{ text: original, beliefs: [] }];
  const pts = new Set([0, original.length]);
  intervals.forEach(iv => { pts.add(iv[0]); pts.add(iv[1]); });
  const sorted = [...pts].sort((a, b) => a - b);
  const pieces = [];
  for (let i = 0; i < sorted.length - 1; i++) {
    const a = sorted[i], b = sorted[i + 1];
    if (a >= b) continue;
    const active = new Set();
    intervals.forEach(iv => { if (iv[0] <= a && b <= iv[1]) active.add(iv[2]); });
    pieces.push({ text: original.slice(a, b), beliefs: [...active].sort((x, y) => x - y) });
  }
  return pieces;
}
function renderMessagePanel(ti, m, beliefsForMsg) {
  const role = m.role || '?';
  const roleLabel = ROLE_LABEL[role] || role;
  const roleClass = (role.toLowerCase().replace(/[^a-z]/g, '')) || 'unknown';
  const content = m.content || '';
  const intervals = [];
  beliefsForMsg.forEach(o => o.ranges.forEach(r => intervals.push([r[0], r[1], o.gid])));
  let body = '';
  sliceWithMarks(content, intervals).forEach(p => {
    const text = escapeHtml(p.text);
    if (!p.beliefs.length) body += text;
    else body += '<span class="ev" data-beliefs="' + p.beliefs.join(' ') + '">' + text + '</span>';
  });
  const n = beliefsForMsg.length;
  const summaryCls = 'msg-belief-count' + (n ? '' : ' empty');
  const summaryText = n ? (n + ' belief' + (n !== 1 ? 's' : '')) : 'no beliefs';
  return '<article class="msg msg-' + roleClass + '" id="msg-' + ti + '">' +
    '<header class="msg-head">' +
    '<span class="msg-role role-' + roleClass + '">' + escapeHtml(roleLabel) + '</span>' +
    '<span class="msg-idx">trajectory_index <b>' + ti + '</b></span>' +
    '<span class="' + summaryCls + '">' + summaryText + '</span>' +
    '</header>' +
    '<pre class="msg-body" data-ti="' + ti + '">' + body + '</pre>' +
    '</article>';
}
function buildTrajectoryHTML() {
  // Place each evidence highlight in the message named by the EVIDENCE's own
  // trajectory_index (falling back to the belief's source). This supports
  // cross-message evidence created during annotation, where a belief sourced
  // in one turn carries evidence pointing at another turn.
  const acc = {};   // ti -> { beliefId -> [[s,e], ...] }
  STATE.beliefs.forEach(b => {
    (b.evidence || []).forEach(ev => {
      const evTi = (ev.source && Number.isInteger(ev.source.trajectory_index))
        ? ev.source.trajectory_index
        : (b.source || {}).trajectory_index;
      if (!Number.isInteger(evTi) || evTi < 0 || evTi >= TRAJ.length) return;
      const content = TRAJ[evTi].content || '';
      const s = ev.start, e = ev.end;
      if (!(Number.isInteger(s) && Number.isInteger(e) && s >= 0 && s < e && e <= content.length)) return;
      acc[evTi] = acc[evTi] || {};
      (acc[evTi][b.id] = acc[evTi][b.id] || []).push([s, e]);
    });
  });
  const byMsg = {};
  Object.keys(acc).forEach(ti => {
    byMsg[ti] = Object.keys(acc[ti]).map(gid => ({ gid: Number(gid), ranges: acc[ti][gid] }));
  });
  const panels = TRAJ.map((m, i) => renderMessagePanel(i, m, byMsg[i] || []));
  return panels.length ? panels.join('\n') : '<p class="detail-empty">No trajectory.</p>';
}

/* ---------- graph SVG ---------- */
function markerDefs() {
  let m = '<defs>';
  Object.keys(MARKER_COLORS).forEach(t => {
    m += '<marker id="arr-' + t + '" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">' +
         '<path d="M0,0 L10,5 L0,10 z" fill="' + MARKER_COLORS[t] + '"/></marker>';
  });
  return m + '</defs>';
}
function edgePathD(positions, fid, tid) {
  if (!positions[fid] || !positions[tid]) return null;
  const p1 = positions[fid], p2 = positions[tid];
  const x1 = p1[0], y1 = p1[1], x2 = p2[0], y2 = p2[1];
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.max(1, Math.sqrt(dx * dx + dy * dy));
  const px = -dy / len, py = dx / len;
  const off = Math.max(30, Math.min(80, Math.abs(dx) * 0.25 + 20));
  return 'M ' + x1.toFixed(1) + ',' + y1.toFixed(1) + ' Q ' +
         (mx + px * off).toFixed(1) + ',' + (my + py * off).toFixed(1) + ' ' +
         x2.toFixed(1) + ',' + y2.toFixed(1);
}
function edgeMarkup(r, dir, positions) {
  const fid = r.from_id, tid = r.to_id;
  const d = edgePathD(positions, fid, tid);
  if (!d) return '';
  const t = r.type || 'temporal';
  const mt = MARKER_COLORS[t] ? t : 'default';
  const key = fid + '->' + tid + ':' + t;
  return '<path class="edge edge-' + dir + ' type-' + t + (r._manual ? ' manual' : '') + '" d="' + d + '" ' +
    'marker-end="url(#arr-' + mt + ')" ' +
    'data-from-id="' + fid + '" data-to-id="' + tid + '" data-type="' + t + '" data-dir="' + dir + '" data-key="' + key + '" ' +
    'onclick="selectEdge(' + fid + ',' + tid + ',&quot;' + t + '&quot;,&quot;' + dir + '&quot;)"/>';
}
function buildGraphSVG(layout) {
  const positions = layout.positions, width = layout.width, height = layout.height;
  const sortedTraj = layout.sortedTraj, colW = layout.colW, padX = layout.padX;
  const parts = [];
  sortedTraj.forEach((ti, ci) => {
    const cx = padX + colW / 2 + ci * colW;
    parts.push('<text class="col-label" x="' + cx + '" y="' + (height - 12) + '">traj[' + ti + ']</text>');
  });
  parts.push(markerDefs());
  STATE.forward_edges.forEach(r => parts.push(edgeMarkup(r, 'forward', positions)));
  STATE.backward_edges.forEach(r => parts.push(edgeMarkup(r, 'backward', positions)));
  STATE.beliefs.forEach(b => {
    const p = positions[b.id];
    if (!p) return;
    const x = p[0], y = p[1];
    const src = (b.source || {}).type || 'unknown';
    const conf = Number(b.confidence || 0);
    parts.push('<g class="node source-' + src + (b._manual ? ' manual' : '') + '" data-id="' + b.id +
      '" transform="translate(' + x.toFixed(1) + ',' + y.toFixed(1) + ')" onclick="onNodeClick(' + b.id + ')">' +
      '<rect class="node-pill" x="-36" y="-15" width="72" height="30"/>' +
      '<text class="node-id" x="0" y="-3">#' + String(b.id + 1).padStart(2, '0') + '</text>' +
      '<text class="node-conf" x="0" y="9">' + conf.toFixed(2) + '</text>' +
      '</g>');
  });
  return '<svg class="belief-graph' + (showBackward ? ' show-backward' : '') + '" xmlns="http://www.w3.org/2000/svg" ' +
    'viewBox="0 0 ' + width + ' ' + height + '" width="' + width + '" height="' + height + '">' +
    parts.join('') + '</svg>';
}

/* ---------- stats + legend ---------- */
function renderStatsHTML() {
  const SRC = ['user', 'assistant', 'tool'];
  const counts = { user: 0, assistant: 0, tool: 0 };
  STATE.beliefs.forEach(b => { const t = (b.source || {}).type; if (t in counts) counts[t]++; });
  const total = STATE.beliefs.length;
  let html = '<span><b>' + total + '</b> belief' + (total !== 1 ? 's' : '') + '</span>';
  SRC.forEach(k => { if (counts[k]) html += '<span><span class="stat-dot ' + k + '"></span><b>' + counts[k] + '</b> ' + SOURCE_LABEL[k] + '</span>'; });
  html += '<span><b>' + STATE.forward_edges.length + '</b> forward</span>';
  html += '<span><b>' + STATE.backward_edges.length + '</b> backward</span>';
  html += '<span><b>' + TRAJ.length + '</b> messages</span>';
  return html;
}
function renderLegend() {
  const host = byId('legend-host');
  if (host) host.innerHTML = REL_TYPES.map(t => '<span><span class="swatch sw-' + t + '"></span>' + REL_SHORT[t] + '</span>').join('');
}

/* ---------- selection highlight ---------- */
function clearHighlightClasses() {
  document.querySelectorAll('.node.active, .edge.active').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.node.dimmed, .edge.dimmed').forEach(el => el.classList.remove('dimmed'));
  document.querySelectorAll('.ev.active').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.msg.flash').forEach(el => el.classList.remove('flash'));
}
function clearSelection() {
  currentSelection = null;
  clearHighlightClasses();
  renderDetailEmpty();
}
function highlightNode(id) {
  const connected = new Set([id]);
  unifiedEdges().forEach(e => { if (e.from_id === id) connected.add(e.to_id); if (e.to_id === id) connected.add(e.from_id); });
  document.querySelectorAll('.node').forEach(el => {
    const nid = parseInt(el.dataset.id, 10);
    if (nid === id) el.classList.add('active');
    else if (!connected.has(nid)) el.classList.add('dimmed');
  });
  document.querySelectorAll('.edge').forEach(el => {
    const f = parseInt(el.dataset.fromId, 10), t = parseInt(el.dataset.toId, 10);
    if (f === id || t === id) el.classList.add('active'); else el.classList.add('dimmed');
  });
  let firstEv = null;
  document.querySelectorAll('.ev').forEach(s => {
    const ids = (s.dataset.beliefs || '').split(' ').filter(Boolean);
    if (ids.includes(String(id))) { s.classList.add('active'); if (!firstEv) firstEv = s; }
  });
  return firstEv;
}
function highlightEdge(from, to, type, dir) {
  const key = from + '->' + to + ':' + type;
  document.querySelectorAll('.edge').forEach(el => {
    if (el.dataset.key === key && el.dataset.dir === dir) el.classList.add('active'); else el.classList.add('dimmed');
  });
  document.querySelectorAll('.node').forEach(el => {
    const nid = parseInt(el.dataset.id, 10);
    if (nid === from || nid === to) el.classList.add('active'); else el.classList.add('dimmed');
  });
}

/* ---------- click entry points ---------- */
function onNodeClick(id) {
  if (connectMode) {
    if (pendingSource == null) { pendingSource = id; markPending(id); updateConnectBanner(); return; }
    if (pendingSource === id) { pendingSource = null; clearPending(); updateConnectBanner(); return; }
    openAddEdgeModal(pendingSource, id);
    return;
  }
  selectNode(id);
}
function selectNode(id) {
  if (currentSelection && currentSelection.kind === 'node' && currentSelection.id === id) { clearSelection(); return; }
  clearHighlightClasses();
  currentSelection = { kind: 'node', id: id };
  const firstEv = highlightNode(id);
  const b = beliefById(id);
  if (b) {
    const ti = (b.source || {}).trajectory_index;
    if (typeof ti === 'number' && ti >= 0) {
      const panel = byId('msg-' + ti);
      if (panel) panel.classList.add('flash');
      if (firstEv) firstEv.scrollIntoView({ behavior: 'smooth', block: 'center' });
      else if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }
  renderNodeDetail(id);
}
function selectEdge(from, to, type, dir) {
  if (currentSelection && currentSelection.kind === 'edge' && currentSelection.from === from &&
      currentSelection.to === to && currentSelection.type === type && currentSelection.direction === dir) {
    clearSelection(); return;
  }
  clearHighlightClasses();
  currentSelection = { kind: 'edge', from: from, to: to, type: type, direction: dir };
  highlightEdge(from, to, type, dir);
  renderEdgeDetail(from, to, type, dir);
}
function reselect() {
  if (!currentSelection) { renderDetailEmpty(); return; }
  if (currentSelection.kind === 'node') {
    if (!beliefById(currentSelection.id)) { currentSelection = null; renderDetailEmpty(); return; }
    highlightNode(currentSelection.id);
    renderNodeDetail(currentSelection.id);
  } else {
    const from = currentSelection.from, to = currentSelection.to, type = currentSelection.type, direction = currentSelection.direction;
    const pool = direction === 'forward' ? STATE.forward_edges : STATE.backward_edges;
    if (!pool.find(e => e.from_id === from && e.to_id === to && e.type === type)) { currentSelection = null; renderDetailEmpty(); return; }
    highlightEdge(from, to, type, direction);
    renderEdgeDetail(from, to, type, direction);
  }
}

/* ---------- inspector ---------- */
function renderDetailEmpty() {
  byId('detail').innerHTML = '<div class="detail-empty">点击节点查看 belief，或点击边查看关系。</div>';
}
function renderNodeDetail(id) {
  const b = beliefById(id);
  if (!b) { renderDetailEmpty(); return; }
  const src = b.source || {};
  const badges = [
    '<span class="badge source-' + src.type + '">' + escapeHtml(SOURCE_LABEL[src.type] || src.type || 'unknown') + '</span>',
    '<span class="badge stance-' + b.stance + '">' + escapeHtml(b.stance || '') + '</span>'
  ];
  if (b._manual) badges.push('<span class="badge manual-badge">人工新增</span>');
  const breadcrumb = (typeof src.trajectory_index === 'number' && src.trajectory_index >= 0)
    ? 'traj[' + src.trajectory_index + '] · ' + escapeHtml(src.role || src.type || '?')
    : 'external (no trajectory)';
  const evs = b.evidence || [];
  let evHtml;
  if (evs.length) {
    evHtml = '<h4>证据 / Evidence</h4><ul class="ev-list">' + evs.map((ev, i) => {
      const loc = Number.isInteger(ev.start) && Number.isInteger(ev.end);
      const where = (ev.source && ev.source.trajectory_index != null) ? ('traj[' + ev.source.trajectory_index + ']') : '';
      const tag = (ev.match === 'manual') ? '<span class="ev-tag">manual</span>' : '';
      return '<li><span class="ev-snip">“' + escapeHtml(truncate(ev.text, 80)) + '”</span>' +
        '<span class="ev-pos">' + (loc ? (ev.start + '–' + ev.end) : 'no offset') + ' ' + where + ' ' + tag + '</span>' +
        '<button class="mini-del" title="删除此证据" onclick="deleteEvidence(' + b.id + ',' + i + ')">✕</button></li>';
    }).join('') + '</ul>';
  } else {
    evHtml = '<h4>证据 / Evidence</h4><div class="detail-empty" style="margin:0">暂无证据 · 在左侧拖选原文添加</div>';
  }
  byId('detail').innerHTML =
    '<div class="detail-card">' +
      '<h3>Belief #' + String(b.id + 1).padStart(2, '0') + '</h3>' +
      '<div class="detail-meta">' + badges.join('') + '</div>' +
      '<div class="detail-belief-text">' + escapeHtml(b.belief) + '</div>' +
      '<h4>Source</h4><div class="mono-soft">' + breadcrumb + '</div>' +
      evHtml +
      '<div class="danger-row"><button class="btn-danger" onclick="deleteNode(' + b.id + ')">删除此节点</button>' +
      '<span class="danger-note">将级联删除其所有关联边</span></div>' +
    '</div>';
}
function renderEdgeDetail(from, to, type, dir) {
  const a = beliefById(from), z = beliefById(to);
  if (!a || !z) { renderDetailEmpty(); return; }
  const pool = dir === 'forward' ? STATE.forward_edges : STATE.backward_edges;
  const rel = pool.find(e => e.from_id === from && e.to_id === to && e.type === type) || {};
  const note = rel.note || '';
  const isLegacy = !REL_TYPES.includes(type);
  let opts = REL_TYPES.map(t => '<option value="' + t + '"' + (t === type ? ' selected' : '') + '>' + t + ' · ' + REL_LABEL[t] + '</option>').join('');
  if (isLegacy) opts += '<option value="' + escapeHtml(type) + '" selected>' + escapeHtml(type) + ' (legacy)</option>';
  const dirLabel = dir === 'forward' ? 'forward · 生成方向 earlier → later' : 'backward · 回溯评价 later → earlier';
  const otherDir = dir === 'forward' ? 'backward' : 'forward';
  byId('detail').innerHTML =
    '<div class="detail-card">' +
      '<h3>Relation <span class="dir-tag">' + escapeHtml(dirLabel) + '</span>' + (rel._manual ? ' <span class="badge manual-badge">人工新增</span>' : '') + '</h3>' +
      '<div class="edge-detail-rel">' +
        '<span>#' + String(from + 1).padStart(2, '0') + '</span><span class="arrow">→</span>' +
        '<span class="rel-pill type-' + type + '">' + escapeHtml(type) + '</span>' +
        '<span class="arrow">→</span><span>#' + String(to + 1).padStart(2, '0') + '</span>' +
      '</div>' +
      '<h4>关系类型</h4>' +
      '<select class="edit-select" onchange="changeEdgeType(' + from + ',' + to + ',&quot;' + type + '&quot;,&quot;' + dir + '&quot;,this.value)">' + opts + '</select>' +
      '<h4>方向</h4>' +
      '<button class="btn-ghost sm" onclick="flipEdgeDirection(' + from + ',' + to + ',&quot;' + type + '&quot;,&quot;' + dir + '&quot;)">切换为 ' + otherDir + '</button>' +
      '<h4>备注 / Note</h4>' +
      '<textarea id="edge-note" class="edit-text" rows="2">' + escapeHtml(note) + '</textarea>' +
      '<button class="btn-ghost sm" onclick="saveEdgeNote(' + from + ',' + to + ',&quot;' + type + '&quot;,&quot;' + dir + '&quot;)">保存备注</button>' +
      '<h4>Beliefs</h4>' +
      '<div class="edge-detail-pair">' +
        '<div class="side"><div class="side-label">from · #' + String(from + 1).padStart(2, '0') + ' · ' + escapeHtml(a.stance || '') + '</div><div class="side-text">' + escapeHtml(a.belief) + '</div></div>' +
        '<div class="side"><div class="side-label">to · #' + String(to + 1).padStart(2, '0') + ' · ' + escapeHtml(z.stance || '') + '</div><div class="side-text">' + escapeHtml(z.belief) + '</div></div>' +
      '</div>' +
      '<div class="danger-row"><button class="btn-danger" onclick="deleteEdge(' + from + ',' + to + ',&quot;' + type + '&quot;,&quot;' + dir + '&quot;)">删除此边</button></div>' +
    '</div>';
}

/* ---------- edit operations ---------- */
function deleteNode(id) {
  if (!beliefById(id)) return;
  if (!confirm('删除节点 #' + (id + 1) + ' 及其所有关联边？')) return;
  STATE.beliefs = STATE.beliefs.filter(x => x.id !== id);
  STATE.forward_edges = STATE.forward_edges.filter(e => e.from_id !== id && e.to_id !== id);
  STATE.backward_edges = STATE.backward_edges.filter(e => e.from_id !== id && e.to_id !== id);
  currentSelection = null;
  rerender();
  toast('已删除节点 #' + (id + 1));
}
function deleteEdge(from, to, type, dir) {
  const pool = dir === 'forward' ? 'forward_edges' : 'backward_edges';
  STATE[pool] = STATE[pool].filter(e => !(e.from_id === from && e.to_id === to && e.type === type));
  currentSelection = null;
  rerender();
  toast('已删除边');
}
function changeEdgeType(from, to, oldType, dir, newType) {
  if (newType === oldType) return;
  const pool = dir === 'forward' ? STATE.forward_edges : STATE.backward_edges;
  const e = pool.find(x => x.from_id === from && x.to_id === to && x.type === oldType);
  if (!e) return;
  if (pool.some(x => x !== e && x.from_id === from && x.to_id === to && x.type === newType)) { toast('已存在同方向同类型的边'); reselect(); return; }
  e.type = newType;
  currentSelection = { kind: 'edge', from: from, to: to, type: newType, direction: dir };
  rerender();
  toast('关系类型已改为 ' + newType);
}
function flipEdgeDirection(from, to, type, dir) {
  const fromPool = dir === 'forward' ? 'forward_edges' : 'backward_edges';
  const toPool = dir === 'forward' ? 'backward_edges' : 'forward_edges';
  const newDir = dir === 'forward' ? 'backward' : 'forward';
  const e = STATE[fromPool].find(x => x.from_id === from && x.to_id === to && x.type === type);
  if (!e) return;
  STATE[fromPool] = STATE[fromPool].filter(x => x !== e);
  if (!STATE[toPool].some(x => x.from_id === from && x.to_id === to && x.type === type)) STATE[toPool].push(e);
  currentSelection = { kind: 'edge', from: from, to: to, type: type, direction: newDir };
  if (newDir === 'backward' && !showBackward) toggleBackward();
  rerender();
  toast('方向已切换为 ' + newDir);
}
function saveEdgeNote(from, to, type, dir) {
  const pool = dir === 'forward' ? STATE.forward_edges : STATE.backward_edges;
  const e = pool.find(x => x.from_id === from && x.to_id === to && x.type === type);
  if (!e) return;
  e.note = val('edge-note');
  toast('备注已保存');
}
function deleteEvidence(beliefId, idx) {
  const b = beliefById(beliefId);
  if (!b || !b.evidence) return;
  const ev = b.evidence[idx];
  b.evidence.splice(idx, 1);
  if (ev && Array.isArray(b.supporting_excerpts)) {
    const stillUsed = (b.evidence || []).some(x => x.text === ev.text);
    if (!stillUsed) b.supporting_excerpts = b.supporting_excerpts.filter(t => t !== ev.text);
  }
  rerender();
  toast('已删除证据');
}
function attachEvidence(node, ti, start, end, text) {
  const m = TRAJ[ti] || {};
  node.evidence = node.evidence || [];
  node.evidence.push({
    text: text, start: start, end: end, match: 'manual', via: 'manual_annotation',
    source: { type: (m.role === 'function' ? 'tool' : m.role), role: m.role, turn_index: (m.turn_index != null ? m.turn_index : ti), trajectory_index: ti }
  });
  node.supporting_excerpts = node.supporting_excerpts || [];
  if (!node.supporting_excerpts.includes(text)) node.supporting_excerpts.push(text);
}
function addNodeCommit(f) {
  const id = maxId() + 1;
  const m = TRAJ[f.ti] || {};
  const node = {
    id: id, belief: f.belief, stance: f.stance,
    event_time: null, time_text: null,
    source: { type: f.srctype, role: f.srctype, trajectory_index: f.ti, turn_index: (m.turn_index != null ? m.turn_index : f.ti) },
    evidence: [], supporting_excerpts: [],
    confidence: 1.0, initial_confidence: 1.0, confidence_history: [],
    _manual: true
  };
  STATE.beliefs.push(node);
  if (f.evidence) attachEvidence(node, f.evidence.ti, f.evidence.start, f.evidence.end, f.evidence.text);
  currentSelection = { kind: 'node', id: id };
  rerender();
  toast('已新增节点 #' + (id + 1));
}
function addEdgeCommit(from, to, dir, type, note) {
  if (from === to) { toast('不能连接节点自身'); return false; }
  const pool = dir === 'forward' ? STATE.forward_edges : STATE.backward_edges;
  if (pool.some(e => e.from_id === from && e.to_id === to && e.type === type)) { toast('该边已存在'); return false; }
  pool.push({ from_id: from, to_id: to, type: type, note: note || '', _manual: true });
  pendingSource = null; clearPending(); updateConnectBanner();
  if (dir === 'backward' && !showBackward) toggleBackward();
  currentSelection = { kind: 'edge', from: from, to: to, type: type, direction: dir };
  rerender();
  toast('已新增边 #' + (from + 1) + ' → #' + (to + 1));
  return true;
}
function addEvidenceCommit(beliefId, ti, start, end, text) {
  const b = beliefById(beliefId);
  if (!b) return false;
  if (b.source && b.source.trajectory_index != null && b.source.trajectory_index !== ti) {
    if (!confirm('证据所在消息 traj[' + ti + '] 与节点来源 traj[' + b.source.trajectory_index + '] 不一致，仍要添加吗？')) return false;
  }
  attachEvidence(b, ti, start, end, text);
  currentSelection = { kind: 'node', id: beliefId };
  window.getSelection().removeAllRanges();
  rerender();
  toast('已为 #' + (beliefId + 1) + ' 添加证据');
  return true;
}

/* ---------- connection mode ---------- */
function toggleConnectMode() {
  connectMode = !connectMode;
  pendingSource = null; clearPending();
  const btn = document.querySelector('.ct-connect');
  if (btn) btn.classList.toggle('active', connectMode);
  if (connectMode) clearSelection();
  updateConnectBanner();
}
function markPending(id) {
  clearPending();
  const el = document.querySelector('.node[data-id="' + id + '"]');
  if (el) el.classList.add('pending');
}
function clearPending() {
  document.querySelectorAll('.node.pending').forEach(el => el.classList.remove('pending'));
}
function updateConnectBanner() {
  const el = byId('connect-banner');
  if (!el) return;
  if (!connectMode) { el.textContent = ''; el.classList.remove('show'); return; }
  el.classList.add('show');
  el.textContent = (pendingSource == null) ? '连线模式：点击源节点'
    : '已选源 #' + (pendingSource + 1) + ' · 点击目标节点（Esc 取消）';
}

/* ---------- modal ---------- */
let _modalConfirm = null, _modalClose = null;
function openModal(title, bodyHTML, onConfirm, onClose) {
  _modalConfirm = onConfirm || null; _modalClose = onClose || null;
  byId('modal-title').textContent = title;
  byId('modal-body').innerHTML = bodyHTML;
  byId('modal-overlay').classList.add('open');
  const f = byId('modal-body').querySelector('textarea,input,select');
  if (f) setTimeout(() => f.focus(), 30);
}
function modalConfirm() {
  if (_modalConfirm && _modalConfirm() === false) return;
  _modalClose = null;
  hideModal();
}
function cancelModal() {
  const oc = _modalClose;
  hideModal();
  if (oc) oc();
}
function hideModal() {
  _modalConfirm = null; _modalClose = null;
  byId('modal-overlay').classList.remove('open');
  byId('modal-body').innerHTML = '';
}
function openAddNodeModal(prefill) {
  prefill = prefill || {};
  const tiOpts = TRAJ.map((m, i) => '<option value="' + i + '"' + (prefill.ti === i ? ' selected' : '') + '>' + i + ' · ' + escapeHtml(m.role || '?') + '</option>').join('');
  const srcOpts = ['user', 'assistant', 'tool'].map(s => '<option value="' + s + '"' + (prefill.srctype === s ? ' selected' : '') + '>' + s + '</option>').join('');
  const stOpts = STANCES.map(s => '<option value="' + s + '">' + s + '</option>').join('');
  const evNote = prefill.evidence
    ? '<div class="form-note">将自动绑定证据：“' + escapeHtml(truncate(prefill.evidence.text, 60)) + '” · traj[' + prefill.evidence.ti + '] ' + prefill.evidence.start + '–' + prefill.evidence.end + '</div>'
    : '';
  const body =
    '<label class="fld">belief 文本<textarea id="f-belief" rows="3" placeholder="用一句话陈述这个 belief">' + escapeHtml(prefill.belief || '') + '</textarea></label>' +
    '<div class="form-row">' +
      '<label class="fld">来源类型<select id="f-srctype">' + srcOpts + '</select></label>' +
      '<label class="fld">trajectory_index<select id="f-ti">' + tiOpts + '</select></label>' +
      '<label class="fld">stance<select id="f-stance">' + stOpts + '</select></label>' +
    '</div>' + evNote;
  openModal('增加节点', body, () => {
    const belief = val('f-belief').trim();
    if (!belief) { toast('请填写 belief 文本'); return false; }
    addNodeCommit({ belief: belief, srctype: val('f-srctype'), ti: parseInt(val('f-ti'), 10), stance: val('f-stance'), evidence: prefill.evidence });
  });
}
function openAddEdgeModal(fromId, toId) {
  const a = beliefById(fromId), z = beliefById(toId);
  if (!a || !z) return;
  const ord = b => { const ti = (b.source || {}).trajectory_index; return (ti == null ? -1 : ti) * 100000 + b.id; };
  const guess = ord(a) <= ord(z) ? 'forward' : 'backward';
  const typeOpts = REL_TYPES.map(t => '<option value="' + t + '">' + t + ' · ' + REL_LABEL[t] + '</option>').join('');
  const body =
    '<div class="edge-preview">#' + (fromId + 1) + ' <span class="arrow">→</span> #' + (toId + 1) + '</div>' +
    '<div class="form-pair">' +
      '<div class="mini-belief">' + escapeHtml(truncate(a.belief, 90)) + '</div>' +
      '<div class="mini-belief">' + escapeHtml(truncate(z.belief, 90)) + '</div>' +
    '</div>' +
    '<label class="fld">关系类型<select id="f-etype">' + typeOpts + '</select></label>' +
    '<label class="fld">方向<select id="f-edir">' +
      '<option value="forward"' + (guess === 'forward' ? ' selected' : '') + '>forward · 生成方向 (earlier → later)</option>' +
      '<option value="backward"' + (guess === 'backward' ? ' selected' : '') + '>backward · 回溯评价 (later → earlier)</option>' +
    '</select></label>' +
    '<label class="fld">备注（可选）<input id="f-enote" type="text" placeholder="为什么建立这条关系"></label>';
  openModal('增加边', body,
    () => addEdgeCommit(fromId, toId, val('f-edir'), val('f-etype'), val('f-enote')),
    () => { pendingSource = null; clearPending(); updateConnectBanner(); });
}
function openAttributeEvidenceModal(ti, start, end, text) {
  const m = TRAJ[ti] || {};
  const nodeOpts = STATE.beliefs.slice().sort((a, b) => a.id - b.id)
    .map(b => '<option value="' + b.id + '">#' + String(b.id + 1).padStart(2, '0') + ' · ' + escapeHtml(truncate(b.belief, 70)) + '</option>').join('');
  const body =
    '<div class="form-note">选区：traj[' + ti + '] · ' + escapeHtml(m.role || '?') + ' · 字符 ' + start + '–' + end + '</div>' +
    '<div class="sel-preview">“' + escapeHtml(text) + '”</div>' +
    '<label class="fld">归属到节点<select id="f-evnode"><option value="__new__">＋ 新建节点并归属…</option>' + nodeOpts + '</select></label>';
  openModal('添加证据', body, () => {
    const v = val('f-evnode');
    if (v === '__new__') {
      window.getSelection().removeAllRanges();
      const srctype = (m.role === 'function' ? 'tool' : (['user', 'assistant', 'tool'].includes(m.role) ? m.role : 'assistant'));
      const pre = { ti: ti, belief: text, srctype: srctype, evidence: { ti: ti, start: start, end: end, text: text } };
      setTimeout(() => openAddNodeModal(pre), 0);
      return true;
    }
    return addEvidenceCommit(parseInt(v, 10), ti, start, end, text);
  }, () => { window.getSelection().removeAllRanges(); });
}

/* ---------- box-select evidence (left panel) ---------- */
function ancestorMsgBody(node) {
  let el = (node && node.nodeType === 3) ? node.parentElement : node;
  while (el && el !== document.body) {
    if (el.classList && el.classList.contains('msg-body')) return el;
    el = el.parentElement;
  }
  return null;
}
function caretOffset(container, node, offset) {
  const r = document.createRange();
  r.selectNodeContents(container);
  try { r.setEnd(node, offset); } catch (e) { return -1; }
  return r.toString().length;
}
document.addEventListener('mouseup', () => {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) return;
  const range = sel.getRangeAt(0);
  if (!sel.toString()) return;
  const sBody = ancestorMsgBody(range.startContainer);
  const eBody = ancestorMsgBody(range.endContainer);
  if (!sBody && !eBody) return;
  if (!sBody || !eBody || sBody !== eBody) { toast('选区需在单条消息内部，已忽略'); return; }
  const ti = parseInt(sBody.dataset.ti, 10);
  const start = caretOffset(sBody, range.startContainer, range.startOffset);
  const end = caretOffset(sBody, range.endContainer, range.endOffset);
  if (start < 0 || end < 0 || end <= start) return;
  const content = TRAJ[ti].content || '';
  const slice = content.slice(start, end);
  if (!slice) return;
  openAttributeEvidenceModal(ti, start, end, slice);
});

/* ---------- evidence span click -> jump to node ---------- */
document.addEventListener('click', (e) => {
  const ev = e.target.closest && e.target.closest('.ev');
  if (ev && !window.getSelection().toString()) {
    const ids = (ev.dataset.beliefs || '').split(' ').filter(Boolean);
    if (ids.length) selectNode(parseInt(ids[0], 10));
  }
});

/* ---------- keyboard ---------- */
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (byId('modal-overlay').classList.contains('open')) { cancelModal(); return; }
  if (connectMode) {
    if (pendingSource != null) { pendingSource = null; clearPending(); updateConnectBanner(); }
    else toggleConnectMode();
    return;
  }
  clearSelection();
});

/* ---------- backward edge toggle ---------- */
function toggleBackward() {
  showBackward = !showBackward;
  const svg = document.querySelector('.belief-graph');
  if (svg) svg.classList.toggle('show-backward', showBackward);
  const btn = document.querySelector('.toggle-bwd');
  if (btn) { btn.classList.toggle('active', showBackward); btn.textContent = showBackward ? '✓ backward edges' : '+ backward edges'; }
  if (!showBackward && currentSelection && currentSelection.kind === 'edge' && currentSelection.direction === 'backward') clearSelection();
}

/* ---------- export / reset ---------- */
function cleanObj(o) { const r = {}; for (const k in o) { if (!k.startsWith('_')) r[k] = o[k]; } return r; }
function exportJSON() {
  const out = deepCopy(RAW);
  out.all_beliefs = STATE.beliefs.map(cleanObj);
  out.forward_relations = STATE.forward_edges.map(cleanObj);
  out.backward_relations = STATE.backward_edges.map(cleanObj);
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const d = new Date();
  const pad = n => String(n).padStart(2, '0');
  const ts = d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + '_' + pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds());
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'result_annotated_' + ts + '.json';
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  toast('已导出 ' + a.download);
}
function resetAll() {
  if (!confirm('放弃所有人工修改，恢复到原始 LLM 标注？')) return;
  STATE = initState();
  currentSelection = null; pendingSource = null;
  if (connectMode) toggleConnectMode();
  rerender();
  toast('已重置为原始标注');
}

/* ---------- toast ---------- */
let _toastTimer = null;
function toast(msg) {
  const t = byId('toast');
  if (!t) return;
  t.textContent = msg; t.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.classList.remove('show'), 2400);
}

/* ---------- render loop ---------- */
function rerender() {
  const layout = computeLayout(STATE.beliefs);
  byId('graph-host').innerHTML = buildGraphSVG(layout);
  byId('traj-host').innerHTML = buildTrajectoryHTML();
  byId('stats-host').innerHTML = renderStatsHTML();
  reselect();
}
function init() {
  const btn = document.querySelector('.toggle-bwd');
  if (btn) { btn.classList.toggle('active', showBackward); btn.textContent = showBackward ? '✓ backward edges' : '+ backward edges'; }
  renderLegend();
  rerender();
}
document.addEventListener('DOMContentLoaded', init);
"""


# =============================================================
# HTML rendering
# =============================================================

def render_message_panel(
    traj_idx: int, msg: Dict[str, Any],
    beliefs_for_msg: List[Tuple[int, Dict[str, Any], List[Tuple[int, int]]]],
    # beliefs_for_msg = list of (belief_global_id, belief, list_of_(start,end)_intervals_within_msg)
) -> str:
    role = msg.get('role') or '?'
    content = msg.get('content', '') or ''
    role_label = ROLE_LABEL.get(role, role)
    role_class = re.sub(r'[^a-z]', '', role.lower()) or 'unknown'

    intervals: List[Tuple[int, int, int]] = []
    for gid, _b, ranges in beliefs_for_msg:
        for (s, e) in ranges:
            intervals.append((s, e, gid))

    pieces = slice_with_marks(content, intervals)
    body_parts: List[str] = []
    for p in pieces:
        text = html_lib.escape(p['text'])
        if not p['beliefs']:
            body_parts.append(text)
        else:
            ids = ' '.join(str(b) for b in p['beliefs'])
            body_parts.append(f'<span class="ev" data-beliefs="{ids}">{text}</span>')
    body_html = ''.join(body_parts)

    summary_cls = 'msg-belief-count' + ('' if beliefs_for_msg else ' empty')
    summary_text = (f'{len(beliefs_for_msg)} belief'
                    + ('s' if len(beliefs_for_msg) != 1 else '')) if beliefs_for_msg else 'no beliefs'

    return (
        f'<article class="msg msg-{role_class}" id="msg-{traj_idx}">'
        f'<header class="msg-head">'
        f'<span class="msg-role role-{role_class}">{html_lib.escape(role_label)}</span>'
        f'<span class="msg-idx">trajectory_index <b>{traj_idx}</b></span>'
        f'<span class="{summary_cls}">{summary_text}</span>'
        f'</header>'
        f'<pre class="msg-body">{body_html}</pre>'
        f'</article>'
    )


def render_graph_svg(
    beliefs: List[Dict[str, Any]],
    forward_relations: List[Dict[str, Any]],
    backward_relations: List[Dict[str, Any]],
    positions: Dict[int, Tuple[float, float]],
    width: int, height: int,
    sorted_traj: List[int], col_w: int, pad_x: int,
) -> str:
    parts: List[str] = []

    # Column labels at the bottom (single tick per traj message).
    for ci, ti in enumerate(sorted_traj):
        cx = pad_x + col_w / 2 + ci * col_w
        parts.append(f'<text class="col-label" x="{cx}" y="{height - 12}">traj[{ti}]</text>')

    # Arrow markers (one per edge type so color follows)
    parts.append(
        '<defs>'
        '<marker id="arr-confirms" viewBox="0 0 10 10" refX="9" refY="5" '
        ' markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '  <path d="M0,0 L10,5 L0,10 z" fill="#1f6a35"/>'
        '</marker>'
        '<marker id="arr-contradicts" viewBox="0 0 10 10" refX="9" refY="5" '
        ' markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '  <path d="M0,0 L10,5 L0,10 z" fill="#c84a3e"/>'
        '</marker>'
        '<marker id="arr-extends" viewBox="0 0 10 10" refX="9" refY="5" '
        ' markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '  <path d="M0,0 L10,5 L0,10 z" fill="#b3500e"/>'
        '</marker>'
        '<marker id="arr-informs" viewBox="0 0 10 10" refX="9" refY="5" '
        ' markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        '  <path d="M0,0 L10,5 L0,10 z" fill="#6b89b8"/>'
        '</marker>'
        '</defs>'
    )

    def _edge_path(fid: int, tid: int) -> Optional[str]:
        if fid not in positions or tid not in positions:
            return None
        x1, y1 = positions[fid]
        x2, y2 = positions[tid]
        mx = (x1 + x2) / 2; my = (y1 + y2) / 2
        dx = x2 - x1; dy = y2 - y1
        length = max(1.0, (dx*dx + dy*dy) ** 0.5)
        px = -dy / length; py = dx / length
        offset = max(30, min(80, abs(dx) * 0.25 + 20))
        cx = mx + px * offset; cy = my + py * offset
        return f'M {x1:.1f},{y1:.1f} Q {cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}'

    # Forward edges first (so backward edges sit on top — they're the louder
    # ones epistemically and should be more visible).
    for r in forward_relations:
        fid = r['from_id']; tid = r['to_id']
        path = _edge_path(fid, tid)
        if not path: continue
        rtype = r.get('type', 'informs')
        edge_key = f'{fid}->{tid}:{rtype}'
        parts.append(
            f'<path class="edge edge-forward type-{rtype}" d="{path}" '
            f'marker-end="url(#arr-{rtype})" '
            f'data-from-id="{fid}" data-to-id="{tid}" data-type="{rtype}" data-dir="forward" '
            f'data-key="{edge_key}" '
            f'onclick="selectEdge({fid},{tid},\'{rtype}\',\'forward\')"/>'
        )

    # Backward edges
    for r in backward_relations:
        fid = r['from_id']; tid = r['to_id']
        path = _edge_path(fid, tid)
        if not path: continue
        rtype = r.get('type', 'extends')
        edge_key = f'{fid}->{tid}:{rtype}'
        parts.append(
            f'<path class="edge edge-backward type-{rtype}" d="{path}" '
            f'marker-end="url(#arr-{rtype})" '
            f'data-from-id="{fid}" data-to-id="{tid}" data-type="{rtype}" data-dir="backward" '
            f'data-key="{edge_key}" '
            f'onclick="selectEdge({fid},{tid},\'{rtype}\',\'backward\')"/>'
        )

    # Nodes
    for b in beliefs:
        bid = b['id']
        if bid not in positions:
            continue
        x, y = positions[bid]
        src_type = (b.get('source') or {}).get('type', 'unknown')
        conf = float(b.get('confidence', 0.0))
        parts.append(
            f'<g class="node source-{src_type}" data-id="{bid}" transform="translate({x:.1f},{y:.1f})" '
            f' onclick="selectNode({bid})">'
            f'<rect class="node-pill" x="-36" y="-15" width="72" height="30"/>'
            f'<text class="node-id" x="0" y="-3">#{bid+1:02d}</text>'
            f'<text class="node-conf" x="0" y="9">{conf:.2f}</text>'
            f'</g>'
        )

    return (
        f'<svg class="belief-graph" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        + ''.join(parts) +
        '</svg>'
    )


def render_html(data: Dict[str, Any], src_path: str) -> str:
    model = data.get('model', '') or ''
    prompt_name = data.get('prompt_name', 'construct_beliefs') or 'construct_beliefs'

    # Embed the COMPLETE original document untouched. The annotation frontend
    # keeps a mutable copy in JS (STATE) and, on export, overwrites only
    # all_beliefs / forward_relations / backward_relations — every other field
    # is passed through verbatim (per the frozen spec). Guard against premature
    # </script> termination and JS line-separator chars inside string content.
    raw_json = json.dumps(data, ensure_ascii=False)
    raw_json = (raw_json.replace('</', '<\\/')
                        .replace('\u2028', '\\u2028')
                        .replace('\u2029', '\\u2029'))

    return (
        '<!doctype html>\n<html lang="en"><head>'
        '<meta charset="utf-8">'
        f'<title>Belief Graph · {html_lib.escape(prompt_name)} · annotate</title>'
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'
        f'<style>{CSS}</style>'
        '</head><body>'
        '<header class="app-header">'
        f'<h1>Belief Graph <span class="sep">·</span> <span class="tag">{html_lib.escape(prompt_name)}</span> <span class="mode-badge">annotate</span></h1>'
        '<div class="file-meta">'
        f'<span>source · <b>{html_lib.escape(Path(src_path).name)}</b></span>'
        f'<span>model · <b>{html_lib.escape(model)}</b></span>'
        '</div></header>'
        '<div class="stats" id="stats-host"></div>'
        '<div class="anno-toolbar">'
        '  <button class="ct ct-add" onclick="openAddNodeModal()">＋ 节点</button>'
        '  <button class="ct ct-connect" onclick="toggleConnectMode()">🔗 连线模式</button>'
        '  <button class="ct ct-reset" onclick="resetAll()">↺ 重置</button>'
        '  <span id="connect-banner" class="connect-banner"></span>'
        '  <span class="anno-spacer"></span>'
        '  <button class="ct ct-export" onclick="exportJSON()">💾 导出 JSON</button>'
        '</div>'
        '<div class="layout">'
        '<div class="left-panel"><h2 class="section-title">Conversation trajectory <span class="sel-hint">· 拖选原文可添加证据</span></h2><div id="traj-host"></div></div>'
        '<div class="right-panel">'
        '<div class="graph-wrap">'
        '  <div class="graph-toolbar"><button class="toggle-bwd" onclick="toggleBackward()">backward edges</button></div>'
        '  <div class="legend" id="legend-host"></div>'
        '  <div id="graph-host"></div>'
        '</div>'
        '<div class="detail-wrap"><h2 class="section-title">Inspector</h2><div id="detail"></div></div>'
        '</div></div>'
        '<div class="hint">'
        '点击节点查看 belief · 点击边查看/编辑关系 · 拖选左侧原文为节点添加证据 · '
        '开「连线模式」后依次点两个节点连边 · 虚线边框节点 = 人工新增 · '
        '<kbd>Esc</kbd> 取消 / 清除。'
        '</div>'
        '<div id="modal-overlay" class="modal-overlay" onclick="if(event.target===this)cancelModal()">'
        '  <div class="modal" role="dialog" aria-modal="true">'
        '    <div class="modal-head"><h3 id="modal-title"></h3><button class="modal-x" onclick="cancelModal()">✕</button></div>'
        '    <div id="modal-body" class="modal-body"></div>'
        '    <div class="modal-foot"><button class="btn-ghost" onclick="cancelModal()">取消</button><button class="btn-primary" onclick="modalConfirm()">确定</button></div>'
        '  </div>'
        '</div>'
        '<div id="toast"></div>'
        '<script>'
        f'window.RAW_DATA = {raw_json};'
        f'{JS}'
        '</script>'
        '</body></html>'
    )

def main():
    parser = argparse.ArgumentParser(description='Graph visualizer for construct_beliefs result.json')
    parser.add_argument('input', help='Path to result.json')
    parser.add_argument('--output', '-o', default=None, help='Output HTML path')
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f'[error] file not found: {in_path}', file=sys.stderr)
        sys.exit(1)
    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    html = render_html(data, str(in_path))
    out_path = Path(args.output) if args.output else in_path.parent / f'graph_{in_path.stem}.html'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'[ok] wrote {out_path}  ({out_path.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
