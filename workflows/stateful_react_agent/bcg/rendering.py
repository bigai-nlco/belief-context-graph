"""Shared graph rendering and the exact character budget used by selectors."""

from __future__ import annotations

import math
import os
import re
from html import escape
from typing import Any

from .client import GraphError

RENDER_STYLES = ("v0", "split", "split_confidence")
SPLIT_STYLES = frozenset({"split", "split_confidence"})

GRAPH_OVERHEAD = len("<belief_graph>\n## Beliefs\n\n## Relations\n</belief_graph>")

SPLIT_ACTIONS_HEADER = "## Completed actions - already done, do not repeat"
# Worst case: the actions section is present. Selectors budget against this,
# so a strict split render can never exceed what they computed. With no action
# node the block is shorter by len(SPLIT_ACTIONS_HEADER) + 2.
SPLIT_GRAPH_OVERHEAD = len(
    "<belief_graph>\n" + SPLIT_ACTIONS_HEADER + "\n\n## Findings\n\n## Relations\n</belief_graph>"
)


def graph_overhead(style: str = "v0") -> int:
    return SPLIT_GRAPH_OVERHEAD if style in SPLIT_STYLES else GRAPH_OVERHEAD


def node_line(node: dict[str, Any]) -> str:
    content = node.get("belief") or node.get("decision")
    if not isinstance(content, str) or not content:
        raise GraphError("selected_node_has_no_text")
    confidence = node.get("confidence")
    suffix = (
        f" (confidence {confidence})"
        if isinstance(confidence, (float, int))
        and not isinstance(confidence, bool)
        and math.isfinite(confidence)
        else ""
    )
    return f"- [B{node['id']}] {escape(content, quote=False)}{suffix}"


# Node classification is a presentation decision made here, not on the graph
# server. Only nodes that record ISSUING a call are actions. A node describing
# what a tool returned is a finding: it carries evidence the model must read.
_CALL_RE = re.compile(
    r"(is using (?P<tool1>\w+) to (search|fetch|call|query|look)"
    r"|is calling the (?P<tool2>\w+) tool"
    r"|tool call to (?P<tool3>\w+))",
    re.I,
)
# Which tool a result node came from, used only to label a dead end.
_TOOL_NAME_RE = re.compile(r"\bthe (?P<tool>\w+) tool\b", re.I)
# The query inside an action node: ... to search for \"foo bar\".
_QUERY_RE = re.compile(
    r'(?:search(?:ing)? for|query|fetch(?:ing)?)\s*[:\s]\s*\\?"(.+?)\\?"\s*\.?\s*$',
    re.I,
)
# A result that carried no usable content; recorded as a dead end.
_BLOCKED_RE = re.compile(
    r"\[BLOCKED\]|source is unavailable|blocked for this task|unavailable for this task",
    re.I,
)


def classify_node(node: dict[str, Any]) -> dict[str, Any]:
    """Return how the split style presents one node (action/blocked/finding)."""
    content = node.get("belief") or node.get("decision")
    if not isinstance(content, str) or not content:
        raise GraphError("selected_node_has_no_text")
    call = _CALL_RE.search(content)
    blocked = bool(_BLOCKED_RE.search(content))
    tool = node.get("tool_name")
    if not tool and call:
        tool = next((call.group(g) for g in ("tool1", "tool2", "tool3") if call.group(g)), None)
    if not tool and blocked:
        named = _TOOL_NAME_RE.search(content)
        tool = named.group("tool") if named else None
    query = None
    if call:
        found = _QUERY_RE.search(content)
        if found:
            query = found.group(1)
    source = node.get("source")
    turn_id = source.get("turn_id") if isinstance(source, dict) else None
    return {
        "is_action": bool(call) or blocked,
        "blocked": blocked,
        "tool": str(tool) if tool else None,
        "query": query,
        "turn_id": turn_id if type(turn_id) is int else None,
    }


def action_line(node: dict[str, Any], info: dict[str, Any] | None = None) -> str:
    """Compact action line: template prose and confidence are dropped."""
    info = info or classify_node(node)
    tool = escape(info["tool"] or "tool", quote=False)
    if info["query"]:
        query = escape(info["query"], quote=False)
        line = f'- [B{node["id"]}] {tool} "{query}"'
        if info["blocked"]:
            line += " (blocked source)"
        return line
    if info["blocked"]:
        return f"- [B{node['id']}] {tool} returned a blocked source"
    content = node.get("belief") or node.get("decision")
    if not isinstance(content, str) or not content:
        raise GraphError("selected_node_has_no_text")
    return f"- [B{node['id']}] {escape(content, quote=False)}"


# split_confidence: the number is replaced by a three-tier label. The tier is
# read from the final propagated confidence only (source/stance prior, plus
# independent evidence and relation factors), so it stays consistent with
# whatever prior table the construction server uses. Ordered strongest first.
CONFIDENCE_LABELS = ("high confidence", "medium confidence", "low confidence")
# Inclusive lower bounds. With the default priors: scholarly/official sources
# stated as confirmed land in high; single general-web sources and the agent's
# own inferences in medium; possible/doubtful claims and failed results in low.
HIGH_CONFIDENCE_THRESHOLD = 0.70
LOW_CONFIDENCE_THRESHOLD = 0.50


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def confidence_thresholds() -> tuple[float, float]:
    """(high, low) bounds; BCG_CONFIDENCE_HIGH / BCG_CONFIDENCE_LOW override."""
    def read(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number in [0, 1]") from exc
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be a number in [0, 1]")
        return value

    high = read("BCG_CONFIDENCE_HIGH", HIGH_CONFIDENCE_THRESHOLD)
    low = read("BCG_CONFIDENCE_LOW", LOW_CONFIDENCE_THRESHOLD)
    if low > high:
        raise ValueError("BCG_CONFIDENCE_LOW must not exceed BCG_CONFIDENCE_HIGH")
    return high, low


def confidence_label(node: dict[str, Any]) -> str | None:
    """high/medium/low confidence for one finding, or None without a number."""
    confidence = _number(node.get("confidence"))
    if confidence is None:
        return None
    high, low = confidence_thresholds()
    if confidence >= high:
        return "high confidence"
    if confidence >= low:
        return "medium confidence"
    return "low confidence"


def finding_line(node: dict[str, Any], style: str = "v0") -> str:
    """A Findings / Beliefs line. Only split_confidence changes the suffix."""
    if style != "split_confidence":
        return node_line(node)
    content = node.get("belief") or node.get("decision")
    if not isinstance(content, str) or not content:
        raise GraphError("selected_node_has_no_text")
    label = confidence_label(node)
    suffix = f" ({label})" if label else ""
    return f"- [B{node['id']}] {escape(content, quote=False)}{suffix}"


def styled_node_line(node: dict[str, Any], style: str = "v0") -> str:
    """The exact line a node occupies in the rendered block of this style."""
    if style in SPLIT_STYLES and classify_node(node)["is_action"]:
        return action_line(node)
    return finding_line(node, style)


def relation_line(edge: dict[str, Any]) -> str:
    line = f"- [B{edge['from_id']}] {edge['type']} [B{edge['to_id']}]"
    if isinstance(edge.get("note"), str) and edge["note"]:
        line += ": " + escape(edge["note"], quote=False)
    return line


def _nodes(snapshot: dict[str, Any]) -> dict[int, dict[str, Any]]:
    nodes = snapshot.get("nodes")
    if nodes is None:
        beliefs, decisions = snapshot.get("beliefs", []), snapshot.get("decisions", [])
        if not isinstance(beliefs, list) or not isinstance(decisions, list):
            raise GraphError("invalid_graph_nodes")
        nodes = beliefs + decisions
    if not isinstance(nodes, list):
        raise GraphError("invalid_graph_nodes")
    return {
        node["id"]: node for node in nodes if isinstance(node, dict) and type(node.get("id")) is int
    }


def full_graph_selection(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Include the entire acknowledged snapshot, without ranking or filtering."""
    relations = snapshot.get("relations", [])
    if not isinstance(relations, list) or any(
        not isinstance(edge, dict) or type(edge.get("id")) is not int for edge in relations
    ):
        raise GraphError("invalid_graph_relations")
    return {
        "strategy": "all",
        "problem_id": snapshot.get("problem_id"),
        "node_ids": sorted(_nodes(snapshot)),
        "relation_ids": sorted(edge["id"] for edge in relations),
    }


def _finding_order(ids: list[int], selection: dict[str, Any]) -> list[int]:
    """Most relevant first: PCST prizes when present, else the selector's order."""
    order = list(dict.fromkeys(ids))
    prizes = selection.get("selected_prizes")
    if isinstance(prizes, dict) and prizes:
        def prize(node_id: int) -> float:
            value = prizes.get(str(node_id), prizes.get(node_id, 0.0))
            return float(value) if isinstance(value, (int, float)) else 0.0

        order.sort(key=lambda i: (-prize(i), i))
    return order


def render_graph(
    snapshot: dict[str, Any],
    selection: dict[str, Any],
    budget: int | None,
    *,
    strict: bool = False,
    style: str = "v0",
) -> tuple[str, list[int], list[int]]:
    """Render real nodes/edges; None disables the cap for full-graph injection."""
    if style not in RENDER_STYLES:
        raise GraphError("invalid_render_style")
    ids, relation_ids = selection.get("node_ids"), selection.get("relation_ids")
    if (
        not isinstance(ids, list)
        or not isinstance(relation_ids, list)
        or not all(type(i) is int for i in ids + relation_ids)
    ):
        raise GraphError("invalid_selection_ids")
    nodes = _nodes(snapshot)
    selected_nodes = set(ids)
    if not selected_nodes <= nodes.keys():
        raise GraphError("selected_node_missing")
    relations = snapshot.get("relations", [])
    if not isinstance(relations, list):
        raise GraphError("invalid_graph_relations")
    edges = {r["id"]: r for r in relations if isinstance(r, dict) and type(r.get("id")) is int}
    if not set(relation_ids) <= edges.keys():
        raise GraphError("selected_relation_missing")
    for relation_id in relation_ids:
        edge = edges[relation_id]
        if (
            edge.get("from_id") not in selected_nodes
            or edge.get("to_id") not in selected_nodes
            or edge.get("type") not in {"depends_on", "supplements", "contradicts"}
        ):
            raise GraphError("invalid_selected_relation")
    node_lines: dict[int, str] = {}
    for node_id in dict.fromkeys(ids):
        node_lines[node_id] = node_line(nodes[node_id])

    def relations_for(kept: list[int]) -> list[int]:
        keeping = set(kept)
        return [
            rid
            for rid in dict.fromkeys(relation_ids)
            if edges[rid]["from_id"] in keeping and edges[rid]["to_id"] in keeping
        ]

    def block_v0(kept: list[int]) -> tuple[str, list[int]]:
        """Baseline layout; byte-identical to the pre-split renderer."""
        chosen = relations_for(kept)
        lines = [
            "<belief_graph>",
            "## Beliefs",
            *(node_lines[i] for i in sorted(kept)),
            "",
            "## Relations",
        ]
        for rid in chosen:
            lines.append(relation_line(edges[rid]))
        return "\n".join([*lines, "</belief_graph>"]), chosen

    if style == "v0":
        block = block_v0
        # Selector already budgets node text. Include confidence/relations/
        # escaping in the final hard cap as well. Remove whole nodes.
        kept = list(node_lines)
    else:
        info = {i: classify_node(nodes[i]) for i in node_lines}
        finding_lines = {
            i: finding_line(nodes[i], style) for i in node_lines if not info[i]["is_action"]
        }
        action_lines = {i: action_line(nodes[i], info[i]) for i in node_lines if info[i]["is_action"]}
        order = _finding_order(list(node_lines), selection)

        def block_split(kept: list[int]) -> tuple[str, list[int]]:
            """Completed actions (chronological), findings (relevance), relations."""
            chosen = relations_for(kept)
            keeping = set(kept)
            actions = sorted(
                (i for i in kept if i in action_lines),
                key=lambda i: (
                    info[i]["turn_id"] if info[i]["turn_id"] is not None else 10**9, i,
                ),
            )
            findings = [i for i in order if i in keeping and i not in action_lines]
            lines = ["<belief_graph>"]
            if actions:
                lines += [SPLIT_ACTIONS_HEADER, *(action_lines[i] for i in actions), ""]
            lines += ["## Findings", *(finding_lines[i] for i in findings), "", "## Relations"]
            lines += [relation_line(edges[rid]) for rid in chosen]
            return "\n".join([*lines, "</belief_graph>"]), chosen

        block = block_split
        # Trimming drops the least relevant node, whatever the display order.
        kept = list(order)
    if budget is None:
        text, chosen = block(kept)
        return text, sorted(kept), chosen
    while kept:
        text, chosen = block(kept)
        if len(text) <= budget:
            return text, sorted(kept), chosen
        if strict:
            raise GraphError("selection_exceeds_render_budget")
        kept.pop()
    raise GraphError("empty_or_oversized_selection")
