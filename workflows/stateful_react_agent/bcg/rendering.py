"""Shared graph rendering and the exact character budget used by selectors."""

from __future__ import annotations

import math
from html import escape
from typing import Any

from .client import GraphError

GRAPH_OVERHEAD = len("<belief_graph>\n## Beliefs\n\n## Relations\n</belief_graph>")


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


def render_graph(
    snapshot: dict[str, Any], selection: dict[str, Any], budget: int, *, strict: bool = False
) -> tuple[str, list[int], list[int]]:
    """Bound the block without truncating beliefs or inventing relation edges."""
    ids, relation_ids = selection.get("node_ids"), selection.get("relation_ids")
    if (
        not isinstance(ids, list)
        or not isinstance(relation_ids, list)
        or not all(type(i) is int for i in ids + relation_ids)
    ):
        raise GraphError("invalid_selection_ids")
    nodes = _nodes(snapshot)
    if not set(ids) <= nodes.keys():
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
            edge.get("from_id") not in ids
            or edge.get("to_id") not in ids
            or edge.get("type") not in {"depends_on", "supplements", "contradicts"}
        ):
            raise GraphError("invalid_selected_relation")
    node_lines: dict[int, str] = {}
    for node_id in dict.fromkeys(ids):
        node_lines[node_id] = node_line(nodes[node_id])

    def block(kept: list[int]) -> tuple[str, list[int]]:
        chosen = [
            rid
            for rid in dict.fromkeys(relation_ids)
            if edges[rid]["from_id"] in kept and edges[rid]["to_id"] in kept
        ]
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

    # Selector already budgets node text. Include confidence/relations/escaping
    # in the final hard cap as well. Remove whole nodes with incident edges.
    kept = list(node_lines)
    while kept:
        text, chosen = block(kept)
        if len(text) <= budget:
            return text, sorted(kept), chosen
        if strict:
            raise GraphError("selection_exceeds_render_budget")
        kept.pop()
    raise GraphError("empty_or_oversized_selection")
