"""Query-aware selection of a small, connected view of a belief graph.

The graph itself remains the source of truth.  This module only selects
existing nodes and relations for an Agent-facing context window; it never
rewrites beliefs or invents edges.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

DEFAULT_NODE_CHAR_BUDGET = 6_600
RELATION_WEIGHTS = {"depends_on": 1.0, "supplements": 0.82, "contradicts": 0.92}
_EMPTY_SEARCH = re.compile(
    r"^The (?:web_search|\S+ tool) (?:tool )?returned no results\.?$", re.I
)


def _text(node: dict[str, Any]) -> str:
    return str(node.get("belief") or node.get("decision") or "").strip()


def _source_turn(node: dict[str, Any]) -> int:
    source = node.get("source")
    if isinstance(source, dict) and isinstance(source.get("turn_id"), int):
        return int(source["turn_id"])
    return -1


def _confidence(node: dict[str, Any]) -> float:
    value = node.get("confidence")
    return float(value) if isinstance(value, (int, float)) else 0.5


def _is_search(node: dict[str, Any]) -> bool:
    return node.get("extraction_method") == "rule_tool_call" or isinstance(
        node.get("query"), str
    )


def _node_line_cost(node: dict[str, Any]) -> int:
    suffix = "" if _is_search(node) else f" (confidence {_confidence(node):.2f})"
    return len(f"- [B{node['id']}] {_text(node)}{suffix}") + 1


def _eligible(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    values = snapshot.get("beliefs") or snapshot.get("nodes") or []
    return [
        node
        for node in values
        if isinstance(node, dict)
        and isinstance(node.get("id"), int)
        and _text(node)
        and (_source_turn(node) < 0 or _source_turn(node) > 1)
        and not _EMPTY_SEARCH.match(_text(node))
    ]


def _embedding_units(node: dict[str, Any]) -> list[str]:
    """Use snippets as retrieval units without changing the visible node."""
    units = [_text(node)]
    items = node.get("tool_result_items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            value = " — ".join(part for part in (title, snippet) if part)
            if value:
                units.append(value)
    return units


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _lexical_similarity(query: str, text: str) -> float:
    query_terms = set(re.findall(r"[\w-]+", query.casefold()))
    text_terms = set(re.findall(r"[\w-]+", text.casefold()))
    return (
        len(query_terms & text_terms) / math.sqrt(len(query_terms) * len(text_terms))
        if query_terms and text_terms
        else 0.0
    )


def _similarities(
    query: str, nodes: list[dict[str, Any]], embedder: Any | None
) -> tuple[dict[int, float], str]:
    units: list[str] = [query]
    spans: dict[int, tuple[int, int]] = {}
    for node in nodes:
        start = len(units)
        units.extend(_embedding_units(node))
        spans[node["id"]] = (start, len(units))
    if embedder is None:
        return {
            node["id"]: max(
                _lexical_similarity(query, unit) for unit in _embedding_units(node)
            )
            for node in nodes
        }, "lexical"
    vectors = embedder.embed(units, purpose="compact_context_selection")
    query_vector = vectors[0]
    return {
        node_id: max(_cosine(query_vector, vector) for vector in vectors[start:end])
        for node_id, (start, end) in spans.items()
    }, "embedding"


def _normalize(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _pagerank(
    node_ids: set[int],
    outgoing: dict[int, list[tuple[int, float]]],
    personalization: dict[int, float],
    *,
    alpha: float = 0.78,
) -> dict[int, float]:
    if not node_ids:
        return {}
    total = sum(max(0.001, value) for value in personalization.values())
    p = {node_id: max(0.001, personalization[node_id]) / total for node_id in node_ids}
    rank = dict(p)
    for _ in range(100):
        updated = {node_id: (1.0 - alpha) * p[node_id] for node_id in node_ids}
        dangling = 0.0
        for source in node_ids:
            edges = outgoing.get(source, [])
            if not edges:
                dangling += rank[source]
                continue
            weight_sum = sum(weight for _, weight in edges)
            for target, weight in edges:
                updated[target] += alpha * rank[source] * weight / weight_sum
        for node_id in node_ids:
            updated[node_id] += alpha * dangling * p[node_id]
        if sum(abs(updated[node_id] - rank[node_id]) for node_id in node_ids) < 1e-8:
            rank = updated
            break
        rank = updated
    return _normalize(rank)


def _adjustment(node: dict[str, Any]) -> float:
    method = node.get("extraction_method")
    value = 0.12 if method == "compact_llm_tool_result" else 0.0
    if method == "rule_tool_result":
        value -= 0.06
    if _is_search(node):
        value -= 0.04
    value -= min(0.08, max(0, len(_text(node)) - 320) / 12_000)
    return value


def _path_proposals(
    outgoing: dict[int, list[tuple[int, float]]],
    seeds: list[int],
    scores: dict[int, float],
    costs: dict[int, int],
    *,
    max_depth: int,
    beam_width: int = 8,
) -> list[tuple[float, tuple[int, ...]]]:
    proposals: list[tuple[float, tuple[int, ...]]] = []

    def utility(score: float, path: tuple[int, ...]) -> float:
        return score / ((1.0 + sum(costs[item] for item in path) / 240.0) ** 0.34)

    for seed in seeds:
        beam = [(scores[seed], (seed,))]
        proposals.append((utility(scores[seed], (seed,)), (seed,)))
        for _ in range(max_depth):
            expanded: list[tuple[float, tuple[int, ...]]] = []
            for score, path in beam:
                for target, weight in outgoing.get(path[-1], []):
                    if target in path:
                        continue
                    next_score = score + weight * (0.35 + 0.65 * scores[target])
                    expanded.append((next_score, (*path, target)))
            if not expanded:
                break
            expanded.sort(key=lambda item: utility(*item), reverse=True)
            beam = expanded[:beam_width]
            proposals.extend((utility(score, path), path) for score, path in beam)
    return sorted(proposals, key=lambda item: (item[0], len(item[1])), reverse=True)


def select_connected_context(
    snapshot: dict[str, Any],
    query: str,
    *,
    embedder: Any | None,
    node_char_budget: int = DEFAULT_NODE_CHAR_BUDGET,
    max_depth: int = 4,
) -> dict[str, Any]:
    """Return a cost-aware connected subgraph selected for ``query``."""
    nodes = _eligible(snapshot)
    by_id = {node["id"]: node for node in nodes}
    node_ids = set(by_id)
    costs = {node_id: _node_line_cost(node) for node_id, node in by_id.items()}
    relations = [
        relation
        for relation in snapshot.get("relations", [])
        if relation.get("from_id") in node_ids and relation.get("to_id") in node_ids
    ]
    if not nodes:
        return {
            "strategy": "connected",
            "retrieval": "none",
            "node_ids": [],
            "relation_ids": [],
            "node_chars": 0,
        }
    if sum(costs.values()) <= node_char_budget:
        selected = set(node_ids)
        return {
            "strategy": "connected",
            "retrieval": "all_fit",
            "node_ids": [node["id"] for node in nodes],
            "relation_ids": [
                relation["id"]
                for relation in relations
                if isinstance(relation.get("id"), int)
            ],
            "node_chars": sum(costs.values()),
        }

    similarities, retrieval = _similarities(query, nodes, embedder)
    turns = [_source_turn(node) for node in nodes]
    low, high = min(turns), max(turns)
    span = max(1, high - low)
    recency = {node["id"]: (_source_turn(node) - low) / span for node in nodes}
    # Treat repeated relation records as one directed edge. Otherwise a
    # duplicate endpoint pair receives artificial PageRank/path weight.
    outgoing_by_target: dict[int, dict[int, float]] = defaultdict(dict)
    for relation in relations:
        outgoing_by_target[relation["from_id"]][relation["to_id"]] = (
            RELATION_WEIGHTS.get(str(relation.get("type")), 0.72)
        )
    outgoing = {
        source: list(targets.items()) for source, targets in outgoing_by_target.items()
    }
    ppr = _pagerank(
        node_ids,
        outgoing,
        {
            node_id: 0.78 * similarities[node_id] + 0.22 * recency[node_id]
            for node_id in node_ids
        },
    )
    scores = {
        node_id: max(
            0.001,
            0.44 * similarities[node_id]
            + 0.27 * ppr.get(node_id, 0.0)
            + 0.16 * recency[node_id]
            + 0.13 * _confidence(by_id[node_id])
            + _adjustment(by_id[node_id]),
        )
        for node_id in node_ids
    }
    newest_turn = max(turns)
    seed_pool = [
        node_id
        for node_id in node_ids
        if _source_turn(by_id[node_id]) >= newest_turn - 2
        or similarities[node_id] >= 0.45
    ]
    seeds = sorted(
        seed_pool,
        key=lambda node_id: scores[node_id] / ((1.0 + costs[node_id] / 240.0) ** 0.34),
        reverse=True,
    )[:12]
    proposals = _path_proposals(outgoing, seeds, scores, costs, max_depth=max_depth)

    selected: list[int] = []
    selected_set: set[int] = set()
    used = 0
    raw_results = 0
    newest_raw = [
        node
        for node in nodes
        if node.get("extraction_method") == "rule_tool_result"
        and _source_turn(node) == newest_turn
        and "Search budget exhausted" not in _text(node)
    ]
    if newest_raw:
        reserved = max(
            newest_raw,
            key=lambda node: (
                0.72 * similarities[node["id"]]
                + 0.18 * _confidence(node)
                + 0.10 * scores[node["id"]]
            ),
        )
        if costs[reserved["id"]] <= node_char_budget:
            selected.append(reserved["id"])
            selected_set.add(reserved["id"])
            used += costs[reserved["id"]]
            raw_results = 1

    for _, path in proposals:
        additions = [node_id for node_id in path if node_id not in selected_set]
        size = sum(costs[node_id] for node_id in additions)
        new_raw = sum(
            by_id[node_id].get("extraction_method") == "rule_tool_result"
            for node_id in additions
        )
        if not additions or raw_results + new_raw > 2 or used + size > node_char_budget:
            continue
        selected.extend(additions)
        selected_set.update(additions)
        used += size
        raw_results += new_raw

    def connected_bonus(node_id: int) -> float:
        adjacent = any(
            target in selected_set for target, _ in outgoing.get(node_id, [])
        )
        if not adjacent:
            adjacent = any(
                node_id == target and source in selected_set
                for source, edges in outgoing.items()
                for target, _ in edges
            )
        return 0.16 if adjacent else 0.0

    remaining = sorted(
        node_ids - selected_set,
        key=lambda node_id: (
            (scores[node_id] + connected_bonus(node_id))
            / ((1.0 + costs[node_id] / 240.0) ** 0.34)
        ),
        reverse=True,
    )
    for node_id in remaining:
        is_raw = by_id[node_id].get("extraction_method") == "rule_tool_result"
        if is_raw and raw_results >= 2:
            continue
        if used + costs[node_id] > node_char_budget:
            continue
        selected.append(node_id)
        selected_set.add(node_id)
        used += costs[node_id]
        raw_results += int(is_raw)

    return {
        "strategy": "connected",
        "retrieval": retrieval,
        "node_ids": selected,
        "relation_ids": [
            relation["id"]
            for relation in relations
            if relation.get("from_id") in selected_set
            and relation.get("to_id") in selected_set
            and isinstance(relation.get("id"), int)
        ],
        "node_chars": used,
    }
