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
FOCUSED_SEARCH_CHAR_BUDGET = 650
FOCUSED_SEARCH_LIMIT = 4
FOCUSED_RELATION_LIMIT = 24
RELATION_WEIGHTS = {"depends_on": 1.0, "supplements": 0.82, "contradicts": 0.92}
_EMPTY_SEARCH = re.compile(
    r"^The (?:web_search|web_fetch|\S+ tool) (?:tool )?returned no results\.?$", re.I
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


def _question_constraints(snapshot: dict[str, Any], question: str) -> list[str]:
    """Recover atomic task constraints from the permanent initial user turn.

    Initial user beliefs are deliberately excluded from the rendered compact
    graph because the raw user request remains in every Agent call. They are,
    however, the best available decomposition of a long question and can guide
    selection without being re-injected.
    """
    values = snapshot.get("beliefs") or snapshot.get("nodes") or []
    constraints: list[str] = []
    seen: set[str] = set()
    for node in values:
        if not isinstance(node, dict) or _source_turn(node) != 1:
            continue
        if node.get("role") not in {None, "user"}:
            continue
        value = _text(node)
        folded = value.casefold()
        if not value or folded in seen:
            continue
        # Execution and response-format instructions are not evidence constraints.
        if (
            "web_search" in folded
            or "web_fetch" in folded
            or "final answer" in folded
            or folded.startswith("external evidence ")
            or folded.startswith("the response must ")
        ):
            continue
        seen.add(folded)
        constraints.append(value)
        if len(constraints) >= 12:
            break
    return constraints or [question]


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


def _selection_features(
    query: str,
    focus_query: str,
    question: str,
    nodes: list[dict[str, Any]],
    constraints: list[str],
    embedder: Any | None,
) -> tuple[
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[int, list[float]],
    dict[int, list[float]] | None,
    str,
]:
    """Embed one selector request once and expose endpoint vectors."""
    if embedder is None:
        query_scores = {
            node["id"]: max(
                _lexical_similarity(query, unit) for unit in _embedding_units(node)
            )
            for node in nodes
        }
        question_scores = {
            node["id"]: max(
                _lexical_similarity(question, unit) for unit in _embedding_units(node)
            )
            for node in nodes
        }
        focus_scores = {
            node["id"]: max(
                _lexical_similarity(focus_query, unit)
                for unit in _embedding_units(node)
            )
            for node in nodes
        }
        constraint_scores = {
            node["id"]: [
                max(
                    _lexical_similarity(constraint, unit)
                    for unit in _embedding_units(node)
                )
                for constraint in constraints
            ]
            for node in nodes
        }
        return (
            query_scores,
            focus_scores,
            question_scores,
            constraint_scores,
            None,
            "lexical",
        )

    units = [query, focus_query, question, *constraints]
    node_offset = len(units)
    spans: dict[int, tuple[int, int]] = {}
    for node in nodes:
        start = len(units)
        units.extend(_embedding_units(node))
        spans[node["id"]] = (start, len(units))
    vectors = embedder.embed(units, purpose="compact_context_selection")
    query_vector, focus_vector, question_vector = vectors[:3]
    query_scores = {
        node_id: max(_cosine(query_vector, vector) for vector in vectors[start:end])
        for node_id, (start, end) in spans.items()
    }
    question_scores = {
        node_id: max(_cosine(question_vector, vector) for vector in vectors[start:end])
        for node_id, (start, end) in spans.items()
    }
    focus_scores = {
        node_id: max(_cosine(focus_vector, vector) for vector in vectors[start:end])
        for node_id, (start, end) in spans.items()
    }
    constraint_vectors = vectors[3:node_offset]
    constraint_scores = {
        node_id: [
            max(_cosine(constraint_vector, vector) for vector in vectors[start:end])
            for constraint_vector in constraint_vectors
        ]
        for node_id, (start, end) in spans.items()
    }
    node_vectors = {node_id: vectors[start] for node_id, (start, _end) in spans.items()}
    return (
        query_scores,
        focus_scores,
        question_scores,
        constraint_scores,
        node_vectors,
        "embedding",
    )


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


def _connected_from_similarities(
    nodes: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    similarities: dict[int, float],
    retrieval: str,
    *,
    node_char_budget: int,
    max_depth: int,
    strategy: str = "connected",
) -> dict[str, Any]:
    by_id = {node["id"]: node for node in nodes}
    node_ids = set(by_id)
    costs = {node_id: _node_line_cost(node) for node_id, node in by_id.items()}
    turns = [_source_turn(node) for node in nodes]
    low, high = min(turns), max(turns)
    span = max(1, high - low)
    recency = {node["id"]: (_source_turn(node) - low) / span for node in nodes}
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
        "strategy": strategy,
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
    return _connected_from_similarities(
        nodes,
        relations,
        similarities,
        retrieval,
        node_char_budget=node_char_budget,
        max_depth=max_depth,
    )


def _coherent_relations(
    relations: list[dict[str, Any]],
    by_id: dict[int, dict[str, Any]],
    node_vectors: dict[int, list[float]] | None,
) -> list[tuple[int, int, float, dict[str, Any]]]:
    """Return one retrieval edge per endpoint pair, suppressing noisy conflicts."""
    best: dict[tuple[int, int], tuple[float, dict[str, Any]]] = {}
    for relation in relations:
        source = relation.get("from_id")
        target = relation.get("to_id")
        if source not in by_id or target not in by_id:
            continue
        coherence = (
            max(0.0, _cosine(node_vectors[source], node_vectors[target]))
            if node_vectors is not None
            else _lexical_similarity(_text(by_id[source]), _text(by_id[target]))
        )
        relation_type = str(relation.get("type"))
        if relation_type == "contradicts" and coherence < 0.28:
            continue
        role_factor = (
            0.78 if _is_search(by_id[source]) or _is_search(by_id[target]) else 1.0
        )
        weight = (
            RELATION_WEIGHTS.get(relation_type, 0.72)
            * (0.30 + 0.70 * coherence)
            * role_factor
        )
        key = (source, target)
        if key not in best or weight > best[key][0]:
            best[key] = (weight, relation)
    return [
        (source, target, weight, relation)
        for (source, target), (weight, relation) in best.items()
    ]


def _relation_role(
    source: int,
    target: int,
    relation: dict[str, Any],
    by_id: dict[int, dict[str, Any]],
) -> int:
    """Rank answer-changing and validating edges before search provenance."""
    if relation.get("type") == "contradicts":
        return 0
    source_search = _is_search(by_id[source])
    target_search = _is_search(by_id[target])
    if not source_search and not target_search:
        return 1
    if relation.get("weight") != 0:
        return 2
    return 3


def _fact_adjacency(
    coherent: list[tuple[int, int, float, dict[str, Any]]],
    by_id: dict[int, dict[str, Any]],
) -> dict[int, list[tuple[int, float, dict[str, Any]]]]:
    adjacency: dict[int, list[tuple[int, float, dict[str, Any]]]] = defaultdict(list)
    for source, target, weight, relation in coherent:
        if _is_search(by_id[source]) or _is_search(by_id[target]):
            continue
        adjacency[source].append((target, weight, relation))
        adjacency[target].append((source, weight, relation))
    return adjacency


def _branch_from_seed(
    seed: int,
    adjacency: dict[int, list[tuple[int, float, dict[str, Any]]]],
    node_scores: dict[int, float],
    *,
    max_depth: int,
    max_nodes: int = 8,
) -> list[int]:
    """Follow the strongest factual relations without crossing search hubs."""
    branch: list[int] = []
    seen = {seed}
    frontier = [(seed, 0)]
    while frontier and len(branch) < max_nodes:
        current, depth = frontier.pop(0)
        branch.append(current)
        if depth >= max_depth:
            continue
        neighbors = sorted(
            adjacency.get(current, []),
            key=lambda item: (
                item[2].get("type") == "contradicts",
                item[1] * (0.35 + 0.65 * node_scores[item[0]]),
            ),
            reverse=True,
        )
        for neighbor, _weight, _relation in neighbors:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            frontier.append((neighbor, depth + 1))
    return branch


def _constraint_ratios(
    node_ids: list[int] | set[int],
    constraint_scores: dict[int, list[float]],
    constraint_best: list[float],
) -> list[float]:
    if not constraint_best:
        return []
    return [
        min(
            1.0,
            max(
                (max(0.0, constraint_scores[node_id][index]) for node_id in node_ids),
                default=0.0,
            )
            / max(0.08, best),
        )
        for index, best in enumerate(constraint_best)
    ]


def _selected_relation_ids(
    selected: set[int],
    coherent: list[tuple[int, int, float, dict[str, Any]]],
    by_id: dict[int, dict[str, Any]],
) -> list[int]:
    candidates = [
        item
        for item in coherent
        if item[0] in selected
        and item[1] in selected
        and isinstance(item[3].get("id"), int)
    ]
    candidates.sort(
        key=lambda item: (
            _relation_role(item[0], item[1], item[3], by_id),
            -item[2],
            -max(_source_turn(by_id[item[0]]), _source_turn(by_id[item[1]])),
            item[3]["id"],
        )
    )
    result: list[int] = []
    retained_provenance_for_search: set[int] = set()
    for source, target, _weight, relation in candidates:
        role = _relation_role(source, target, relation, by_id)
        if role == 3:
            search_id = source if _is_search(by_id[source]) else target
            if search_id in retained_provenance_for_search:
                continue
            retained_provenance_for_search.add(search_id)
        result.append(relation["id"])
        if len(result) >= FOCUSED_RELATION_LIMIT:
            break
    return result


def select_focused_context(
    snapshot: dict[str, Any],
    query: str,
    focus_query: str,
    question: str,
    *,
    embedder: Any | None,
    node_char_budget: int = DEFAULT_NODE_CHAR_BUDGET,
    max_depth: int = 4,
) -> dict[str, Any]:
    """Select answer evidence first and retain only useful search connectors."""
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
            "strategy": "focused",
            "retrieval": "none",
            "node_ids": [],
            "relation_ids": [],
            "node_chars": 0,
        }
    if sum(costs.values()) <= node_char_budget:
        return {
            "strategy": "focused",
            "retrieval": "all_fit",
            "node_ids": [node["id"] for node in nodes],
            "relation_ids": [
                relation["id"]
                for relation in relations
                if isinstance(relation.get("id"), int)
            ],
            "node_chars": sum(costs.values()),
        }
    constraints = _question_constraints(snapshot, question)
    (
        query_similarities,
        focus_similarities,
        question_similarities,
        constraint_similarities,
        node_vectors,
        retrieval,
    ) = _selection_features(
        query,
        focus_query,
        question,
        nodes,
        constraints,
        embedder,
    )
    coherent = _coherent_relations(relations, by_id, node_vectors)
    fact_ids = [node_id for node_id in node_ids if not _is_search(by_id[node_id])]
    if not fact_ids:
        return {
            "strategy": "focused",
            "retrieval": retrieval,
            "node_ids": [],
            "relation_ids": [],
            "node_chars": 0,
        }

    turns = [_source_turn(node) for node in nodes]
    low, high = min(turns), max(turns)
    span = max(1, high - low)
    recency = {node["id"]: (_source_turn(node) - low) / span for node in nodes}
    constraint_best = [
        max(max(0.0, constraint_similarities[node_id][index]) for node_id in fact_ids)
        for index in range(len(constraints))
    ]

    def node_constraint_value(node_id: int) -> tuple[float, float]:
        ratios = _constraint_ratios([node_id], constraint_similarities, constraint_best)
        return max(ratios, default=0.0), sum(ratios) / max(1, len(ratios))

    node_scores: dict[int, float] = {}
    for node_id in fact_ids:
        constraint_max, constraint_mean = node_constraint_value(node_id)
        node_scores[node_id] = max(
            0.001,
            0.24 * max(0.0, question_similarities[node_id])
            + 0.22 * max(0.0, focus_similarities[node_id])
            + 0.10 * max(0.0, query_similarities[node_id])
            + 0.20 * constraint_max
            + 0.09 * constraint_mean
            + 0.08 * recency[node_id]
            + 0.07 * _confidence(by_id[node_id])
            + _adjustment(by_id[node_id]),
        )

    adjacency = _fact_adjacency(coherent, by_id)
    seed_ids: list[int] = []
    for index in range(len(constraints)):
        seed_ids.extend(
            sorted(
                fact_ids,
                key=lambda node_id, position=index: (
                    constraint_similarities[node_id][position],
                    node_scores[node_id],
                ),
                reverse=True,
            )[:2]
        )
    seed_ids.extend(sorted(fact_ids, key=node_scores.get, reverse=True)[:12])
    for source, target, _weight, relation in coherent:
        if relation.get("type") == "contradicts":
            if not _is_search(by_id[source]):
                seed_ids.append(source)
            if not _is_search(by_id[target]):
                seed_ids.append(target)
    seed_ids = list(dict.fromkeys(seed_ids))

    branches: list[list[int]] = []
    seen_branches: set[tuple[int, ...]] = set()
    for seed in seed_ids:
        branch = _branch_from_seed(
            seed,
            adjacency,
            node_scores,
            max_depth=min(3, max_depth),
        )
        key = tuple(sorted(branch))
        if key and key not in seen_branches:
            seen_branches.add(key)
            branches.append(branch)

    selected: list[int] = []
    selected_set: set[int] = set()
    used = 0
    raw_results = 0
    fact_budget = max(
        0,
        node_char_budget - min(FOCUSED_SEARCH_CHAR_BUDGET, node_char_budget // 5),
    )

    def can_add_fact(node_id: int) -> bool:
        nonlocal raw_results
        is_raw = by_id[node_id].get("extraction_method") == "rule_tool_result"
        return (
            not (is_raw and raw_results >= 2) and used + costs[node_id] <= fact_budget
        )

    def add_fact(node_id: int) -> bool:
        nonlocal raw_results, used
        if node_id in selected_set or not can_add_fact(node_id):
            return False
        selected.append(node_id)
        selected_set.add(node_id)
        used += costs[node_id]
        raw_results += int(
            by_id[node_id].get("extraction_method") == "rule_tool_result"
        )
        return True

    newest_turn = max(turns)
    newest_raw_results = [
        node_id
        for node_id in fact_ids
        if by_id[node_id].get("extraction_method") == "rule_tool_result"
        and _source_turn(by_id[node_id]) == newest_turn
        and "Search budget exhausted" not in _text(by_id[node_id])
    ]
    if newest_raw_results:
        add_fact(max(newest_raw_results, key=node_scores.get))

    current_coverage = _constraint_ratios(
        selected_set, constraint_similarities, constraint_best
    )
    remaining_branches = list(branches)
    for _ in range(6):
        best_branch: list[int] | None = None
        best_utility = -1.0
        for branch in remaining_branches:
            additions = [node_id for node_id in branch if node_id not in selected_set]
            feasible = [node_id for node_id in additions if can_add_fact(node_id)]
            if not feasible:
                continue
            coverage = _constraint_ratios(
                set(selected_set) | set(feasible),
                constraint_similarities,
                constraint_best,
            )
            marginal = sum(
                max(0.0, value - current_coverage[index])
                for index, value in enumerate(coverage)
            ) / max(1, len(coverage))
            breadth = sum(value >= 0.80 for value in coverage) / max(1, len(coverage))
            branch_quality = (
                0.48 * (sum(coverage) / max(1, len(coverage)))
                + 0.22 * breadth
                + 0.20 * max(node_scores[node_id] for node_id in feasible)
                + 0.10 * max(recency[node_id] for node_id in feasible)
            )
            branch_cost = sum(costs[node_id] for node_id in feasible)
            utility = (0.62 * marginal + 0.38 * branch_quality) / (
                (1.0 + branch_cost / 900.0) ** 0.28
            )
            if utility > best_utility:
                best_branch = branch
                best_utility = utility
        if best_branch is None:
            break
        remaining_branches.remove(best_branch)
        ordered = [
            best_branch[0],
            *sorted(best_branch[1:], key=node_scores.get, reverse=True),
        ]
        for node_id in ordered:
            add_fact(node_id)
        current_coverage = _constraint_ratios(
            selected_set, constraint_similarities, constraint_best
        )

    def redundancy(node_id: int) -> float:
        if not selected_set:
            return 0.0
        if node_vectors is not None:
            return max(
                max(0.0, _cosine(node_vectors[node_id], node_vectors[other]))
                for other in selected_set
            )
        return max(
            _lexical_similarity(_text(by_id[node_id]), _text(by_id[other]))
            for other in selected_set
        )

    # Spend the remaining evidence budget on nodes that add a missing constraint
    # or directly extend one of the selected factual branches.
    remaining_facts = set(fact_ids) - selected_set
    while remaining_facts:

        def fill_utility(
            node_id: int,
            baseline_coverage: tuple[float, ...] = tuple(current_coverage),
        ) -> float:
            candidate_ratios = _constraint_ratios(
                [node_id], constraint_similarities, constraint_best
            )
            candidate_coverage = [
                max(value, candidate_ratios[index])
                for index, value in enumerate(baseline_coverage)
            ]
            marginal = sum(
                max(0.0, value - baseline_coverage[index])
                for index, value in enumerate(candidate_coverage)
            ) / max(1, len(candidate_coverage))
            connected = any(
                neighbor in selected_set
                for neighbor, _weight, _relation in adjacency.get(node_id, [])
            )
            return (
                node_scores[node_id]
                + 0.24 * marginal
                + 0.12 * float(connected)
                - 0.16 * redundancy(node_id)
            ) / ((1.0 + costs[node_id] / 240.0) ** 0.34)

        best = max(remaining_facts, key=fill_utility)
        remaining_facts.remove(best)
        if not add_fact(best):
            continue
        current_coverage = _constraint_ratios(
            selected_set, constraint_similarities, constraint_best
        )

    # Search history is useful only as provenance for a retained evidence branch.
    search_neighbors: dict[int, list[tuple[int, float, dict[str, Any]]]] = defaultdict(
        list
    )
    for source, target, weight, relation in coherent:
        if _is_search(by_id[source]) and target in selected_set:
            search_neighbors[source].append((target, weight, relation))
        elif _is_search(by_id[target]) and source in selected_set:
            search_neighbors[target].append((source, weight, relation))
    search_candidates = set(search_neighbors)
    search_chars = 0
    while (
        search_candidates
        and len([item for item in selected if _is_search(by_id[item])])
        < FOCUSED_SEARCH_LIMIT
    ):

        def search_utility(node_id: int) -> tuple[float, int]:
            neighbors = search_neighbors[node_id]
            evidence_count = (
                min(3, len({neighbor for neighbor, _weight, _rel in neighbors})) / 3.0
            )
            strongest_edge = max(weight for _neighbor, weight, _relation in neighbors)
            produced_evidence = any(
                relation.get("from_id") == neighbor and relation.get("to_id") == node_id
                for neighbor, _weight, relation in neighbors
            )
            score = (
                0.28 * evidence_count
                + 0.22 * strongest_edge
                + 0.18 * max(0.0, focus_similarities[node_id])
                + 0.12 * max(0.0, question_similarities[node_id])
                + 0.12 * recency[node_id]
                + 0.08 * float(produced_evidence)
            )
            return score, _source_turn(by_id[node_id])

        best_search = max(search_candidates, key=search_utility)
        search_candidates.remove(best_search)
        cost = costs[best_search]
        if (
            search_chars + cost > FOCUSED_SEARCH_CHAR_BUDGET
            or used + cost > node_char_budget
        ):
            continue
        selected.append(best_search)
        selected_set.add(best_search)
        used += cost
        search_chars += cost

    return {
        "strategy": "focused",
        "retrieval": retrieval,
        "node_ids": selected,
        "relation_ids": _selected_relation_ids(selected_set, coherent, by_id),
        "node_chars": used,
    }
