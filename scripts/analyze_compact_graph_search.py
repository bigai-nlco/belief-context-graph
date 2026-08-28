#!/usr/bin/env python3
"""Replay Compact Graph selection algorithms on recorded benchmark snapshots.

The replay is deliberately offline: every strategy sees the same Graph snapshot
and the same Agent request state.  Ground-truth answers are used only for
evaluation and are never passed to a selector.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

FACT_BUDGET = 4_900
SEARCH_BUDGET = 1_700
NODE_BUDGET = FACT_BUDGET + SEARCH_BUDGET
RELATION_WEIGHTS = {
    "depends_on": 1.0,
    "supplements": 0.82,
    "contradicts": 0.92,
}
EMPTY_SEARCH_RE = re.compile(
    r"^The (?:web_search|\S+ tool) (?:tool )?returned no results\.?$", re.I
)
BELIEF_ID_RE = re.compile(r"\[B(\d+)\]")


@dataclass
class ReplayItem:
    task_id: str
    call_id: int
    stream_turn_index: int
    correct: bool
    query: str
    next_action: str
    question: str
    answer: str
    snapshot: dict[str, Any]
    baseline_ids: set[int]
    baseline_chars: int
    final_support_ids: set[int]


@dataclass
class Selection:
    strategy: str
    node_ids: list[int]
    relation_ids: list[int]
    chars: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for block in value:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif kind == "thinking" and isinstance(block.get("thinking"), str):
            parts.append(block["thinking"])
        elif kind == "toolCall":
            parts.append(
                f"{block.get('name', 'tool')} "
                f"{json.dumps(block.get('arguments', {}), ensure_ascii=False)}"
            )
    return "\n".join(parts)


def request_query(
    request: dict[str, Any], question: str, max_chars: int = 12_000
) -> str:
    messages = request.get("payload", {}).get("messages", [])
    parts = [question]
    for message in messages:
        if message.get("role") == "system":
            continue
        content = text_content(message.get("content"))
        if content and question not in content:
            parts.append(content)
    combined = "\n\n".join(parts)
    if len(combined) <= max_chars:
        return combined
    # Keep the permanent question and the newest state when tool output is long.
    remaining = max(0, max_chars - len(question) - 2)
    return question + "\n\n" + combined[-remaining:]


def response_text(response: dict[str, Any]) -> str:
    return text_content(response.get("message", {}).get("content"))


def source_turn(node: dict[str, Any]) -> int:
    source = node.get("source")
    if isinstance(source, dict) and isinstance(source.get("turn_id"), int):
        return source["turn_id"]
    return -1


def node_text(node: dict[str, Any]) -> str:
    return str(node.get("belief") or node.get("decision") or "").strip()


def node_embedding_texts(node: dict[str, Any]) -> list[str]:
    """Return retrieval units without changing the displayed belief node."""
    texts = [node_text(node)]
    items = node.get("tool_result_items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            text = " — ".join(part for part in (title, snippet) if part)
            if text:
                texts.append(text)
    return texts


def is_search(node: dict[str, Any]) -> bool:
    return node.get("extraction_method") == "rule_tool_call" or isinstance(
        node.get("query"), str
    )


def eligible_nodes(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = snapshot.get("beliefs") or snapshot.get("nodes") or []
    return [
        node
        for node in nodes
        if isinstance(node, dict)
        and isinstance(node.get("id"), int)
        and node_text(node)
        and (source_turn(node) < 0 or source_turn(node) > 1)
        and not EMPTY_SEARCH_RE.match(node_text(node))
    ]


def confidence(node: dict[str, Any]) -> float:
    value = node.get("confidence")
    return float(value) if isinstance(value, (int, float)) else 0.5


def normalize_scores(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def cosine_scores(
    query_vector: np.ndarray, vectors: dict[int, np.ndarray]
) -> dict[int, float]:
    return {
        node_id: float(np.dot(query_vector, vector))
        for node_id, vector in vectors.items()
    }


def node_line(node: dict[str, Any], include_confidence: bool = True) -> str:
    suffix = f" (confidence {confidence(node):.2f})" if include_confidence else ""
    return f"- [B{node['id']}] {node_text(node)}{suffix}"


def select_ranked_with_split_budget(
    strategy: str,
    nodes: list[dict[str, Any]],
    scores: dict[int, float],
) -> Selection:
    selected: list[int] = []
    chars = 0
    for search_group, budget in ((False, FACT_BUDGET), (True, SEARCH_BUDGET)):
        used = 0
        ordered = sorted(
            (node for node in nodes if is_search(node) is search_group),
            key=lambda node: (
                scores.get(node["id"], 0.0),
                source_turn(node),
                node["id"],
            ),
            reverse=True,
        )
        for node in ordered:
            line = node_line(node, include_confidence=not search_group)
            size = len(line) + 1
            if used + size > budget:
                continue
            selected.append(node["id"])
            used += size
            chars += size
    return Selection(strategy, selected, [], chars)


def baseline_selection(item: ReplayItem) -> Selection:
    relations = item.snapshot.get("relations", [])
    ids = item.baseline_ids
    relation_ids = [
        relation["id"]
        for relation in relations
        if relation.get("from_id") in ids
        and relation.get("to_id") in ids
        and isinstance(relation.get("id"), int)
    ]
    return Selection("current", sorted(ids), relation_ids, item.baseline_chars)


def legacy_ranked_selection(nodes: list[dict[str, Any]]) -> Selection:
    """Reproduce the production Compact Graph selector before query awareness."""
    scores: dict[int, float] = {}
    for node in nodes:
        if is_search(node):
            scores[node["id"]] = float(source_turn(node) * 10_000 + node["id"])
            continue
        method = node.get("extraction_method")
        semantic_bonus = 100.0 if method == "compact_llm_tool_result" else 0.0
        raw_penalty = -100.0 if method == "rule_tool_result" else 0.0
        scores[node["id"]] = confidence(node) * 1_000 + semantic_bonus + raw_penalty
    selected = select_ranked_with_split_budget("legacy_ranked", nodes, scores)
    return selected


def recency_scores(nodes: list[dict[str, Any]]) -> dict[int, float]:
    turns = [source_turn(node) for node in nodes]
    low = min(turns, default=0)
    high = max(turns, default=0)
    span = max(1, high - low)
    return {node["id"]: (source_turn(node) - low) / span for node in nodes}


def semantic_selection(
    nodes: list[dict[str, Any]], similarities: dict[int, float]
) -> Selection:
    recency = recency_scores(nodes)
    scores = {
        node["id"]: 0.72 * similarities[node["id"]]
        + 0.18 * confidence(node)
        + 0.10 * recency[node["id"]]
        for node in nodes
    }
    return select_ranked_with_split_budget("semantic", nodes, scores)


def graph_for_snapshot(snapshot: dict[str, Any], node_ids: set[int]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    for relation in snapshot.get("relations", []):
        source = relation.get("from_id")
        target = relation.get("to_id")
        if source not in node_ids or target not in node_ids:
            continue
        weight = RELATION_WEIGHTS.get(str(relation.get("type")), 0.72)
        graph.add_edge(source, target, weight=weight, relation=relation)
    return graph


def pagerank_scores(
    snapshot: dict[str, Any],
    nodes: list[dict[str, Any]],
    similarities: dict[int, float],
) -> dict[int, float]:
    node_ids = {node["id"] for node in nodes}
    graph = graph_for_snapshot(snapshot, node_ids)
    recency = recency_scores(nodes)
    personalization = {
        node["id"]: max(
            0.001, 0.78 * similarities[node["id"]] + 0.22 * recency[node["id"]]
        )
        for node in nodes
    }
    if graph.number_of_edges() == 0:
        return normalize_scores(personalization)
    ranks = nx.pagerank(
        graph,
        alpha=0.78,
        personalization=personalization,
        weight="weight",
        max_iter=200,
    )
    return normalize_scores(ranks)


def ppr_selection(
    snapshot: dict[str, Any],
    nodes: list[dict[str, Any]],
    similarities: dict[int, float],
) -> Selection:
    ppr = pagerank_scores(snapshot, nodes, similarities)
    recency = recency_scores(nodes)
    scores = {
        node["id"]: 0.55 * ppr.get(node["id"], 0.0)
        + 0.28 * similarities[node["id"]]
        + 0.10 * confidence(node)
        + 0.07 * recency[node["id"]]
        for node in nodes
    }
    return select_ranked_with_split_budget("personalized_pagerank", nodes, scores)


def path_proposals(
    graph: nx.DiGraph,
    seeds: list[int],
    node_scores: dict[int, float],
    node_costs: dict[int, int] | None = None,
    max_depth: int = 4,
    beam_width: int = 8,
) -> list[tuple[float, tuple[int, ...]]]:
    proposals: list[tuple[float, tuple[int, ...]]] = []
    costs = node_costs or {node_id: 1 for node_id in graph.nodes}

    def utility(score: float, path: tuple[int, ...]) -> float:
        char_cost = sum(costs[node_id] for node_id in path)
        return score / ((1.0 + char_cost / 240.0) ** 0.34)

    for seed in seeds:
        beam: list[tuple[float, tuple[int, ...]]] = [(node_scores[seed], (seed,))]
        proposals.append((utility(node_scores[seed], (seed,)), (seed,)))
        for _ in range(max_depth):
            expanded: list[tuple[float, tuple[int, ...]]] = []
            for score, path in beam:
                for target in graph.successors(path[-1]):
                    if target in path:
                        continue
                    weight = float(graph[path[-1]][target].get("weight", 0.72))
                    next_score = score + weight * (0.35 + 0.65 * node_scores[target])
                    next_path = (*path, target)
                    expanded.append((next_score, next_path))
            if not expanded:
                break
            expanded.sort(key=lambda item: utility(item[0], item[1]), reverse=True)
            beam = expanded[:beam_width]
            proposals.extend((utility(score, path), path) for score, path in beam)
    proposals.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return proposals


def chain_selection(
    snapshot: dict[str, Any],
    nodes: list[dict[str, Any]],
    similarities: dict[int, float],
) -> Selection:
    by_id = {node["id"]: node for node in nodes}
    node_ids = set(by_id)
    graph = graph_for_snapshot(snapshot, node_ids)
    ppr = pagerank_scores(snapshot, nodes, similarities)
    recency = recency_scores(nodes)
    scores = {
        node["id"]: 0.42 * similarities[node["id"]]
        + 0.30 * ppr.get(node["id"], 0.0)
        + 0.18 * recency[node["id"]]
        + 0.10 * confidence(node)
        for node in nodes
    }
    newest_turn = max((source_turn(node) for node in nodes), default=-1)
    seed_pool = [
        node["id"]
        for node in nodes
        if source_turn(node) >= newest_turn - 2 or similarities[node["id"]] >= 0.45
    ]
    seeds = sorted(seed_pool, key=lambda node_id: scores[node_id], reverse=True)[:10]
    proposals = path_proposals(graph, seeds, scores)

    selected: list[int] = []
    selected_set: set[int] = set()
    used = 0
    for _, path in proposals:
        additions = [node_id for node_id in path if node_id not in selected_set]
        size = sum(
            len(
                node_line(
                    by_id[node_id], include_confidence=not is_search(by_id[node_id])
                )
            )
            + 1
            for node_id in additions
        )
        if not additions or used + size > NODE_BUDGET:
            continue
        selected.extend(additions)
        selected_set.update(additions)
        used += size

    # Use any remaining budget for high-value nodes. A node adjacent to the
    # retained subgraph receives a bonus so the final view stays path-like.
    ordered = sorted(
        node_ids - selected_set,
        key=lambda node_id: (
            scores[node_id]
            + 0.18
            * any(
                neighbor in selected_set
                for neighbor in set(graph.predecessors(node_id))
                | set(graph.successors(node_id))
            )
        ),
        reverse=True,
    )
    for node_id in ordered:
        node = by_id[node_id]
        size = len(node_line(node, include_confidence=not is_search(node))) + 1
        if used + size > NODE_BUDGET:
            continue
        selected.append(node_id)
        selected_set.add(node_id)
        used += size

    relation_ids = [
        relation["id"]
        for relation in snapshot.get("relations", [])
        if relation.get("from_id") in selected_set
        and relation.get("to_id") in selected_set
        and isinstance(relation.get("id"), int)
    ]
    return Selection("continuous_chains", selected, relation_ids, used)


def compact_node_adjustment(node: dict[str, Any]) -> float:
    """Prefer distilled facts while allowing uniquely relevant raw evidence."""
    method = node.get("extraction_method")
    adjustment = 0.0
    if method == "compact_llm_tool_result":
        adjustment += 0.12
    elif method == "rule_tool_result":
        adjustment -= 0.06
    if is_search(node):
        adjustment -= 0.04
    length = len(node_text(node))
    adjustment -= min(0.08, max(0, length - 320) / 12_000)
    return adjustment


def cost_aware_chain_selection(
    snapshot: dict[str, Any],
    nodes: list[dict[str, Any]],
    similarities: dict[int, float],
) -> Selection:
    """Select coherent paths by value per visible character, not node count."""
    by_id = {node["id"]: node for node in nodes}
    node_ids = set(by_id)
    graph = graph_for_snapshot(snapshot, node_ids)
    ppr = pagerank_scores(snapshot, nodes, similarities)
    recency = recency_scores(nodes)
    costs = {
        node["id"]: len(node_line(node, include_confidence=not is_search(node))) + 1
        for node in nodes
    }
    if sum(costs.values()) <= NODE_BUDGET:
        selected = [node["id"] for node in nodes]
        selected_set = set(selected)
        return Selection(
            "cost_aware_chains",
            selected,
            induced_relation_ids(snapshot, selected_set),
            sum(costs.values()),
        )
    scores = {
        node["id"]: max(
            0.001,
            0.44 * similarities[node["id"]]
            + 0.27 * ppr.get(node["id"], 0.0)
            + 0.16 * recency[node["id"]]
            + 0.13 * confidence(node)
            + compact_node_adjustment(node),
        )
        for node in nodes
    }
    newest_turn = max((source_turn(node) for node in nodes), default=-1)
    seed_pool = [
        node["id"]
        for node in nodes
        if source_turn(node) >= newest_turn - 2 or similarities[node["id"]] >= 0.45
    ]
    seeds = sorted(
        seed_pool,
        key=lambda node_id: scores[node_id] / ((1.0 + costs[node_id] / 240.0) ** 0.34),
        reverse=True,
    )[:12]
    proposals = path_proposals(graph, seeds, scores, node_costs=costs)

    selected: list[int] = []
    selected_set: set[int] = set()
    used = 0
    raw_results = 0
    # If compact extraction fell back to raw summaries, preserve the most
    # query-relevant result from the newest evidence turn. This prevents a
    # long but uniquely useful result from losing solely on characters (for
    # example, the only snippet that names the eventual answer).
    newest_raw = [
        node
        for node in nodes
        if node.get("extraction_method") == "rule_tool_result"
        and source_turn(node) == newest_turn
        and "Search budget exhausted" not in node_text(node)
    ]
    if newest_raw:
        reserved = max(
            newest_raw,
            key=lambda node: (
                0.72 * similarities[node["id"]]
                + 0.18 * confidence(node)
                + 0.10 * scores[node["id"]]
            ),
        )
        reserved_id = reserved["id"]
        if costs[reserved_id] <= NODE_BUDGET:
            selected.append(reserved_id)
            selected_set.add(reserved_id)
            used += costs[reserved_id]
            raw_results = 1
    for _, path in proposals:
        additions = [node_id for node_id in path if node_id not in selected_set]
        size = sum(costs[node_id] for node_id in additions)
        new_raw = sum(
            by_id[node_id].get("extraction_method") == "rule_tool_result"
            for node_id in additions
        )
        # Avoid allowing several verbose fallback summaries to crowd out the
        # semantic facts distilled from those same results.
        if raw_results + new_raw > 2 or not additions or used + size > NODE_BUDGET:
            continue
        selected.extend(additions)
        selected_set.update(additions)
        used += size
        raw_results += new_raw

    ordered = sorted(
        node_ids - selected_set,
        key=lambda node_id: (
            (
                scores[node_id]
                + 0.16
                * any(
                    neighbor in selected_set
                    for neighbor in set(graph.predecessors(node_id))
                    | set(graph.successors(node_id))
                )
            )
            / ((1.0 + costs[node_id] / 240.0) ** 0.34)
        ),
        reverse=True,
    )
    for node_id in ordered:
        node = by_id[node_id]
        is_raw = node.get("extraction_method") == "rule_tool_result"
        if is_raw and raw_results >= 2:
            continue
        size = costs[node_id]
        if used + size > NODE_BUDGET:
            continue
        selected.append(node_id)
        selected_set.add(node_id)
        used += size
        raw_results += int(is_raw)

    relation_ids = induced_relation_ids(snapshot, selected_set)
    return Selection("cost_aware_chains", selected, relation_ids, used)


def induced_relation_ids(snapshot: dict[str, Any], selected: set[int]) -> list[int]:
    return [
        relation["id"]
        for relation in snapshot.get("relations", [])
        if relation.get("from_id") in selected
        and relation.get("to_id") in selected
        and isinstance(relation.get("id"), int)
    ]


def rendered_selection(
    snapshot: dict[str, Any], nodes: list[dict[str, Any]], selected: set[int]
) -> tuple[int, list[int]]:
    """Match the production compact renderer's visible payload shape."""
    by_id = {node["id"]: node for node in nodes}
    facts = [by_id[node_id] for node_id in selected if not is_search(by_id[node_id])]
    searches = [by_id[node_id] for node_id in selected if is_search(by_id[node_id])]
    lines: list[str] = []
    if facts:
        lines.append("#### Factual beliefs")
        lines.extend(node_line(node, include_confidence=True) for node in facts)
    if searches:
        lines.extend(["", "#### Search-history beliefs"])
        lines.extend(node_line(node, include_confidence=False) for node in searches)
    relation_by_id = {
        relation.get("id"): relation for relation in snapshot.get("relations", [])
    }
    relation_lines: list[tuple[int, str]] = []
    for relation_id in induced_relation_ids(snapshot, selected):
        relation = relation_by_id[relation_id]
        relation_lines.append(
            (
                relation_id,
                f"- [B{relation['from_id']}] {relation.get('type', 'informs')} "
                f"[B{relation['to_id']}]",
            )
        )
    displayed_relation_ids: list[int] = []
    if relation_lines:
        lines.extend(["", "#### Retained relations"])
        heading = "### Earlier investigation memory"
        used = len(heading) + 1 + len("\n".join(lines))
        for relation_id, line in relation_lines:
            if used + len(line) + 1 > 8_000:
                break
            lines.append(line)
            displayed_relation_ids.append(relation_id)
            used += len(line) + 1
    heading = "### Earlier investigation memory"
    body = "\n".join(lines)
    payload = f"{heading}\n\n{body}"
    chars = len(
        "<｜begin▁of▁sentence｜><｜User｜>"
        + payload
        + "<｜Assistant｜><｜end▁of▁sentence｜>"
    )
    return chars, displayed_relation_ids


def normalized_answer(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def answer_node_ids(nodes: list[dict[str, Any]], answer: str) -> set[int]:
    answer_norm = normalized_answer(answer)
    if not answer_norm:
        return set()
    answer_tokens = answer_norm.split()
    result: set[int] = set()
    for node in nodes:
        content = normalized_answer(node_text(node))
        if answer_norm in content or (
            len(answer_tokens) > 1
            and all(token in content.split() for token in answer_tokens)
        ):
            result.add(node["id"])
    return result


def component_metrics(
    graph: nx.DiGraph, selected: set[int]
) -> tuple[float, float, float]:
    if not selected:
        return 0.0, 1.0, 0.0
    induced = graph.subgraph(selected).to_undirected()
    components = list(nx.connected_components(induced))
    largest = max((len(component) for component in components), default=0) / len(
        selected
    )
    isolated = sum(1 for node_id in selected if induced.degree(node_id) == 0) / len(
        selected
    )
    nodes_with_graph_edges = [
        node_id for node_id in selected if graph.degree(node_id) > 0
    ]
    retained = (
        sum(1 for node_id in nodes_with_graph_edges if induced.degree(node_id) > 0)
        / len(nodes_with_graph_edges)
        if nodes_with_graph_edges
        else 0.0
    )
    return largest, isolated, retained


def support_closure(final_graph: dict[str, Any]) -> set[int]:
    nodes = final_graph.get("nodes") or final_graph.get("beliefs") or []
    decisions = {
        node["id"]
        for node in nodes
        if node.get("node_type") == "decision" and isinstance(node.get("id"), int)
    }
    outgoing: dict[int, list[int]] = defaultdict(list)
    for relation in final_graph.get("relations", []):
        if isinstance(relation.get("from_id"), int) and isinstance(
            relation.get("to_id"), int
        ):
            outgoing[relation["from_id"]].append(relation["to_id"])
    seen = set(decisions)
    queue: deque[tuple[int, int]] = deque((node_id, 0) for node_id in decisions)
    while queue:
        node_id, depth = queue.popleft()
        if depth >= 5:
            continue
        for target in outgoing.get(node_id, []):
            if target not in seen:
                seen.add(target)
                queue.append((target, depth + 1))
    return seen - decisions


def discover_graph_sessions(graphs_dir: Path) -> dict[str, Path]:
    sessions: dict[str, Path] = {}
    for directory in graphs_dir.iterdir():
        latest = directory / "belief_graph_latest.json"
        if not latest.exists():
            continue
        data = json.loads(latest.read_text(encoding="utf-8"))
        problem_id = str(data.get("problem_id", ""))
        session_id = problem_id.split(":", 1)[0]
        if session_id:
            sessions[session_id] = directory
    return sessions


def collect_replay_items(run_dir: Path, graphs_dir: Path) -> list[ReplayItem]:
    mode_dir = run_dir / "browsecomp" / "bcg"
    graph_sessions = discover_graph_sessions(graphs_dir)
    items: list[ReplayItem] = []
    for task_path in sorted((mode_dir / "tasks").glob("*.json")):
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task_id = task["task_id"]
        trajectory_path = mode_dir / "trajectories" / f"browsecomp-bcg-{task_id}.jsonl"
        first = read_jsonl(trajectory_path)
        if not first:
            continue
        session_id = str(first[0].get("id", ""))
        graph_dir = graph_sessions.get(session_id)
        if graph_dir is None:
            continue
        snapshots = {
            row.get("stream_turn_index"): row
            for row in read_jsonl(graph_dir / "belief_graph.jsonl")
        }
        final_graph_path = graph_dir / "final_graph.json"
        final_graph = (
            json.loads(final_graph_path.read_text(encoding="utf-8"))
            if final_graph_path.exists()
            else {}
        )
        support_ids = support_closure(final_graph) if task.get("correct") else set()
        traces = read_jsonl(
            mode_dir / "graph-contexts" / f"browsecomp-bcg-{task_id}.jsonl"
        )
        model_rows = read_jsonl(
            mode_dir / "model-io" / f"browsecomp-bcg-{task_id}.jsonl"
        )
        requests = [row for row in model_rows if row.get("type") == "request"]
        responses = {
            row.get("call_id"): row
            for row in model_rows
            if row.get("type") == "response"
        }
        nonempty_traces = [trace for trace in traces if trace.get("text")]
        for request in requests:
            system_text = "\n".join(
                str(message.get("content") or "")
                for message in request.get("payload", {}).get("messages", [])
                if message.get("role") == "system"
            )
            matching_traces = [
                trace
                for trace in nonempty_traces
                if str(trace.get("text")) in system_text
            ]
            if not matching_traces:
                continue
            # A request contains one injected Graph block. Prefer the longest
            # exact match defensively if one snapshot text is a prefix of another.
            trace = max(matching_traces, key=lambda row: len(str(row.get("text"))))
            stream_index = trace.get("streamTurnIndex")
            snapshot = snapshots.get(stream_index)
            if not snapshot:
                continue
            call_id = int(request.get("call_id", 0))
            items.append(
                ReplayItem(
                    task_id=task_id,
                    call_id=call_id,
                    stream_turn_index=int(stream_index),
                    correct=bool(task.get("correct")),
                    query=request_query(request, task["question"]),
                    next_action=response_text(responses.get(call_id, {})),
                    question=task["question"],
                    answer=str(task.get("reference_answers", [""])[0]),
                    snapshot=snapshot,
                    baseline_ids={
                        int(match)
                        for match in BELIEF_ID_RE.findall(trace.get("text", ""))
                    },
                    baseline_chars=int(trace.get("chars", 0)),
                    final_support_ids=support_ids,
                )
            )
    return items


def strategy_metrics(
    item: ReplayItem,
    selection: Selection,
    nodes: list[dict[str, Any]],
    similarities: dict[int, float],
    answer_similarities: dict[int, float],
    action_similarities: dict[int, float],
) -> dict[str, Any]:
    selected = set(selection.node_ids)
    rendered_chars, displayed_relations = rendered_selection(
        item.snapshot, nodes, selected
    )
    selection.relation_ids = displayed_relations
    if selection.strategy != "current":
        selection.chars = rendered_chars
    graph = graph_for_snapshot(item.snapshot, {node["id"] for node in nodes})
    largest, isolated, retained = component_metrics(graph, selected)
    answer_ids = answer_node_ids(nodes, item.answer)
    available_support = item.final_support_ids & {node["id"] for node in nodes}
    selected_values = [similarities[node_id] for node_id in selected]
    answer_values = [answer_similarities[node_id] for node_id in selected]
    action_values = [action_similarities[node_id] for node_id in selected]
    return {
        "task_id": item.task_id,
        "call_id": item.call_id,
        "stream_turn_index": item.stream_turn_index,
        "correct": item.correct,
        "strategy": selection.strategy,
        "selected_nodes": len(selected),
        "selected_relations": len(selection.relation_ids),
        "chars": selection.chars,
        "query_similarity_mean": float(np.mean(selected_values))
        if selected_values
        else 0.0,
        "answer_similarity_top3": float(
            np.mean(sorted(answer_values, reverse=True)[:3])
        )
        if answer_values
        else 0.0,
        "next_action_similarity_top3": float(
            np.mean(sorted(action_values, reverse=True)[:3])
        )
        if action_values
        else 0.0,
        "answer_nodes_available": len(answer_ids),
        "answer_nodes_selected": len(answer_ids & selected),
        "answer_node_recall": len(answer_ids & selected) / len(answer_ids)
        if answer_ids
        else None,
        "support_nodes_available": len(available_support),
        "support_nodes_selected": len(available_support & selected),
        "support_node_recall": (
            len(available_support & selected) / len(available_support)
            if available_support
            else None
        ),
        "largest_component_ratio": largest,
        "isolated_node_ratio": isolated,
        "relation_endpoint_retention": retained,
        "selected_node_ids": sorted(selected),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)
    summary: dict[str, dict[str, Any]] = {}
    numeric_fields = [
        "selected_nodes",
        "selected_relations",
        "chars",
        "query_similarity_mean",
        "answer_similarity_top3",
        "next_action_similarity_top3",
        "answer_node_recall",
        "support_node_recall",
        "largest_component_ratio",
        "isolated_node_ratio",
        "relation_endpoint_retention",
    ]
    for strategy, values in grouped.items():
        result: dict[str, Any] = {"snapshots": len(values)}
        for field in numeric_fields:
            present = [
                float(row[field]) for row in values if row.get(field) is not None
            ]
            result[field] = float(np.mean(present)) if present else None
            result[f"{field}_n"] = len(present)
        summary[strategy] = result
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--graphs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--task-id", action="append", default=[])
    args = parser.parse_args()

    items = collect_replay_items(args.run_dir, args.graphs_dir)
    if args.task_id:
        requested = set(args.task_id)
        items = [item for item in items if item.task_id in requested]
    if args.task_limit > 0:
        allowed = set(sorted({item.task_id for item in items})[: args.task_limit])
        items = [item for item in items if item.task_id in allowed]
    if not items:
        raise SystemExit("No aligned Graph snapshots were found.")

    model = SentenceTransformer(args.model, device=args.device)
    embedding_cache: dict[str, np.ndarray] = {}

    def encode_cached(texts: list[str]) -> np.ndarray:
        missing = list(
            dict.fromkeys(text for text in texts if text not in embedding_cache)
        )
        if missing:
            vectors = model.encode(
                missing,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embedding_cache.update(zip(missing, vectors, strict=False))
        return np.asarray([embedding_cache[text] for text in texts])

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        nodes = eligible_nodes(item.snapshot)
        if not nodes:
            continue
        texts: list[str] = []
        spans: dict[int, tuple[int, int]] = {}
        for node in nodes:
            start = len(texts)
            texts.extend(node_embedding_texts(node))
            spans[node["id"]] = (start, len(texts))
        encoded = encode_cached(
            [
                *texts,
                item.query,
                f"{item.question}\nCorrect answer: {item.answer}",
                item.next_action,
            ]
        )
        query_vector, answer_vector, action_vector = encoded[-3:]
        similarities = {
            node_id: max(
                float(np.dot(query_vector, vector)) for vector in encoded[start:end]
            )
            for node_id, (start, end) in spans.items()
        }
        answer_similarities = {
            node_id: max(
                float(np.dot(answer_vector, vector)) for vector in encoded[start:end]
            )
            for node_id, (start, end) in spans.items()
        }
        action_similarities = {
            node_id: max(
                float(np.dot(action_vector, vector)) for vector in encoded[start:end]
            )
            for node_id, (start, end) in spans.items()
        }
        selections = [
            baseline_selection(item),
            legacy_ranked_selection(nodes),
            semantic_selection(nodes, similarities),
            ppr_selection(item.snapshot, nodes, similarities),
            chain_selection(item.snapshot, nodes, similarities),
            cost_aware_chain_selection(item.snapshot, nodes, similarities),
        ]
        for selection in selections:
            rows.append(
                strategy_metrics(
                    item,
                    selection,
                    nodes,
                    similarities,
                    answer_similarities,
                    action_similarities,
                )
            )
        if index % 50 == 0:
            print(f"replayed {index}/{len(items)} snapshots", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_snapshot.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "graphs_dir": str(args.graphs_dir.resolve()),
        "tasks": len({item.task_id for item in items}),
        "snapshots": len(items),
        "strategies": aggregate(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
