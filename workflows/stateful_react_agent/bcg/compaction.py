"""Replace completed steps with their graph content, keeping only window edges."""

from __future__ import annotations

from collections import deque
from typing import Any

from bcg.construct.unified.task_state import READABLE_VERSIONS, build_task_state, render_task_state
from frontier_agent.core.messages import Message

from .client import GraphError
from .messages import Projection, has_explicit_user_provenance, is_human_user_message
from .recent import CompactionTurns, step_window_selection
from .rendering import (
    RENDER_STYLES,
    _nodes,
    _v1_action,
    _v1_blocked_result,
    checked_graph_selection,
    render_graph,
    render_v1_block,
)


def _active_task_state_seeds(snapshot: dict[str, Any], rendered_task_nodes: list[int]) -> list[int]:
    """Select active belief IDs from the rendered <task_state>, excluding completed work and closed gaps."""
    state = snapshot.get("task_state") or {}
    if state.get("version") == 1:
        hops = state.get("max_hops", 3)
        state = build_task_state(snapshot, max_hops=hops if type(hops) is int and hops >= 0 else 3)
    primary_active: set[int] = set()
    associated_active: set[int] = set()
    closure_ids = {
        cid
        for c in state.get("gap_closures", [])
        if isinstance(c, dict)
        for cid in (c.get("gap_id"), c.get("source_id"))
        if type(cid) is int
    }
    root = state.get("root") or {}
    if type(root.get("status_source_id")) is int:
        primary_active.add(root["status_source_id"])
    for key in ("open_gap_ids", "plan_ids"):
        primary_active.update(i for i in root.get(key, []) if type(i) is int)
    primary_active.update(i for i in state.get("unassigned_gap_ids", []) if type(i) is int)
    for task in state.get("nodes", []):
        if task.get("status") in {"reported_complete", "cancelled"}:
            continue
        for key in ("belief_id", "status_source_id"):
            if type(task.get(key)) is int:
                primary_active.add(task[key])
        for key in ("open_gap_ids", "plan_ids"):
            primary_active.update(i for i in task.get(key, []) if type(i) is int)
        associated_active.update(i for i in task.get("associated_belief_ids", []) if type(i) is int)
    active = primary_active | (associated_active - closure_ids)
    return sorted(set(rendered_task_nodes) & active)


def _khop_reachable(
    all_nodes: dict[int, dict[str, Any]],
    relations: list[dict[str, Any]],
    seeds: list[int],
    hops: int,
) -> set[int]:
    """Expand undirected real-relation neighborhood up to k hops across the full snapshot."""
    if not seeds:
        return set()
    adjacent: dict[int, list[int]] = {nid: [] for nid in all_nodes}
    for edge in relations:
        if edge.get("type") not in {"depends_on", "supplements", "contradicts"}:
            continue
        a, b = edge.get("from_id"), edge.get("to_id")
        if a in adjacent and b in adjacent:
            adjacent[a].append(b)
            adjacent[b].append(a)
    seen = {s for s in seeds if s in adjacent}
    queue = deque((s, 0) for s in sorted(seen))
    while queue:
        node_id, dist = queue.popleft()
        if dist >= hops:
            continue
        for nb in adjacent[node_id]:
            if nb not in seen:
                seen.add(nb)
                queue.append((nb, dist + 1))
    return seen


def _pair_tool_action(
    result_id: int,
    step_actions: list[int],
    all_nodes: dict[int, dict[str, Any]],
    outgoing_depends: dict[int, list[int]],
    *,
    fallback_first: bool,
) -> int | None:
    """Match a tool result node to its issuing tool action using the same rules as v1 rendering."""
    if not step_actions:
        return None
    action_set = set(step_actions)
    result = all_nodes[result_id]
    tcid = result.get("tool_call_id")
    if tcid:
        match = next((aid for aid in step_actions if all_nodes[aid].get("tool_call_id") == tcid), None)
        if match is not None:
            return match
    match = next((tid for tid in outgoing_depends.get(result_id, ()) if tid in action_set), None)
    if match is not None:
        return match
    tname = result.get("tool_name")
    if tname:
        match = next((aid for aid in step_actions if all_nodes[aid].get("tool_name") == tname), None)
        if match is not None:
            return match
    return step_actions[0] if fallback_first else None


def compact_history(
    messages: list[Message], plan: Projection, snapshot: dict[str, Any], count: CompactionTurns,
    style: str, task_budget: int | None,
    *, task_state_filter: bool = False, task_state_hops: int = 2,
) -> tuple[list[Message], dict[str, Any]]:
    """Render each window node once; no retrieval, synthetic facts or raw history.

    Edges are validated against the UNION of nodes in the retained complete steps,
    then printed once in their source node's block. A cross-step edge must not
    disappear merely because its endpoints are in different compressed messages.
    """
    if (not isinstance(snapshot.get("task_state"), dict)
            or snapshot["task_state"].get("version") not in READABLE_VERSIONS):
        raise GraphError("graph_server_task_state_required")
    if len(plan.evicted_step_message_indices) != plan.evicted_steps:
        raise GraphError("invalid_compaction_step_positions")
    if style not in RENDER_STYLES:
        raise GraphError("invalid_render_style")
    if task_state_filter and (type(task_state_hops) is not int or task_state_hops < 0):
        raise GraphError("invalid_compaction_task_state_hops")
    task_text, task_nodes, task_edges, task_ids = render_task_state(snapshot, task_budget)
    selection, owners = step_window_selection(snapshot, plan, count)
    window_steps = list(selection["agent_step_numbers"])
    if task_state_filter:
        all_nodes = _nodes(snapshot)
        relations = snapshot.get("relations", [])
        seeds = _active_task_state_seeds(snapshot, task_nodes)
        reachable = _khop_reachable(all_nodes, relations, seeds, task_state_hops)
        step_nodes: dict[int, list[int]] = {}
        for nid in selection["node_ids"]:
            step_nodes.setdefault(owners[nid], []).append(nid)
        outgoing_depends: dict[int, list[int]] = {}
        for edge in sorted(relations, key=lambda e: e["id"]):
            if edge.get("type") == "depends_on":
                outgoing_depends.setdefault(edge["from_id"], []).append(edge["to_id"])
        latest_step_index = plan.evicted_steps - 1 if plan.evicted_steps > 0 else -1
        kept_nodes: set[int] = set()
        for agent_step in window_steps:
            index = agent_step - 1
            step_ids = step_nodes.get(index, [])
            if index == latest_step_index:
                step_kept = set(step_ids)
            else:
                step_kept = {nid for nid in step_ids if nid in reachable}
                if style == "v1":
                    # Blocked results only survive as a status signal attached to a retained action.
                    step_kept = {nid for nid in step_kept if not _v1_blocked_result(all_nodes[nid])}
                step_actions = [nid for nid in step_ids if _v1_action(all_nodes[nid])]
                for rid in sorted(step_kept):
                    node = all_nodes[rid]
                    if (node.get("role") == "tool" and not _v1_action(node)
                            and not (style == "v1" and _v1_blocked_result(node))):
                        aid = _pair_tool_action(
                            rid, step_actions, all_nodes, outgoing_depends, fallback_first=True,
                        )
                        if aid is not None:
                            step_kept.add(aid)
                if style == "v1":
                    for bid in step_ids:
                        if _v1_blocked_result(all_nodes[bid]):
                            aid = _pair_tool_action(
                                bid, step_actions, all_nodes, outgoing_depends, fallback_first=False,
                            )
                            if aid is not None and aid in step_kept:
                                step_kept.add(bid)
            kept_nodes.update(step_kept)
        selection.update(
            strategy="compaction_task_state",
            node_ids=sorted(kept_nodes),
            relation_ids=sorted(
                edge["id"] for edge in relations
                if edge.get("from_id") in kept_nodes and edge.get("to_id") in kept_nodes
                and edge.get("type") in {"depends_on", "supplements", "contradicts"}
            ),
            task_state_filter=True,
            task_state_hops=task_state_hops,
            task_state_seed_ids=seeds,
            retained_latest_step=plan.evicted_steps if plan.evicted_steps > 0 else None,
            window_agent_step_numbers=window_steps,
        )
    else:
        selection["strategy"] = "compaction"
    # Index the union once: an unbounded history must not rescan or render
    # the entire graph separately for every past interaction.
    nodes, edges = checked_graph_selection(snapshot, selection)
    grouped_ids: dict[int, list[int]] = {}
    hidden: set[int] = set()
    for nid in selection["node_ids"]:
        if style == "v1" and _v1_blocked_result(nodes[nid]):
            hidden.add(nid)
        grouped_ids.setdefault(owners[nid], []).append(nid)
    visible = [nid for nid in selection["node_ids"] if nid not in hidden]
    relation_ids = [rid for rid in selection["relation_ids"]
                    if edges[rid]["from_id"] not in hidden and edges[rid]["to_id"] not in hidden]
    grouped_edges: dict[int, list[int]] = {}
    for rid in relation_ids:
        grouped_edges.setdefault(owners[edges[rid]["from_id"]], []).append(rid)
    compacted = []
    replacements: dict[int, Message] = {}
    for agent_step in window_steps:
        index = agent_step - 1
        render_ids = grouped_ids.get(index, [])
        ids = [nid for nid in render_ids if nid not in hidden]
        if task_state_filter and not ids:
            continue
        owned_edges = grouped_edges.get(index, [])
        if style == "v1":
            # A cross-step relation belongs inline to its source node even
            # though the target is rendered in another compaction message.
            # Include suppressed blocked-result nodes only as status signals;
            # render_v1_block never prints or returns those redundant nodes.
            block, _ = render_v1_block(nodes, render_ids,
                                       [edges[rid] for rid in owned_edges],
                                       allow_external_edges=True,
                                       step_distance=plan.evicted_steps - index)
        else:
            block, _, _ = render_graph({"nodes": [nodes[nid] for nid in ids]}, {"node_ids": ids, "relation_ids": []},
                                       None, strict=True, style=style)
            # Preserve the exact legacy compaction output for older styles.
            if owned_edges:
                from .rendering import relation_line
                block = block.removesuffix("</belief_graph>") + "\n".join(
                    relation_line(edges[rid]) for rid in owned_edges) + "\n</belief_graph>"
        # Request projection only: the durable transcript (and thus the list of
        # pinned human inputs or graph writes) never receives this memory.
        message: Message = {"role": "user", "content": block}
        position = plan.evicted_step_message_indices[index]
        replacements[position] = message
        compacted.append({"agent_step": index + 1, "original_message_index": position,
                          "graph_turn_ids": [i + 1 for i in plan.evicted_step_turn_indices[index]],
                          "role": message["role"], "content": block,
                          "node_ids": ids, "relation_ids": owned_edges})
    if task_state_filter:
        selection["agent_step_numbers"] = [block["agent_step"] for block in compacted]

    task_message: Message | None = (
        {"role": "user", "content": "<task_state>\n" + task_text + "\n</task_state>"} if task_text else None
    )
    explicit_prov = has_explicit_user_provenance(messages)
    retained = set(plan.retained_indices)
    result: list[Message] = []
    inserted = False
    for index, message in enumerate(messages):
        if index in retained:
            result.append(message)
            if not inserted and is_human_user_message(message, explicit_provenance=explicit_prov):
                if task_message is not None:
                    result.append(task_message)
                inserted = True
        elif index in replacements:
            result.append(replacements[index])
    if not inserted:
        raise GraphError("compaction_missing_original_user")
    return result, {
        "context_mode": "bcg_compaction", "compaction_turns": count,
        "compaction_task_state_filter": task_state_filter,
        "compaction_task_state_hops": task_state_hops if task_state_filter else None,
        "retained_indices": plan.retained_indices, "evicted_steps": plan.evicted_steps,
        "selector_result": selection, "selected_node_ids": visible,
        "selected_relation_ids": relation_ids,
        "graph_chars": sum(len(block["content"]) for block in compacted),
        "compacted_messages": compacted, "task_state_message": task_message,
        "task_state": {"available": True, "injected": task_message is not None,
                       "scope": "full_graph", "placement": "user_after_initial_input",
                       "task_ids": task_ids, "source_node_ids": task_nodes,
                       "relation_ids": task_edges, "chars": len(task_text)},
    }
