"""Replace completed steps with their graph content, keeping only window edges."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from bcg.construct.unified.task_state import render_task_state
from frontier_agent.core.messages import Message

from .client import GraphError
from .messages import Projection
from .recent import recent_step_selection
from .rendering import _nodes, render_graph, render_v1_block


def compact_history(
    messages: list[Message], plan: Projection, snapshot: dict[str, Any], count: int,
    style: str, task_budget: int | None,
) -> tuple[list[Message], dict[str, Any]]:
    """Render each window node once; no retrieval, synthetic facts or raw history.

    Edges are validated against the UNION of nodes in the last n complete steps,
    then printed once in their source node's block. A cross-step edge must not
    disappear merely because its endpoints are in different compressed messages.
    """
    if (not isinstance(snapshot.get("task_state"), dict)
            or snapshot["task_state"].get("version") not in {1, 2}):
        raise GraphError("graph_server_task_state_required")
    if len(plan.evicted_step_message_indices) != plan.evicted_steps:
        raise GraphError("invalid_compaction_step_positions")
    selection = recent_step_selection(snapshot, plan, count)
    selection["strategy"] = "compaction"
    # Validate the complete selected graph using the unchanged renderer. The
    # per-step rendering below must keep exactly these nodes and internal edges.
    _, visible, relation_ids = render_graph(snapshot, selection, None, strict=True, style=style)
    first = max(0, plan.evicted_steps - count)
    owners: dict[int, int] = {}
    for index in range(first, plan.evicted_steps):
        single = replace(plan, evicted_steps=1,
                         evicted_step_turn_indices=[plan.evicted_step_turn_indices[index]])
        for node_id in recent_step_selection(snapshot, single, 1)["node_ids"]:
            owners[node_id] = index  # a merged representative appears only once
    original_owners = dict(owners)
    if style == "v1":
        owners = {node_id: owner for node_id, owner in owners.items() if node_id in visible}
    if set(owners) != set(visible):
        raise GraphError("compaction_window_membership_mismatch")
    edges = {edge["id"]: edge for edge in snapshot.get("relations", [])}
    compacted = []
    replacements: dict[int, Message] = {}
    for index in range(first, plan.evicted_steps):
        ids = sorted(node_id for node_id, owner in owners.items() if owner == index)
        owned_edges = [rid for rid in relation_ids if owners[edges[rid]["from_id"]] == index]
        if style == "v1":
            # A cross-step relation belongs inline to its source node even
            # though the target is rendered in another compaction message.
            # Include suppressed blocked-result nodes only as status signals;
            # render_v1_block never prints or returns those redundant nodes.
            from .rendering import _v1_blocked_result
            all_nodes = _nodes(snapshot)
            render_ids = ids + [node_id for node_id, owner in original_owners.items()
                                if owner == index and node_id not in owners and
                                _v1_blocked_result(all_nodes[node_id])]
            block, _ = render_v1_block(all_nodes, render_ids,
                                       [edges[rid] for rid in owned_edges],
                                       allow_external_edges=True,
                                       step_distance=plan.evicted_steps - index)
        else:
            block, _, _ = render_graph(snapshot, {"node_ids": ids, "relation_ids": []},
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

    task_text, task_nodes, task_edges, task_ids = render_task_state(snapshot, task_budget)
    task_message: Message = {"role": "user", "content": "<task_state>\n" + task_text + "\n</task_state>"}
    retained = set(plan.retained_indices)
    result: list[Message] = []
    inserted = False
    for index, message in enumerate(messages):
        if index in retained:
            result.append(message)
            if not inserted and message.get("role") == "user" and message.get("bcg_origin") != "agent_report":
                result.append(task_message)
                inserted = True
        elif index in replacements:
            result.append(replacements[index])
    if not inserted:
        raise GraphError("compaction_missing_original_user")
    return result, {
        "context_mode": "bcg_compaction", "compaction_turns": count,
        "retained_indices": plan.retained_indices, "evicted_steps": plan.evicted_steps,
        "selector_result": selection, "selected_node_ids": visible,
        "selected_relation_ids": relation_ids,
        "graph_chars": sum(len(block["content"]) for block in compacted),
        "compacted_messages": compacted, "task_state_message": task_message,
        "task_state": {"available": True, "scope": "full_graph", "placement": "user_after_initial_input",
                       "task_ids": task_ids, "source_node_ids": task_nodes,
                       "relation_ids": task_edges, "chars": len(task_text)},
    }
