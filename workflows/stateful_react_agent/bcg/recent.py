"""Select a window of complete evicted steps, without retrieval or graph mutation."""

from __future__ import annotations

from typing import Any

from .client import GraphError
from .messages import Projection
from .rendering import _nodes, full_graph_selection


def recent_step_selection(
    snapshot: dict[str, Any], plan: Projection, count: int, *, turn_offset: int = 1,
) -> dict[str, Any]:
    """Return all active representatives of the last `count` evicted steps.

    One step contains the assistant and its entire parallel tool-result batch.
    Graph source turn IDs are message-layer IDs, not Agent step numbers. Merged
    beliefs retain their canonical ID, content and rewired relations; archived
    merge members establish whether they also belong to the requested window.
    """
    if count < 1 or len(plan.evicted_step_turn_indices) != plan.evicted_steps:
        raise GraphError("invalid_recent_graph_step_window")
    window = plan.evicted_step_turn_indices[-count:]
    turn_ids = {index + turn_offset for group in window for index in group}
    nodes = _nodes(snapshot)
    provenance = dict(nodes)
    for merge in snapshot.get("merges", []):
        for member in merge.get("absorbed_snapshots", []):
            if isinstance(member, dict) and type(member.get("id")) is int:
                provenance[member["id"]] = member

    def belongs(node: dict[str, Any]) -> bool:
        represented = [node["id"], *(node.get("merged_from") or [])]
        for node_id in represented:
            source = (provenance.get(node_id) or {}).get("source") or {}
            if type(source.get("turn_id")) is int and source["turn_id"] in turn_ids:
                return True
        return False

    ids = {node_id for node_id, node in nodes.items() if belongs(node)}
    selection = full_graph_selection(snapshot)  # validate the snapshot, no ranking
    selection.update(
        strategy="recent_steps",
        node_ids=sorted(ids),
        relation_ids=sorted(
            edge["id"] for edge in snapshot.get("relations", [])
            if edge.get("from_id") in ids and edge.get("to_id") in ids
        ),
        graph_recent_turns=count,
        # Human-facing one-based complete interaction numbers, not graph layers.
        agent_step_numbers=list(range(plan.evicted_steps - len(window) + 1, plan.evicted_steps + 1)),
        graph_turn_ids=sorted(turn_ids),
    )
    return selection
