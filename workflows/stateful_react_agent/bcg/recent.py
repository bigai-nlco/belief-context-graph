"""Select a window of complete evicted steps, without retrieval or graph mutation."""

from __future__ import annotations

from typing import Any, Literal

from .client import GraphError
from .messages import Projection
from .rendering import _nodes, full_graph_selection


CompactionTurns = int | Literal["all"]


def parse_bool_flag(value: object, *, name: str) -> bool:
    """Parse boolean config/env values strictly; never coerce integers like 1 or 0."""
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def parse_compaction_task_state_hops(value: object) -> int:
    """Parse Task-State graph expansion hops (0..10) for BCG Compaction."""
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if type(value) is int and 0 <= value <= 10:
        return value
    raise ValueError("BCG_COMPACTION_TASK_STATE_HOPS must be an integer between 0 and 10")


def parse_compaction_turns(value: object) -> CompactionTurns:
    """Use the same explicit 'all' or positive count in YAML, env, CLI and run records."""
    if value == "all":
        return "all"
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if type(value) is int and value > 0:
        return value
    raise ValueError("BCG_COMPACTION_TURNS must be a positive integer or 'all'")


def recent_step_selection(
    snapshot: dict[str, Any], plan: Projection, count: int, *, turn_offset: int = 1,
) -> dict[str, Any]:
    return step_window_selection(snapshot, plan, count, turn_offset=turn_offset)[0]


def step_window_selection(
    snapshot: dict[str, Any], plan: Projection, count: CompactionTurns, *, turn_offset: int = 1,
) -> tuple[dict[str, Any], dict[int, int]]:
    """Select complete interactions and map each node to its latest owning step.

    One step contains the assistant and its entire parallel tool-result batch.
    Graph source turn IDs are message-layer IDs, not Agent step numbers. Merged
    beliefs retain their canonical ID, content and rewired relations; archived
    merge members establish whether they also belong to the requested window.
    """
    if ((count != "all" and (type(count) is not int or count < 1))
            or len(plan.evicted_step_turn_indices) != plan.evicted_steps):
        raise GraphError("invalid_recent_graph_step_window")
    first = 0 if count == "all" else max(0, plan.evicted_steps - count)
    turn_steps = {turn + turn_offset: step for step in range(first, plan.evicted_steps)
                  for turn in plan.evicted_step_turn_indices[step]}
    nodes = _nodes(snapshot)
    provenance = dict(nodes)
    for merge in snapshot.get("merges", []):
        for member in merge.get("absorbed_snapshots", []):
            if isinstance(member, dict) and type(member.get("id")) is int:
                provenance[member["id"]] = member

    owners: dict[int, int] = {}
    for node_id, node in nodes.items():
        for member in (node_id, *(node.get("merged_from") or [])):
            source = (provenance.get(member) or {}).get("source") or {}
            turn = source.get("turn_id")
            if type(turn) is int and turn in turn_steps:
                owners[node_id] = max(owners.get(node_id, -1), turn_steps[turn])
    selection = full_graph_selection(snapshot)  # validate the snapshot, no ranking
    selection.update(
        strategy="recent_steps",
        node_ids=sorted(owners),
        relation_ids=sorted(
            edge["id"] for edge in snapshot.get("relations", [])
            if edge.get("from_id") in owners and edge.get("to_id") in owners
        ),
        graph_recent_turns=count,
        # Human-facing one-based complete interaction numbers, not graph layers.
        agent_step_numbers=list(range(first + 1, plan.evicted_steps + 1)),
        graph_turn_ids=sorted(turn_steps),
    )
    return selection, owners
