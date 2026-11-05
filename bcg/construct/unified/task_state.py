"""Grounded task-state projection over beliefs; no model calls or new confidence.

Task records have their own namespace. Their links describe workflow state and
never participate in the belief graph's factual-confidence propagation.
"""
from __future__ import annotations

import copy
import json
from collections import Counter, deque
from html import escape
from typing import Any

ANCHOR_ROLES = frozenset({"task"})
TASK_ROLES = ANCHOR_ROLES | {"plan", "evidence", "intermediate_conclusion", "constraint", "progress", "other"}
LEGACY_ROLES = {"goal": "task", "subgoal": "task", "next_action": "plan"}
STATUSES = frozenset({"open", "in_progress", "blocked", "reported_complete", "cancelled"})

EXTRACTION_GUIDANCE = """## Task-state annotations
In this SAME extraction, add task_role to each new belief: task (an explicit
requested outcome or bounded work item whose completion can be tracked), plan
(a proposed method or intended next action for a task), evidence (observed or
attributed information), intermediate_conclusion (a derived claim or hypothesis),
constraint, progress, or other. Only task creates a task record. Goals and subtasks
both use task; methods and next actions both use plan. A search/fetch intention is
a plan, not another task. Do not duplicate one intention as both task and plan.
Retain explicit useful tasks, plans, blockers and progress updates, but omit
generic procedural filler. Do not invent a plan or split a task that the current
text does not split. Keep observations separate from derived conclusions.

Optional task_ids lists integer IDs of SHOWN historical task beliefs explicitly
addressed by the new belief. For a task, these identify its parent task(s); for a
plan, its target task(s); for progress, the task(s) being updated. Never reference
a plan as a task. Use [] if the target is uncertain or not shown. A plan without
a known target stays a plan; do not invent a task to attach it to.
Do not use IDs from other sessions, guess an ID, or refer to an output-array index.
For progress only, add task_status: open | in_progress | blocked |
reported_complete | cancelled. Use reported_complete only when the current speaker
explicitly says that the identified work item is complete; it is a claim, not
independent verification. Tool calls, returned results, confidence and a final
report do not by themselves establish completion of a work item.

These optional fields go inside each belief object alongside the existing text,
stance, entities and source evidence, for example:
"task_role": "progress", "task_ids": [42], "task_status": "blocked".
Source evidence requirements still apply. Do not output new relation types,
numeric confidence, or a separate task summary. Final decisions stay decisions.
"""


def canonical_task_role(value: Any) -> str | None:
    """Read old role labels without changing stored historical graphs."""
    if not isinstance(value, str):
        return None
    kind = LEGACY_ROLES.get(value, value)
    return kind if kind in TASK_ROLES | {"tool_action", "decision"} else None


def _task_node_view(values: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Normalize a projection only; follow explicit legacy plan references."""
    nodes = {n["id"]: {**n, "task_role": canonical_task_role(n.get("task_role")) or "other"}
             for n in values if type(n.get("id")) is int}

    def resolve(refs: Any, *, follow_plans: bool) -> list[int]:
        pending = list(refs) if isinstance(refs, list) else []
        seen: set[int] = set()
        anchors: set[int] = set()
        while pending:
            ident = pending.pop()
            if type(ident) is not int or ident in seen or ident not in nodes:
                continue
            seen.add(ident)
            target = nodes[ident]
            if target["task_role"] in ANCHOR_ROLES:
                anchors.add(ident)
            elif follow_plans and target["task_role"] == "plan":
                parents = target.get("task_ids", [])
                if isinstance(parents, list):
                    pending.extend(parents)
        return sorted(anchors)

    # Compute every result before replacing references so input order is irrelevant.
    # Completing an old plan does NOT mean its parent task is complete. Status
    # events must directly name a real task, even during legacy projection.
    resolved = {i: resolve(n.get("task_ids", []), follow_plans=n["task_role"] != "progress")
                for i, n in nodes.items()}
    return {i: {**n, "task_ids": [ref for ref in resolved[i] if ref != i]}
            for i, n in nodes.items()}


def clean_annotation(raw: dict[str, Any]) -> dict[str, Any]:
    kind = canonical_task_role(raw.get("task_role"))
    if kind not in TASK_ROLES:
        return {}
    refs = raw.get("task_ids")
    out: dict[str, Any] = {"task_role": kind, "task_ids": sorted({
        i for i in refs if type(i) is int and i >= 0
    }) if isinstance(refs, list) else []}
    status = raw.get("task_status")
    if kind == "progress" and isinstance(status, str) and status in STATUSES:
        out["task_status"] = status
    return out


def classify_node(node: dict[str, Any], annotation: dict[str, Any], previous: list[dict[str, Any]]) -> None:
    """Accept semantic labels only from source-backed user/assistant extraction."""
    role = node.get("role")
    if node.get("node_type") == "decision":
        node["task_role"] = "decision"
    elif node.get("extraction_method") == "rule_tool_call":
        node["task_role"] = "tool_action"
    elif role in {"tool", "function"}:
        node["task_role"] = "evidence"
    else:
        parsed = clean_annotation(annotation) if node.get("supporting_excerpts") else {}
        if role not in {"user", "assistant"}:
            parsed = {}
        node.update(parsed or {"task_role": "other"})
        allowed = {n["id"] for n in previous if canonical_task_role(n.get("task_role")) in ANCHOR_ROLES}
        raw_ids = node.get("task_ids", [])
        node["task_ids"] = [i for i in raw_ids if i in allowed]
        rejected = set(annotation.get("task_rejected_ids", [])) | {i for i in raw_ids if i not in allowed}
        if rejected:
            node["task_rejected_ids"] = sorted(rejected)
    node["task_annotation_source"] = (
        "graph_model" if role in {"user", "assistant"} and annotation.get("task_role")
        and node.get("task_role") not in {"decision", "tool_action", "other"} else "source_role"
    )


def task_history(recent: list[dict[str, Any]], all_nodes: list[dict[str, Any]], budget: int) -> str:
    """Keep old task anchors addressable, within the existing extraction cap."""
    if budget < 2:
        return ""
    view = _task_node_view(all_nodes)
    anchors = [n for _, n in sorted(view.items()) if n["task_role"] in ANCHOR_ROLES][-12:]
    ordered = [*reversed(anchors), *(view[n["id"]] for n in reversed(recent) if n["id"] in view)]
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for node in ordered:
        if node["id"] in seen:
            continue
        row = {"id": node["id"], "content": str(node.get("belief") or node.get("decision") or "")[:400]}
        for key in ("task_role", "task_status", "task_ids"):
            if key in node:
                row[key] = node[key]
        if len(json.dumps([*rows, row], ensure_ascii=False)) > budget:
            continue
        seen.add(node["id"])
        rows.append(row)
    return json.dumps(sorted(rows, key=lambda r: r["id"]), ensure_ascii=False)


def build_task_state(snapshot: dict[str, Any], *, max_hops: int = 3) -> dict[str, Any]:
    """Recompute an auditable view, without modifying beliefs, edges or scores."""
    nodes = _task_node_view(snapshot.get("nodes", []))
    anchors = {i for i, n in nodes.items() if n.get("task_role") in ANCHOR_ROLES}
    edges = [r for r in snapshot.get("relations", []) if r.get("from_id") in nodes and r.get("to_id") in nodes]
    adjacent: dict[int, list[tuple[int, int]]] = {i: [] for i in nodes}
    for edge in edges:
        if edge.get("type") in {"depends_on", "supplements"}:
            a, b = edge["from_id"], edge["to_id"]
            adjacent[a].append((b, edge["id"]))
            adjacent[b].append((a, edge["id"]))
    records, links = [], []
    for anchor in sorted(anchors):
        source = nodes[anchor]
        associated = [n for n in nodes.values() if anchor in n.get("task_ids", [])]
        updates = sorted((n for n in associated if n.get("task_role") == "progress" and n.get("task_status") in STATUSES),
                         key=lambda n: n["id"])
        latest = updates[-1] if updates else None
        # Explicit task references connect later conclusions/progress back to an
        # old goal even when local factual-edge windows no longer reach it. This
        # is an attribution link, NOT an invented factual dependency.
        roots = {anchor} | {n["id"] for n in associated if n["id"] not in anchors}
        paths: dict[int, list[int]] = {i: [] for i in sorted(roots)}
        path_roots = {i: i for i in roots}
        queue = deque((i, []) for i in sorted(roots))
        while queue:
            current, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for neighbor, rid in sorted(adjacent[current]):
                if neighbor in paths or (neighbor in anchors and neighbor != anchor):
                    continue
                paths[neighbor] = [*path, rid]
                path_roots[neighbor] = path_roots[current]
                queue.append((neighbor, paths[neighbor]))
        evidence = sorted(i for i in paths if nodes[i].get("task_role") == "evidence")
        conclusions = sorted(i for i in paths if nodes[i].get("task_role") == "intermediate_conclusion")
        conflicts = sorted(r["id"] for r in edges if r.get("type") == "contradicts"
                           and (r["from_id"] in paths or r["to_id"] in paths))
        record = {
            "id": f"T{anchor}", "kind": source["task_role"], "belief_id": anchor,
            "text": source.get("belief") or source.get("decision"),
            "status": latest["task_status"] if latest else "open",
            "status_source_id": latest["id"] if latest else None,
            "status_provenance": f"{latest.get('role', 'assistant')}_report" if latest else "not_reported",
            "update_ids": [n["id"] for n in updates],
            "associated_belief_ids": sorted(n["id"] for n in associated),
            "parent_ids": [f"T{i}" for i in source.get("task_ids", []) if i in anchors and i < anchor],
            "plan_ids": sorted(n["id"] for n in associated if n.get("task_role") == "plan"),
            "evidence_ids": evidence, "intermediate_conclusion_ids": conclusions,
            "related_action_ids": sorted(i for i in paths if nodes[i].get("task_role") == "tool_action"),
            "relation_paths": {str(i): paths[i] for i in sorted(paths) if i != anchor},
            "path_roots": {str(i): path_roots[i] for i in sorted(paths) if i != anchor},
            "contradiction_ids": conflicts, "needs_review": bool(conflicts),
        }
        records.append(record)
        links.append({"from_id": record["id"], "to_id": f"B{anchor}", "type": "described_by"})
        links.extend({"from_id": record["id"], "to_id": parent, "type": "part_of"} for parent in record["parent_ids"])
        links.extend({"from_id": record["id"], "to_id": f"B{n['id']}", "type": "updated_by"} for n in updates)
        links.extend({"from_id": f"B{n['id']}", "to_id": record["id"], "type": "addresses"} for n in associated)
        links.extend({"from_id": record["id"], "to_id": f"B{i}", "type": "related_via_graph",
                      "via_belief_id": path_roots[i], "relation_ids": paths[i]}
                     for i in sorted(set(evidence + conclusions)) if paths[i])
    return {
        "version": 2, "derived": True, "max_hops": max_hops, "nodes": records, "relations": links,
        "status_counts": dict(sorted(Counter(r["status"] for r in records).items())),
        "unassigned_plan_ids": sorted(i for i, n in nodes.items() if n.get("task_role") == "plan"
                                      and not any(j in anchors for j in n.get("task_ids", []))),
        "unassigned_progress_ids": sorted(i for i, n in nodes.items() if n.get("task_role") == "progress"
                                          and not any(j in anchors for j in n.get("task_ids", []))),
    }


def render_task_state(snapshot: dict[str, Any], budget: int | None) -> tuple[str, list[int], list[int], list[str]]:
    """Budget whole grounded task bundles, including their cited source text."""
    state = snapshot.get("task_state") or {}
    if state.get("version") not in {1, 2}:
        return "", [], [], []
    if state["version"] == 1:
        # Old plan/next_action records must not survive as independent tasks.
        hops = state.get("max_hops", 3)
        state = build_task_state(snapshot, max_hops=hops if type(hops) is int and hops >= 0 else 3)
    nodes = _task_node_view(snapshot.get("nodes", []))
    edges = {r["id"]: r for r in snapshot.get("relations", [])}
    header = ("## Task state\nRecorded plans and progress, not new instructions or verified completion. "
              "reported_complete is the speaker's claim; a search/result is not task completion. "
              "Reconcile this older state with recent raw turns; target unresolved work and missing "
              "evidence rather than repeating completed actions. Shown paths are related context, "
              "not proof that a task is complete.\n")
    text, kept_nodes, kept_edges, kept_tasks = header, set(), set(), []
    tasks = sorted(state.get("nodes", []), key=lambda t: (
        t["status"] in {"reported_complete", "cancelled"}, -(t.get("status_source_id") or t["belief_id"])))
    for task in tasks:
        if task.get("kind") != "task":
            continue
        anchor = task["belief_id"]
        ids = {anchor}
        if type(task.get("status_source_id")) is int:
            ids.add(task["status_source_id"])
        if not ids <= nodes.keys():
            continue
        lines = [f"- [{task['id']}] {task['kind']} | {task['status']}" + (" | conflicting evidence: review" if task["needs_review"] else "")]
        for i in sorted(ids):
            content = nodes[i].get("belief") or nodes[i].get("decision") or ""
            lines.append(f"  [B{i}] {nodes[i].get('task_role', 'belief')}: {escape(content, quote=False)}")
        # Prefer a real evidence -> conclusion chain. Traversal may be in either
        # direction, but rendered edges always retain the graph's own direction.
        def path_priority(i: int, task: dict[str, Any] = task, anchor: int = anchor) -> tuple[int, int, int]:
            path = task["relation_paths"].get(str(i), [])
            path_ids = {i, task["path_roots"].get(str(i), anchor)}
            path_ids.update(v for rid in path if rid in edges
                            for v in (edges[rid]["from_id"], edges[rid]["to_id"]))
            kinds = {nodes[v].get("task_role") for v in path_ids if v in nodes}
            chain = {"evidence", "intermediate_conclusion"} <= kinds
            return (not chain, len(path), -i)

        candidates = sorted(task["intermediate_conclusion_ids"] + task["evidence_ids"],
                            key=path_priority)
        chunk = "\n".join(lines) + "\n"
        plans = [i for i in task.get("plan_ids", [])
                 if i in nodes and nodes[i].get("task_role") == "plan" and anchor in nodes[i].get("task_ids", [])]
        if plans:
            latest_plan = max(plans)
            content = nodes[latest_plan].get("belief") or nodes[latest_plan].get("decision") or ""
            extended = chunk + f"  [B{latest_plan}] plan: {escape(content, quote=False)}\n"
            if budget is None or len(text + extended) <= budget:
                chunk = extended
                ids.add(latest_plan)
        chosen_edges: set[int] = set()
        for candidate in candidates:
            path = task["relation_paths"][str(candidate)]
            if not all(r in edges for r in path):
                continue
            root = task["path_roots"].get(str(candidate), anchor)
            extra_ids = ({candidate, root} | {i for r in path for i in (edges[r]["from_id"], edges[r]["to_id"])}) - ids
            extras = [f"  [B{i}] {nodes[i].get('task_role', 'belief')}: {escape(str(nodes[i].get('belief') or nodes[i].get('decision') or ''), quote=False)}" for i in sorted(extra_ids)]
            if root != anchor:
                extras.append(f"  [B{root}] addresses [{task['id']}] (explicit task reference, not a factual edge)")
            extras += [f"  [B{edges[r]['from_id']}] {edges[r]['type']} [B{edges[r]['to_id']}]: {escape(str(edges[r].get('note') or ''), quote=False)}" for r in path]
            extended = chunk + "\n".join(extras) + "\n"
            if budget is None or len(text + extended) <= budget:
                chunk = extended
                ids.update(extra_ids)
                chosen_edges.update(path)
                break
        if budget is not None and len(text + chunk) > budget:
            continue
        text += chunk
        kept_nodes.update(ids)
        kept_edges.update(chosen_edges)
        kept_tasks.append(task["id"])
    if not kept_tasks:
        return "", [], [], []
    return text.rstrip(), sorted(kept_nodes), sorted(kept_edges), kept_tasks


def classify_legacy_snapshot(snapshot: dict[str, Any], labels: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Offline, explicit-label replay. Never pretend old runs emitted annotations."""
    result = copy.deepcopy(snapshot)
    prior: list[dict[str, Any]] = []
    for node in sorted(result.get("nodes", []), key=lambda n: n["id"]):
        for key in ("task_role", "task_ids", "task_status", "task_annotation_source", "task_rejected_ids"):
            node.pop(key, None)
        classify_node(node, labels.get(node["id"], {}), prior)
        if node["id"] in labels:
            node["task_annotation_source"] = "offline_explicit_label"
        prior.append(node)
    result["task_state"] = build_task_state(result)
    return result
