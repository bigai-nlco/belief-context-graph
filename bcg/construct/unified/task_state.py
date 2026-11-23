"""Grounded task-state projection over beliefs; no model calls or new confidence.

Task records have their own namespace. Their links describe workflow state and
never participate in the belief graph's factual-confidence propagation.
"""
from __future__ import annotations

import copy
import json
import re
import unicodedata
from collections import Counter, deque
from collections.abc import Sequence
from html import escape
from typing import Any

ANCHOR_ROLES = frozenset({"task"})
TASK_ROLES = ANCHOR_ROLES | {"plan", "evidence", "intermediate_conclusion", "constraint", "progress", "other"}
LEGACY_ROLES = {"goal": "task", "subgoal": "task", "next_action": "plan"}
STATUSES = frozenset({"open", "in_progress", "blocked", "reported_complete", "cancelled"})
ROOT_REF = "ROOT"  # A reference to the original input, never a factual node or a T ID.
READABLE_VERSIONS = frozenset({1, 2, 3, 4})

_ASSISTANT_WORKFLOW = """## Task-state annotations
First extract the substantive beliefs of THIS turn. Keep independently checkable findings, figures,
constraints and unresolved questions separate; a progress summary does not replace these facts.
Then label EVERY belief: task = a checkable work item the assistant chooses; plan = an intended method
or next action; evidence = reported observation; intermediate_conclusion = inference; constraint =
restriction; progress = explicit change in work status; other = none of these.
ROOT is the original assignment, a read-only reference. Restating it or listing its requested outputs is
not a new task. Selecting a bounded part can be: cite this choice and the part of ROOT still outstanding
(narrower_scope), or a separately checkable new output (new_deliverable), in root_refinement.
A search method is a plan, not a task. For a plan, gap or progress update, identify its target:
use task_ids for a shown task it addresses; use task_scope="root" when it addresses the original
assignment (including a specific subject's gap or plan when no shown child task covers it); leave
unattached if uncertain. When the assistant explicitly notes missing substantive evidence for a named
subject, program, country or analytical dimension, preserve the belief and attach a verbatim gap_quote
that includes both the subject and what is missing (do not bury per-subject gaps inside generic
progress, and do not mark routine tool/fetch error counts without a missing subject as gap_quote). For
a concrete next step, next_action_quote must name the specific subject or mechanism to check: select
the informative clause or bullet, not its heading or trailing generic intentions to keep searching.
Copy quotes verbatim from a cited current sentence.
Only explicit reported progress sets task_status. A gap is not necessarily a blocker. A tool result,
confidence or report submission never proves completion. Only close a shown gap when the current
speaker explicitly answers that same gap; cite the answer. The output shows the available fields:
use only applicable ones, never copy placeholders, and do not create nodes just to populate a field."""

_USER_DELIVERABLES = """When the user explicitly requests separately checkable
outcomes for named subjects, retain those outcomes as distinct, self-contained
task beliefs rather than merging all subjects into one large task."""

_ROLE_GUIDANCE = """## Task-state annotations
In this SAME extraction, add task_role to each new belief: task (an explicit
requested outcome or bounded work item whose completion can be tracked), plan
(a proposed method or intended next action for a task), evidence (observed or
attributed information), intermediate_conclusion (a derived claim or hypothesis),
constraint, progress, or other. Only task creates a task record. Goals and subtasks
both use task; methods and next actions both use plan. A search/fetch intention is
a plan, not another task. Do not duplicate one intention as both task and plan.
Retain explicit useful tasks, plans, blockers and progress updates, but omit
generic procedural filler. """ + _USER_DELIVERABLES + """ Preserve each
outcome's scope and constraints. Do not invent subtasks or methods not stated in
the current turn. Keep observations separate from derived conclusions.
For a plan or blocker, preserve the specific missing evidence or unresolved
question and the target task when explicit. Generic intentions to keep searching
are not progress. A tool failure records a failed attempt, not completion or
absence of the sought fact.

Optional task_ids lists integer IDs of SHOWN historical task beliefs explicitly
addressed by the new belief. For a task, these identify its parent task(s); for a
plan, its target task(s); for progress, the task(s) being updated. Never reference
a plan as a task. Use [] if the target is uncertain or not shown. A plan without
a known target stays a plan; do not invent a task to attach it to.
Do not use IDs from other sessions, guess an ID, or refer to an output-array index.
"""
_REFINEMENT_GUIDANCE = """Rephrasing an existing outcome, shortening its description, dropping examples,
or switching from "the user wants" to "the report must" is NOT a subtask.
Keep a source-backed restatement as other with the existing task_ids, not as a
new root, a new task or a progress update. Use plan for a proposed method.
Before making a child task, ask: if it were finished, what part of the parent's
requested work would still be open? For task_role=task with nonempty task_ids,
task_refinements is REQUIRED inside that belief, one object per proposed parent,
as in the Output format below. Copy short contiguous spans in their original
languages. 42 in that format is an example number, not a suggested parent.
For narrower_scope, child_quote names the bounded work in the current source;
parent_quote names the wider scope or a different part still outstanding in the
shown parent's text. Do not cherry-pick one subject when the child covers them all.
Alternatively use kind=new_deliverable: child_quote names a distinct, independently
checkable output first specified here; parent_quote names the objective it serves.
If the output was already listed among several, use narrower_scope instead.
Both quotes must support this particular belief. If neither test holds, do not
claim a child. Omit task_refinements for independent tasks and all non-task roles.
"""
_STATUS_GUIDANCE = """For progress only, add task_status: open | in_progress | blocked |
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
EXTRACTION_GUIDANCE = _ROLE_GUIDANCE + _STATUS_GUIDANCE


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
    if kind == "task":
        proofs = raw.get("task_refinements")
        if isinstance(proofs, list):
            out["task_refinements"] = [
                {k: proof[k] for k in ("parent_id", "kind", "parent_quote", "child_quote")}
                for proof in proofs if isinstance(proof, dict)
                and type(proof.get("parent_id")) is int and proof["parent_id"] in out["task_ids"]
                and proof.get("kind") in ("narrower_scope", "new_deliverable")
                and all(isinstance(proof.get(k), str) and proof[k].strip() for k in ("parent_quote", "child_quote"))
            ]
        if (refs or proofs) and not out["task_ids"]:
            # A malformed claim about a parent must not turn into an independent task.
            return {"task_role": "other", "task_ids": []}
    status = raw.get("task_status")
    if kind == "progress" and isinstance(status, str) and status in STATUSES:
        out["task_status"] = status
    if raw.get("task_scope") == "root" and not refs:
        out["task_scope"] = "root"
    proof = raw.get("root_refinement")
    if (kind == "task" and isinstance(proof, dict) and proof.get("kind") in ("narrower_scope", "new_deliverable")
            and all(isinstance(proof.get(k), str) and proof[k].strip() for k in ("parent_quote", "child_quote"))):
        out["root_refinement"] = {k: proof[k] for k in ("kind", "parent_quote", "child_quote")}
    for field in ("gap_quote", "next_action_quote"):
        if field == "next_action_quote" and kind != "plan":
            continue
        if isinstance(raw.get(field), str) and raw[field].strip():
            out[field] = raw[field].strip()
    if isinstance(raw.get("closes_gaps"), list):
        out["closes_gaps"] = [{"id": p["id"], "quote": p["quote"].strip()} for p in raw["closes_gaps"]
                              if isinstance(p, dict) and type(p.get("id")) is int
                              and isinstance(p.get("quote"), str) and p["quote"].strip()]
    return out


def _text_key(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()).strip(" .。!?！？")


def _has_quote(text: str, quote: str) -> bool:
    text, quote = _text_key(text), _text_key(quote)
    if not quote or not any(ch.isalnum() for ch in quote):
        return False
    # E.g. do not ground 'UK' in 'Ukraine'. CJK spans need no spaces.
    left = r"(?<![a-z0-9])" if quote[0].isascii() and quote[0].isalnum() else ""
    right = r"(?![a-z0-9])" if quote[-1].isascii() and quote[-1].isalnum() else ""
    return re.search(left + re.escape(quote) + right, text) is not None


def _is_abridgement(quote: str, text: str) -> bool:
    """Order-preserving omissions are not evidence of a NEW deliverable.

    Use only to reject a claimed contrast, never to discover parents. Keep
    numbers and word order; don't treat character overlap as Chinese semantics.
    """
    if _has_quote(text, quote):
        return True
    def words(value: str) -> list[str]:
        return re.findall(r"[^\W_]+", re.sub(r"^\d+(?:[.)](?=\s)|、)\s*", "", _text_key(value)))
    target = words(quote)
    if len(target) < 2:
        return False
    tokens = iter(words(text))
    return all(token in tokens for token in target)


def shown_task_texts(graph_nodes: str) -> dict[int, str]:
    try:
        rows = json.loads(graph_nodes)
    except (TypeError, ValueError):
        return {}
    if not isinstance(rows, list):
        return {}
    return {r["id"]: r.get("content") or r.get("belief") or "" for r in rows
            if isinstance(r, dict) and type(r.get("id")) is int
            and canonical_task_role(r.get("task_role")) in ANCHOR_ROLES}


def extraction_guidance(has_parents: bool, *, grounded_assistant: bool = False) -> str:
    # An Assistant chooses work; a user listing desired outcomes is not a decomposition.
    if grounded_assistant:
        return _ASSISTANT_WORKFLOW + ("\n" + _REFINEMENT_GUIDANCE if has_parents else "")
    return _ROLE_GUIDANCE + (_REFINEMENT_GUIDANCE if has_parents else "") + _STATUS_GUIDANCE


def shown_workflow(history: str) -> tuple[str, set[int]]:
    try:
        rows = json.loads(history)
    except (ValueError, TypeError):
        return "", set()
    rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
    root = next((r["content"] for r in rows if r.get("ref") == ROOT_REF
                 and isinstance(r.get("content"), str)), "")
    return root, {r["id"] for r in rows if type(r.get("id")) is int and r.get("gap_quote")}


def restrict_to_shown_tasks(nodes: list[dict[str, Any]], anchors: dict[int, str], *,
                            root: str = "", gaps: set[int] | None = None) -> None:
    """Historical IDs and quoted parent text must actually be in this model request."""
    for node in nodes:
        refs = node.get("task_ids", [])
        node["task_rejected_ids"] = [i for i in refs if i not in anchors]
        node["task_ids"] = [i for i in refs if i in anchors]
        if not root:
            for k in ("task_scope", "root_refinement", "gap_quote", "closes_gaps", "next_action_quote"):
                node.pop(k, None)
        else:
            if not _has_quote(root, node.get("root_refinement", {}).get("parent_quote", "")):
                node.pop("root_refinement", None)
            if "closes_gaps" in node:
                node["closes_gaps"] = [p for p in node["closes_gaps"] if p["id"] in (gaps or set())]
        if node.get("task_role") != "task":
            continue
        key = _text_key(node.get("belief", ""))
        duplicates = [i for i, text in anchors.items() if key and _text_key(text) == key]
        if duplicates:
            node.update(task_role="other", task_ids=duplicates)
            node.pop("task_refinements", None)
        elif "task_refinements" in node:
            node["task_refinements"] = [p for p in node["task_refinements"]
                                        if p["parent_id"] in anchors and _has_quote(anchors[p["parent_id"]], p["parent_quote"])]


def _valid_refinement(proof: dict[str, Any], child: dict[str, Any], parent: dict[str, Any]) -> bool:
    claim, objective = child.get("belief", ""), parent.get("belief", "")
    focus, wider = proof["child_quote"], proof["parent_quote"]
    if not claim or not objective or _text_key(claim) == _text_key(objective) or not _has_quote(objective, wider):
        return False
    if not any(_has_quote(s, focus) for s in child.get("supporting_excerpts", [])):
        return False
    if _text_key(focus) == _text_key(wider):
        return False
    parent_texts = [objective, *parent.get("supporting_excerpts", [])]
    already_specified = any(_is_abridgement(focus, s) for s in parent_texts)
    if proof["kind"] == "narrower_scope":
        if already_specified and (_is_abridgement(focus, wider) or
                                  any(_text_key(wider) == _text_key(s) for s in parent_texts)):
            # Quoting the whole old outcome alongside an abridged copy proves
            # no boundary. A previously named component can still cite a
            # specific outstanding peer (e.g. Nepal vs Pakistan).
            return False
        if not _has_quote(claim, wider) and not _has_quote(focus, wider):
            return True  # An explicit remaining part of the parent is outside the child.
        # A new literal boundary can refine the same named subject (e.g.
        # "Nepal" -> "Nepal, 2018–2020"). Require the qualifier in the claim
        # AND its cited source, not just a shared subject or different wording.
        return not already_specified and not _has_quote(focus, wider) and _has_quote(claim, focus)
    # An already specified output is not a NEW deliverable. An explicitly
    # separated component of an existing list can still pass the scope test.
    return not already_specified


def classify_node(node: dict[str, Any], annotation: dict[str, Any], previous: list[dict[str, Any]], *,
                  original_task: str = "") -> None:
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
        allowed = {n["id"]: n for n in previous if canonical_task_role(n.get("task_role")) in ANCHOR_ROLES}
        raw_ids = node.get("task_ids", [])
        node["task_ids"] = [i for i in raw_ids if i in allowed]
        rejected = set(annotation.get("task_rejected_ids", [])) | {i for i in raw_ids if i not in allowed}
        if rejected:
            node["task_rejected_ids"] = sorted(rejected)
        proofs = node.pop("task_refinements", [])
        if node["task_role"] == "task" and (raw_ids or rejected):
            valid = {}
            for proof in proofs:
                i = proof["parent_id"]
                if i in node["task_ids"] and _valid_refinement(proof, node, allowed[i]):
                    valid.setdefault(i, proof)
            unproven = sorted(set(node["task_ids"]) - valid.keys())
            if unproven:
                node["task_unproven_parent_ids"] = unproven
            if valid:
                node["task_ids"] = sorted(valid)
                node["task_refinements"] = [valid[i] for i in sorted(valid)]
            else:
                # Keep the sourced belief and its valid addresses links, but do
                # not create a child OR silently promote it to a top-level task.
                node["task_role"] = "other"
        if original_task and role == "assistant" and parsed:
            cited = node.get("supporting_excerpts", [])
            if node["task_role"] == "task" and not node["task_ids"]:
                proof = node.get("root_refinement")
                if (node.get("task_scope") != "root" or not proof
                        or not _valid_refinement(proof, node, {"belief": original_task})):
                    node["task_role"] = "other"
            if node["task_role"] != "task" or node.get("task_ids"):
                node.pop("root_refinement", None)
            for field in ("next_action_quote", "gap_quote"):
                if field in node and not any(_has_quote(s, node[field]) for s in cited):
                    node.pop(field)
            action = node.get("next_action_quote", "")
            if action.endswith((":", "：")) and any(_text_key(s) == _text_key(action) for s in cited):
                # A list heading by itself does not name the following targets.
                # Preserve the plan belief; never substitute an unquoted target.
                node.pop("next_action_quote")
            if "closes_gaps" in node:
                node["closes_gaps"] = [p for p in node["closes_gaps"] if any(_has_quote(s, p["quote"]) for s in cited)]
            # Never promote an invalid named target to ROOT, even after filtering.
            if rejected or node.get("task_ids"):
                node.pop("task_scope", None)
            if node["task_role"] == "plan":
                node["task_workflow_version"] = 4  # Distinguish generic new plans from legacy plans.
        else:
            # Disabled, legacy and classifier paths cannot volunteer workflow-v4 annotations.
            for field in ("task_scope", "root_refinement", "gap_quote", "next_action_quote", "closes_gaps"):
                node.pop(field, None)
    node["task_annotation_source"] = (
        "graph_model" if role in {"user", "assistant"} and annotation.get("task_role")
        and node.get("task_role") not in {"decision", "tool_action", "other"} else "source_role"
    )


def task_history(recent: list[dict[str, Any]], all_nodes: list[dict[str, Any]], budget: int, *,
                 original_task: str = "") -> str:
    """Keep old task anchors addressable, within the existing extraction cap."""
    if budget < 2:
        return ""
    view = _task_node_view(all_nodes)
    anchors = [n for _, n in sorted(view.items()) if n["task_role"] in ANCHOR_ROLES][-12:]
    unresolved, _ = _gap_ledger(view) if original_task else ({}, {})
    ordered = [*reversed(anchors), *(view[i] for i in sorted(unresolved, reverse=True)[:8]),
               *(view[n["id"]] for n in reversed(recent) if n["id"] in view)]
    rows: list[dict[str, Any]] = []
    root_rows = []
    if original_task:
        cap = min(4000, budget // 3)
        excerpt = original_task if len(original_task) <= cap else (
            original_task[:cap * 2 // 3] + "\n[…]\n" + original_task[-max(1, cap // 3):])
        row = {"ref": ROOT_REF, "content": excerpt}
        if cap and len(json.dumps([row], ensure_ascii=False)) <= budget:
            root_rows.append(row)
    seen: set[int] = set()
    for node in ordered:
        if node["id"] in seen:
            continue
        row = {"id": node["id"], "content": str(node.get("belief") or node.get("decision") or "")[:400]}
        for key in ("task_role", "task_status", "task_ids"):
            if key in node:
                row[key] = node[key]
        if node["id"] in unresolved:
            row["gap_quote"] = node["gap_quote"]
        if len(json.dumps([*root_rows, *rows, row], ensure_ascii=False)) > budget:
            continue
        seen.add(node["id"])
        rows.append(row)
    return json.dumps(root_rows + sorted(rows, key=lambda r: r["id"]), ensure_ascii=False)


def _targets(node: dict[str, Any]) -> set[int | str]:
    return set(node.get("task_ids", ())) or ({ROOT_REF} if node.get("task_scope") == "root" else set())


def _gap_ledger(nodes: dict[int, dict[str, Any]]) -> tuple[dict[int, set[int | str]], dict[int, int]]:
    """A closure must cite an earlier open gap and cover its explicit scope (if any)."""
    opened, closed = {}, {}
    for i, n in sorted(nodes.items()):
        targets = _targets(n)
        for p in n.get("closes_gaps", []):
            j = p["id"]
            if j in opened and opened[j] <= targets:
                closed[j] = i
                del opened[j]
        if n.get("gap_quote") and (targets or not n.get("task_rejected_ids")):
            opened[i] = targets
    return opened, closed


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
    open_gaps, closed_gaps = _gap_ledger(nodes)
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
            "open_gap_ids": sorted(i for i, targets in open_gaps.items() if anchor in targets),
        }
        records.append(record)
        links.append({"from_id": record["id"], "to_id": f"B{anchor}", "type": "described_by"})
        proofs = {f"T{p['parent_id']}": p for p in source.get("task_refinements", []) if isinstance(p, dict) and "parent_id" in p}
        for parent in record["parent_ids"]:
            link: dict[str, Any] = {"from_id": record["id"], "to_id": parent, "type": "part_of"}
            if parent in proofs:  # Old snapshots have no refinement evidence; don't invent it.
                link["basis"] = {k: proofs[parent][k] for k in ("kind", "parent_quote", "child_quote") if k in proofs[parent]}
            links.append(link)
        links.extend({"from_id": record["id"], "to_id": f"B{n['id']}", "type": "updated_by"} for n in updates)
        links.extend({"from_id": f"B{n['id']}", "to_id": record["id"], "type": "addresses"} for n in associated)
        links.extend({"from_id": record["id"], "to_id": f"B{i}", "type": "related_via_graph",
                      "via_belief_id": path_roots[i], "relation_ids": paths[i]}
                     for i in sorted(set(evidence + conclusions)) if paths[i])
    global_items = [n for n in nodes.values() if _targets(n) == {ROOT_REF}]
    root_updates = sorted(n["id"] for n in global_items if n.get("task_role") == "progress"
                          and n.get("task_status") in STATUSES)
    return {
        "version": 4, "derived": True, "max_hops": max_hops, "nodes": records, "relations": links,
        "root": {"ref": ROOT_REF, "plan_ids": sorted(n["id"] for n in global_items if n.get("task_role") == "plan"),
                 "update_ids": root_updates, "status_source_id": root_updates[-1] if root_updates else None,
                 "open_gap_ids": sorted(i for i, targets in open_gaps.items() if ROOT_REF in targets)},
        "gap_closures": [{"gap_id": j, "source_id": i} for j, i in sorted(closed_gaps.items())],
        "status_counts": dict(sorted(Counter(r["status"] for r in records).items())),
        "unassigned_gap_ids": sorted(i for i, targets in open_gaps.items() if not targets),
        "unassigned_plan_ids": sorted(i for i, n in nodes.items() if n.get("task_role") == "plan"
                                      and not _targets(n)),
        "unassigned_progress_ids": sorted(i for i, n in nodes.items() if n.get("task_role") == "progress"
                                          and not _targets(n)),
    }


_GENERIC_ACTION_WORDS = frozenset({
    "a", "about", "access", "across", "additional", "after", "again", "all", "also", "an", "analysis",
    "and", "another", "any", "are", "aren", "aspect", "aspects", "available", "be", "been", "before",
    "begin", "beyond", "both", "but", "by", "cached", "can", "categories", "category", "check",
    "checking", "collect", "collected", "collecting", "compile", "compiling", "comprehensive",
    "continue", "continuing", "could", "countries", "country", "cover", "d", "data", "detailed",
    "details", "do", "does", "doesn", "doing", "don", "each", "eight", "ensure", "exclusion",
    "exclusions", "existing", "explore", "explored", "failure", "failures", "fetch", "fetching",
    "file", "files", "fill", "filling", "finalize", "finalizing", "find", "finding", "findings",
    "first", "five", "focus", "focusing", "for", "four", "from", "further", "gap", "gaps", "gather",
    "gathering", "get", "getting", "has", "hasn", "have", "haven", "i", "in", "info", "information",
    "into", "is", "isn", "issue", "issues", "it", "item", "items", "key", "let", "ll", "look",
    "looking", "m", "made", "main", "make", "me", "mentioned", "might", "more", "my", "need", "needed",
    "next", "nine", "now", "of", "on", "one", "or", "organize", "organizing", "other", "prepare",
    "proceed", "program", "programs", "provide", "queries", "query", "re", "read", "reading",
    "relevant", "remaining", "report", "requested", "research", "researching", "result", "results",
    "returned", "s", "saved", "search", "searches", "searching", "section", "sections", "seek", "seven",
    "should", "six", "some", "source", "sources", "specific", "start", "still", "structure",
    "structuring", "such", "synthesize", "synthesizing", "systematic", "systematically", "t",
    "targeted", "ten", "that", "the", "then", "these", "this", "those", "three", "to", "topic",
    "topics", "try", "trying", "two", "use", "various", "ve", "was", "we", "web", "web_fetch",
    "web_search", "were", "what", "which", "while", "will", "with", "would", "write", "writing", "yet",
})
_GENERIC_CJK_TERMS = (
    "继续搜索", "进一步搜索", "继续检索", "继续查找", "搜索", "检索", "查找", "收集", "获取", "整理", "汇总",
    "阅读", "查看", "读取", "编写", "撰写", "完成", "更多", "相关", "具体", "详细", "全面", "进一步", "已有",
    "剩余", "信息", "资料", "数据", "结果", "内容", "来源", "报告", "结构", "文件", "问题", "缺口", "下一步",
    "接下来", "需要", "准备", "开始", "尝试", "进行", "我们", "我将", "先", "再", "并", "的", "了", "和", "与",
    "以", "及", "等", "对", "中", "在", "将", "把",
)
_LEADING_LIST_MARKER_RE = re.compile(r"^(?:[-*•]+|\d+(?:[.)]|、))\s+")


def _clean_item_quote(value: str) -> str:
    """Strip orphan list bullets/numbers copied from enumerated source sentences."""
    return _LEADING_LIST_MARKER_RE.sub("", (value or "").strip()).strip()


def _gap_text(node: dict[str, Any]) -> str:
    """Keep the subject alongside a short gap clause using its verbatim source sentence when compact."""
    quote = (node.get("gap_quote") or "").strip()
    if not quote:
        return ""
    for s in node.get("supporting_excerpts") or ():
        if isinstance(s, str) and len(s.strip()) <= 360 and _has_quote(s, quote):
            return _clean_item_quote(s)
    belief = (node.get("belief") or "").strip()
    if belief and not _has_quote(quote, belief):
        return belief if _has_quote(belief, quote) else f"{belief} ({_clean_item_quote(quote)})"
    return _clean_item_quote(quote)


def _is_generic_action(quote: str) -> bool:
    """Detect pure procedural search/report transition filler that names no concrete target."""
    text = _clean_item_quote(quote)
    if not text:
        return True
    if re.search(r"\d{2,}|/[A-Za-z0-9_.-]+", text):
        return False
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]+", text))
    if cjk:
        rest = cjk
        for term in _GENERIC_CJK_TERMS:
            rest = rest.replace(term, "")
        if rest:
            return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]*", text)
    if any(len(w) >= 2 and any(c.isupper() for c in w[1:]) for w in words):
        return False
    return all(w.casefold() in _GENERIC_ACTION_WORDS for w in words)


def _item_signature(text: str) -> frozenset[str]:
    """Distinctive non-boilerplate tokens for collapsing reworded duplicate gaps/plans."""
    cleaned = _clean_item_quote(text)
    tokens = {
        w.casefold()
        for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9_/-]*", cleaned)
        if w.casefold() not in _GENERIC_ACTION_WORDS
    }
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]+", cleaned))
    if cjk:
        rest = cjk
        for term in _GENERIC_CJK_TERMS:
            rest = rest.replace(term, "")
        tokens.update(rest)
    return frozenset(tokens)


def _is_redundant_signature(sig: frozenset[str], seen_sigs: list[frozenset[str]]) -> bool:
    if not sig:
        return False
    return any(
        sig <= prev or (len(sig & prev) / max(1, len(sig | prev)) >= 0.75)
        for prev in seen_sigs
    )


_STATUS_PHRASES = {
    "open": "open (reported)",
    "in_progress": "in progress (reported)",
    "blocked": "blocked (reported)",
    "reported_complete": "complete (reported, not verified)",
    "cancelled": "cancelled (reported)",
}
_DONE_STATUSES = frozenset({"reported_complete", "cancelled"})


def _task_status_text(task: dict[str, Any]) -> str:
    """Explain the recorded state without upgrading it to verified progress."""
    if task["status"] == "open" and task.get("status_provenance") == "not_reported":
        return "no progress reported"
    return _STATUS_PHRASES[task["status"]]


# Display-only: the stored belief keeps the extractor's third-person wording.
_SELF_REPORT_RE = re.compile(
    r"^(?:the assistant|the agent|the report)\s+(?:"
    r"(?:reports|reported|concludes|assesses|notes|finds|found|states|indicates|observes|confirms|"
    r"determines|believes|judges|infers|acknowledges|explains|says|records|summarizes|attributes|describes|"
    r"considers|recognizes|recommends|proposes)(?:\s+that)?"
    r"|has (?:found|gathered|collected|obtained)(?: information stating)?(?:\s+that)?"
    r"|identifies a need (?:to|for)"
    r"|plans to|planned to|intends to|will|has a task to|has the task to|is tasked to|decides to|aims to)\s+"
    r"|^(?:basic information has been found|research has already covered)\s*:\s*",
    re.I,
)
_SELF_MODAL_RE = re.compile(
    r"^(?:the assistant|the agent)\s+(?=(?:must|should|cannot|can't|may not|needs to|still needs|lacks)\b)",
    re.I,
)
# A note about the research PROCESS hitting a wall, as opposed to a finding
# about the subject (e.g. "elite capture blocked benefits" is not matched).
_ATTEMPT_FAILURE_RE = re.compile(
    r"\b(?:search\w*|quer(?:y|ies)|fetch\w*|url\w*|source\w*|site\w*|page\w*|article\w*|access\w*)\b"
    r"[^.;]{0,80}\b(?:blocked|failed|failing|unavailable|inaccessible|timed out|denied)\b"
    r"|\b(?:blocked|failed)\b[^.;]{0,30}\b(?:search\w*|quer(?:y|ies)|fetch\w*|url\w*)\b"
    r"|\bweb_fetch\b[^.;]{0,60}\b(?:fail\w*|error\w*|unavailable)"
    r"|\b(?:failed|unable) to (?:fetch|access|retrieve|load|open)\b",
    re.I,
)
# Runtime / sandbox / scratchpad mechanics that are not domain findings.
_PROCESS_META_RE = re.compile(
    r"\b(?:pip|pandas|numpy|yfinance|virtual environment|python environment|python code|run_python_code|"
    r"sandbox|tool_call|tool calls|structured tool_call)\b[^.;]{0,80}"
    r"\b(?:not installed|unavailable|lacks?|failed|cannot|prevent\w*|used to compile|did not)\b"
    r"|\bdid not (?:make|emit|issue) any tool calls\b"
    r"|\bS\d+\s+(?:mentions separate files|provides only a README|is README documentation)\b"
    r"|^(?:having (?:found|gathered|collected|obtained)\b[^0-9%;:]*$)"
    r"|^(?:from search results|search results include|finding mentions of)\b",
    re.I,
)
_TOOL_BLOCKED_RE = re.compile(r"\[BLOCKED\]|source is unavailable|blocked for this task|unavailable for this task", re.I)
_TOOL_FAILED_RE = re.compile(r"fetch unavailable|failed to fetch|\bHTTP [45]\d\d\b|timed out", re.I)

_FAILURE_WORDS = frozenset({
    "blocked", "block", "failed", "failing", "failure", "failures", "fail", "unavailable", "inaccessible",
    "error", "errors", "repeatedly", "multiple", "times", "attempts", "attempt", "searches", "search",
    "web_fetch", "fetches", "fetch", "queries", "query", "valueerror", "were", "have", "been", "has",
})
_INVENTORY_WORDS = frozenset({
    "belief", "graph", "current", "knowledge", "base", "includes", "include", "including", "contains",
    "findings", "finding", "collected", "gathered", "information", "reported", "problems", "issues",
    "studies", "results", "mixed", "some", "substantial", "programme", "programs", "program",
})
_CONSTRAINT_WORDS = frozenset({
    "assistant", "agent", "user", "report", "must", "not", "cannot", "avoid", "using", "use", "cite",
    "citing", "access", "accessing", "view", "viewing", "quote", "quoting", "article", "review", "paper",
    "excluded", "blocked", "specified", "mentioned", "instructions", "priority", "rule", "urls", "url",
    "while", "researching", "requires", "prohibited", "lacks", "lack", "unavailable", "preventing",
})
_GAP_BOILERPLATE_WORDS = frozenset({
    "assistant", "agent", "lacks", "lack", "missing", "still", "needs", "need", "needed", "identifies",
    "outstanding", "unresolved", "open", "gaps", "gap", "specific", "detailed", "details", "detail",
    "authoritative", "explanations", "explanation", "studies", "study", "academic", "identified",
    "complete", "completing", "section", "table", "part", "possible", "methods", "method", "obtain",
    "obtaining", "difficult", "challenges", "facing", "accessing", "figures", "figure",
})
_ACTIVE_GAP_HORIZON = 8  # Matches task_history's maximum closable gap window.
_RECENT_FINDING_WINDOW_TURNS = 3  # Covered by the recent <belief_graph> window below <task_state>.
_STATE_HEADER = ("## Task state\n"
                 "Your own research state from earlier turns, rebuilt from the full graph. "
                 "Statuses, findings and plans are self-reported, not independently verified.")
_SECTIONS = (
    "### Task status and constraints",
    "### Established findings (your earlier syntheses)",
    "### Open gaps (evidence still missing)",
    "### Blocked or failed attempts (do not repeat as-is)",
    "### Latest plan (intended next steps, not verified as done)",
)


def _display(node: dict[str, Any]) -> str:
    text = _clean_item_quote(" ".join(str(node.get("belief") or node.get("decision") or "").split()))
    match = _SELF_REPORT_RE.match(text) or _SELF_MODAL_RE.match(text)
    if match and match.end() < len(text):
        stripped_that = match.group(0).rstrip().lower().endswith("that")
        rest = re.sub(r"^that,\s*", "", text[match.end():], flags=re.I)
        if rest != text[match.end():]:
            stripped_that = True
        text = re.sub(r"^(?:it|its)\s+", "", rest, flags=re.I)
        if stripped_that:  # "reports that A, and that B" -> "A, and B"
            text = re.sub(r"\b(and|but|while) that\s+", r"\1 ", text)
        if "_" not in text.split(" ", 1)[0]:
            text = text[:1].upper() + text[1:]
    return escape(text, quote=False)


def _content_tokens(text: str, *, extra_stop: frozenset[str] = frozenset()) -> frozenset[str]:
    words = re.findall(r"\d+(?:\.\d+)?%?|[^\W\d_][\w-]*", unicodedata.normalize("NFKC", text).casefold())
    return frozenset(w for w in words if len(w) > 2 and w not in _GENERIC_ACTION_WORDS and w not in extra_stop)


def _turn(node: dict[str, Any]) -> int | None:
    source = node.get("source")
    turn = source.get("turn_id") if isinstance(source, dict) else None
    return turn if type(turn) is int else None


def _distinct(
    ids: Sequence[int],
    nodes: dict[int, dict[str, Any]],
    *,
    newest_first: bool = True,
    cover_threshold: float = 0.65,
    jaccard_threshold: float = 0.55,
    extra_stop: frozenset[str] = frozenset(),
) -> list[int]:
    """Collapse reworded duplicates. Newer statements usually supersede older ones."""
    chosen: list[int] = []
    seen_keys: set[str] = set()
    seen_sigs: list[frozenset[str]] = []
    for i in sorted(set(ids), reverse=newest_first):
        text = _display(nodes[i])
        key = _text_key(text)
        if not key or key in seen_keys:
            continue
        sig = _content_tokens(text, extra_stop=extra_stop) or _item_signature(text)
        if any(
            sig and (
                sig <= prev
                or len(sig & prev) / max(1, len(sig | prev)) >= jaccard_threshold
                or len(sig & prev) / len(sig) >= cover_threshold
            )
            for prev in seen_sigs
        ):
            continue
        seen_keys.add(key)
        if sig:
            seen_sigs.append(sig)
        chosen.append(i)
    return sorted(chosen)


def _is_substantive_finding(node: dict[str, Any], text: str) -> bool:
    """Keep conclusion-grade or multi-fact/quantitative syntheses, not scratchpad chatter."""
    if _PROCESS_META_RE.search(text) or _PROCESS_META_RE.search(str(node.get("belief") or "")):
        return False
    if node["task_role"] == "intermediate_conclusion":
        return True
    own = _content_tokens(text, extra_stop=_INVENTORY_WORDS)
    if re.search(r"\d+(?:\.\d+)?%|\$\d+", text) or len(re.findall(r"\b\d+(?:\.\d+)?\b", text)) >= 2:
        return len(own) >= 4
    entities = node.get("entities") or ()
    return len(own) >= 8 and (len(entities) >= 2 or ";" in text or text.count(",") >= 2)


def _normalize_attempt_target(tool_name: str, target: str) -> str:
    cleaned = target.strip().strip("[] ")
    if tool_name == "web_fetch":
        cleaned = re.sub(r"^https?://(?:www\.)?", "", cleaned, flags=re.I).rstrip("/")
    return _text_key(cleaned)


def _call_target(text: str) -> str:
    """The exact query/URL of a rule-extracted tool call, as the Agent sent it."""
    match = re.search(r"arguments\s*(\{.*\})", text)
    if match:
        try:
            args = json.loads(match[1])
        except ValueError:
            args = None
        if isinstance(args, dict):
            for key in ("url", "query", "q", "urls", "queries"):
                value = args.get(key)
                if isinstance(value, list):
                    value = "; ".join(str(v) for v in value)
                if isinstance(value, str) and value.strip():
                    return value
    quoted = re.search(r'(?:search for|call|fetch)\s+"([^"]+)"', text)
    return quoted[1] if quoted else ""


def render_task_state(snapshot: dict[str, Any], budget: int | None) -> tuple[str, list[int], list[int], list[str]]:
    """Render the workspace in five fixed sections; subtasks are a tree inside the first.

    Every rendered item cites its real [Bn]; one belief body appears once. With a
    budget, whole items are kept in priority order (status, constraints, subtasks,
    plan, gaps, blocked attempts, findings) and never truncated mid-sentence.
    """
    state = snapshot.get("task_state") or {}
    if state.get("version") not in READABLE_VERSIONS:
        return "", [], [], []
    if state["version"] == 1:
        hops = state.get("max_hops", 3)
        state = build_task_state(snapshot, max_hops=hops if type(hops) is int and hops >= 0 else 3)
    nodes = _task_node_view(snapshot.get("nodes", []))
    edges = {r["id"]: r for r in snapshot.get("relations", []) if type(r.get("id")) is int}
    root = state.get("root") or {}
    tasks = {t["belief_id"]: t for t in state.get("nodes", [])
             if t.get("kind") == "task" and t.get("belief_id") in nodes}
    active_tasks = {i: t for i, t in tasks.items() if t["status"] not in _DONE_STATUSES}
    # (section, priority, order, lines, belief ids, edge ids, task ids)
    items: list[tuple[int, int, int, list[str], set[int], set[int], list[str]]] = []
    shown: set[int] = set()

    def tag(node: dict[str, Any]) -> str:
        refs = [f"[T{i}]" for i in node.get("task_ids", []) if i in tasks]
        return f" (for {', '.join(refs)})" if refs else ""

    def add(section: int, priority: int, order: int, lines: list[str], ids: set[int],
            edge_ids: set[int] = frozenset(), task_ids: Sequence[str] = ()) -> None:
        items.append((section, priority, order, lines, set(ids), set(edge_ids), list(task_ids)))
        shown.update(ids)

    # 1. Overall status, constraints and the subtask tree.
    status_id = root.get("status_source_id")
    if status_id in nodes:
        phrase = _STATUS_PHRASES.get(nodes[status_id].get("task_status"), "reported")
        add(0, 0, 0, [f"- Overall task (the original user request): {phrase} [B{status_id}] — "
                      f"{_display(nodes[status_id])}"], {status_id})
    elif not tasks:
        loose = [i for i in state.get("unassigned_progress_ids", []) if i in nodes]
        if loose:
            add(0, 0, 0, [f"- Latest reported progress [B{max(loose)}]: {_display(nodes[max(loose)])}"], {max(loose)})
    constraints = [i for i, n in nodes.items() if n["task_role"] == "constraint"
                   and n.get("role") in {"user", "assistant"} and i not in shown]
    for i in _distinct(constraints, nodes, cover_threshold=0.50, jaccard_threshold=0.45, extra_stop=_CONSTRAINT_WORDS):
        add(0, 1, i, [f"- Constraint [B{i}]: {_display(nodes[i])}"], {i})
    if tasks:
        children: dict[int | None, list[int]] = {}
        for i, t in tasks.items():
            parents = [int(p[1:]) for p in t.get("parent_ids", []) if p[1:].isdigit() and int(p[1:]) in tasks]
            children.setdefault(max(parents) if parents else None, []).append(i)
        for kids in children.values():
            kids.sort(key=lambda i: (tasks[i]["status"] in _DONE_STATUSES, i))
        add(0, 2, 0, ["- Subtasks:"], set())
        stack = [(i, 1) for i in reversed(children.get(None, []))]
        visited: set[int] = set()
        order = 1
        while stack:
            i, depth = stack.pop()
            if i in visited:
                continue
            visited.add(i)
            t, pad = tasks[i], "  " * depth
            lines = [f"{pad}- [T{i}] {_task_status_text(t)} — {_display(nodes[i])} [B{i}]"]
            ids = {i}
            update = t.get("status_source_id")
            if update in nodes:
                body = "" if update in shown else f": {_display(nodes[update])}"
                lines.append(f"{pad}  latest update [B{update}]{body}")
                ids.add(update)
            add(0, 2, order, lines, ids, task_ids=[t["id"]])
            order += 1
            stack.extend((k, depth + 1) for k in reversed(children.get(i, [])))

    # Classify open gaps and process-level failure notes up front so findings exclude both.
    task_gap_ids = {i for t in active_tasks.values() for i in t.get("open_gap_ids", []) if i in nodes}
    global_gap_ids = {i for i in (*root.get("open_gap_ids", []), *state.get("unassigned_gap_ids", [])) if i in nodes}
    open_gaps = task_gap_ids | global_gap_ids
    failure_notes = {i for i in open_gaps if _ATTEMPT_FAILURE_RE.search(nodes[i].get("belief") or "")}
    for i, n in nodes.items():
        if (i not in shown and n.get("role") == "assistant" and _ATTEMPT_FAILURE_RE.search(n.get("belief") or "")
                and n["task_role"] in {"evidence", "intermediate_conclusion", "progress", "other"}):
            failure_notes.add(i)

    # 2. Findings: consolidated conclusions and substantive syntheses from earlier turns.
    assistant_turns = sorted({t for n in nodes.values() if n.get("role") == "assistant" and (t := _turn(n)) is not None})
    recent_window = min(_RECENT_FINDING_WINDOW_TURNS, len(assistant_turns) - 1) if len(assistant_turns) >= 2 else 0
    recent_assistant_turns = set(assistant_turns[-recent_window:]) if recent_window else set()
    closed_gaps = {c.get("gap_id") for c in state.get("gap_closures", []) if isinstance(c, dict)}
    substantive_findings = [
        i for i, n in nodes.items()
        if i not in shown and i not in open_gaps and i not in failure_notes
        and (not n.get("gap_quote") or i in closed_gaps)
        and n.get("role") == "assistant"
        and n["task_role"] in {"evidence", "intermediate_conclusion"}
        and _is_substantive_finding(n, _display(n))
    ]
    all_finding_tokens = {
        i: _content_tokens(_display(nodes[i]), extra_stop=_INVENTORY_WORDS)
        for i in substantive_findings
    }
    findings = [
        i for i in substantive_findings
        if not (nodes[i]["task_role"] == "evidence" and _turn(nodes[i]) in recent_assistant_turns)
    ]
    kept = _distinct(findings, nodes, cover_threshold=0.60, jaccard_threshold=0.50, extra_stop=_INVENTORY_WORDS)
    for i in sorted(kept, reverse=True):
        # An inventory ("the graph has findings on A, B and C") adds nothing once A, B, C are shown.
        own = all_finding_tokens[i]
        rest = set().union(*(all_finding_tokens[j] for j in kept if j != i))
        if len(kept) > 2 and own and len(own & rest) / len(own) >= 0.65:
            kept.remove(i)
    for i in kept:
        add(1, 6, i, [f"- [B{i}]{tag(nodes[i])} {_display(nodes[i])}"], {i})

    # 3. Open gaps: bounded to the active closable horizon and retired once later findings cover them.
    active_task_gaps = [
        i for i in task_gap_ids - failure_notes
        if i not in shown and not _is_generic_action(_display(nodes[i]))
    ]
    recent_global_gaps = sorted(
        (
            i for i in global_gap_ids - task_gap_ids - failure_notes
            if i not in shown and not _is_generic_action(_display(nodes[i]))
        ),
        reverse=True,
    )[:_ACTIVE_GAP_HORIZON]
    candidate_gaps = _distinct(
        [*active_task_gaps, *recent_global_gaps],
        nodes,
        cover_threshold=0.60,
        jaccard_threshold=0.50,
        extra_stop=_GAP_BOILERPLATE_WORDS,
    )
    for i in candidate_gaps:
        gap_sig = _content_tokens(_display(nodes[i]), extra_stop=_GAP_BOILERPLATE_WORDS)
        later_tokens = set().union(*(all_finding_tokens[fid] for fid in substantive_findings if fid > i), set())
        if gap_sig and len(gap_sig & later_tokens) / len(gap_sig) >= 0.60:
            continue
        add(2, 4, i, [f"- [B{i}]{tag(nodes[i])} {_display(nodes[i])}"], {i})

    # 5. Plans from the most recent turn that recorded any; older intentions retire.
    plan_ids = {*root.get("plan_ids", [])}
    for t in active_tasks.values():
        plan_ids.update(t.get("plan_ids", []))
    plan_ids = {i for i in plan_ids if i in nodes and i not in shown}
    if plan_ids:
        turns = {i: _turn(nodes[i]) for i in plan_ids}
        latest = max((v for v in turns.values() if v is not None), default=None)
        current = [i for i in plan_ids if turns[i] == latest] if latest is not None else [max(plan_ids)]
        concrete = [i for i in current if not _is_generic_action(_display(nodes[i]))]
        for i in _distinct(concrete or current, nodes):
            add(4, 3, i, [f"- [B{i}]{tag(nodes[i])} {_display(nodes[i])}"], {i})

    # 4. Blocked or failed attempts: cluster exact failed tool calls; omit redundant prose notes.
    calls = {n["tool_call_id"]: i for i, n in nodes.items()
             if n.get("extraction_method") == "rule_tool_call" and n.get("tool_call_id")}
    raw_attempts: list[tuple[int, str, str, str, str, frozenset[str]]] = []
    for _, n in sorted(nodes.items()):
        body = str(n.get("belief") or "")
        if n.get("role") != "tool" or n.get("tool_call_id") not in calls:
            continue
        outcome = "blocked" if _TOOL_BLOCKED_RE.search(body) else "failed" if _TOOL_FAILED_RE.search(body) else ""
        aid = calls[n["tool_call_id"]]
        action = nodes[aid]
        tname = str(action.get("tool_name") or "tool")
        target = str(action.get("query") or action.get("url") or _call_target(str(action.get("belief") or ""))).strip()
        if not outcome or not target or aid in shown:
            continue
        norm = _normalize_attempt_target(tname, target)
        raw_attempts.append((aid, tname, outcome, target, norm, _content_tokens(target)))

    # Cluster newest-first so the representative line cites the latest attempt ID and wording.
    clusters: list[list[Any]] = []
    for aid, tname, outcome, target, norm, sig in reversed(raw_attempts):
        matched = None
        for cluster in clusters:
            c_tname, c_outcome, c_norms, c_sig = cluster[1], cluster[2], cluster[4], cluster[5]
            if tname != c_tname or outcome != c_outcome:
                continue
            if norm in c_norms or (
                tname == "web_search" and sig and c_sig and (
                    len(sig & c_sig) / max(1, len(sig | c_sig)) >= 0.50
                    or len(sig & c_sig) / len(sig) >= 0.65
                )
            ):
                matched = cluster
                break
        if matched is not None:
            matched[4].add(norm)
            matched[6] += 1
        else:
            clusters.append([aid, tname, outcome, target, {norm}, sig, 0])
    for aid, tname, outcome, target, _, _, extra_count in reversed(clusters):
        suffix = f" (+{extra_count} similar)" if extra_count else ""
        add(3, 5, 10**9 + aid, [
            f"- [B{aid}]{suffix} {tname}: \"{escape(target, quote=False)}\" → {outcome}"], {aid})

    if not clusters:
        for i in _distinct(failure_notes, nodes):
            add(3, 5, i, [f"- [B{i}]{tag(nodes[i])} {_display(nodes[i])}"], {i})
    else:
        for i in _distinct([fid for fid in failure_notes if tag(nodes[fid])], nodes):
            add(3, 5, i, [f"- [B{i}]{tag(nodes[i])} {_display(nodes[i])}"], {i})
    conflict_scope = shown | {i for t in tasks.values() for i in t.get("associated_belief_ids", [])}
    conflict_ids = {r for t in tasks.values() for r in t.get("contradiction_ids", [])}

    def ref(v: int) -> str:
        return f"[B{v}]" if v in shown else f"[B{v}] \"{_display(nodes[v])}\""

    for rid, e in sorted(edges.items()):
        a, b = e.get("from_id"), e.get("to_id")
        if e.get("type") != "contradicts" or a not in nodes or b not in nodes:
            continue
        if rid not in conflict_ids and a not in conflict_scope and b not in conflict_scope:
            continue
        note = f": {escape(e['note'], quote=False)}" if e.get("note") else ""
        add(1, 6, max(a, b), [f"- Conflict: {ref(a)} contradicts {ref(b)}{note}"], {a, b}, {rid})

    if not items:
        return "", [], [], []
    selected = items
    if budget is not None:
        selected, size, used = [], len(_STATE_HEADER) + 1, set()
        for item in sorted(items, key=lambda x: (x[1], -x[2] if x[1] == 6 else x[2])):
            extra = sum(len(line) + 1 for line in item[3]) + (0 if item[0] in used else len(_SECTIONS[item[0]]) + 1)
            if size + extra > budget:
                continue
            size += extra
            used.add(item[0])
            selected.append(item)
        if not selected or all(not x[4] for x in selected):
            return "", [], [], []
    # A subtask header without any surviving subtask is noise.
    if not any(x[6] for x in selected):
        selected = [x for x in selected if x[3] != ["- Subtasks:"]]
    lines = [_STATE_HEADER]
    for section, title in enumerate(_SECTIONS):
        block = sorted((x for x in selected if x[0] == section), key=lambda x: (x[1], x[2]))
        if block:
            lines.append(title)
            lines.extend(line for x in block for line in x[3])
    ids = sorted(set().union(*(x[4] for x in selected)))
    edge_ids = sorted(set().union(*(x[5] for x in selected)))
    task_ids = [t for x in sorted(selected, key=lambda x: (x[1], x[2])) for t in x[6]]
    return "\n".join(lines), ids, edge_ids, task_ids


def classify_legacy_snapshot(snapshot: dict[str, Any], labels: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Offline, explicit-label replay. Never pretend old runs emitted annotations."""
    result = copy.deepcopy(snapshot)
    prior: list[dict[str, Any]] = []
    for node in sorted(result.get("nodes", []), key=lambda n: n["id"]):
        for key in ("task_role", "task_ids", "task_status", "task_annotation_source", "task_rejected_ids",
                    "task_refinements", "task_unproven_parent_ids"):
            node.pop(key, None)
        classify_node(node, labels.get(node["id"], {}), prior)
        if node["id"] in labels:
            node["task_annotation_source"] = "offline_explicit_label"
        prior.append(node)
    result["task_state"] = build_task_state(result)
    return result
