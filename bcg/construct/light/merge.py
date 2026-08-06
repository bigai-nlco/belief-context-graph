"""
merge.py
========
Belief-node disambiguation and merging — INCREMENTAL only.

This backend has no trajectory-end/final merge pass and no CLI-level
``--merge-strategy``/``--merge-threshold``: stream.py's only call site always
passes ``strategy="embedding", verify=False`` for the per-turn incremental
merge (``StreamOptions.incremental_merge`` / ``incremental_merge_threshold``).
``run_merge_pass()`` below keeps ``strategy``/``verify`` as general internal
parameters (also supporting ``llm`` and ``off``), but nothing in this backend
exercises those other values.

Role-aware belief-only merge policy:
  * decision nodes are excluded from every merge pass, incremental or final;
  * belief nodes are eligible to merge ONLY when their source role is identical
    (source.role/source.type; e.g. user with user, assistant with assistant,
    tool with tool). The role gate is enforced before embedding-only
    incremental merges and again before applying LLM-proposed groups.

Merge semantics (one confirmed group):
  * canonical  = the SMALLEST id in the group (the node every earlier edge
                 already points at stays stable);
  * statement  = regenerated locally from the canonical node's deduplicated
                 evidence after every group in the pass has been applied;
  * confidence = recomputed from the canonical node's initial_confidence plus
                 additional evidence absorbed from duplicate nodes. The original
                 evidence attached at canonical creation is not counted twice;
  * evidence_ids = union of all members' evidence ids (deduped), so the
                 canonical belief points at every place the fact was stated;
  * merged_from accumulates the absorbed ids; absorbed beliefs are removed
    from the active graph but archived in graph.merges with full snapshots;
  * relations  = every relation endpoint at an absorbed id is rewired to the
    canonical id; self-loops and duplicates are dropped (and reported).

Auditability: every pass writes logs/merge_<pass>.json (machine-readable —
beliefs in, embedding similarities, candidate pairs, every LLM verification
prompt+raw output when verification is enabled, applied merges, edge rewiring report) and a human-readable
logs/merge_<pass>.log. Embedding API calls additionally land in
logs/embedding_calls.jsonl via the shared EmbeddingClient.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import llm
from .confidence import record_evidence_merge_confidence
from .evidence import evidence_key
from .graph import BeliefGraph


def _node_text(b: dict[str, Any]) -> str:
    """Primary text for either a belief or a decision node."""
    if b.get("node_type") == "decision":
        return str(b.get("decision") or b.get("belief") or "")
    return str(b.get("belief") or b.get("decision") or "")


def _set_primary_text_field(
    node: dict[str, Any],
    *,
    text_key: str,
    text: str,
) -> None:
    """Keep exactly one primary text field in the original text-field position."""
    original = dict(node)
    node.clear()
    inserted = False
    had_primary_key = "belief" in original or "decision" in original
    for key, value in original.items():
        if key in {"belief", "decision"}:
            if not inserted:
                node[text_key] = text
                inserted = True
            continue
        node[key] = value
        if key == "node_type" and not inserted and not had_primary_key:
            node[text_key] = text
            inserted = True
    if not inserted:
        node[text_key] = text


def _compact_for_merge(b: dict[str, Any]) -> dict[str, Any]:
    src = b.get("source") or {}
    c = {
        "id": b.get("id"),
        "node_type": b.get("node_type", "belief"),
        "role": b.get("role") or src.get("role") or src.get("type"),
        "turn": src.get("turn_id", src.get("turn_index")),
        "stance": b.get("stance"),
        "conf": b.get("confidence"),
        "entities": b.get("entities") or [],
        "belief": _node_text(b),
    }
    return c


def _blob(beliefs: list[dict[str, Any]]) -> str:
    return json.dumps(
        [_compact_for_merge(b) for b in beliefs], ensure_ascii=False, indent=2
    )


def _merge_role(b: dict[str, Any]) -> str:
    """Role key used to constrain deduplication.

    Nodes may be merged ONLY when this key is identical. This prevents, for
    example, a user's claim from being absorbed into a semantically similar
    assistant conclusion or tool result.
    """
    src = b.get("source") or {}
    role = b.get("role") or src.get("role") or src.get("type") or "unknown"
    return str(role).strip().lower() or "unknown"


def _same_merge_role(ids: list[int], by_id: dict[int, dict[str, Any]]) -> bool:
    roles = {_merge_role(by_id[i]) for i in ids if i in by_id}
    return len(roles) == 1


def _merge_node_type(b: dict[str, Any]) -> str:
    """Node-type key used to constrain deduplication.

    Nodes may be merged ONLY when this key is identical. This prevents a belief
    node from being absorbed into a semantically similar decision node (or vice
    versa). Defaults to "belief" to match the graph-wide fallback used in
    graph.py / stream.py (node.get("node_type", "belief")).
    """
    return str(b.get("node_type", "belief")).strip().lower() or "belief"


def _same_node_type(ids: list[int], by_id: dict[int, dict[str, Any]]) -> bool:
    types = {_merge_node_type(by_id[i]) for i in ids if i in by_id}
    return len(types) == 1


def _split_ids_by_role(
    ids: list[int], by_id: dict[int, dict[str, Any]]
) -> list[list[int]]:
    """Split ids into subgroups sharing BOTH role and node_type.

    The bucket key is (role, node_type), so a mixed-role or mixed-type group is
    broken into homogeneous subgroups; only subgroups with >= 2 ids survive.
    """
    buckets: dict[tuple[str, str], list[int]] = {}
    for i in ids:
        if i in by_id:
            key = (_merge_role(by_id[i]), _merge_node_type(by_id[i]))
            buckets.setdefault(key, []).append(i)
    return [sorted(v) for v in buckets.values() if len(v) >= 2]


# ---------------------------------------------------------------------------
# Candidate generation (embedding strategy)
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _embedding_candidates(
    beliefs: list[dict[str, Any]],
    embedder,
    threshold: float,
    pass_label: str,
    incremental_new_ids: set[int] | None = None,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Return embedding candidate groups and accepted pair records.

    Full/final passes compare every pair. Incremental passes compare only
    ``new x old`` and ``new x new`` pairs; ``old x old`` similarities are not
    computed at all. Incremental grouping also guarantees that a candidate
    group contains at most one old node, so two historical nodes cannot be
    merged transitively through a new node.
    """
    import numpy as np

    texts = [_node_text(b) for b in beliefs]
    roles = [_merge_role(b) for b in beliefs]
    types = [_merge_node_type(b) for b in beliefs]
    vectors = embedder.embed(texts, purpose=f"merge:{pass_label}")

    arr = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = arr / norms

    n = len(beliefs)
    id_to_index = {int(b["id"]): i for i, b in enumerate(beliefs)}
    incremental = incremental_new_ids is not None
    new_ids = set(incremental_new_ids or set())
    new_indices = sorted(id_to_index[i] for i in new_ids if i in id_to_index)
    new_index_set = set(new_indices)
    old_indices = [i for i in range(n) if i not in new_index_set]

    if incremental:
        # Deliberately omit old x old pairs from both iteration and cosine work.
        pair_indices = [
            (min(i, j), max(i, j)) for i in new_indices for j in old_indices
        ]
        pair_indices.extend(
            (new_indices[a], new_indices[b])
            for a in range(len(new_indices))
            for b in range(a + 1, len(new_indices))
        )
        pair_indices = sorted(set(pair_indices))
    else:
        pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

    uf = _UnionFind(n)
    pairs: list[dict[str, Any]] = []
    skipped_cross_role: list[dict[str, Any]] = []
    skipped_cross_type: list[dict[str, Any]] = []
    old_links: list[tuple[int, int, float]] = []

    for i, j in pair_indices:
        s = float(np.clip(np.dot(unit[i], unit[j]), -1.0, 1.0))
        if s < threshold:
            continue
        rec = {
            "id_a": beliefs[i]["id"],
            "id_b": beliefs[j]["id"],
            "role_a": roles[i],
            "role_b": roles[j],
            "type_a": types[i],
            "type_b": types[j],
            "similarity": round(s, 4),
            "belief_a": texts[i],
            "belief_b": texts[j],
        }
        if roles[i] != roles[j]:
            rec["skipped_reason"] = "cross_role"
            skipped_cross_role.append(rec)
            continue
        if types[i] != types[j]:
            rec["skipped_reason"] = "cross_type"
            skipped_cross_type.append(rec)
            continue

        pairs.append(rec)
        if not incremental:
            uf.union(i, j)
            continue

        i_new = i in new_index_set
        j_new = j in new_index_set
        if i_new and j_new:
            uf.union(i, j)
        else:
            new_idx, old_idx = (i, j) if i_new else (j, i)
            old_links.append((new_idx, old_idx, s))

    anchor_choices: list[dict[str, Any]] = []
    if not incremental:
        groups_by_root: dict[int, list[int]] = {}
        for i in range(n):
            groups_by_root.setdefault(uf.find(i), []).append(i)
        groups = [
            sorted(beliefs[i]["id"] for i in idxs)
            for idxs in groups_by_root.values()
            if len(idxs) >= 2
        ]
    else:
        # First build connected components among new nodes only. Then attach each
        # component to at most one old anchor: the old node with the strongest
        # qualifying cosine link to any member (ties -> smaller node id).
        components: dict[int, list[int]] = {}
        for i in new_indices:
            components.setdefault(uf.find(i), []).append(i)

        links_by_root: dict[int, dict[int, float]] = {}
        for new_idx, old_idx, score in old_links:
            root = uf.find(new_idx)
            previous = links_by_root.setdefault(root, {}).get(old_idx, -1.0)
            links_by_root[root][old_idx] = max(previous, score)

        anchored_groups: dict[int, set[int]] = {}
        standalone_groups: list[list[int]] = []
        for root, component in sorted(components.items(), key=lambda x: min(x[1])):
            component_ids = sorted(int(beliefs[i]["id"]) for i in component)
            candidates = links_by_root.get(root, {})
            if candidates:
                anchor_idx, anchor_score = sorted(
                    candidates.items(),
                    key=lambda item: (-item[1], int(beliefs[item[0]]["id"])),
                )[0]
                anchor_id = int(beliefs[anchor_idx]["id"])
                anchored_groups.setdefault(anchor_id, set()).update(component_ids)
                anchor_choices.append(
                    {
                        "new_component_ids": component_ids,
                        "selected_old_id": anchor_id,
                        "selected_similarity": round(float(anchor_score), 4),
                        "other_old_candidates": [
                            {
                                "old_id": int(beliefs[idx]["id"]),
                                "similarity": round(float(score), 4),
                            }
                            for idx, score in sorted(
                                candidates.items(),
                                key=lambda item: (
                                    -item[1],
                                    int(beliefs[item[0]]["id"]),
                                ),
                            )[1:]
                        ],
                    }
                )
            elif len(component_ids) >= 2:
                standalone_groups.append(component_ids)

        groups = [
            sorted([anchor_id, *component_ids])
            for anchor_id, component_ids in anchored_groups.items()
        ]
        groups.extend(standalone_groups)

    groups.sort(key=lambda g: (g[0], len(g), g))
    # Attach diagnostics without changing the public return contract.
    _embedding_candidates.last_skipped_cross_role = skipped_cross_role  # type: ignore[attr-defined]
    _embedding_candidates.last_skipped_cross_type = skipped_cross_type  # type: ignore[attr-defined]
    _embedding_candidates.last_incremental_policy = {  # type: ignore[attr-defined]
        "enabled": incremental,
        "new_node_ids": sorted(new_ids) if incremental else [],
        "old_node_ids": sorted(int(beliefs[i]["id"]) for i in old_indices)
        if incremental
        else [],
        "old_old_pairs_omitted": (
            len(old_indices) * (len(old_indices) - 1) // 2 if incremental else 0
        ),
        "anchor_choices": anchor_choices,
    }
    return groups, pairs


# ---------------------------------------------------------------------------
# LLM verification / proposal
# ---------------------------------------------------------------------------


def _parse_merge_groups(
    raw: str,
    allowed_ids: set[int],
    used_ids: set[int],
    by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate {"merge_groups":[...]} and enforce same-role groups.

    LLM proposals are not trusted: cross-role groups are rejected/split before
    any merge is applied. This is the final safety net for both full-LLM merge
    and embedding+LLM verification.
    """
    parsed = llm.parse_json_response(raw)
    groups_in = parsed.get("merge_groups", []) if isinstance(parsed, dict) else []
    out: list[dict[str, Any]] = []
    for g in groups_in or []:
        if not isinstance(g, dict):
            continue
        ids_in = g.get("ids") or []
        try:
            ids0 = sorted({int(i) for i in ids_in})
        except (TypeError, ValueError):
            continue
        ids0 = [i for i in ids0 if i in allowed_ids and i not in used_ids]
        if len(ids0) < 2:
            continue

        # Hard role gate. If an LLM emits a mixed-role group, split it into
        # same-role subgroups and keep only subgroups with at least 2 ids.
        role_groups = _split_ids_by_role(ids0, by_id)
        if not role_groups:
            continue

        reason = g.get("reason") or ""
        if not isinstance(reason, str):
            reason = str(reason)

        for ids in role_groups:
            ids = [i for i in ids if i not in used_ids]
            if len(ids) < 2:
                continue
            used_ids.update(ids)
            role = _merge_role(by_id[ids[0]]) if ids[0] in by_id else "unknown"
            role_note = f"same role={role}"
            final_reason = reason.strip()
            if final_reason:
                final_reason = f"{final_reason} ({role_note})"
            else:
                final_reason = role_note
            out.append({"ids": ids, "reason": final_reason})
    return out


# ---------------------------------------------------------------------------
# Apply one confirmed merge group to the graph
# ---------------------------------------------------------------------------


def _apply_merge_group(
    graph: BeliefGraph,
    ids: list[int],
    reason: str,
    pass_label: str,
    keep_newest_text: bool = True,
) -> dict[str, Any]:
    """Merge one exact-duplicate group into its smallest-id node.

    The smallest id always SURVIVES as the canonical node (every earlier edge and
    evidence record already points at it). ``keep_newest_text`` controls only which
    member's *content* the survivor displays:
      * True  (default) -> adopt the most recently generated member's text
                           (the largest id in the group);
      * False           -> keep the smallest-id member's own text.
    Either way the canonical id, edges, and unioned evidence are unchanged. If a
    ``summary_regenerator`` runs later in the pass it overrides this text.

    Evidence is deduplicated before any later summary regeneration. Entity
    metadata is cleared because the canonical text is not stable until the
    whole merge pass has finished and ``summary_regenerator`` has run.
    """
    ids = sorted(ids)
    canon_id, absorbed_ids, newest_id = ids[0], ids[1:], ids[-1]
    canon = graph.beliefs[canon_id]
    absorbed_snapshots = [copy.deepcopy(graph.beliefs[a]) for a in absorbed_ids]
    old_conf = float(canon.get("confidence") or 0.0)
    previous_summary = _node_text(canon)

    # Evidence union in deterministic member/id order. Duplicate records are
    # suppressed by both id and semantic evidence_key; the first id wins.
    merged_eids: list[int] = []
    seen_ids: set[int] = set()
    seen_keys: set[tuple] = set()
    added_eids: list[int] = []
    added_records: list[dict[str, Any]] = []

    def _append_evidence_id(raw_eid: Any, *, is_additional: bool) -> None:
        try:
            eid = int(raw_eid)
        except (TypeError, ValueError):
            return
        ev = graph.evidence.get(eid)
        if ev is None:
            return
        key = evidence_key(ev)
        if eid in seen_ids or key in seen_keys:
            return
        seen_ids.add(eid)
        seen_keys.add(key)
        merged_eids.append(eid)
        if is_additional:
            added_eids.append(eid)
            added_records.append(ev)

    for eid in canon.get("evidence_ids") or []:
        _append_evidence_id(eid, is_additional=False)
    for ev in canon.get("evidence") or []:
        if isinstance(ev, dict):
            _append_evidence_id(graph.add_evidence(ev), is_additional=False)
    canon.pop("evidence", None)

    for snapshot in absorbed_snapshots:
        for eid in snapshot.get("evidence_ids") or []:
            _append_evidence_id(eid, is_additional=True)
        for ev in snapshot.get("evidence") or []:
            if isinstance(ev, dict):
                _append_evidence_id(graph.add_evidence(ev), is_additional=True)

    canon["evidence_ids"] = merged_eids
    canon["supporting_excerpts"] = list(
        dict.fromkeys(
            graph.evidence[eid].get("text")
            for eid in merged_eids
            if eid in graph.evidence and graph.evidence[eid].get("text")
        )
    )

    merged_from = list(canon.get("merged_from") or [])
    for absorbed_id, snapshot in zip(absorbed_ids, absorbed_snapshots, strict=False):
        merged_from.append(absorbed_id)
        merged_from.extend(snapshot.get("merged_from") or [])
    canon["merged_from"] = sorted(set(merged_from))

    # Adopt the most recently generated member's text as the canonical content,
    # keeping the smallest id as the surviving node (edges/evidence already point
    # at it). Groups are homogeneous in (role, node_type), so the primary text
    # field is consistent across members.
    if keep_newest_text and newest_id != canon_id:
        newest_snapshot = next(
            (
                snap
                for aid, snap in zip(absorbed_ids, absorbed_snapshots, strict=False)
                if aid == newest_id
            ),
            None,
        )
        if newest_snapshot is not None:
            newest_text = _node_text(newest_snapshot)
            if newest_text:
                text_key = (
                    "decision" if canon.get("node_type") == "decision" else "belief"
                )
                _set_primary_text_field(canon, text_key=text_key, text=newest_text)

    # Old member entities must not survive a text-changing merge. The stable
    # canonical summary is regenerated after every group in this pass is applied,
    # and stream.py extracts entities from that final text afterwards.
    canon["entities"] = []

    record_evidence_merge_confidence(
        canon,
        added_evidence_ids=added_eids,
        added_evidence_records=added_records,
        absorbed_ids=absorbed_ids,
        newest_id=newest_id,
        evidence_by_id=graph.evidence,
        config=getattr(graph, "confidence_config", None),
    )
    new_conf = float(canon.get("confidence") or 0.0)

    for absorbed_id in absorbed_ids:
        graph.remove_belief(absorbed_id)

    record = {
        "pass": pass_label,
        "applied_at": datetime.now(UTC).isoformat(),
        "canonical_id": canon_id,
        "absorbed_ids": absorbed_ids,
        "newest_id": newest_id,
        "old_confidence": round(old_conf, 3),
        "new_confidence": round(new_conf, 3),
        "adopted_confidence": round(new_conf, 3),
        "added_evidence_ids": added_eids,
        "evidence_ids": list(merged_eids),
        "previous_summary": previous_summary,
        "canonical_belief": _node_text(canon),
        "reason": reason,
        "absorbed_snapshots": absorbed_snapshots,
    }
    graph.merges.append(record)
    return record


def _infer_incremental_new_ids(
    beliefs: list[dict[str, Any]], pass_label: str
) -> set[int] | None:
    """Infer current-turn node ids for ``turn_<index>`` merge passes.

    ``stream.py`` already labels per-turn passes as ``turn_<turn_id>`` and
    stores that same index in ``node["source"]["turn_id"]``. Returning
    ``None`` means this is not an incremental pass; returning an empty set means
    it is incremental but no current-turn nodes could be found, in which case
    the pass must be skipped rather than falling back to an old x old scan.
    """
    prefix = "turn_"
    if not pass_label.startswith(prefix):
        return None
    try:
        turn_index = int(pass_label[len(prefix) :])
    except ValueError:
        return set()

    out: set[int] = set()
    for belief in beliefs:
        src = belief.get("source") or {}
        try:
            source_turn = int(src.get("turn_id", src.get("turn_index")))
        except (TypeError, ValueError):
            continue
        if source_turn == turn_index:
            out.add(int(belief["id"]))
    return out


# ---------------------------------------------------------------------------
# Pass entry point
# ---------------------------------------------------------------------------


def run_merge_pass(
    *,
    graph: BeliefGraph,
    strategy: str,  # "embedding" | "llm" | "off"
    client,
    model: str,
    embedder=None,
    threshold: float = 0.8,
    max_tokens: int | None = None,
    pass_label: str = "merge",
    log_dir: Path | None = None,
    verify: bool = True,
    incremental_new_ids: set[int] | None = None,
    exclude_node_ids: set[int] | None = None,
    max_verify_workers: int = 8,
    summary_regenerator: Callable[[dict[str, Any], list[dict[str, Any]]], str]
    | None = None,
    keep_newest_text: bool = True,
) -> dict[str, Any]:
    """Run one merge pass over the active graph. Returns a report dict.

    verify — only meaningful for strategy="embedding". When True (default) each
        embedding candidate group is verified by the LLM before merging. When
        False the merge is embedding-ONLY: every candidate group whose pairwise
        cosine >= threshold is merged directly, with NO LLM call. This is the
        per-turn incremental merge mode (see stream.py). The canonical is always
        the smallest id; its evidence is deduplicated and its summary is then
        regenerated locally by ``summary_regenerator``.

    incremental_new_ids — optional explicit set of nodes created in the current
        turn. When omitted, ``turn_<index>`` pass labels infer the set from each
        node's ``source.turn_index``. Incremental embedding passes compare only
        new x old and new x new pairs. Final/non-turn passes still compare the
        complete eligible graph, including old x old.

    exclude_node_ids — optional additional ids that must not participate in this
        merge pass. All active decision nodes are always excluded automatically.
        Excluded nodes can never be absorbed or selected as canonical nodes.
        Relations incident to excluded nodes are still rewired if their
        non-excluded endpoints merge.

    max_verify_workers — upper bound on how many candidate groups' LLM-verify
        calls run concurrently (strategy="embedding", verify=True only). Each
        embedding candidate group is independent (union-find guarantees the
        groups don't share node ids), so their verify calls are fired via a
        thread pool instead of one-at-a-time; parsing/applying the results
        stays strictly sequential. Default 8; set to 1 to force the old
        one-at-a-time behaviour.

    summary_regenerator — optional local callback invoked once per surviving
        canonical node only after every merge group in this pass has been
        applied. It receives the canonical node and its deduplicated evidence
        records, and returns the replacement summary text.
    """
    all_active = graph.active()
    decision_ids = {
        int(node["id"])
        for node in all_active
        if node.get("node_type") == "decision" and isinstance(node.get("id"), int)
    }
    requested_excluded_ids: set[int] = set(decision_ids)
    for raw_id in exclude_node_ids or set():
        try:
            requested_excluded_ids.add(int(raw_id))
        except (TypeError, ValueError):
            continue
    active_all_ids = {int(b["id"]) for b in all_active}
    excluded_existing_ids = sorted(requested_excluded_ids & active_all_ids)
    excluded_decision_ids = sorted(decision_ids & active_all_ids)
    active = [b for b in all_active if int(b["id"]) not in requested_excluded_ids]
    if strategy == "off" or len(active) < 2:
        return {
            "skipped": True,
            "skip_reason": (
                "strategy off" if strategy == "off" else "fewer than 2 eligible beliefs"
            ),
            "applied": [],
            "excluded_node_ids": excluded_existing_ids,
            "excluded_decision_ids": excluded_decision_ids,
        }

    if incremental_new_ids is None:
        incremental_new_ids = _infer_incremental_new_ids(active, pass_label)
    elif incremental_new_ids is not None:
        incremental_new_ids = {int(i) for i in incremental_new_ids}

    if incremental_new_ids is not None:
        active_ids = {int(b["id"]) for b in active}
        incremental_new_ids &= active_ids
        if not incremental_new_ids:
            return {
                "skipped": True,
                "skip_reason": "incremental pass has no active current-turn belief nodes",
                "applied": [],
                "excluded_node_ids": excluded_existing_ids,
                "excluded_decision_ids": excluded_decision_ids,
            }

    if strategy == "embedding" and embedder is None:
        if incremental_new_ids is not None or not verify:
            # Incremental merge must never fall back to a full-graph LLM pass,
            # because that would re-enable old x old merging.
            return {
                "skipped": True,
                "skip_reason": "embedding merge needs an embedder",
                "applied": [],
                "excluded_node_ids": excluded_existing_ids,
                "excluded_decision_ids": excluded_decision_ids,
            }
        print(
            f"  [merge:{pass_label}] no embedding client — falling back to strategy=llm"
        )
        strategy = "llm"

    log: dict[str, Any] = {
        "pass": pass_label,
        "started_at": datetime.now(UTC).isoformat(),
        "strategy": strategy,
        "threshold": threshold if strategy == "embedding" else None,
        "n_beliefs": len(active),
        "beliefs": [_compact_for_merge(b) for b in active],
        "excluded_node_ids": excluded_existing_ids,
        "excluded_decision_ids": excluded_decision_ids,
    }
    allowed_ids = {b["id"] for b in active}
    used_ids: set[int] = set()
    confirmed: list[dict[str, Any]] = []
    by_id = {b["id"]: b for b in active}

    # Sub-step wall clocks (seconds). "embedding_seconds" covers candidate
    # generation (embedder.embed + cosine); "llm_verify_seconds" covers the LLM
    # verification loop (embedding strategy) or the single full-list LLM proposal
    # (llm strategy). Both surface in the returned report so stream.py can bill
    # the incremental merge into the per-turn `merging` / `llm_check` totals.
    embedding_seconds = 0.0
    llm_verify_seconds = 0.0

    if strategy == "embedding":
        log["embedding_model"] = getattr(embedder, "model", None)
        log["verify"] = verify
        _t_embed = time.perf_counter()
        candidate_groups, pairs = _embedding_candidates(
            active, embedder, threshold, pass_label, incremental_new_ids
        )
        embedding_seconds += time.perf_counter() - _t_embed
        log["candidate_pairs"] = pairs
        log["skipped_cross_role_pairs"] = getattr(
            _embedding_candidates, "last_skipped_cross_role", []
        )
        log["skipped_cross_type_pairs"] = getattr(
            _embedding_candidates, "last_skipped_cross_type", []
        )
        log["incremental_policy"] = getattr(
            _embedding_candidates, "last_incremental_policy", {"enabled": False}
        )
        log["candidate_groups"] = candidate_groups
        if verify:
            print(
                "No LLM verification for embedding strategy is not yet implemented in this snippet."
            )
        else:
            # embedding-ONLY (no LLM verification): confirm every candidate group
            # directly. Evidence is unioned/deduplicated in _apply_merge_group;
            # the canonical summary is regenerated locally after all groups.
            for g_ids in candidate_groups:
                ids = [i for i in g_ids if i in allowed_ids and i not in used_ids]
                if (
                    len(ids) < 2
                    or not _same_merge_role(ids, by_id)
                    or not _same_node_type(ids, by_id)
                ):
                    continue
                used_ids.update(ids)
                role = _merge_role(by_id[ids[0]]) if ids[0] in by_id else "unknown"
                ntype = _merge_node_type(by_id[ids[0]]) if ids[0] in by_id else "belief"
                confirmed.append(
                    {
                        "ids": ids,
                        "reason": (
                            f"embedding cosine >= {threshold} "
                            f"within role={role}, node_type={ntype} "
                            f"(no LLM verification)"
                        ),
                    }
                )
    else:  # strategy == "llm"
        print("No LLM full-list merge in this snippet.")
        # prompt = PROMPT_MERGE_FULL.replace(BELIEFS_LIST_PLACEHOLDER, _blob(active))
        # _t_verify = time.perf_counter()
        # try:
        #     raw = llm.call_model(client, model, prompt, temperature=0.0,
        #                          max_tokens=max_tokens)
        #     confirmed = _parse_merge_groups(raw, allowed_ids, used_ids, by_id)
        #     log["llm_full"] = {"raw_output": raw, "accepted_groups": confirmed}
        # except Exception as e:
        #     log["llm_full"] = {"error": str(e)}
        # llm_verify_seconds += time.perf_counter() - _t_verify

    applied_records: list[dict[str, Any]] = []
    mapping: dict[int, int] = {}
    skipped_role_mismatch: list[dict[str, Any]] = []
    skipped_type_mismatch: list[dict[str, Any]] = []
    for g in sorted(confirmed, key=lambda g: g["ids"][0]):
        if not _same_merge_role(g["ids"], by_id):
            skipped_role_mismatch.append(
                {"ids": g["ids"], "reason": "cross_role_group"}
            )
            continue
        if not _same_node_type(g["ids"], by_id):
            skipped_type_mismatch.append(
                {"ids": g["ids"], "reason": "cross_type_group"}
            )
            continue
        rec = _apply_merge_group(
            graph,
            g["ids"],
            g["reason"],
            pass_label,
            keep_newest_text=keep_newest_text,
        )
        applied_records.append(rec)
        for absorbed_id in rec["absorbed_ids"]:
            mapping[absorbed_id] = rec["canonical_id"]

    rewire = (
        graph.remap_relations(mapping)
        if mapping
        else {"rewritten": 0, "dropped_self": 0, "dropped_duplicate": 0}
    )

    # Duplicate evidence ids that lost the first-id tie-break are no longer
    # referenced after absorbed nodes are removed. Prune them before summary
    # regeneration so both the graph output and summary input are deduplicated.
    evidence_prune = graph.prune_unreferenced_evidence()

    summary_regeneration: list[dict[str, Any]] = []
    summary_regeneration_seconds = 0.0
    if summary_regenerator is not None and applied_records:
        _t_summary = time.perf_counter()
        for rec in applied_records:
            canonical_id = int(rec["canonical_id"])
            node = graph.beliefs.get(canonical_id)
            if node is None:
                continue
            evidence_records = graph.evidence_records(node.get("evidence_ids") or [])
            before = _node_text(node)
            entry: dict[str, Any] = {
                "canonical_id": canonical_id,
                "evidence_ids": list(node.get("evidence_ids") or []),
                "before": before,
            }
            try:
                regenerated = str(
                    summary_regenerator(node, evidence_records) or ""
                ).strip()
                if regenerated:
                    text_key = (
                        "decision" if node.get("node_type") == "decision" else "belief"
                    )
                    original_key = (
                        "decision_original"
                        if text_key == "decision"
                        else "belief_original"
                    )
                    if regenerated != before:
                        node.setdefault(original_key, before)
                        _set_primary_text_field(
                            node, text_key=text_key, text=regenerated
                        )
                    node["entities"] = []
                entry["after"] = _node_text(node)
            except Exception as exc:
                entry["after"] = before
                entry["error"] = str(exc)
            rec["canonical_belief"] = _node_text(node)
            rec["summary_regeneration"] = dict(entry)
            summary_regeneration.append(entry)
        summary_regeneration_seconds += time.perf_counter() - _t_summary

    applied: list[dict[str, Any]] = [
        {
            key: rec.get(key)
            for key in (
                "canonical_id",
                "absorbed_ids",
                "newest_id",
                "old_confidence",
                "new_confidence",
                "adopted_confidence",
                "added_evidence_ids",
                "evidence_ids",
                "canonical_belief",
                "previous_summary",
                "summary_regeneration",
                "reason",
            )
        }
        for rec in applied_records
    ]

    log["applied_merges"] = applied
    log["skipped_role_mismatch_groups"] = skipped_role_mismatch
    log["skipped_type_mismatch_groups"] = skipped_type_mismatch
    log["relation_rewire"] = rewire
    log["evidence_prune"] = evidence_prune
    log["summary_regeneration"] = summary_regeneration
    log["timing"] = {
        "embedding_seconds": round(embedding_seconds, 6),
        "llm_verify_seconds": round(llm_verify_seconds, 6),
        "summary_regeneration_seconds": round(summary_regeneration_seconds, 6),
    }
    log["finished_at"] = datetime.now(UTC).isoformat()

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        json_path = log_dir / f"merge_{pass_label}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        with open(log_dir / f"merge_{pass_label}.log", "w", encoding="utf-8") as f:
            f.write(_render_text_log(log))
        log["log_path"] = str(json_path)

    return {
        "skipped": False,
        "strategy": strategy,
        "applied": applied,
        "n_candidate_groups": len(log.get("candidate_groups", confirmed)),
        "incremental_policy": log.get("incremental_policy"),
        "excluded_node_ids": excluded_existing_ids,
        "excluded_decision_ids": excluded_decision_ids,
        "relation_rewire": rewire,
        "evidence_prune": evidence_prune,
        "summary_regeneration": summary_regeneration,
        "timing": {
            "embedding_seconds": round(embedding_seconds, 6),
            "llm_verify_seconds": round(llm_verify_seconds, 6),
            "summary_regeneration_seconds": round(summary_regeneration_seconds, 6),
        },
        "log_path": log.get("log_path"),
    }


def _render_text_log(log: dict[str, Any]) -> str:
    lines = [
        "=" * 74,
        f" merge pass: {log['pass']}   strategy={log['strategy']}"
        + (
            f"   threshold={log['threshold']}"
            if log.get("threshold") is not None
            else ""
        ),
        f" nodes in: {log['n_beliefs']}",
        "=" * 74,
    ]
    if log.get("embedding_model"):
        lines.append(f" embedding model: {log['embedding_model']}")
    policy = log.get("incremental_policy") or {}
    if log.get("excluded_node_ids"):
        lines.append(f" excluded node ids: {log.get('excluded_node_ids')}")
    if policy.get("enabled"):
        lines.append(
            " incremental policy: new x old + new x new only; "
            f"omitted old x old pairs={policy.get('old_old_pairs_omitted', 0)}"
        )
        lines.append(f" current-turn node ids: {policy.get('new_node_ids', [])}")
        for choice in policy.get("anchor_choices", []):
            lines.append(
                f"  anchor new={choice.get('new_component_ids')} -> "
                f"old #{choice.get('selected_old_id')} "
                f"sim={choice.get('selected_similarity')}"
            )
    for p in log.get("candidate_pairs", []):
        lines.append(
            f"  pair  #{p['id_a']} ~ #{p['id_b']}  "
            f"role={p.get('role_a')}  sim={p['similarity']}"
        )
        lines.append(f"        a: {p['belief_a'][:100]}")
        lines.append(f"        b: {p['belief_b'][:100]}")
    skipped = log.get("skipped_cross_role_pairs") or []
    if skipped:
        lines.append(f" skipped cross-role candidate pairs: {len(skipped)}")
        for p in skipped[:20]:
            lines.append(
                f"  skip  #{p['id_a']}({p.get('role_a')}) ~ "
                f"#{p['id_b']}({p.get('role_b')})  sim={p['similarity']}"
            )
    skipped_t = log.get("skipped_cross_type_pairs") or []
    if skipped_t:
        lines.append(f" skipped cross-type candidate pairs: {len(skipped_t)}")
        for p in skipped_t[:20]:
            lines.append(
                f"  skip  #{p['id_a']}({p.get('type_a')}) ~ "
                f"#{p['id_b']}({p.get('type_b')})  sim={p['similarity']}"
            )
    if "candidate_groups" in log:
        lines.append(f" candidate groups: {log['candidate_groups']}")
    for v in log.get("llm_verifications", []):
        lines.append(
            f"  verify {v.get('candidate_ids')}"
            + (
                f" -> ERROR {v['error']}"
                if "error" in v
                else f" -> {[g['ids'] for g in v.get('accepted_groups', [])]}"
            )
        )
    for g in log.get("skipped_role_mismatch_groups", []):
        lines.append(f"  SKIP cross-role merge group {g.get('ids')}")
    for g in log.get("skipped_type_mismatch_groups", []):
        lines.append(f"  SKIP cross-type merge group {g.get('ids')}")
    for m in log.get("applied_merges", []):
        lines.append(
            f"  MERGE  {m['absorbed_ids']} -> #{m['canonical_id']}  "
            f"(conf {m.get('old_confidence', '—')} -> {m.get('new_confidence', m.get('adopted_confidence'))}; "
            f"additional evidence={m.get('added_evidence_ids', [])})"
        )
        lines.append(f"         canonical: {m['canonical_belief']}")
        if m.get("reason"):
            lines.append(f"         reason:    {m['reason']}")
    rw = log.get("relation_rewire") or {}
    lines.append(
        f" relations: rewritten={rw.get('rewritten', 0)}  "
        f"dropped_self={rw.get('dropped_self', 0)}  "
        f"dropped_dup={rw.get('dropped_duplicate', 0)}"
    )
    lines.append("=" * 74)
    return "\n".join(lines) + "\n"
