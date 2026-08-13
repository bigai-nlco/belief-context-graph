"""
merge.py
========
Belief disambiguation & merging.

The only merge pass that still runs is the per-turn INCREMENTAL merge
(StreamOptions.incremental_merge / incremental_merge_threshold / verify_merge
in stream.py) — there is no more trajectory-end global merge pass or
CLI-level ``--merge-strategy``/``--merge-threshold``. ``run_merge_pass()``
below keeps its ``strategy``/``threshold`` parameters as internal API (the
incremental caller always passes ``strategy="embedding"``); the ``strategy``
values it still understands are:
  * embedding — embed every active belief statement, flag pairs with cosine
    similarity >= ``threshold``, group candidates via union-find, then
    (unless ``verify=False``) have the LLM VERIFY each candidate group (only
    LLM-confirmed duplicates are merged).
  * llm       — the LLM sees the whole belief list and proposes merge groups
    directly (no embeddings needed). Not used by the current pipeline.
  * off       — skip merging.

Role-aware & type-aware merge policy:
  * nodes are eligible to merge ONLY when their source role is identical
    (source.role/source.type; e.g. user with user, assistant with assistant,
    tool with tool) AND their node_type is identical (belief with belief,
    decision with decision). Both hard gates are enforced before embedding-only
    incremental merges and again before applying LLM-proposed groups.

Merge semantics (one confirmed group):
  * canonical  = the SMALLEST id in the group (the node every earlier edge
                 already points at stays stable);
  * statement  = the LLM's `canonical_belief` wording (original wording is
                 preserved in `belief_original` and in the merge record);
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
prompt+raw output, applied merges, edge rewiring report) and a human-readable
logs/merge_<pass>.log. Embedding API calls additionally land in
logs/embedding_calls.jsonl via the shared EmbeddingClient.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import llm
from .confidence import record_evidence_merge_confidence
from .evidence import evidence_key
from .graph import BeliefGraph
from .prompts import (
    BELIEFS_LIST_PLACEHOLDER,
    CANDIDATE_GROUP_PLACEHOLDER,
    PROMPT_MERGE_FULL,
    PROMPT_MERGE_VERIFY,
    PROMPT_MERGE_VERIFY_REWRITE,
)


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
    if b.get("event_time"):
        c["time"] = b.get("event_time")
    if b.get("time_text"):
        c["time_text"] = b.get("time_text")
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
    query_identities = {_query_identity(by_id[i]) for i in ids if i in by_id}
    return len(types) == 1 and len(query_identities) == 1


def _query_identity(node: dict[str, Any]) -> tuple[str, str, str]:
    """Merge key that keeps distinct executed queries as distinct nodes."""

    query = node.get("query")
    tool_name = str(node.get("tool_name") or "tool")
    if isinstance(query, str) and query:
        return ("query", tool_name, query)
    if node.get("extraction_method") == "rule_tool_call":
        arguments = json.dumps(
            node.get("tool_arguments") or {}, ensure_ascii=False, sort_keys=True
        )
        return ("tool_call", tool_name, arguments)
    return ("non_query", "", "")


def _split_ids_by_role(
    ids: list[int], by_id: dict[int, dict[str, Any]]
) -> list[list[int]]:
    """Split ids into subgroups sharing role, node type, and query identity.

    Query nodes merge only when both the exact tool name and query are equal.
    """
    buckets: dict[tuple[str, str, tuple[str, str, str]], list[int]] = {}
    for i in ids:
        if i in by_id:
            key = (
                _merge_role(by_id[i]),
                _merge_node_type(by_id[i]),
                _query_identity(by_id[i]),
            )
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

        canonical = g.get("canonical_belief")
        if not isinstance(canonical, str) or not canonical.strip():
            canonical = None
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
            out.append(
                {"ids": ids, "canonical_belief": canonical, "reason": final_reason}
            )
    return out


# ---------------------------------------------------------------------------
# Apply one confirmed merge group to the graph
# ---------------------------------------------------------------------------


def _apply_merge_group(
    graph: BeliefGraph,
    ids: list[int],
    canonical_belief: str | None,
    reason: str,
    pass_label: str,
) -> dict[str, Any]:
    ids = sorted(ids)
    canon_id, absorbed_ids, newest_id = ids[0], ids[1:], ids[-1]
    canon = graph.beliefs[canon_id]
    absorbed_snapshots = [copy.deepcopy(graph.beliefs[a]) for a in absorbed_ids]
    old_conf = float(canon.get("confidence") or 0.0)

    # evidence_ids union (order-preserving, deduped). Evidence from absorbed
    # duplicate nodes becomes ADDITIONAL evidence for the canonical node.
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
        k = evidence_key(ev)
        if eid in seen_ids or k in seen_keys:
            return
        seen_ids.add(eid)
        seen_keys.add(k)
        merged_eids.append(eid)
        if is_additional:
            added_eids.append(eid)
            added_records.append(ev)

    for eid in canon.get("evidence_ids") or []:
        _append_evidence_id(eid, is_additional=False)

    # Compatibility for very old in-memory nodes that still carry embedded
    # evidence instead of evidence_ids. Newly generated nodes never use this.
    for ev in canon.get("evidence") or []:
        if isinstance(ev, dict):
            _append_evidence_id(graph.add_evidence(ev), is_additional=False)
    canon.pop("evidence", None)

    for snap in absorbed_snapshots:
        for eid in snap.get("evidence_ids") or []:
            _append_evidence_id(eid, is_additional=True)
        for ev in snap.get("evidence") or []:
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

    # merged_from accumulates across passes (absorbed nodes' own merged_from too)
    merged_from = list(canon.get("merged_from") or [])
    for a, snap in zip(absorbed_ids, absorbed_snapshots, strict=False):
        merged_from.append(a)
        merged_from.extend(snap.get("merged_from") or [])
    canon["merged_from"] = sorted(set(merged_from))

    # time fields: fill canonical gaps from absorbed members (newest first)
    for snap in reversed(absorbed_snapshots):
        if not canon.get("event_time") and snap.get("event_time"):
            canon["event_time"] = snap["event_time"]
        if not canon.get("time_text") and snap.get("time_text"):
            canon["time_text"] = snap["time_text"]

    # canonical statement (preserve the original wording once)
    if (
        canonical_belief
        and canonical_belief.strip()
        and canonical_belief.strip() != _node_text(canon)
    ):
        if canon.get("node_type") == "decision":
            canon.setdefault("decision_original", _node_text(canon))
            _set_primary_text_field(
                canon,
                text_key="decision",
                text=canonical_belief.strip(),
            )
        else:
            canon.setdefault("belief_original", _node_text(canon))
            _set_primary_text_field(
                canon,
                text_key="belief",
                text=canonical_belief.strip(),
            )

    # confidence: recompute P_posterior from canonical P_prior plus additional
    # evidence.
    record_evidence_merge_confidence(
        canon,
        added_evidence_ids=added_eids,
        added_evidence_records=added_records,
        absorbed_ids=absorbed_ids,
        newest_id=newest_id,
        evidence_by_id=graph.evidence,
    )
    new_conf = float(canon.get("confidence") or 0.0)

    for a in absorbed_ids:
        graph.remove_belief(a)

    record = {
        "pass": pass_label,
        "applied_at": datetime.now(UTC).isoformat(),
        "canonical_id": canon_id,
        "absorbed_ids": absorbed_ids,
        "newest_id": newest_id,
        "old_confidence": round(old_conf, 3),
        "new_confidence": round(new_conf, 3),
        # Kept for older log consumers; this is now the recomputed posterior,
        # not an adopted newest-member value.
        "adopted_confidence": round(new_conf, 3),
        "added_evidence_ids": added_eids,
        "canonical_belief": canon.get("belief"),
        "canonical_belief_original": canon.get("belief_original"),
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
    verify_rewrite: bool = False,
    incremental_new_ids: set[int] | None = None,
    exclude_node_ids: set[int] | None = None,
    max_verify_workers: int = 8,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Run one merge pass over the active graph. Returns a report dict.

    verify — only meaningful for strategy="embedding". When True (default) each
        embedding candidate group is verified by the LLM before merging. When
        False the merge is embedding-ONLY: every candidate group whose pairwise
        cosine >= threshold is merged directly, with NO LLM call. This is the
        per-turn incremental merge mode (see stream.py). canonical_belief stays
        None in this mode, so the canonical (smallest id == earliest) keeps its
        own wording and evidence_ids are unioned onto it.

    verify_rewrite — only meaningful when verify=True. When True, the LLM
        verification step uses PROMPT_MERGE_VERIFY_REWRITE instead of
        PROMPT_MERGE_VERIFY: in one call the LLM both (1) gates each candidate
        group (only groups it confirms are merged) and (2) returns a
        canonical_belief that must cover the FULL meaning of all merged members,
        which _apply_merge_group then writes onto the surviving node. This is the
        per-turn incremental merge mode enabled by StreamOptions.verify_merge.
        The trajectory-end (final) merge leaves this False and is unchanged.

    incremental_new_ids — optional explicit set of nodes created in the current
        turn. When omitted, ``turn_<index>`` pass labels infer the set from each
        node's ``source.turn_index``. Incremental embedding passes compare only
        new x old and new x new pairs. Final/non-turn passes still compare the
        complete eligible graph, including old x old.

    exclude_node_ids — optional ids that must not participate in this merge pass
        at all. They are omitted from embedding/LLM candidate generation and can
        never be absorbed or selected as canonical nodes. Relations incident to
        excluded nodes are still rewired if their non-excluded endpoints merge.

    max_verify_workers — upper bound on how many candidate groups' LLM-verify
        calls run concurrently (strategy="embedding", verify=True only). Each
        embedding candidate group is independent (union-find guarantees the
        groups don't share node ids), so their verify calls are fired via a
        thread pool instead of one-at-a-time; parsing/applying the results
        stays strictly sequential. Default 8; set to 1 to force the old
        one-at-a-time behaviour.
    """
    all_active = graph.active()
    requested_excluded_ids: set[int] = set()
    for raw_id in exclude_node_ids or set():
        try:
            requested_excluded_ids.add(int(raw_id))
        except (TypeError, ValueError):
            continue
    active_all_ids = {int(b["id"]) for b in all_active}
    excluded_existing_ids = sorted(requested_excluded_ids & active_all_ids)
    active = [b for b in all_active if int(b["id"]) not in requested_excluded_ids]
    if strategy == "off" or len(active) < 2:
        return {
            "skipped": True,
            "skip_reason": (
                "strategy off" if strategy == "off" else "fewer than 2 eligible beliefs"
            ),
            "applied": [],
            "excluded_node_ids": excluded_existing_ids,
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
                "skip_reason": "incremental pass has no active current-turn nodes",
                "applied": [],
            }

    if strategy == "embedding" and embedder is None:
        if incremental_new_ids is not None or not verify:
            # Incremental merge must never fall back to a full-graph LLM pass,
            # because that would re-enable old x old merging.
            return {
                "skipped": True,
                "skip_reason": "embedding merge needs an embedder",
                "applied": [],
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
            verify_template = (
                PROMPT_MERGE_VERIFY_REWRITE if verify_rewrite else PROMPT_MERGE_VERIFY
            )
            log["verify_rewrite"] = verify_rewrite
            log["llm_verifications"] = []
            _t_verify = time.perf_counter()

            # Each candidate group here comes from union-find connected
            # components (see _embedding_candidates): groups are node-disjoint
            # by construction, so verifying them is an embarrassingly
            # parallel, independent LLM call per group. We fire the network
            # calls concurrently and keep the (cheap, CPU-only) parsing /
            # used_ids bookkeeping strictly sequential and in original group
            # order afterwards — that keeps _parse_merge_groups' shared
            # `used_ids` mutation race-free without needing a lock, and keeps
            # logs/merge_<pass>.json deterministic regardless of which call
            # happens to return first.
            #
            # contextvars.ContextVar values (the active USAGE tracker, the
            # prompt/embedding audit log paths — see llm.py) are NOT inherited
            # by threads spawned via ThreadPoolExecutor, so we capture them
            # once in this thread and re-bind them inside each worker before
            # it calls the LLM; otherwise token accounting and prompt auditing
            # would silently go missing for these parallel calls.
            _usage_tracker = llm.current_usage_tracker()
            _prompt_log_path = llm.current_prompt_log_path()

            def _verify_one(
                g_ids: list[int],
            ) -> tuple[list[int], str | None, str | None]:
                u_tok = llm.bind_usage_tracker(_usage_tracker)
                p_tok = (
                    llm.bind_prompt_log_path(_prompt_log_path)
                    if _prompt_log_path is not None
                    else None
                )
                try:
                    group_beliefs = [by_id[i] for i in g_ids if i in by_id]
                    prompt = verify_template.replace(
                        CANDIDATE_GROUP_PLACEHOLDER, _blob(group_beliefs)
                    )
                    try:
                        raw = llm.call_model(
                            client,
                            model,
                            prompt,
                            temperature=0.0,
                            max_tokens=max_tokens,
                            reasoning_effort=reasoning_effort,
                        )
                        return g_ids, raw, None
                    except Exception as e:
                        return g_ids, None, str(e)
                finally:
                    llm.unbind_usage_tracker(u_tok)
                    if p_tok is not None:
                        llm.unbind_prompt_log_path(p_tok)

            n_workers = max(1, min(max_verify_workers, len(candidate_groups)))
            if n_workers <= 1:
                raw_results = [_verify_one(g_ids) for g_ids in candidate_groups]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
                    # map() preserves input order in its output regardless of
                    # completion order, so downstream logging/used_ids stay
                    # identical to the old sequential-loop ordering.
                    raw_results = list(ex.map(_verify_one, candidate_groups))

            for g_ids, raw, err in raw_results:
                if err is not None:
                    log["llm_verifications"].append(
                        {"candidate_ids": g_ids, "error": err}
                    )
                    continue
                groups = _parse_merge_groups(raw, set(g_ids), used_ids, by_id)
                log["llm_verifications"].append(
                    {
                        "candidate_ids": g_ids,
                        "raw_output": raw,
                        "accepted_groups": groups,
                    }
                )
                confirmed.extend(groups)
            llm_verify_seconds += time.perf_counter() - _t_verify
        else:
            # embedding-ONLY (no LLM verification): confirm every candidate group
            # directly. canonical_belief stays None so the canonical (smallest
            # id == earliest) keeps its own wording; evidence_ids are unioned in
            # _apply_merge_group.
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
                        "canonical_belief": None,
                        "reason": (
                            f"embedding cosine >= {threshold} "
                            f"within role={role}, node_type={ntype} "
                            f"(no LLM verification)"
                        ),
                    }
                )
    else:  # strategy == "llm"
        prompt = PROMPT_MERGE_FULL.replace(BELIEFS_LIST_PLACEHOLDER, _blob(active))
        _t_verify = time.perf_counter()
        try:
            raw = llm.call_model(
                client,
                model,
                prompt,
                temperature=0.0,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            confirmed = _parse_merge_groups(raw, allowed_ids, used_ids, by_id)
            log["llm_full"] = {"raw_output": raw, "accepted_groups": confirmed}
        except Exception as e:
            log["llm_full"] = {"error": str(e)}
        llm_verify_seconds += time.perf_counter() - _t_verify

    applied: list[dict[str, Any]] = []
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
            graph, g["ids"], g["canonical_belief"], g["reason"], pass_label
        )
        applied.append(
            {
                k: rec[k]
                for k in (
                    "canonical_id",
                    "absorbed_ids",
                    "newest_id",
                    "old_confidence",
                    "new_confidence",
                    "adopted_confidence",
                    "added_evidence_ids",
                    "canonical_belief",
                    "reason",
                )
            }
        )
        for a in rec["absorbed_ids"]:
            mapping[a] = rec["canonical_id"]

    rewire = (
        graph.remap_relations(mapping)
        if mapping
        else {"rewritten": 0, "dropped_self": 0, "dropped_duplicate": 0}
    )

    log["applied_merges"] = applied
    log["skipped_role_mismatch_groups"] = skipped_role_mismatch
    log["skipped_type_mismatch_groups"] = skipped_type_mismatch
    log["relation_rewire"] = rewire
    log["timing"] = {
        "embedding_seconds": round(embedding_seconds, 6),
        "llm_verify_seconds": round(llm_verify_seconds, 6),
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
        "relation_rewire": rewire,
        "timing": {
            "embedding_seconds": round(embedding_seconds, 6),
            "llm_verify_seconds": round(llm_verify_seconds, 6),
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
