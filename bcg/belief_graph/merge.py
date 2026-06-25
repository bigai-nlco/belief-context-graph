"""Belief merge and deduplication pass."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bcg.belief_graph.confidence import record_merge_confidence
from bcg.belief_graph.evidence import evidence_key
from bcg.belief_graph.extraction import TextGenerator
from bcg.belief_graph.prompts import (
    BELIEFS_LIST_PLACEHOLDER,
    CANDIDATE_GROUP_PLACEHOLDER,
    PROMPT_MERGE_FULL,
    PROMPT_MERGE_VERIFY,
)
from bcg.belief_graph.split import cosine_similarity_matrix
from bcg.belief_graph.utils import save_json
from bcg.graph import BCG
from bcg.llm import parse_json_response
from bcg.utils import utc_now


@dataclass(frozen=True, slots=True)
class MergeResult:
    skipped: bool
    skip_reason: str | None = None
    strategy: str | None = None
    applied: list[dict[str, Any]] | None = None
    relation_rewire: dict[str, Any] | None = None
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "strategy": self.strategy,
            "applied": self.applied or [],
            "relation_rewire": self.relation_rewire,
            "log_path": self.log_path,
        }


async def run_merge_pass(
    *,
    graph: BCG,
    strategy: str,
    generator: TextGenerator,
    embedder: Any | None = None,
    threshold: float = 0.86,
    max_tokens: int | None = None,
    pass_label: str = "merge",
    log_dir: Path | None = None,
) -> MergeResult:
    """Run one conservative merge pass over active BCG belief nodes."""

    active = graph.belief_dicts()
    if strategy == "off" or len(active) < 2:
        return MergeResult(
            skipped=True,
            skip_reason="strategy off" if strategy == "off" else "fewer than 2 beliefs",
            applied=[],
        )
    if strategy == "embedding" and embedder is None:
        strategy = "llm"

    log: dict[str, Any] = {
        "pass": pass_label,
        "started_at": utc_now().isoformat(),
        "strategy": strategy,
        "threshold": threshold if strategy == "embedding" else None,
        "n_beliefs": len(active),
        "beliefs": [_compact_for_merge(belief) for belief in active],
    }
    allowed_ids = {int(belief["id"]) for belief in active}
    used_ids: set[int] = set()
    confirmed: list[dict[str, Any]] = []
    by_id = {int(belief["id"]): belief for belief in active}

    if strategy == "embedding":
        candidate_groups, pairs = _embedding_candidates(
            active,
            embedder,
            threshold,
            pass_label,
        )
        log["embedding_model"] = getattr(embedder, "model", None)
        log["candidate_pairs"] = pairs
        log["candidate_groups"] = candidate_groups
        log["llm_verifications"] = []
        for group_ids in candidate_groups:
            group_beliefs = [by_id[belief_id] for belief_id in group_ids]
            prompt = PROMPT_MERGE_VERIFY.replace(
                CANDIDATE_GROUP_PLACEHOLDER,
                _belief_blob(group_beliefs),
            )
            try:
                raw = await generator.generate_text(
                    prompt,
                    label=f"merge:{pass_label}:verify",
                    temperature=None,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                log["llm_verifications"].append(
                    {"candidate_ids": group_ids, "error": str(exc)}
                )
                continue
            groups = _parse_merge_groups(raw, set(group_ids), used_ids)
            log["llm_verifications"].append(
                {
                    "candidate_ids": group_ids,
                    "raw_output": raw,
                    "accepted_groups": groups,
                }
            )
            confirmed.extend(groups)
    else:
        prompt = PROMPT_MERGE_FULL.replace(
            BELIEFS_LIST_PLACEHOLDER, _belief_blob(active)
        )
        try:
            raw = await generator.generate_text(
                prompt,
                label=f"merge:{pass_label}",
                temperature=None,
                max_tokens=max_tokens,
            )
            confirmed = _parse_merge_groups(raw, allowed_ids, used_ids)
            log["llm_full"] = {"raw_output": raw, "accepted_groups": confirmed}
        except Exception as exc:
            log["llm_full"] = {"error": str(exc)}

    applied: list[dict[str, Any]] = []
    mapping: dict[int, int] = {}
    for group in sorted(confirmed, key=lambda item: item["ids"][0]):
        record = _apply_merge_group(
            graph,
            group["ids"],
            group.get("canonical_belief"),
            group.get("reason", ""),
            pass_label,
        )
        applied.append(
            {
                key: record[key]
                for key in (
                    "canonical_id",
                    "absorbed_ids",
                    "newest_id",
                    "adopted_confidence",
                    "canonical_belief",
                    "reason",
                )
            }
        )
        for absorbed_id in record["absorbed_ids"]:
            mapping[absorbed_id] = record["canonical_id"]

    rewire = graph.remap_relations(mapping) if mapping else None
    log["applied_merges"] = applied
    log["relation_rewire"] = rewire
    log["finished_at"] = utc_now().isoformat()
    log_path = _write_merge_log(log, log_dir) if log_dir is not None else None

    return MergeResult(
        skipped=False,
        strategy=strategy,
        applied=applied,
        relation_rewire=rewire,
        log_path=str(log_path) if log_path else None,
    )


def _embedding_candidates(
    beliefs: list[dict[str, Any]],
    embedder: Any,
    threshold: float,
    pass_label: str,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    texts = [str(belief.get("belief") or "") for belief in beliefs]
    vectors = embedder.embed(texts, purpose=f"merge:{pass_label}")
    sim = cosine_similarity_matrix(vectors)
    union_find = _UnionFind(len(beliefs))
    pairs: list[dict[str, Any]] = []
    for left in range(len(beliefs)):
        for right in range(left + 1, len(beliefs)):
            similarity = float(sim[left][right])
            if similarity < threshold:
                continue
            union_find.union(left, right)
            pairs.append(
                {
                    "id_a": beliefs[left]["id"],
                    "id_b": beliefs[right]["id"],
                    "similarity": round(similarity, 4),
                    "belief_a": texts[left],
                    "belief_b": texts[right],
                }
            )

    groups_by_root: dict[int, list[int]] = {}
    for index in range(len(beliefs)):
        groups_by_root.setdefault(union_find.find(index), []).append(index)
    groups = [
        sorted(int(beliefs[index]["id"]) for index in indices)
        for indices in groups_by_root.values()
        if len(indices) >= 2
    ]
    groups.sort(key=lambda group: group[0])
    return groups, pairs


def _parse_merge_groups(
    raw: str,
    allowed_ids: set[int],
    used_ids: set[int],
) -> list[dict[str, Any]]:
    parsed = parse_json_response(raw)
    groups_in = parsed.get("merge_groups", []) if isinstance(parsed, dict) else []
    groups: list[dict[str, Any]] = []
    for group in groups_in or []:
        if not isinstance(group, dict):
            continue
        try:
            ids = sorted({int(item) for item in group.get("ids") or []})
        except (TypeError, ValueError):
            continue
        ids = [belief_id for belief_id in ids if belief_id in allowed_ids]
        ids = [belief_id for belief_id in ids if belief_id not in used_ids]
        if len(ids) < 2:
            continue
        used_ids.update(ids)
        canonical = group.get("canonical_belief")
        reason = group.get("reason") or ""
        groups.append(
            {
                "ids": ids,
                "canonical_belief": canonical.strip()
                if isinstance(canonical, str) and canonical.strip()
                else None,
                "reason": reason.strip() if isinstance(reason, str) else str(reason),
            }
        )
    return groups


def _apply_merge_group(
    graph: BCG,
    ids: list[int],
    canonical_belief: str | None,
    reason: str,
    pass_label: str,
) -> dict[str, Any]:
    ids = sorted(ids)
    canonical_id = ids[0]
    absorbed_ids = ids[1:]
    newest_id = ids[-1]
    by_id = {belief.id: belief.model_dump(mode="json") for belief in graph.beliefs()}
    canonical = copy.deepcopy(by_id[canonical_id])
    absorbed_snapshots = [copy.deepcopy(by_id[belief_id]) for belief_id in absorbed_ids]
    adopted_confidence = float(by_id[newest_id].get("confidence") or 0.0)
    adopted_dimensions = by_id[newest_id].get("confidence_dimensions")

    seen = {evidence_key(evidence) for evidence in canonical.get("evidence") or []}
    merged_evidence = list(canonical.get("evidence") or [])
    for snapshot in absorbed_snapshots:
        for evidence in snapshot.get("evidence") or []:
            key = evidence_key(evidence)
            if key in seen:
                continue
            seen.add(key)
            merged_evidence.append(evidence)
    canonical["evidence"] = merged_evidence
    canonical["supporting_excerpts"] = list(
        dict.fromkeys(
            evidence.get("text") for evidence in merged_evidence if evidence.get("text")
        )
    )

    merged_from = list(canonical.get("merged_from") or [])
    for absorbed_id, snapshot in zip(absorbed_ids, absorbed_snapshots, strict=False):
        merged_from.append(absorbed_id)
        merged_from.extend(snapshot.get("merged_from") or [])
    canonical["merged_from"] = sorted(set(merged_from))

    for snapshot in reversed(absorbed_snapshots):
        if not canonical.get("event_time") and snapshot.get("event_time"):
            canonical["event_time"] = snapshot["event_time"]
        if not canonical.get("time_text") and snapshot.get("time_text"):
            canonical["time_text"] = snapshot["time_text"]

    if canonical_belief and canonical_belief != canonical.get("belief"):
        canonical.setdefault("belief_original", canonical.get("belief"))
        canonical["belief"] = canonical_belief

    record_merge_confidence(
        canonical,
        adopted_confidence,
        newest_id,
        absorbed_ids,
        reason,
        adopted_dimensions if isinstance(adopted_dimensions, dict) else None,
    )
    graph.update_belief(canonical)
    for absorbed_id in absorbed_ids:
        graph.remove_belief(absorbed_id, drop_edges=False)

    record = {
        "pass": pass_label,
        "applied_at": utc_now().isoformat(),
        "canonical_id": canonical_id,
        "absorbed_ids": absorbed_ids,
        "newest_id": newest_id,
        "adopted_confidence": round(adopted_confidence, 3),
        "canonical_belief": canonical["belief"],
        "canonical_belief_original": canonical.get("belief_original"),
        "reason": reason,
        "absorbed_snapshots": absorbed_snapshots,
    }
    graph.merges.append(record)
    return record


def _compact_for_merge(belief: dict[str, Any]) -> dict[str, Any]:
    source = belief.get("source") or {}
    compact = {
        "id": belief.get("id"),
        "source": source.get("type"),
        "session": source.get("session_index"),
        "turn": source.get("turn_index"),
        "stance": belief.get("stance"),
        "confidence": belief.get("confidence"),
        "belief": belief.get("belief"),
    }
    if belief.get("event_time"):
        compact["event_time"] = belief.get("event_time")
    if belief.get("time_text"):
        compact["time_text"] = belief.get("time_text")
    return compact


def _belief_blob(beliefs: list[dict[str, Any]]) -> str:
    return json.dumps([_compact_for_merge(belief) for belief in beliefs], indent=2)


def _write_merge_log(log: dict[str, Any], log_dir: Path) -> Path:
    path = log_dir / f"merge_{log['pass']}.json"
    save_json(log, path)
    return path


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[max(root_left, root_right)] = min(root_left, root_right)
