"""
merge.py
========
Session-end belief disambiguation & merging.

Strategies (--merge-strategy):
  * embedding (default) — embed every active belief statement, flag pairs with
    cosine similarity >= --merge-threshold, group candidates via union-find,
    then have the LLM VERIFY each candidate group (only LLM-confirmed
    duplicates are merged).
  * llm                  — the LLM sees the whole belief list and proposes
    merge groups directly (no embeddings needed).
  * off                  — skip merging.

Merge semantics (one confirmed group):
  * canonical  = the SMALLEST id in the group (the node every earlier edge
                 already points at stays stable);
  * statement  = the LLM's `canonical_belief` wording (original wording is
                 preserved in `belief_original` and in the merge record);
  * confidence = adopted from the NEWEST member (largest id) — its score
                 reflects the most recent epistemic state; the adoption is
                 recorded in confidence_history (step="merge");
  * evidence   = union of all members' evidence (deduped), so the canonical
                 belief points at every place the fact was stated;
  * merged_from accumulates the absorbed ids; absorbed beliefs are removed
    from the active graph but archived in graph.merges with full snapshots;
  * edges      = every relation endpoint at an absorbed id is rewired to the
    canonical id; self-loops / duplicates / direction-invalid edges are
    dropped (and reported).

Auditability: every pass writes logs/merge_<pass>.json (machine-readable —
beliefs in, embedding similarities, candidate pairs, every LLM verification
prompt+raw output, applied merges, edge rewiring report) and a human-readable
logs/merge_<pass>.log. Embedding API calls additionally land in
logs/embedding_calls.jsonl via the shared EmbeddingClient.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from . import llm
from .confidence import record_merge_confidence
from .evidence import evidence_key
from .graph import BeliefGraph
from .prompts import (
    BELIEFS_LIST_PLACEHOLDER,
    CANDIDATE_GROUP_PLACEHOLDER,
    PROMPT_MERGE_FULL,
    PROMPT_MERGE_VERIFY,
)


def _compact_for_merge(b: Dict[str, Any]) -> Dict[str, Any]:
    src = b.get("source") or {}
    c = {
        "id":     b.get("id"),
        "role":   src.get("type"),
        "turn":   src.get("turn_index"),
        "stance": b.get("stance"),
        "conf":   b.get("confidence"),
        "belief": b.get("belief"),
    }
    if b.get("event_time"):
        c["time"] = b.get("event_time")
    if b.get("time_text"):
        c["time_text"] = b.get("time_text")
    return c


def _blob(beliefs: List[Dict[str, Any]]) -> str:
    return json.dumps([_compact_for_merge(b) for b in beliefs],
                      ensure_ascii=False, indent=2)


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
    beliefs: List[Dict[str, Any]],
    embedder,
    threshold: float,
    pass_label: str,
) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    """Return (candidate groups of belief ids, pair records for the log)."""
    from .llm import cosine_similarity_matrix

    texts = [b.get("belief") or "" for b in beliefs]
    vectors = embedder.embed(texts, purpose=f"merge:{pass_label}")
    sim = cosine_similarity_matrix(vectors)

    n = len(beliefs)
    uf = _UnionFind(n)
    pairs: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if s >= threshold:
                uf.union(i, j)
                pairs.append({
                    "id_a": beliefs[i]["id"], "id_b": beliefs[j]["id"],
                    "similarity": round(s, 4),
                    "belief_a": texts[i], "belief_b": texts[j],
                })

    groups_by_root: Dict[int, List[int]] = {}
    for i in range(n):
        groups_by_root.setdefault(uf.find(i), []).append(i)
    groups = [sorted(beliefs[i]["id"] for i in idxs)
              for idxs in groups_by_root.values() if len(idxs) >= 2]
    groups.sort(key=lambda g: g[0])
    return groups, pairs


# ---------------------------------------------------------------------------
# LLM verification / proposal
# ---------------------------------------------------------------------------

def _parse_merge_groups(
    raw: str,
    allowed_ids: Set[int],
    used_ids: Set[int],
) -> List[Dict[str, Any]]:
    """Validate {"merge_groups":[{ids, canonical_belief, reason}]}."""
    parsed = llm.parse_json_response(raw)
    groups_in = parsed.get("merge_groups", []) if isinstance(parsed, dict) else []
    out: List[Dict[str, Any]] = []
    for g in groups_in or []:
        if not isinstance(g, dict):
            continue
        ids_in = g.get("ids") or []
        try:
            ids = sorted({int(i) for i in ids_in})
        except (TypeError, ValueError):
            continue
        ids = [i for i in ids if i in allowed_ids and i not in used_ids]
        if len(ids) < 2:
            continue
        used_ids.update(ids)
        canonical = g.get("canonical_belief")
        if not isinstance(canonical, str) or not canonical.strip():
            canonical = None
        reason = g.get("reason") or ""
        if not isinstance(reason, str):
            reason = str(reason)
        out.append({"ids": ids, "canonical_belief": canonical, "reason": reason.strip()})
    return out


# ---------------------------------------------------------------------------
# Apply one confirmed merge group to the graph
# ---------------------------------------------------------------------------

def _apply_merge_group(
    graph: BeliefGraph,
    ids: List[int],
    canonical_belief: Optional[str],
    reason: str,
    pass_label: str,
) -> Dict[str, Any]:
    ids = sorted(ids)
    canon_id, absorbed_ids, newest_id = ids[0], ids[1:], ids[-1]
    canon = graph.beliefs[canon_id]
    absorbed_snapshots = [copy.deepcopy(graph.beliefs[a]) for a in absorbed_ids]
    adopted_conf = float(graph.beliefs[newest_id].get("confidence") or 0.0)

    # evidence union (order-preserving, deduped)
    seen = {evidence_key(ev) for ev in (canon.get("evidence") or [])}
    merged_ev = list(canon.get("evidence") or [])
    for snap in absorbed_snapshots:
        for ev in (snap.get("evidence") or []):
            k = evidence_key(ev)
            if k in seen:
                continue
            seen.add(k)
            merged_ev.append(ev)
    canon["evidence"] = merged_ev
    canon["supporting_excerpts"] = list(dict.fromkeys(
        ev.get("text") for ev in merged_ev if ev.get("text")))

    # merged_from accumulates across passes (absorbed nodes' own merged_from too)
    merged_from = list(canon.get("merged_from") or [])
    for a, snap in zip(absorbed_ids, absorbed_snapshots):
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
    if canonical_belief and canonical_belief.strip() and canonical_belief.strip() != canon.get("belief"):
        canon.setdefault("belief_original", canon.get("belief"))
        canon["belief"] = canonical_belief.strip()

    # confidence: adopt newest member's value (user policy), with history entry
    record_merge_confidence(canon, adopted_conf, newest_id, absorbed_ids, reason)

    for a in absorbed_ids:
        graph.remove_belief(a)

    record = {
        "pass": pass_label,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "canonical_id": canon_id,
        "absorbed_ids": absorbed_ids,
        "newest_id": newest_id,
        "adopted_confidence": round(adopted_conf, 3),
        "canonical_belief": canon.get("belief"),
        "canonical_belief_original": canon.get("belief_original"),
        "reason": reason,
        "absorbed_snapshots": absorbed_snapshots,
    }
    graph.merges.append(record)
    return record


# ---------------------------------------------------------------------------
# Pass entry point
# ---------------------------------------------------------------------------

def run_merge_pass(
    *,
    graph: BeliefGraph,
    strategy: str,                       # "embedding" | "llm" | "off"
    client,
    model: str,
    embedder=None,
    threshold: float = 0.86,
    max_tokens: Optional[int] = None,
    pass_label: str = "merge",
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one merge pass over the active graph. Returns a report dict."""
    active = graph.active()
    if strategy == "off" or len(active) < 2:
        return {"skipped": True, "skip_reason": "strategy off" if strategy == "off"
                else "fewer than 2 beliefs", "applied": []}

    if strategy == "embedding" and embedder is None:
        print(f"  [merge:{pass_label}] no embedding client — falling back to strategy=llm")
        strategy = "llm"

    log: Dict[str, Any] = {
        "pass": pass_label,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "threshold": threshold if strategy == "embedding" else None,
        "n_beliefs": len(active),
        "beliefs": [_compact_for_merge(b) for b in active],
    }
    allowed_ids = {b["id"] for b in active}
    used_ids: Set[int] = set()
    confirmed: List[Dict[str, Any]] = []
    by_id = {b["id"]: b for b in active}

    if strategy == "embedding":
        log["embedding_model"] = getattr(embedder, "model", None)
        candidate_groups, pairs = _embedding_candidates(
            active, embedder, threshold, pass_label)
        log["candidate_pairs"] = pairs
        log["candidate_groups"] = candidate_groups
        log["llm_verifications"] = []
        for g_ids in candidate_groups:
            group_beliefs = [by_id[i] for i in g_ids if i in by_id]
            prompt = PROMPT_MERGE_VERIFY.replace(
                CANDIDATE_GROUP_PLACEHOLDER, _blob(group_beliefs))
            try:
                raw = llm.call_model(client, model, prompt, temperature=0.0,
                                     max_tokens=max_tokens)
            except Exception as e:
                log["llm_verifications"].append(
                    {"candidate_ids": g_ids, "error": str(e)})
                continue
            groups = _parse_merge_groups(raw, set(g_ids), used_ids)
            log["llm_verifications"].append({
                "candidate_ids": g_ids,
                "raw_output": raw,
                "accepted_groups": groups,
            })
            confirmed.extend(groups)
    else:  # strategy == "llm"
        prompt = PROMPT_MERGE_FULL.replace(BELIEFS_LIST_PLACEHOLDER, _blob(active))
        try:
            raw = llm.call_model(client, model, prompt, temperature=0.0,
                                 max_tokens=max_tokens)
            confirmed = _parse_merge_groups(raw, allowed_ids, used_ids)
            log["llm_full"] = {"raw_output": raw, "accepted_groups": confirmed}
        except Exception as e:
            log["llm_full"] = {"error": str(e)}

    applied: List[Dict[str, Any]] = []
    mapping: Dict[int, int] = {}
    for g in sorted(confirmed, key=lambda g: g["ids"][0]):
        rec = _apply_merge_group(graph, g["ids"], g["canonical_belief"],
                                 g["reason"], pass_label)
        applied.append({k: rec[k] for k in
                        ("canonical_id", "absorbed_ids", "newest_id",
                         "adopted_confidence", "canonical_belief", "reason")})
        for a in rec["absorbed_ids"]:
            mapping[a] = rec["canonical_id"]

    rewire = graph.remap_relations(mapping) if mapping else {
        "rewritten": 0, "dropped_self": 0, "dropped_direction": [],
        "dropped_duplicate": 0}

    log["applied_merges"] = applied
    log["relation_rewire"] = rewire
    log["finished_at"] = datetime.now(timezone.utc).isoformat()

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        json_path = log_dir / f"merge_{pass_label}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        with open(log_dir / f"merge_{pass_label}.log", "w", encoding="utf-8") as f:
            f.write(_render_text_log(log))
        log["log_path"] = str(json_path)

    return {"skipped": False, "strategy": strategy, "applied": applied,
            "n_candidate_groups": len(log.get("candidate_groups", confirmed)),
            "relation_rewire": rewire,
            "log_path": log.get("log_path")}


def _render_text_log(log: Dict[str, Any]) -> str:
    lines = [
        "=" * 74,
        f" merge pass: {log['pass']}   strategy={log['strategy']}"
        + (f"   threshold={log['threshold']}" if log.get("threshold") is not None else ""),
        f" beliefs in: {log['n_beliefs']}",
        "=" * 74,
    ]
    if log.get("embedding_model"):
        lines.append(f" embedding model: {log['embedding_model']}")
    for p in log.get("candidate_pairs", []):
        lines.append(f"  pair  #{p['id_a']} ~ #{p['id_b']}  sim={p['similarity']}")
        lines.append(f"        a: {p['belief_a'][:100]}")
        lines.append(f"        b: {p['belief_b'][:100]}")
    if "candidate_groups" in log:
        lines.append(f" candidate groups: {log['candidate_groups']}")
    for v in log.get("llm_verifications", []):
        lines.append(f"  verify {v.get('candidate_ids')}"
                     + (f" -> ERROR {v['error']}" if "error" in v else
                        f" -> {[g['ids'] for g in v.get('accepted_groups', [])]}"))
    for m in log.get("applied_merges", []):
        lines.append(f"  MERGE  {m['absorbed_ids']} -> #{m['canonical_id']}  "
                     f"(conf<-{m['adopted_confidence']} from newest #{m['newest_id']})")
        lines.append(f"         canonical: {m['canonical_belief']}")
        if m.get("reason"):
            lines.append(f"         reason:    {m['reason']}")
    rw = log.get("relation_rewire") or {}
    lines.append(f" edges: rewritten={rw.get('rewritten', 0)}  "
                 f"dropped_self={rw.get('dropped_self', 0)}  "
                 f"dropped_dup={rw.get('dropped_duplicate', 0)}  "
                 f"dropped_direction={len(rw.get('dropped_direction', []))}")
    lines.append("=" * 74)
    return "\n".join(lines) + "\n"
