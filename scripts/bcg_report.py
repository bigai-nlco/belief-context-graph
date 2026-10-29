#!/usr/bin/env python3
"""BCG experiment report — what the graph did, and whether the model used it.

PURPOSE
-------
Every change in the BCG iteration plan (rendering format, context guide, trim
policy, selection query, injection position) is supposed to move one of four
behaviours:

    A  the model expands beliefs with bcg_recover instead of guessing
    B  the model notices low confidence and reacts to it
    C  the model stops repeating searches whose results were evicted
    D  the model's answer does not get worse while A/B/C improve

None of those are visible in the benchmark's own summary.json, which reports
report length and citation counts only. This script reads the BCG audit log
(``requests.jsonl``), the graph snapshot (``state.json``), the agent trajectory
and ``agent.log``, and reports the four behaviours plus the health signals that
decide whether a run is comparable at all.

USAGE
-----
    python scripts/bcg_report.py RUN_DIR                    # full report
    python scripts/bcg_report.py RUN_DIR --tasks 23         # one task only
    python scripts/bcg_report.py RUN_DIR --brief            # skip the prose
    python scripts/bcg_report.py VARIANT --baseline BASE    # A/B comparison
    python scripts/bcg_report.py RUN_DIR --json out.json    # archive
    python scripts/bcg_report.py RUN_DIR --dump-memory 7    # injected block
    python scripts/bcg_report.py RUN_DIR --dump-queries     # search history

Layout is auto-detected, so both pipelines work:
    stateful-react-agent   trials/<id>/agent/bcg/        trajectory react_agent.*
    agent_team             .../trajectories/bcg_team/g_*  trajectory main_agent.*
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
BELIEF_REF = re.compile(r"\[B(\d+)\]")
TEAM_REF = re.compile(r"\[(g_[0-9a-f]{6,}):B(\d+)\]")
URL = re.compile(r"https?://[^\s<>()\[\]\"']+")
DONE_LINE = re.compile(
    r"\[ReAct\]\s+done\s+turns=(\d+)\s+tool_calls=(\d+)(?:\s+stopped_by=(\S+))?"
)
TURN_BANNER = re.compile(r"\[ReAct\]\s+Turn\s+(\d+)\s*/\s*(\d+)")
HTTP_ERR = re.compile(r"\b(400|401|403|404|408|409|429|500|502|503|504)\b")
NO_SOURCE = "No recorded source"

# A node that merely records "the assistant ran this tool" rather than asserting
# a fact about the world. These are the raw material for the Phase 1.1 split of
# the belief block into "completed actions" vs "findings".
ACTION_PATTERNS = re.compile(
    r"(is using \w+ to (search|fetch|call)|is calling the \w+ tool"
    r"|the \w+ tool returned|tool call to \w+|assistant (issued|ran|performed) a)",
    re.I,
)
# The query string inside an action node, e.g. ... to search for "foo bar".
# Must match runtime's _QUERY_RE: greedy and anchored at the end, or a query
# containing escaped quotes (very common) is captured only up to its first one
# and never matches the query the model actually sent.
QUERY_IN_NODE = re.compile(
    r'(?:search(?:ing)? for|query|fetch(?:ing)?)\s*[:\s]\s*\\?"(.+?)\\?"\s*\.?\s*$',
    re.I,
)
# Reference-list entries in a report: "1. Author (year)." or "- Author (year)."
REFERENCE_ENTRY = re.compile(r"^\s*(?:\d+[.)]|[-*\u2022])\s+\S", re.M)
REFERENCES_HEADING = re.compile(
    r"^#{1,6}\s*(references|bibliography|sources|works cited|参考文献|参考资料|引用)\b",
    re.I | re.M,
)
DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")

# Reasoning that reacts to reliability rather than just restating a fact.
CONFIDENCE_CUES = re.compile(
    r"(置信度|可信度|信度|不太?可靠|不够可靠|存疑|有疑问|不确定|证据不足|待验证"
    r"|需要(进一步|更多)?(验证|证据|检索|查证|佐证)|交叉验证|相互矛盾|矛盾|冲突"
    r"|confidence|unreliable|low[- ]confidence|uncertain|needs? (more )?(evidence|verification)"
    r"|verify|corroborat|cross-check|contradict|conflicting)",
    re.I,
)
# A stronger cue: explicit intent to act because of unreliability.
ACTION_CUES = re.compile(
    r"(需要(继续|进一步|再)(检索|搜索|查|验证)|再(搜|查|找)一(下|次)|换个来源|另找"
    r"|should (search|verify|check|look) (again|further|for more)"
    r"|let me (verify|double[- ]check|confirm)|need (more|another) source)",
    re.I,
)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows


def word_count(text: str) -> int:
    """CJK characters count individually; Latin runs count as one word each.

    The benchmark's own ``report_words`` splits on whitespace, which badly
    undercounts Chinese reports. Both numbers are kept so you can see the gap.
    """
    return len(CJK.findall(text)) + len(LATIN_WORD.findall(CJK.sub(" ", text)))


def pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return {
        "n": len(ordered),
        "min": round(ordered[0], 1),
        "p50": round(statistics.median(ordered), 1),
        "p95": round(ordered[idx], 1),
        "max": round(ordered[-1], 1),
        "mean": round(statistics.fmean(ordered), 1),
    }


def fmt_stats(s: dict[str, Any] | None, unit: str = "") -> str:
    if not s or not s.get("n"):
        return "no samples"
    return (
        f"n={s['n']}  min={s['min']}{unit}  p50={s['p50']}{unit}  "
        f"p95={s['p95']}{unit}  max={s['max']}{unit}"
    )


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


_STOPWORDS = frozenset(
    "a an the of for and or to in on at is are was were be been how many what "
    "which when where who study research paper article".split()
)
NEAR_DUPLICATE_THRESHOLD = 0.6


def query_tokens(query: str) -> frozenset[str]:
    """Content words only, so a rewording cannot hide a repeated search."""
    return frozenset(re.findall(r"[\w\u4e00-\u9fff]+", query.lower())) - _STOPWORDS


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def normalize_query(text: str) -> str:
    """Lowercase, collapse whitespace, drop surrounding quotes.

    Two searches that differ only in spacing or quoting are the same search as
    far as wasted effort is concerned.
    """
    return re.sub(r"\s+", " ", str(text).strip().strip('"\'').lower())


def walk(node: Any):
    """Yield every dict nested anywhere inside a JSON structure."""
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            yield item
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def dig(node: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        node = (node or {}).get(key) if isinstance(node, dict) else None
    return node


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory parsing (tolerant of several message schemas)
# ─────────────────────────────────────────────────────────────────────────────

def extract_tool_calls(obj: Any) -> list[dict[str, Any]]:
    """Find tool calls whatever shape the trajectory writer used.

    Handles OpenAI ``{"function": {"name", "arguments"}}``, flattened
    ``{"name", "arguments"}`` and native ``{"type": "tool_use", "input"}``.
    A wrapper and its own ``function`` payload both look like a call, so the
    inner one is skipped or every call is counted twice.
    """
    items = list(walk(obj))
    nested = {id(i["function"]) for i in items if isinstance(i.get("function"), dict)}
    found: list[dict[str, Any]] = []
    for item in items:
        if id(item) in nested:
            continue
        func = item.get("function") if isinstance(item.get("function"), dict) else item
        name = func.get("name") if isinstance(func, dict) else None
        if not isinstance(name, str) or not name:
            continue
        raw = None
        for key in ("arguments", "args", "input", "parameters"):
            if key in func:
                raw = func[key]
                break
        if raw is None:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                raw = {"__raw__": raw}
        if not isinstance(raw, dict):
            continue
        found.append(
            {
                "id": item.get("id") or item.get("call_id") or item.get("tool_call_id") or "",
                "name": name,
                "arguments": raw,
            }
        )
    return found


def assistant_blocks(rows: list[Any]) -> list[str]:
    """Visible assistant prose plus exposed reasoning, one string per block."""
    out: list[str] = []
    for item in walk(rows):
        if item.get("role") != "assistant":
            continue
        for key in ("content", "reasoning_content", "reasoning", "thinking"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value)
            elif isinstance(value, list):
                for block in value:
                    if isinstance(block, dict):
                        for sub in ("text", "thinking", "reasoning_text"):
                            if isinstance(block.get(sub), str) and block[sub].strip():
                                out.append(block[sub])
    return out


def tool_results(rows: list[Any]) -> dict[str, str]:
    """Map tool_call_id -> result text, for checking whether a call succeeded."""
    out: dict[str, str] = {}
    for item in walk(rows):
        if item.get("role") != "tool":
            continue
        cid = item.get("tool_call_id") or item.get("id") or ""
        content = item.get("content")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        if cid and isinstance(content, str):
            out[cid] = content
    return out


def load_trajectory(trial: Path) -> list[Any]:
    rows: list[Any] = []
    for path in sorted(trial.rglob("*.jsonl")):
        if path.name == "requests.jsonl":
            continue
        rows.extend(read_jsonl(path))
    if rows:
        return rows
    for path in sorted(trial.rglob("*.json")):
        if path.name in {"result.json", "state.json", "index.json", "run.json"}:
            continue
        if "bcg" in path.parts:
            continue
        value = read_json(path)
        if value is not None:
            rows.append(value)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Graph snapshot analysis
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_nodes(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = snapshot.get("nodes")
    if nodes is None:
        nodes = (snapshot.get("beliefs") or []) + (snapshot.get("decisions") or [])
    return [n for n in nodes if isinstance(n, dict)] if isinstance(nodes, list) else []


def node_text(node: dict[str, Any]) -> str:
    for key in ("belief", "text", "content", "statement", "claim"):
        value = node.get(key)
        if isinstance(value, str):
            return value
    return ""


def classify_node(node: dict[str, Any]) -> str:
    """ACTION = a record of a tool the agent ran. FACT = a claim about the world.

    Phase 1.1 proposes rendering these as two separate sections so the model can
    read "what I already did" without wading through findings. This function is
    the measurement side of that change: it tells you how many nodes would land
    in each section, and therefore whether the split is worth making.
    """
    if ACTION_PATTERNS.search(node_text(node)):
        return "action"
    return "fact"


def analyse_graph(state_files: list[Path]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    problem_ids: list[str] = []
    for path in state_files:
        state = read_json(path)
        if not isinstance(state, dict) or "snapshot" not in state:
            continue
        snapshot = state.get("snapshot") or {}
        nodes.extend(snapshot_nodes(snapshot))
        rels = snapshot.get("relations")
        if isinstance(rels, list):
            relations.extend(r for r in rels if isinstance(r, dict))
        if state.get("problem_id"):
            problem_ids.append(str(state["problem_id"]))

    confidences = [
        float(n["confidence"])
        for n in nodes
        if isinstance(n.get("confidence"), (int, float)) and not isinstance(n.get("confidence"), bool)
    ]
    value_counts = Counter(round(c, 3) for c in confidences)
    buckets = Counter()
    for conf in confidences:
        if conf < 0.45:
            buckets["low <0.45"] += 1
        elif conf < 0.75:
            buckets["mid 0.45-0.75"] += 1
        else:
            buckets["high >=0.75"] += 1

    kinds = Counter(classify_node(n) for n in nodes)
    with_source = sum(
        1 for n in nodes if isinstance(n.get("source"), dict)
        and isinstance(n["source"].get("turn_id"), int)
    )
    with_tool = sum(1 for n in nodes if n.get("tool_name"))
    rel_types = Counter(str(r.get("type")) for r in relations)

    # Queries the graph has recorded. If the model repeats one of these, the
    # graph told it not to and it did anyway — that is the Phase 1.1 target.
    recorded_queries: dict[str, int] = {}
    for node in nodes:
        match = QUERY_IN_NODE.search(node_text(node))
        if match:
            recorded_queries.setdefault(normalize_query(match.group(1)), node.get("id"))

    return {
        "nodes_by_id": {n["id"]: n for n in nodes if isinstance(n.get("id"), int)},
        "problem_ids": sorted(set(problem_ids)),
        "graphs": len(state_files),
        "nodes_total": len(nodes),
        "node_kinds": dict(kinds),
        "action_node_pct": pct(kinds.get("action", 0), len(nodes)),
        "relations_total": len(relations),
        "relation_types": dict(rel_types),
        "contradicts_edges": rel_types.get("contradicts", 0),
        "confidence_present": len(confidences),
        "confidence_missing": len(nodes) - len(confidences),
        "confidence_buckets": dict(buckets),
        "confidence_distinct_values": len(value_counts),
        "confidence_value_counts": dict(value_counts.most_common(12)),
        "confidence_median": round(statistics.median(confidences), 3) if confidences else None,
        "confidence_stdev": round(statistics.pstdev(confidences), 4) if len(confidences) > 1 else None,
        "source_coverage_pct": pct(with_source, len(nodes)),
        "tool_name_coverage_pct": pct(with_tool, len(nodes)),
        "recorded_queries": recorded_queries,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-trial collection
# ─────────────────────────────────────────────────────────────────────────────

def find_trials(run_dir: Path) -> list[Path]:
    trials = sorted(p.parent for p in run_dir.glob("trials/*/result.json"))
    return trials or sorted({p.parent for p in run_dir.rglob("result.json")})


def collect_bcg(trial: Path) -> dict[str, Any]:
    """Everything the BCG audit log knows about this trial."""
    events = [
        e
        for path in sorted(trial.rglob("requests.jsonl"))
        for e in read_jsonl(path)
        if isinstance(e, dict)
    ]
    prepares = [e for e in events if "call" in e]
    flushes = [e for e in events if e.get("event") == "final_graph_flush"]

    fallbacks = Counter(str(e["fallback"]) for e in prepares if e.get("fallback"))
    healthy = [e for e in prepares if not e.get("fallback")]
    injected = [e for e in healthy if e.get("evicted_steps")]

    graph = analyse_graph(sorted(trial.rglob("state.json")))

    # Graph-side token usage is cumulative on each event, so take the last one
    # rather than summing.
    usage: dict[str, Any] = {}
    for event in reversed(prepares):
        totals = (event.get("graph_usage_cumulative") or {}).get("totals")
        if isinstance(totals, dict) and totals:
            usage = {k: v for k, v in totals.items() if isinstance(v, (int, float))}
            break

    # How many of the nodes shown per call were action records. The selector
    # caps search-history nodes on its own (FOCUSED_SEARCH_LIMIT), so a thin
    # completed-actions section may be the selector's doing, not the renderer's.
    kinds = {
        node_id: classify_node(node)
        for node_id, node in (graph.get("nodes_by_id") or {}).items()
    }
    actions_shown = [
        float(sum(1 for i in (e.get("selected_node_ids") or []) if kinds.get(i) == "action"))
        for e in injected
    ]
    trimmed = [e["trim_dropped_nodes"] for e in injected if "trim_dropped_nodes" in e]
    action_chars = [e["action_chars"] for e in injected if "action_chars" in e]
    action_counts = [
        len(e.get("action_node_ids") or []) for e in injected if "action_node_ids" in e
    ]
    # Every call the model was actually shown, pooled over the session. The
    # coverage metric below asks how many of its searches appeared here.
    shown_actions: set[str] = set()
    for event in injected:
        for line in (event.get("system_memory") or "").splitlines():
            found = re.match(
                r'\s*(?:- )?\[B\d+\]\s+(?:\S+\s+)?\\?"(.+?)\\?"\s*(?:\(blocked[^)]*\))?\s*$',
                line,
            )
            if found:
                shown_actions.add(normalize_query(found.group(1)))
    dropped_ids = [i for e in injected for i in (e.get("trim_dropped_node_ids") or [])]
    variants = Counter(
        str(e.get("variant") or e.get("bcg_variant") or "")
        for e in prepares
        if e.get("variant") or e.get("bcg_variant")
    )

    return {
        "prepare_calls": len(prepares),
        "fallback_calls": sum(fallbacks.values()),
        "fallback_reasons": dict(fallbacks),
        "fallback_calls_detail": [
            {"call": e.get("call"), "reason": e.get("fallback"), "ms": e.get("duration_ms")}
            for e in prepares
            if e.get("fallback")
        ],
        "injected_calls": len(injected),
        "first_injection_call": injected[0].get("call") if injected else None,
        "evicted_steps_max": max((e.get("evicted_steps", 0) for e in injected), default=0),
        "graph_turns_written": sum(e.get("graph_turns_written", 0) or 0 for e in prepares),
        "selected_nodes": stats([float(len(e.get("selected_node_ids") or [])) for e in injected]),
        "action_nodes_shown": stats(actions_shown),
        "selected_relations": stats([float(len(e.get("selected_relation_ids") or [])) for e in injected]),
        "graph_chars": stats([float(e.get("graph_chars", 0)) for e in injected]),
        "messages_before": stats([float(e.get("messages_before", 0)) for e in injected]),
        "messages_after": stats([float(e.get("messages_after", 0)) for e in injected]),
        "messages_removed": stats(
            [float(e.get("messages_before", 0) - e.get("messages_after", 0)) for e in injected]
        ),
        "trim_dropped_nodes": stats([float(t) for t in trimmed]) if trimmed else None,
        "action_chars": stats([float(c) for c in action_chars]) if action_chars else None,
        "action_nodes_shown": stats([float(c) for c in action_counts]) if action_counts else None,
        "shown_action_queries": shown_actions,
        "trim_dropped_node_ids": sorted(set(dropped_ids)),
        "prep_duration_ms": stats(
            [float(e["duration_ms"]) for e in prepares if isinstance(e.get("duration_ms"), (int, float))]
        ),
        "final_flush": [e.get("status") for e in flushes],
        "graph_token_usage": usage,
        "variant_tags": dict(variants),
        "graph": graph,
        "timeline": [
            {
                "call": e.get("call"),
                "fallback": e.get("fallback"),
                "evicted": e.get("evicted_steps", 0),
                "before": e.get("messages_before"),
                "after": e.get("messages_after"),
                "nodes": len(e.get("selected_node_ids") or []),
                "rels": len(e.get("selected_relation_ids") or []),
                "chars": e.get("graph_chars", 0),
                "dropped": e.get("trim_dropped_nodes"),
                "ms": e.get("duration_ms"),
            }
            for e in prepares
        ],
    }


def collect_trajectory(
    trial: Path, graph: dict[str, Any], bcg_shown: set[str] | None = None
) -> dict[str, Any]:
    rows = load_trajectory(trial)
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        for call in extract_tool_calls(row):
            key = call["id"] or f"{len(calls)}:{canonical([call['name'], call['arguments']])}"
            if key in seen:
                continue
            seen.add(key)
            calls.append(call)

    results = tool_results(rows)
    by_name = Counter(c["name"] for c in calls)

    # ── duplicates ──────────────────────────────────────────────────────────
    signatures = [canonical([c["name"], c["arguments"]]) for c in calls]
    sig_counts = Counter(signatures)
    duplicate_calls = sum(n - 1 for n in sig_counts.values() if n > 1)
    duplicate_by_name: Counter = Counter()
    for sig, n in sig_counts.items():
        if n > 1:
            duplicate_by_name[json.loads(sig)[0]] += n - 1
    repeated = [
        {"name": json.loads(sig)[0], "args": json.loads(sig)[1], "times": n}
        for sig, n in sig_counts.most_common()
        if n > 1
    ]

    # Search queries specifically: the clearest form of wasted work.
    queries: list[str] = []
    for call in calls:
        if call["name"] not in {"web_search", "search"}:
            continue
        for key in ("q", "query", "question", "search_query"):
            if isinstance(call["arguments"].get(key), str):
                queries.append(normalize_query(call["arguments"][key]))
                break
    qcounts = Counter(queries)
    repeated_queries = {q: n for q, n in qcounts.items() if n > 1}
    # Exact repeats turn out to be rare; rephrasing the same search is not.
    # Compare content words between every pair of distinct queries.
    unique = list(qcounts)
    token_sets = {q: query_tokens(q) for q in unique}
    near_pairs = [
        {"a": unique[i], "b": unique[j], "overlap": round(score, 2)}
        for i in range(len(unique))
        for j in range(i + 1, len(unique))
        if (score := jaccard(token_sets[unique[i]], token_sets[unique[j]]))
        >= NEAR_DUPLICATE_THRESHOLD
    ]
    near_pairs.sort(key=lambda item: item["overlap"], reverse=True)
    # A repeat the graph could have prevented: the query is recorded in the
    # graph as a completed action, and the model ran it again anyway.
    shown = bcg_shown or set()
    shown_tokens = [query_tokens(q) for q in shown]
    # Match on content words: the graph stores a query with escaped quotes and
    # the trajectory does not, and a rephrasing is still the same search.
    covered = sum(
        1
        for q in set(queries)
        if any(jaccard(query_tokens(q), s) >= 0.75 for s in shown_tokens)
    )
    recorded = graph.get("recorded_queries") or {}
    preventable = {
        q: {"times": n, "recorded_as": f"B{recorded[q]}"}
        for q, n in repeated_queries.items()
        if q in recorded
    }

    # ── bcg_recover ─────────────────────────────────────────────────────────
    recover_calls = [c for c in calls if c["name"] == "bcg_recover"]
    recover_miss = sum(
        1 for c in recover_calls if NO_SOURCE in results.get(c["id"], "")
    )
    # Fall back to a text scan when ids did not line up.
    if not recover_miss:
        blob = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        recover_miss = min(len(recover_calls), blob.count(NO_SOURCE))
    recover_ids = [c["arguments"].get("belief_id") for c in recover_calls]
    recover_bad_type = sum(1 for v in recover_ids if not isinstance(v, int))

    # ── belief references in reasoning ──────────────────────────────────────
    blocks = assistant_blocks(rows)
    joined = "\n".join(blocks)
    refs = BELIEF_REF.findall(joined)

    cue_blocks = [b for b in blocks if CONFIDENCE_CUES.search(b)]
    cue_and_ref = [b for b in cue_blocks if BELIEF_REF.search(b)]
    cue_and_action = [b for b in cue_blocks if ACTION_CUES.search(b)]
    strong = [b for b in cue_and_ref if ACTION_CUES.search(b)]

    return {
        "tool_calls_total": len(calls),
        "tool_calls_by_name": dict(by_name.most_common()),
        "search_calls": by_name.get("web_search", 0) + by_name.get("search", 0),
        "fetch_calls": by_name.get("web_fetch", 0) + by_name.get("fetch", 0),
        "distinct_queries": len(qcounts),
        "queries_listed_in_block": covered,
        "query_coverage_pct": pct(covered, len(qcounts)),
        "repeated_queries": repeated_queries,
        "repeated_query_calls": sum(n - 1 for n in qcounts.values() if n > 1),
        "preventable_repeats": preventable,
        "duplicate_tool_calls": duplicate_calls,
        "duplicate_by_name": dict(duplicate_by_name.most_common()),
        "near_duplicate_pairs": len(near_pairs),
        "near_duplicate_queries": near_pairs[:10],
        "duplicate_search_calls": duplicate_by_name.get("web_search", 0)
        + duplicate_by_name.get("search", 0)
        + duplicate_by_name.get("web_fetch", 0),
        "repeated_signatures": repeated[:8],
        "bcg_recover_calls": len(recover_calls),
        "bcg_recover_misses": recover_miss,
        "bcg_recover_bad_arg_type": recover_bad_type,
        "bcg_recover_belief_ids": recover_ids,
        "bcg_team_read_calls": by_name.get("bcg_team_read", 0),
        "assistant_blocks": len(blocks),
        "belief_refs_total": len(refs),
        "belief_refs_distinct": len(set(refs)),
        "belief_refs_ids": sorted(set(int(r) for r in refs)),
        "confidence_cue_blocks": len(cue_blocks),
        "cue_with_belief_ref": len(cue_and_ref),
        "cue_with_action_intent": len(cue_and_action),
        "strong_confidence_reactions": len(strong),
        "samples_strong": [b[:400] for b in strong[:3]],
        "samples_cue": [b[:300] for b in cue_blocks[:2]],
        "queries_in_order": queries,
    }


def collect_log(trial: Path, trajectory: list[Any] | None = None) -> dict[str, Any]:
    logs = list(trial.rglob("agent.log")) or list(trial.rglob("*.log"))
    turns = tool_calls = None
    stopped = None
    turns_source = "none"
    errors: Counter = Counter()
    nudges = 0
    for path in logs:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in DONE_LINE.finditer(text):
            turns, tool_calls = int(match.group(1)), int(match.group(2))
            stopped = match.group(3) or stopped
            turns_source = "agent.log"
        for line in text.splitlines():
            if "LLM error" in line or "llm_error" in line or "exhausted" in line:
                code = HTTP_ERR.search(line)
                errors[code.group(1) if code else "other"] += 1
            if "nudge" in line.lower():
                nudges += 1
    if turns is None:
        # The loop prints "=== [ReAct] Turn n/m" for every turn even when the
        # summary line is missing (crash, timeout, or a different stop path).
        banners = [
            int(m.group(1))
            for path in logs
            for m in TURN_BANNER.finditer(
                path.read_text(encoding="utf-8", errors="replace")
            )
        ]
        if banners:
            turns = max(banners)
            turns_source = "Turn banners in agent.log"
    rows = trajectory if trajectory is not None else load_trajectory(trial)
    event_types = Counter(
        str(row.get("t")) for row in rows if isinstance(row, dict) and "t" in row
    )
    if turns is None:
        counted = sum(1 for item in walk(rows) if item.get("role") == "assistant")
        if not counted:
            # Trajectories written as an event stream carry no role field; count
            # whichever event type looks like one model response per turn.
            for key in ("llm", "assistant", "turn", "response", "message"):
                if event_types.get(key):
                    counted = event_types[key]
                    turns_source = f"trajectory events t={key}"
                    break
        else:
            turns_source = "assistant blocks (approximate)"
        turns = counted or None
        if turns is None:
            turns_source = "none"
    return {
        "turns": turns,
        "turns_source": turns_source,
        "tool_calls_logged": tool_calls,
        "stopped_by_log": stopped,
        "llm_error_lines": dict(errors),
        "nudge_lines": nudges,
        "trajectory_event_types": dict(event_types.most_common(12)),
    }


def collect_report(trial: Path, result: dict[str, Any]) -> dict[str, Any]:
    paths = list(trial.glob("report.md")) or list(trial.rglob("report*.md"))
    text = ""
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            break
        except OSError:
            continue
    metrics = result.get("diagnostic_metrics") or result.get("metrics") or {}
    urls = URL.findall(text)
    # The benchmark counts URLs. An academic reference list carries none, so
    # urls_cited reads 0 for a report that is in fact well sourced. Count the
    # entries under the references heading as well.
    heading = REFERENCES_HEADING.search(text)
    tail = text[heading.end():] if heading else ""
    entries = len(REFERENCE_ENTRY.findall(tail))
    return {
        "path": str(paths[0]) if paths else None,
        "chars": len(text),
        "words_counted": word_count(text),
        "words_reported": metrics.get("report_words"),
        "headings": metrics.get("report_headings", text.count("\n#")),
        "urls_in_text": len(urls),
        "urls_distinct": len(set(urls)),
        "urls_cited": metrics.get("urls_cited"),
        "domains_cited": metrics.get("domains_cited"),
        "has_references_section": metrics.get("has_references_section"),
        "references_heading": bool(heading),
        "reference_entries": entries,
        "doi_mentions": len(set(DOI.findall(text))),
        "blocked_citation_hits": metrics.get("blocked_citation_hits"),
    }


def find_tokens(result: dict[str, Any]) -> dict[str, Any]:
    """Pick the run total, not one call's usage.

    Per-call usage records sit in the same file as the run total, so the first
    match is usually a single call. The largest input count is the total.
    """
    candidates = [
        item
        for item in walk(result)
        if {"input_tokens", "output_tokens"} <= item.keys()
        and isinstance(item.get("input_tokens"), (int, float))
    ]
    agent = max(candidates, key=lambda d: d.get("input_tokens") or 0, default={})
    tokens = result.get("total_tokens") or result.get("tokens") or {}
    declared = tokens.get("agent")
    if isinstance(declared, dict) and isinstance(declared.get("input_tokens"), (int, float)):
        agent = declared
    return {"agent": agent, "graph_declared": tokens.get("graph")}


def collect_trial(trial: Path) -> dict[str, Any]:
    result = read_json(trial / "result.json") or {}
    bcg = collect_bcg(trial)
    rows = load_trajectory(trial)
    traj = collect_trajectory(trial, bcg["graph"], bcg.pop("shown_action_queries", None))
    return {
        "task": trial.name,
        "path": str(trial),
        "status": result.get("status"),
        "stopped_by": result.get("stopped_by"),
        "answer_status": result.get("answer_status"),
        "answer_source": result.get("answer_source"),
        "duration_s": result.get("duration_seconds") or result.get("duration"),
        "tokens": find_tokens(result),
        "log": collect_log(trial, rows),
        "report": collect_report(trial, result),
        "trajectory": traj,
        "bcg": bcg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def total(path: str) -> float:
        out = 0.0
        for row in rows:
            value = dig(row, path)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out += value
        return out

    n = max(1, len(rows))
    prepares = total("bcg.prepare_calls")
    fallbacks = total("bcg.fallback_calls")
    injected = total("bcg.injected_calls")
    calls = total("trajectory.tool_calls_total")
    recover = total("trajectory.bcg_recover_calls")
    agent_in = total("tokens.agent.input_tokens")
    agent_out = total("tokens.agent.output_tokens")

    fallback_reasons: Counter = Counter()
    conf_buckets: Counter = Counter()
    node_kinds: Counter = Counter()
    rel_types: Counter = Counter()
    for row in rows:
        fallback_reasons.update(row["bcg"]["fallback_reasons"])
        conf_buckets.update(row["bcg"]["graph"]["confidence_buckets"])
        node_kinds.update(row["bcg"]["graph"]["node_kinds"])
        rel_types.update(row["bcg"]["graph"]["relation_types"])

    nodes_total = total("bcg.graph.nodes_total")
    graph_tokens = sum(
        (row["bcg"]["graph_token_usage"] or {}).get("total_tokens", 0) for row in rows
    )

    return {
        "tasks": len(rows),
        "task_ids": [r["task"] for r in rows],
        "status_counts": dict(Counter(str(r["status"]) for r in rows)),
        "stopped_by": dict(Counter(str(r["stopped_by"]) for r in rows)),
        "health": {
            "prepare_calls": int(prepares),
            "fallback_calls": int(fallbacks),
            "fallback_rate_pct": pct(fallbacks, prepares),
            "fallback_reasons": dict(fallback_reasons),
            "injected_calls": int(injected),
            "injection_rate_pct": pct(injected, prepares),
            "llm_errors": sum(sum(r["log"]["llm_error_lines"].values()) for r in rows),
        },
        "graph": {
            "nodes_total": int(nodes_total),
            "nodes_per_task": round(nodes_total / n, 1),
            "node_kinds": dict(node_kinds),
            "action_node_pct": pct(node_kinds.get("action", 0), nodes_total),
            "relations_total": int(total("bcg.graph.relations_total")),
            "relation_types": dict(rel_types),
            "contradicts_edges": int(total("bcg.graph.contradicts_edges")),
            "confidence_buckets": dict(conf_buckets),
            "confidence_distinct_values": max(
                (r["bcg"]["graph"]["confidence_distinct_values"] for r in rows), default=0
            ),
            "confidence_spread_pct_low": pct(conf_buckets.get("low <0.45", 0), nodes_total),
            "source_coverage_pct": round(
                statistics.fmean([r["bcg"]["graph"]["source_coverage_pct"] for r in rows] or [0]), 1
            ),
            "tool_name_coverage_pct": round(
                statistics.fmean([r["bcg"]["graph"]["tool_name_coverage_pct"] for r in rows] or [0]), 1
            ),
        },
        "core": {
            "A_recover_calls": int(recover),
            "A_recover_per_task": round(recover / n, 1),
            "A_recover_success_pct": pct(recover - total("trajectory.bcg_recover_misses"), recover),
            "A_recover_bad_arg_type": int(total("trajectory.bcg_recover_bad_arg_type")),
            "B_belief_refs": int(total("trajectory.belief_refs_total")),
            "B_belief_refs_distinct": int(total("trajectory.belief_refs_distinct")),
            "B_cue_blocks": int(total("trajectory.confidence_cue_blocks")),
            "B_cue_with_ref": int(total("trajectory.cue_with_belief_ref")),
            "B_strong_reactions": int(total("trajectory.strong_confidence_reactions")),
            "C_duplicate_calls": int(total("trajectory.duplicate_tool_calls")),
            "C_duplicate_search_calls": int(total("trajectory.duplicate_search_calls")),
            "C_duplicate_rate_pct": pct(total("trajectory.duplicate_tool_calls"), calls),
            "C_repeated_query_calls": int(total("trajectory.repeated_query_calls")),
            "C_preventable_repeats": sum(
                len(r["trajectory"]["preventable_repeats"]) for r in rows
            ),
            "C_near_duplicate_pairs": int(total("trajectory.near_duplicate_pairs")),
            "C_query_coverage_pct": round(
                statistics.fmean([r["trajectory"]["query_coverage_pct"] for r in rows] or [0]), 1
            ),
        },
        "efficiency": {
            "turns": int(total("log.turns")),
            "turns_per_task": round(total("log.turns") / n, 1),
            "tool_calls": int(calls),
            "search_calls": int(total("trajectory.search_calls")),
            "fetch_calls": int(total("trajectory.fetch_calls")),
            "duration_s": round(total("duration_s"), 1),
            "duration_per_task_s": round(total("duration_s") / n, 1),
            "agent_input_tokens": int(agent_in),
            "agent_output_tokens": int(agent_out),
            "agent_total_tokens": int(agent_in + agent_out),
            "agent_tokens_per_task": round((agent_in + agent_out) / n),
            "graph_total_tokens": int(graph_tokens),
        },
        "output": {
            "words_counted": int(total("report.words_counted")),
            "words_per_task": round(total("report.words_counted") / n),
            "headings": int(total("report.headings")),
            "urls_cited": int(total("report.urls_cited")),
            "domains_cited": int(total("report.domains_cited")),
            "with_references": int(total("report.has_references_section")),
            "reference_entries": int(total("report.reference_entries")),
            "doi_mentions": int(total("report.doi_mentions")),
            "blocked_hits": int(total("report.blocked_citation_hits")),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

RULE = "─" * 78
BAR = "═" * 78


def section(num: str, title: str, why: str, brief: bool) -> list[str]:
    out = ["", BAR, f"§{num}  {title}", BAR]
    if not brief:
        out += ["", *(f"  {line}" for line in why.strip().splitlines()), ""]
    return out


def render(run: dict[str, Any], brief: bool = False, detail: bool = True) -> str:
    agg = run["aggregate"]
    rows = run["tasks"]
    one = rows[0] if len(rows) == 1 else None
    out: list[str] = []

    # ── §0 identity ─────────────────────────────────────────────────────────
    meta = run["meta"]
    out += [
        BAR,
        "BCG EXPERIMENT REPORT",
        BAR,
        f"  run dir     {run['run_dir']}",
        f"  pipeline    {meta.get('pipeline')}        profile  {meta.get('profile')}",
        f"  agent model {meta.get('agent_model')}",
        f"  graph url   {meta.get('graph_url')}   recent_turns={meta.get('recent_turns')}"
        f"  graph_chars={meta.get('graph_chars')}",
        f"  tasks       {agg['tasks']}  ({', '.join(agg['task_ids'])})",
    ]
    tags = one["bcg"]["variant_tags"] if one else {}
    if tags:
        out.append(f"  variant     {tags}")
    else:
        out.append("  variant     (untagged — see Phase 0.5 variant switches)")

    # ── §1 health gate ──────────────────────────────────────────────────────
    h = agg["health"]
    gate_ok = h["fallback_rate_pct"] == 0 and h["injected_calls"] > 0
    out += section(
        "1",
        "HEALTH GATE — is this run comparable at all?",
        """WHAT: how often BCG.prepare() succeeded, and how often it actually put a
graph block into the request.
WHY: on any failure prepare() returns the full message list unchanged, so the
context is not compressed and the model never sees the graph. A run with
fallbacks is measuring the fallback path, not your change. Treat a non-zero
fallback rate as "this run is void" rather than as a small defect.
PASS: fallback_rate = 0% AND injected_calls > 0.""",
        brief,
    )
    out += [
        f"  prepare calls          {h['prepare_calls']}",
        f"  fallbacks              {h['fallback_calls']}  ({h['fallback_rate_pct']}%)",
        f"  injected (graph shown) {h['injected_calls']}  ({h['injection_rate_pct']}%)",
        f"  LLM errors in log      {h['llm_errors']}",
        f"  status                 {agg['status_counts']}",
        f"  stopped_by             {agg['stopped_by']}",
        "",
        f"  >>> GATE {'PASS' if gate_ok else 'FAIL'}",
    ]
    if h["fallback_reasons"]:
        out.append(f"      reasons: {h['fallback_reasons']}")
        out.append("      graph_retry_cooldown entries are knock-on skips after a real")
        out.append("      failure, not independent failures. Lower BCG_FAILURE_COOLDOWN")
        out.append("      and fix the first-cause reason instead.")
    if one and one["bcg"]["fallback_calls_detail"]:
        out.append("      failing calls:")
        for item in one["bcg"]["fallback_calls_detail"][:10]:
            out.append(f"        call {item['call']:>3}  {item['reason']:<24} {item['ms']} ms")
        out.append("      <10 ms  = connection refused, the server was already down")
        out.append("      >1000 ms = the request started and then failed (crash or timeout)")
    if one and one["bcg"]["final_flush"]:
        out.append(f"      final flush: {one['bcg']['final_flush']}")

    # ── §2 graph quality ────────────────────────────────────────────────────
    g = agg["graph"]
    out += section(
        "2",
        "GRAPH QUALITY — does the graph carry the signals the experiment needs?",
        """WHAT: what the construction server actually produced.
WHY: three downstream changes depend on properties of the graph, not on the
rendering code. If the property is missing, the corresponding change cannot
work no matter how it is written:
  - confidence spread  -> goal B (react to low confidence) and tiered rendering
  - source coverage    -> goal A (bcg_recover returns something useful)
  - action-node share  -> the completed-actions section that targets goal C
Fix these in the construction config (belief_graph.extractor) before changing
any rendering code.""",
        brief,
    )
    out += [
        f"  nodes                  {g['nodes_total']}  ({g['nodes_per_task']} per task)",
        f"  node kinds             {g['node_kinds']}   action share {g['action_node_pct']}%",
        f"  relations              {g['relations_total']}  {g['relation_types']}",
        f"  contradicts edges      {g['contradicts_edges']}",
        "",
        f"  confidence buckets     {g['confidence_buckets']}",
        f"  distinct values        {g['confidence_distinct_values']}",
        f"  source coverage        {g['source_coverage_pct']}%",
        f"  tool_name coverage     {g['tool_name_coverage_pct']}%",
    ]
    if one:
        gg = one["bcg"]["graph"]
        out += [
            f"  confidence median      {gg['confidence_median']}   stdev {gg['confidence_stdev']}",
            f"  confidence histogram   {gg['confidence_value_counts']}",
            f"  queries recorded       {len(gg['recorded_queries'])}",
            f"  problem id             {', '.join(gg['problem_ids']) or '(none)'}",
        ]
    out.append("")
    if g["confidence_spread_pct_low"] < 20:
        out += [
            f"  !! only {g['confidence_spread_pct_low']}% of beliefs sit below 0.45.",
            "     Goal B has almost no signal to work with, and a three-tier",
            "     rendering would leave one or two tiers empty. Raise the spread in",
            "     the extractor before investing in confidence-driven rendering.",
        ]
    if one and one["bcg"]["graph"]["confidence_distinct_values"] <= 8:
        out += [
            f"  !! confidence takes only {one['bcg']['graph']['confidence_distinct_values']} distinct values.",
            "     That is a category label per node type, not a reliability estimate.",
            "     Check whether the extractor is assigning fixed values by node kind.",
        ]
    if g["source_coverage_pct"] < 70:
        out += [
            f"  !! source coverage {g['source_coverage_pct']}% — bcg_recover will often",
            "     return nothing, which trains the model to stop calling it.",
            "     Check belief_graph.extractor.require_excerpt.",
        ]
    if g["tool_name_coverage_pct"] < 20 and g["action_node_pct"] < 10:
        out += [
            "  !! few nodes record which tool produced them, and few read as action",
            "     records. A 'completed actions' section would have little to show.",
        ]
    if g["contradicts_edges"] == 0:
        out += [
            "  ·  no contradicts edges in this run. Contradiction is a stronger",
            "     trigger for re-verification than a low score, so rendering it",
            "     prominently will not help until the graph produces some.",
        ]

    # ── §3 injection timeline ───────────────────────────────────────────────
    out += section(
        "3",
        "INJECTION — how much context was replaced by how much graph",
        """WHAT: per call, how many messages were dropped and what the graph block
cost in characters.
WHY: this is the mechanism the whole system rests on. messages_removed shows
the compression actually achieved; graph_chars against the budget shows whether
the client-side hard cap is binding. trim_dropped_nodes is the number the
selector offered that did not fit — if it is always 0 the budget is not the
constraint and a smarter trim policy would change nothing.""",
        brief,
    )
    if one:
        b = one["bcg"]
        out += [
            f"  first injection at call  {b['first_injection_call']}",
            f"  max evicted steps        {b['evicted_steps_max']}",
            f"  graph turns written      {b['graph_turns_written']}",
            "",
            f"  messages before   {fmt_stats(b['messages_before'])}",
            f"  messages after    {fmt_stats(b['messages_after'])}",
            f"  messages removed  {fmt_stats(b['messages_removed'])}",
            f"  nodes shown       {fmt_stats(b['selected_nodes'])}",
            f"  relations shown   {fmt_stats(b['selected_relations'])}",
            f"  graph chars       {fmt_stats(b['graph_chars'])}  (budget {meta.get('graph_chars', '?')})",
            f"  prepare latency   {fmt_stats(b['prep_duration_ms'], ' ms')}",
        ]
        if b["action_chars"] is not None:
            out += [
                f"  action section    {fmt_stats(b['action_chars'], ' chars')}",
                f"  actions listed    {fmt_stats(b['action_nodes_shown'])}",
            ]
        if b["trim_dropped_nodes"] is None:
            out += [
                "  trim dropped      not instrumented",
                "                    add trim_dropped_nodes to the prepare event to see",
                "                    how many selected nodes the budget threw away.",
            ]
        else:
            out.append(f"  trim dropped      {fmt_stats(b['trim_dropped_nodes'])}")
            if b["trim_dropped_node_ids"]:
                out.append(f"                    dropped ids: {b['trim_dropped_node_ids'][:20]}")
            elif b["trim_dropped_nodes"]["max"] == 0:
                out.append("                    always 0 — the budget never bound, so trim")
                out.append("                    policy changes cannot move anything yet.")
        if detail and b["timeline"]:
            out += ["", "  per-call timeline:", "", 
                    "   call  fallback                 evict  msgs      nodes rel  chars  drop   ms",
                    "   " + RULE[:72]]
            for t in b["timeline"]:
                fb = (t["fallback"] or "")[:22]
                msgs = f"{t['before']}->{t['after']}" if t["before"] is not None else "-"
                out.append(
                    f"   {str(t['call']):>4}  {fb:<24}{t['evicted']:>5}  {msgs:<9}"
                    f"{t['nodes']:>5}{t['rels']:>4}{t['chars']:>7}"
                    f"{'' if t['dropped'] is None else t['dropped']:>6}{t['ms']:>6}"
                )

    # ── §4 core A ───────────────────────────────────────────────────────────
    c = agg["core"]
    out += section(
        "4",
        "GOAL A — does the model expand beliefs instead of guessing?",
        """WHAT: bcg_recover usage and whether those calls returned anything.
WHY: recover is the only way the model can get back the exact text behind a
belief once the original step has been evicted. Low usage can mean two very
different things, and the success rate tells them apart: if calls succeed but
are rare, the rendering is not advertising recoverability; if calls fail, the
graph lacks source turn ids and the model has learned the tool is useless.
A failed recover is worse than no recover, so success rate is the gate and
call count is the target.""",
        brief,
    )
    out += [
        f"  bcg_recover calls      {c['A_recover_calls']}  ({c['A_recover_per_task']} per task)",
        f"  returned a source      {c['A_recover_success_pct']}%",
        f"  non-integer belief_id  {c['A_recover_bad_arg_type']}",
    ]
    if one:
        out.append(f"  belief ids requested   {one['trajectory']['bcg_recover_belief_ids']}")
    if c["A_recover_bad_arg_type"]:
        out += [
            "  !! the model passed belief_id as a non-integer. Widen the signature to",
            "     (belief_id: int | str) and coerce, or every call fails for a reason",
            "     that has nothing to do with the rendering you are testing.",
        ]
    if c["A_recover_calls"] == 0:
        out += [
            "  ·  zero calls. Baseline for goal A. The rendering currently gives no",
            "     indication of which beliefs can be expanded.",
        ]

    # ── §5 core B ───────────────────────────────────────────────────────────
    out += section(
        "5",
        "GOAL B — does the model read the graph and react to confidence?",
        """WHAT: three nested signals, weakest to strongest.
  belief refs      the model wrote [Bnn] in its reasoning — it is reading the
                   block at all, and can point at a specific node.
  cue blocks       reasoning containing reliability language (uncertain,
                   contradictory, needs verification...). Cheap regex, noisy.
  cue + ref        the reliability language appears next to a belief id.
  strong           cue + belief id + an explicit intent to go get more evidence.
                   This is the behaviour the whole confidence effort is for.
WHY: the strong count is the real metric; the others exist to show where the
chain breaks. Refs at zero means the rendering is not being read; refs high but
strong at zero means it is read but confidence is being ignored.""",
        brief,
    )
    out += [
        f"  assistant blocks       {int(sum(r['trajectory']['assistant_blocks'] for r in rows))}",
        f"  belief refs [Bnn]      {c['B_belief_refs']} total / {c['B_belief_refs_distinct']} distinct",
        f"  reliability cues       {c['B_cue_blocks']} blocks",
        f"  cue + belief ref       {c['B_cue_with_ref']} blocks",
        f"  strong reactions       {c['B_strong_reactions']} blocks   <<< the metric",
    ]
    if one and one["trajectory"]["belief_refs_ids"]:
        out.append(f"  ids referenced         {one['trajectory']['belief_refs_ids']}")
    if one and detail:
        for sample in one["trajectory"]["samples_strong"]:
            out += ["", "  strong sample:", *(f"    | {ln}" for ln in sample.splitlines()[:6])]
        if not one["trajectory"]["samples_strong"]:
            for sample in one["trajectory"]["samples_cue"]:
                out += ["", "  cue-only sample:", *(f"    | {ln}" for ln in sample.splitlines()[:5])]
    if c["B_belief_refs"] == 0:
        out += [
            "",
            "  ·  zero belief references. The model never pointed at a node. Either it",
            "     is not reading the block, or nothing in the rendering invites a",
            "     reference. Asking for [Bnn] citations in the guide would raise this",
            "     number but also contaminate it — keep that as a separate variant.",
        ]

    # ── §6 core C ───────────────────────────────────────────────────────────
    out += section(
        "6",
        "GOAL C — does the model stop repeating work the graph remembers?",
        """WHAT: identical tool calls, repeated search queries, and the subset of
repeats where the graph had already recorded that exact query.
WHY: this is the most concrete cost of evicting history. Once a search result
is gone from the context, nothing stops the model running the same search
again — except the graph telling it the search happened. preventable_repeats
is the sharp metric: the query was in the graph, in front of the model, and it
searched again anyway. That number going to zero is what a completed-actions
section is supposed to achieve.""",
        brief,
    )
    out += [
        f"  tool calls             {agg['efficiency']['tool_calls']}",
        f"  identical repeats      {c['C_duplicate_calls']}  ({c['C_duplicate_rate_pct']}% of calls)",
        f"  of those, search/fetch {c['C_duplicate_search_calls']}   <<< the ones that cost work",
        f"  repeated search calls  {c['C_repeated_query_calls']}",
        f"  preventable repeats    {c['C_preventable_repeats']}   <<< the metric",
        f"  near-duplicate pairs   {c['C_near_duplicate_pairs']}  "
        f"(content-word overlap >= {NEAR_DUPLICATE_THRESHOLD})",
    ]
    if one:
        t = one["trajectory"]
        out.append(f"  duplicates by tool     {t['duplicate_by_name']}")
        out.append(f"  distinct queries       {t['distinct_queries']} of {t['search_calls']} searches")
        out.append(
            f"  queries listed in block {t['queries_listed_in_block']} "
            f"({t['query_coverage_pct']}%)   <<< actions-section coverage"
        )
        if t["duplicate_by_name"].get("bcg_recover"):
            out += [
                "  ·  repeated bcg_recover calls inflate the identical-repeats count.",
                "     Re-reading a belief is cheap and is goal A working, not goal C",
                "     regressing; compare the search/fetch line across variants.",
            ]
        if t["preventable_repeats"] and detail:
            out += ["", "  repeats the graph had already recorded:"]
            for query, info in list(t["preventable_repeats"].items())[:8]:
                out.append(f"    {info['recorded_as']:<6} x{info['times']}  {query[:60]}")
        if t["near_duplicate_queries"] and detail:
            out += ["", "  searches that asked nearly the same thing:"]
            for pair in t["near_duplicate_queries"][:6]:
                out.append(f"    {pair['overlap']}  {pair['a'][:52]}")
                out.append(f"          {pair['b'][:52]}")
        if t["repeated_signatures"] and detail:
            out += ["", "  most repeated call signatures:"]
            for sig in t["repeated_signatures"][:5]:
                args = canonical(sig["args"])[:70]
                out.append(f"    x{sig['times']}  {sig['name']}  {args}")

    # ── §7 efficiency ───────────────────────────────────────────────────────
    e = agg["efficiency"]
    out += section(
        "7",
        "COST — did the gains cost more than they are worth?",
        """WHAT: turns, tool calls, wall time and tokens on both sides.
WHY: goal A pushes recover calls up and goal C pushes searches down, and both
change the turn count. A variant that triples recover usage while adding 30%
more turns and no better answer is a regression. Agent input tokens are the
most sensitive number here: if BCG is working they should stay roughly flat as
the session grows, and if they climb linearly the projection is not compressing
(usually a fallback, see §1).""",
        brief,
    )
    out += [
        f"  turns                  {e['turns']}  ({e['turns_per_task']} per task)"
        + (f"   [{one['log']['turns_source']}]" if one else ""),
        f"  tool calls             {e['tool_calls']}  (search {e['search_calls']} / fetch {e['fetch_calls']})",
        f"  wall time              {e['duration_s']}s  ({e['duration_per_task_s']}s per task)",
        f"  agent tokens           in {e['agent_input_tokens']}  out {e['agent_output_tokens']}"
        f"  total {e['agent_total_tokens']}",
        f"  agent tokens/task      {e['agent_tokens_per_task']}",
        f"  graph tokens           {e['graph_total_tokens'] or 'unavailable'}",
    ]
    if one:
        out.append(f"  tool breakdown         {one['trajectory']['tool_calls_by_name']}")

    # ── §8 output ───────────────────────────────────────────────────────────
    o = agg["output"]
    out += section(
        "8",
        "OUTPUT — did the answer survive?",
        """WHAT: the benchmark's deterministic diagnostics plus a CJK-aware word
count.
WHY: these are not accuracy scores and the official rubric is not implemented
in this runner, so treat them as a regression guard rather than a result. The
useful signal is relative: a variant that improves A/B/C while citations or
length collapse has traded the answer for the metric.""",
        brief,
    )
    out += [
        f"  words (CJK-aware)      {o['words_counted']}  ({o['words_per_task']} per task)",
        f"  headings               {o['headings']}",
        f"  urls cited             {o['urls_cited']}   domains {o['domains_cited']}",
        f"  reference entries      {o['reference_entries']}   DOIs {o['doi_mentions']}",
        f"  references section     {o['with_references']}/{agg['tasks']}",
        f"  blocked-source hits    {o['blocked_hits']}",
    ]
    if o["reference_entries"] and not o["urls_cited"]:
        out += [
            "  ·  urls_cited is 0 but the report has a reference list. The benchmark",
            "     metric counts URLs only, so an academic bibliography scores zero.",
            "     Track reference entries instead when comparing variants.",
        ]
    if one:
        r = one["report"]
        out += [
            f"  words as benchmark counts  {r['words_reported']}  "
            f"(whitespace split; undercounts Chinese)",
            f"  urls in text               {r['urls_in_text']} ({r['urls_distinct']} distinct)",
            f"  answer_status              {one['answer_status'] or '(none)'}"
            f"   source {one['answer_source'] or '(none)'}",
        ]
        if one["stopped_by"] == "no_tool":
            out += [
                "  ·  stopped_by=no_tool. On stateful-react-agent the loop policy is",
                "     no_tool_behavior='stop', so a turn without tool calls IS the",
                "     answer — this is a normal finish. On agent_team the same reason",
                "     means the nudge budget ran out, which is a truncated run.",
            ]

    # ── §9 verdict ──────────────────────────────────────────────────────────
    out += ["", BAR, "§9  WHAT TO DO NEXT", BAR, ""]
    todo: list[str] = []
    if not gate_ok:
        todo.append("Fix the health gate first; nothing below §1 is comparable.")
    if g["confidence_spread_pct_low"] < 20:
        todo.append(
            "Confidence has no spread. Either fix it in the extractor, or drop the "
            "tiered-rendering variant and pursue goals A and C first."
        )
    if g["source_coverage_pct"] >= 70 and c["A_recover_calls"] <= 1:
        todo.append(
            "Recover works but is unused — mark recoverable beliefs in the rendering "
            "(goal A has a clear path here)."
        )
    if c["C_preventable_repeats"] > 0:
        todo.append(
            f"{c['C_preventable_repeats']} searches repeated a query the graph had "
            "recorded — a completed-actions section targets exactly this."
        )
    if one and ((one["bcg"]["action_nodes_shown"] or {}).get("p50") or 0) <= 2 \
            and one["bcg"]["graph"]["node_kinds"].get("action", 0) > 6:
        todo.append(
            "The graph holds many action nodes but only a couple reach the model. "
            "The selector caps search history (FOCUSED_SEARCH_LIMIT) and keeps it "
            "only as provenance for a retained finding, so the completed-actions "
            "section is starved at the source, not by the renderer."
        )
    if one and one["bcg"]["trim_dropped_nodes"] is None:
        todo.append("Instrument trim_dropped_nodes so budget pressure becomes visible.")
    if not tags:
        todo.append("Tag runs with a variant id so archived reports stay distinguishable.")
    if not todo:
        todo.append("Healthy baseline. Archive it and start the first variant.")
    out += [f"  {i}. {line}" for i, line in enumerate(todo, 1)]
    out.append("")
    return "\n".join(out)


def render_per_task(run: dict[str, Any]) -> str:
    head = (
        f"{'task':<8}{'status':<16}{'turns':>6}{'tools':>6}{'dup':>5}{'prev':>5}"
        f"{'rec':>5}{'refs':>6}{'strong':>7}{'fb':>5}{'inj':>5}{'words':>7}{'refs#':>6}"
    )
    lines = ["", head, "─" * len(head)]
    for row in run["tasks"]:
        t, b = row["trajectory"], row["bcg"]
        lines.append(
            f"{row['task']:<8}{str(row['status'])[:15]:<16}{row['log']['turns'] or 0:>6}"
            f"{t['tool_calls_total']:>6}{t['duplicate_tool_calls']:>5}"
            f"{len(t['preventable_repeats']):>5}{t['bcg_recover_calls']:>5}"
            f"{t['belief_refs_total']:>6}{t['strong_confidence_reactions']:>7}"
            f"{b['fallback_calls']:>5}{b['injected_calls']:>5}"
            f"{row['report']['words_counted']:>7}{row['report']['reference_entries'] or 0:>6}"
        )
    lines += [
        "",
        "  dup=identical repeated calls   prev=repeats the graph had recorded",
        "  rec=bcg_recover calls   refs=[Bnn] mentions   strong=confidence reactions",
        "  fb=fallback prepares    inj=prepares that showed a graph",
    ]
    return "\n".join(lines)


COMPARE = [
    ("health.fallback_rate_pct", "fallback %", "must stay 0"),
    ("core.A_recover_calls", "A recover calls", "higher"),
    ("core.A_recover_success_pct", "A recover success %", "higher"),
    ("core.B_belief_refs", "B belief refs", "higher"),
    ("core.B_strong_reactions", "B strong reactions", "higher"),
    ("core.C_duplicate_calls", "C duplicate calls", "lower"),
    ("core.C_duplicate_search_calls", "C duplicate searches", "lower"),
    ("core.C_preventable_repeats", "C preventable repeats", "lower"),
    ("core.C_near_duplicate_pairs", "C near-duplicate pairs", "lower"),
    ("core.C_query_coverage_pct", "C query coverage %", "higher"),
    ("efficiency.turns", "turns", "watch"),
    ("efficiency.tool_calls", "tool calls", "watch"),
    ("efficiency.agent_total_tokens", "agent tokens", "lower"),
    ("efficiency.duration_s", "wall seconds", "watch"),
    ("graph.nodes_total", "graph nodes", "should not move"),
    ("output.words_counted", "report words", "should not drop"),
    ("output.urls_cited", "urls cited", "should not drop"),
    ("output.reference_entries", "reference entries", "should not drop"),
]


def render_compare(base: dict[str, Any], variant: dict[str, Any]) -> str:
    lines = [
        "",
        BAR,
        f"COMPARE   baseline {base['run_dir']}",
        f"          variant  {variant['run_dir']}",
        BAR,
        "",
        f"  {'metric':<24}{'baseline':>11}{'variant':>11}{'change':>12}   want",
        "  " + RULE[:72],
    ]
    for path, label, want in COMPARE:
        a, b = dig(base["aggregate"], path), dig(variant["aggregate"], path)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        delta = b - a
        rel = f"{pct(delta, a):+.0f}%" if a else "n/a"
        mark = "=" if delta == 0 else ("+" if delta > 0 else "-")
        lines.append(f"  {label:<24}{a:>11.1f}{b:>11.1f}{mark + f'{delta:+.1f} ' + rel:>12}   {want}")
    lines += [
        "",
        "  'should not move' rows are sanity checks: if the graph node count changed,",
        "  the construction config changed too and this is not a clean A/B.",
        "  One run per variant is not a result — repeat 3x and compare medians.",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

def load_run(run_dir: Path, only: set[str] | None = None) -> dict[str, Any]:
    meta = read_json(run_dir / "run.json") or {}
    summary = read_json(run_dir / "summary.json") or {}
    trials = find_trials(run_dir)
    if only:
        trials = [t for t in trials if t.name in only]
    if not trials:
        print(f"warning: no trials under {run_dir}"
              + (f" matching {sorted(only)}" if only else ""), file=sys.stderr)
    rows = [collect_trial(t) for t in trials]
    for row in rows:
        row["bcg"]["graph"].pop("nodes_by_id", None)
    merged = {**summary, **meta}
    return {
        "run_dir": str(run_dir),
        "meta": merged,
        "tasks": rows,
        "aggregate": aggregate(rows),
    }


def dump_memory(run_dir: Path, call: int, only: set[str] | None) -> str:
    out: list[str] = []
    for path in sorted(run_dir.rglob("requests.jsonl")):
        if only and not any(part in only for part in path.parts):
            continue
        for event in read_jsonl(path):
            if not isinstance(event, dict) or event.get("call") != call:
                continue
            out += [
                BAR,
                f"{path}   call={call}",
                f"nodes offered {event.get('selector_node_ids')}",
                f"nodes shown   {event.get('selected_node_ids')}",
                f"dropped       {event.get('trim_dropped_node_ids')}",
                f"chars         {event.get('graph_chars')}",
                BAR,
                event.get("system_memory") or "(no memory block on this call)",
            ]
    return "\n".join(out) or f"no prepare event with call={call}"


def dump_queries(run_dir: Path, only: set[str] | None) -> str:
    out: list[str] = []
    for trial in find_trials(run_dir):
        if only and trial.name not in only:
            continue
        bcg = collect_bcg(trial)
        traj = collect_trajectory(trial, bcg["graph"], bcg.get("shown_action_queries"))
        out += [BAR, f"task {trial.name}", BAR, "", "  searches in order:"]
        seen: Counter = Counter()
        for i, query in enumerate(traj["queries_in_order"], 1):
            seen[query] += 1
            mark = f"  REPEAT#{seen[query]}" if seen[query] > 1 else ""
            out.append(f"   {i:>3}. {query[:88]}{mark}")
        recorded = bcg["graph"]["recorded_queries"]
        out += ["", f"  queries the graph recorded ({len(recorded)}):"]
        for query, nid in list(recorded.items())[:40]:
            out.append(f"    B{nid:<5} {query[:80]}")
    return "\n".join(out) or "no searches found"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--tasks", help="comma-separated task ids, e.g. 23 or 5,23")
    parser.add_argument("--baseline", type=Path, help="compare each run against this one")
    parser.add_argument("--json", type=Path, help="write the structured record here")
    parser.add_argument("--per-task", action="store_true", help="add a one-line-per-task table")
    parser.add_argument("--brief", action="store_true", help="omit the WHAT/WHY prose")
    parser.add_argument("--no-detail", action="store_true", help="omit timelines and samples")
    parser.add_argument("--dump-memory", type=int, metavar="CALL",
                        help="print the injected block for this prepare call")
    parser.add_argument("--dump-queries", action="store_true",
                        help="print every search in order and what the graph recorded")
    args = parser.parse_args()

    only = {t.strip() for t in args.tasks.split(",")} if args.tasks else None

    if args.dump_memory is not None:
        for run_dir in args.run_dirs:
            print(dump_memory(run_dir, args.dump_memory, only))
        return
    if args.dump_queries:
        for run_dir in args.run_dirs:
            print(dump_queries(run_dir, only))
        return

    runs = [load_run(d, only) for d in args.run_dirs]
    baseline = load_run(args.baseline, only) if args.baseline else None

    for run in runs:
        print(render(run, brief=args.brief, detail=not args.no_detail))
        if args.per_task or len(run["tasks"]) > 1:
            print(render_per_task(run))
    if baseline:
        for run in runs:
            print(render_compare(baseline, run))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"runs": runs, "baseline": baseline}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()