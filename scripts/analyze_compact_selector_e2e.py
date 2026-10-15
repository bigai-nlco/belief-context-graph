#!/usr/bin/env python3
"""Compare two paired Compact Graph benchmark runs without causal overclaiming."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized_query(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def query_similarity(left: str, right: str) -> float:
    a = set(normalized_query(left).split())
    b = set(normalized_query(right).split())
    return len(a & b) / len(a | b) if a and b else 0.0


def model_queries(path: Path) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for row in read_jsonl(path):
        if row.get("type") != "response":
            continue
        call_id = int(row.get("call_id", 0))
        content = row.get("message", {}).get("content") or []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "toolCall":
                continue
            arguments = block.get("arguments")
            if isinstance(arguments, dict) and isinstance(arguments.get("query"), str):
                result.append((call_id, arguments["query"]))
    return result


def repeated_queries(queries: list[tuple[int, str]], threshold: float = 0.72) -> int:
    previous: list[str] = []
    repeated = 0
    for _, query in queries:
        if any(query_similarity(query, item) >= threshold for item in previous):
            repeated += 1
        previous.append(query)
    return repeated


def model_calls(path: Path) -> int:
    return sum(row.get("type") == "request" for row in read_jsonl(path))


def task_files(run_dir: Path) -> Path:
    return run_dir / "browsecomp" / "bcg" / "tasks"


def mode_dir(run_dir: Path) -> Path:
    return run_dir / "browsecomp" / "bcg"


def trace_stats(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    retrieval = Counter(str(row.get("selectionRetrieval") or "unrecorded") for row in rows)
    nonempty = [row for row in rows if row.get("text")]
    return {
        "calls": len(rows),
        "nonempty": len(nonempty),
        "retrieval": dict(retrieval),
        "embedding_calls": retrieval.get("embedding", 0) + retrieval.get("lexical", 0),
        "mean_chars": (
            sum(int(row.get("chars", 0)) for row in nonempty) / len(nonempty)
            if nonempty
            else 0.0
        ),
    }


def first_selector_call(path: Path) -> int | None:
    rows = read_jsonl(path)
    for index, row in enumerate(rows, 1):
        if row.get("selectionRetrieval") in {"embedding", "lexical"}:
            return index
    return None


def total_tokens(task: dict[str, Any]) -> int:
    return int(task.get("usage", {}).get("total", 0)) + int(
        task.get("graph_usage", {}).get("total", 0)
    )


def compare_task(
    task_name: str, ranked_dir: Path, connected_dir: Path
) -> dict[str, Any]:
    ranked_mode = mode_dir(ranked_dir)
    connected_mode = mode_dir(connected_dir)
    ranked = json.loads((task_files(ranked_dir) / task_name).read_text(encoding="utf-8"))
    connected = json.loads(
        (task_files(connected_dir) / task_name).read_text(encoding="utf-8")
    )
    task_id = str(connected["task_id"])
    ranked_io = ranked_mode / "model-io" / f"browsecomp-bcg-{task_id}.jsonl"
    connected_io = connected_mode / "model-io" / f"browsecomp-bcg-{task_id}.jsonl"
    connected_trace = (
        connected_mode / "graph-contexts" / f"browsecomp-bcg-{task_id}.jsonl"
    )
    ranked_queries = model_queries(ranked_io)
    connected_queries = model_queries(connected_io)
    selector_call = first_selector_call(connected_trace)
    before_selector = (
        [query for call, query in connected_queries if call < selector_call]
        if selector_call is not None
        else [query for _, query in connected_queries]
    )
    ranked_prefix = [query for _, query in ranked_queries[: len(before_selector)]]
    diverged_before_selector = bool(before_selector) and [
        normalized_query(query) for query in before_selector
    ] != [normalized_query(query) for query in ranked_prefix]
    return {
        "task_id": task_id,
        "question": connected.get("question"),
        "reference_answer": (connected.get("reference_answers") or [""])[0],
        "ranked_correct": bool(ranked.get("correct")),
        "connected_correct": bool(connected.get("correct")),
        "ranked_answer": ranked.get("extracted_answer"),
        "connected_answer": connected.get("extracted_answer"),
        "ranked_calls": model_calls(ranked_io),
        "connected_calls": model_calls(connected_io),
        "ranked_searches": int(ranked.get("search_calls", 0)),
        "connected_searches": int(connected.get("search_calls", 0)),
        "ranked_repeated_queries": repeated_queries(ranked_queries),
        "connected_repeated_queries": repeated_queries(connected_queries),
        "ranked_total_tokens": total_tokens(ranked),
        "connected_total_tokens": total_tokens(connected),
        "connected_trace": trace_stats(connected_trace),
        "first_selector_call": selector_call,
        "diverged_before_selector": diverged_before_selector,
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranked-run", type=Path, required=True)
    parser.add_argument("--connected-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ranked_names = {path.name for path in task_files(args.ranked_run).glob("*.json")}
    connected_names = {path.name for path in task_files(args.connected_run).glob("*.json")}
    names = sorted(ranked_names & connected_names)
    rows = [compare_task(name, args.ranked_run, args.connected_run) for name in names]
    matrix = Counter(
        (row["ranked_correct"], row["connected_correct"]) for row in rows
    )
    exclusive = [
        row for row in rows if row["ranked_correct"] != row["connected_correct"]
    ]
    selector_active_exclusive = [
        row
        for row in exclusive
        if row["connected_trace"]["embedding_calls"] > 0
    ]
    summary = {
        "ranked_run": str(args.ranked_run.resolve()),
        "connected_run": str(args.connected_run.resolve()),
        "tasks": len(rows),
        "paired_matrix": {
            "both_correct": matrix[(True, True)],
            "ranked_only": matrix[(True, False)],
            "connected_only": matrix[(False, True)],
            "both_wrong": matrix[(False, False)],
        },
        "mean": {
            "ranked_calls": mean(rows, "ranked_calls"),
            "connected_calls": mean(rows, "connected_calls"),
            "ranked_searches": mean(rows, "ranked_searches"),
            "connected_searches": mean(rows, "connected_searches"),
            "ranked_repeated_queries": mean(rows, "ranked_repeated_queries"),
            "connected_repeated_queries": mean(rows, "connected_repeated_queries"),
            "ranked_total_tokens": mean(rows, "ranked_total_tokens"),
            "connected_total_tokens": mean(rows, "connected_total_tokens"),
        },
        "exclusive_tasks": len(exclusive),
        "selector_active_exclusive_tasks": len(selector_active_exclusive),
        "exclusive_diverged_before_selector": sum(
            bool(row["diverged_before_selector"]) for row in exclusive
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_task.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Paired Compact selector end-to-end diagnostic",
        "",
        "## Paired outcome",
        "",
        "| Both correct | Ranked only | Connected only | Both wrong |",
        "|---:|---:|---:|---:|",
        f"| {matrix[(True, True)]} | {matrix[(True, False)]} | {matrix[(False, True)]} | {matrix[(False, False)]} |",
        "",
        "## Exclusive outcomes",
        "",
        "| Task | Winner | Selector calls | Diverged before selector | Calls R/C | Searches R/C | Repeated queries R/C | Tokens R/C |",
        "|---|---|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in exclusive:
        winner = "ranked" if row["ranked_correct"] else "connected"
        lines.append(
            f"| {row['task_id']} | {winner} | {row['connected_trace']['embedding_calls']} | "
            f"{'yes' if row['diverged_before_selector'] else 'no'} | "
            f"{row['ranked_calls']}/{row['connected_calls']} | "
            f"{row['ranked_searches']}/{row['connected_searches']} | "
            f"{row['ranked_repeated_queries']}/{row['connected_repeated_queries']} | "
            f"{row['ranked_total_tokens']:,}/{row['connected_total_tokens']:,} |"
        )
    lines.extend(
        [
            "",
            "## Causal boundary",
            "",
            f"Only {len(selector_active_exclusive)} of {len(exclusive)} exclusive outcomes entered query-aware selection at all. "
            f"In {sum(bool(row['diverged_before_selector']) for row in exclusive)} exclusive outcomes, the search trajectory had already diverged before the first query-aware selector call. "
            "Independent end-to-end runs therefore measure total-system variance and utility, not the isolated causal effect of a selector. Fixed-snapshot replay is required for selector attribution.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
