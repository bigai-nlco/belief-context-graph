"""Compute pass^k from existing rollout results.

Reads results.json files produced by the agent rollout runner and reports
pass^k (probability all k samples are correct) alongside pass@k for
various k values.

Usage:
    python scripts/compute_passk.py output/Qwen3-4B-Thinking-2507/gpqa_diamond/results_20260515-061009.json
    python scripts/compute_passk.py output/Qwen3-4B-Thinking-2507/  # all benchmarks under a model dir
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return 1.0 if c == n else 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def estimate_pass_hat_k(n: int, c: int, k: int) -> float:
    if n < k:
        return 1.0 if c == n else 0.0
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


def analyze_results(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)

    records = data.get("records", [])
    if not records:
        return {}

    num_samples = records[0].get("num_samples", 1)
    ks = sorted({k for k in [1, 2, 4, 8, 16, 32, num_samples] if 1 <= k <= num_samples})

    per_task_passk = {k: [] for k in ks}
    per_task_passhatk = {k: [] for k in ks}

    for rec in records:
        n = rec["num_samples"]
        c = rec["num_correct"]
        for k in ks:
            per_task_passk[k].append(estimate_pass_at_k(n, c, k))
            per_task_passhatk[k].append(estimate_pass_hat_k(n, c, k))

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def std(vals):
        m = mean(vals)
        return (
            math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))
            if len(vals) > 1
            else 0.0
        )

    summary = {
        "file": str(path),
        "benchmark": data.get("summary", {}).get("benchmark", path.parent.name),
        "num_tasks": len(records),
        "num_samples": num_samples,
    }
    for k in ks:
        summary[f"pass@{k}"] = mean(per_task_passk[k])
        summary[f"pass@{k}_std"] = std(per_task_passk[k])
        summary[f"pass^{k}"] = mean(per_task_passhatk[k])
        summary[f"pass^{k}_std"] = std(per_task_passhatk[k])

    return summary


def print_table(summaries: list[dict]) -> None:
    if not summaries:
        return

    num_samples = summaries[0].get("num_samples", 1)
    ks = sorted({k for k in [1, 2, 4, 8, 16, 32, num_samples] if 1 <= k <= num_samples})

    tbl = Table(title="Pass@k and Pass^k Analysis", show_lines=False)
    tbl.add_column("Benchmark", style="cyan")
    tbl.add_column("N", justify="right")
    tbl.add_column("Samples", justify="right")
    for k in ks:
        tbl.add_column(f"pass@{k}", justify="right")
    for k in ks:
        tbl.add_column(f"pass^{k}", justify="right")

    def fmt(m, s):
        return f"{m*100:.1f} ({s*100:.1f})"

    for s in summaries:
        row = [s["benchmark"], str(s["num_tasks"]), str(s["num_samples"])]
        for k in ks:
            row.append(fmt(s.get(f"pass@{k}", 0), s.get(f"pass@{k}_std", 0)))
        for k in ks:
            row.append(fmt(s.get(f"pass^{k}", 0), s.get(f"pass^{k}_std", 0)))
        tbl.add_row(*row)
    console.print(tbl)


def main():
    parser = argparse.ArgumentParser(description="Compute pass^k from rollout results")
    parser.add_argument(
        "paths", nargs="+", help="results.json files or model directories"
    )
    args = parser.parse_args()

    result_files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_file():
            result_files.append(path)
        elif path.is_dir():
            result_files.extend(sorted(path.rglob("results*.json")))

    if not result_files:
        console.print("[red]No results files found.")
        sys.exit(1)

    summaries = []
    for f in result_files:
        s = analyze_results(f)
        if s:
            summaries.append(s)

    print_table(summaries)

    if len(summaries) == 1:
        console.print()
        console.print(json.dumps(summaries[0], indent=2))


if __name__ == "__main__":
    main()
