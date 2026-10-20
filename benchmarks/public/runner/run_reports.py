"""Run DRB2 / DRBench through FrontierAgent's BCG ReAct or Agent Team.

One process per task isolates runtime registries, task workspaces and Graphs.
This runner collects reports and deterministic diagnostics, NOT official scores.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import signal
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.public.reports import drb2, drbench
from benchmarks.public.reports.models import BenchmarkTask
from benchmarks.public.reports.prepare import prepare
from benchmarks.public.reports.report_scoring import ReportScoringContext, score_report_task
from benchmarks.public.reports.runtime import ReportPolicy, report_policy
from benchmarks.public.reports.selection import parse_task_ids, select_tasks

_ROOT = Path(__file__).resolve().parents[3]
_PIPELINES = ("stateful-react-agent", "agent_team")
_ADAPTER_REVISION = 1


def canonicalize(name: str) -> str:
    if name == "drb2":
        return drb2.NAME
    if name not in {drb2.NAME, drbench.NAME}:
        raise ValueError(f"Unsupported report benchmark: {name}")
    return name


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def load_tasks(args: argparse.Namespace) -> list[BenchmarkTask]:
    if args.benchmark == drb2.NAME:
        tasks = drb2.load_tasks(
            args.data_root, data_file=args.data_file, language=args.language, theme=args.theme
        )
    else:
        tasks = drbench.load_tasks(
            args.data_root, data_file=args.data_file, subset=args.subset, difficulty=args.difficulty
        )
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in ids
    ):
        raise ValueError("Task IDs must be unique and safe directory names.")
    selection = parse_task_ids(args.task_ids, canonicalize=canonicalize)
    unexpected = set(selection.scoped) - {args.benchmark}
    if unexpected:
        raise ValueError(f"Task IDs were scoped to a different benchmark: {sorted(unexpected)}")
    if requested := selection.ids_for(args.benchmark):
        tasks, missing = select_tasks(tasks, requested)
        if missing:
            raise ValueError(f"Unknown or filtered task IDs: {missing}")
    elif args.seed is not None:
        random.Random(args.seed).shuffle(tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("No tasks selected.")
    return tasks


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode()
    ).hexdigest()


def _token_usage(raw: dict[str, Any]) -> dict[str, int | None]:
    def number(*names: str) -> int | None:
        return next(
            (int(raw[name]) for name in names if isinstance(raw.get(name), (int, float))), None
        )

    inputs = number("input_tokens", "prompt_tokens")
    outputs = number("output_tokens", "completion_tokens")
    total = number("total_tokens")
    if total is None and inputs is not None and outputs is not None:
        total = inputs + outputs
    return {"input_tokens": inputs, "output_tokens": outputs, "total_tokens": total}


def _sum_usage(items: list[dict[str, int | None]]) -> dict[str, int | None]:
    return {
        key: (
            sum(item[key] or 0 for item in items)
            if items and all(item[key] is not None for item in items)
            else None
        )
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }


def collect_usage(trial: Path) -> dict[str, Any]:
    # Only JSONL llm events, never the duplicate JSON trajectory envelope.
    agent_calls: list[dict[str, int | None]] = []
    fallback_count = 0
    for path in (trial / "agent").rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # abrupt worker termination can leave a partial last line
            if event.get("t") == "llm":
                agent_calls.append(_token_usage(event.get("usage") or {}))
            if path.name == "requests.jsonl" and event.get("fallback"):
                fallback_count += 1
    graphs = []
    for path in (trial / "agent").rglob("state.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if "snapshot" in value and "problem_id" in value:
            raw = (value.get("snapshot") or {}).get("token_usage") or {}
            graphs.append(
                {
                    "path": str(path.relative_to(trial)),
                    "problem_id": value["problem_id"],
                    "usage": raw,
                }
            )
    # Keep total construction usage (including handoff/tail flushes) explicit;
    # online-only counters remain in the native Graph records for separate audit.
    graph_usage = _sum_usage([_token_usage(g["usage"]) for g in graphs])
    agent_usage = _sum_usage(agent_calls)
    return {
        "agent_model_calls": len(agent_calls),
        "agent": agent_usage,
        "graph": graph_usage,
        "combined": _sum_usage([agent_usage, graph_usage]),
        "graph_fallback_calls": fallback_count,
        "graphs": graphs,
        "accounting": "Observed LLM response usage; missing usage is null, never estimated. Input includes cache. Output includes reasoning. Graph includes final/handoff flushes.",
    }


async def execute_task(
    task: BenchmarkTask,
    config: dict[str, Any],
    trial: Path,
    *,
    session_factory: Any = None,
) -> dict[str, Any]:
    """Shared worker body; the injectable session seam supports offline tests."""
    from benchmarks.public.core.kernel_adapter import BenchmarkSession

    started = time.monotonic()
    trial.mkdir(parents=True, exist_ok=True)
    policy = ReportPolicy(
        task,
        trial / "sandbox" / "worktree",
        trial / "tools.jsonl",
        search_calls=config["search_calls"],
        fetch_calls=config["fetch_calls"],
        fetch_chars=config["fetch_chars"],
    )
    meta = {
        "_trial_dir": str(trial),
        "profile": config["profile"],
        "pipeline_id": config["pipeline"],
        "benchmark": task.benchmark,
        "question_id": task.task_id,
        "answer_type": "report",
        "_sys_prompt_addendum": policy.prompt,
        # Keep report delivery inside the research agent; no separate writer
        # can accidentally fall back to short-answer extraction.
        "profile_overrides": {"agent": {"reporter": False}},
    }
    if config.get("max_turns"):
        meta["profile_overrides"]["agent"].update(
            main_max_turns=config["max_turns"],
            sub_max_turns=config["max_turns"],
        )
    if Path(config["profile"]).is_file():
        import yaml

        meta["profile_inline"] = yaml.safe_load(Path(config["profile"]).read_text())
    status = "completed"
    error = None
    state: dict[str, Any] = {}
    try:
        with report_policy(policy):
            async with (session_factory or BenchmarkSession)() as session:
                state = await session.run(
                    policy.instruction(),
                    meta=meta,
                    pipeline_id=config["pipeline"],
                    extra_input={"language": task.metadata.get("language", "auto")},
                )
    except Exception as exc:
        status, error = "agent_error", type(exc).__name__
    report = str(state.get("final_answer") or state.get("final_content") or "").strip()
    if not report and status == "completed":
        status = "empty_report"
    elif state.get("llm_error") or state.get("stopped_by") in {
        "llm_error",
        "user_rejected",
        "aborted",
    }:
        status = "agent_error"
    (trial / "report.md").write_text(report, encoding="utf-8")
    metrics, scoring_errors = score_report_task(
        task,
        report,
        ReportScoringContext(
            tool_trace=policy.trace,
            blocklist=policy.blocklist,
            citation_keys=policy.citation_keys,
        ),
    )
    if scoring_errors and status == "completed":
        status = "score_error"
    usage = collect_usage(trial)
    if usage["graph_fallback_calls"] and status == "completed":
        status = "graph_fallback"
    result = {
        "benchmark": task.benchmark,
        "task_id": task.task_id,
        "pipeline": config["pipeline"],
        "mode": "bcg",
        "status": status,
        "error": error,
        "duration_seconds": round(time.monotonic() - started, 3),
        "official_score": None,
        "scoring_status": "official_pending",
        "metrics": metrics,
        "scoring_errors": scoring_errors,
        "usage": usage,
        "stopped_by": state.get("stopped_by"),
        "answer_status": state.get("answer_status"),
        "report": "report.md",
        "tool_trace": "tools.jsonl",
        "graph_records": "agent/",
    }
    write_json(trial / "result.json", result)
    return result


def _worker(output: Path, task_id: str) -> None:
    config = json.loads((output / "run.json").read_text())
    raw = json.loads((output / "_cases" / f"{task_id}.json").read_text())
    task = BenchmarkTask(**raw)
    trial = output / "trials" / task_id
    # Native deployment mode still receives this task's paths, never repo cwd.
    os.environ["FRONTIER_AGENT_WORKSPACE_DIR"] = str(trial / "sandbox" / "worktree")
    os.environ["FRONTIER_AGENT_OUTPUTS_DIR"] = str(trial / "sandbox" / "outputs")
    os.environ["FRONTIER_AGENT_INPUTS_DIR"] = str(trial / "sandbox" / "inputs")
    os.environ["AGENT_CORE_TRAJECTORY_FORMATS"] = "json,jsonl"
    for name in ("outputs", "inputs"):
        (trial / "sandbox" / name).mkdir(parents=True, exist_ok=True)
    asyncio.run(execute_task(task, config, trial))


def summarize(output: Path, task_ids: list[str]) -> dict[str, Any]:
    results = []
    for task_id in task_ids:
        path = output / "trials" / task_id / "result.json"
        if path.is_file():
            results.append(json.loads(path.read_text()))
    metrics = {}
    for key in {k for result in results for k in result.get("metrics", {})}:
        values = [
            r["metrics"][key]
            for r in results
            if isinstance(r.get("metrics", {}).get(key), (int, float))
        ]
        if values:
            metrics[key] = sum(values) / len(values)
    sums = {}
    means = {}
    for kind in ("agent", "graph", "combined"):
        totals = _sum_usage([r.get("usage", {}).get(kind, _token_usage({})) for r in results])
        sums[kind] = totals
        means[kind] = {
            k: value / len(results) if value is not None else None for k, value in totals.items()
        }
    summary = {
        "selected": len(task_ids),
        "finished": len(results),
        "status_counts": dict(Counter(r["status"] for r in results)),
        "official_score": None,
        "scoring_status": "official_pending",
        "mean_diagnostic_metrics": metrics,
        "total_tokens": sums,
        "mean_tokens_per_task": means,
        "mean_duration_seconds": sum(r["duration_seconds"] for r in results) / len(results)
        if results
        else None,
        "note": "Deterministic diagnostics are not accuracy or official rubric/insight recall scores.",
    }
    write_json(output / "summary.json", summary)
    return summary


async def run_worker(
    task_id: str, config: dict[str, Any], output: Path, sem: asyncio.Semaphore
) -> None:
    trial = output / "trials" / task_id
    if (trial / "result.json").exists():
        return  # completed/error attempts are all terminal; no silent selective reruns
    async with sem:
        if trial.exists() and any(trial.iterdir()):
            raise ValueError(
                f"Interrupted trial {task_id} has artifacts. Use a new output directory; do not mix attempts."
            )
        trial.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with (trial / "agent.log").open("w", encoding="utf-8") as log:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "benchmarks.public.runner.run_reports",
                "_worker",
                "--out",
                str(output),
                "--task-id",
                task_id,
                cwd=_ROOT,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            failure = "worker_error"
            try:
                await asyncio.wait_for(proc.wait(), timeout=config["timeout"] or None)
            except TimeoutError:
                failure = "timeout"
            finally:
                if proc.returncode is None:
                    # Also terminate descendants/sandboxes created by this task.
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(proc.wait(), 5)
                    except TimeoutError:
                        os.killpg(proc.pid, signal.SIGKILL)
                        await proc.wait()
            if not (trial / "result.json").exists():
                write_json(
                    trial / "result.json",
                    {
                        "task_id": task_id,
                        "benchmark": config["benchmark"],
                        "pipeline": config["pipeline"],
                        "mode": "bcg",
                        "status": failure,
                        "duration_seconds": time.monotonic() - start,
                        "official_score": None,
                        "scoring_status": "official_pending",
                        "usage": collect_usage(trial),
                    },
                )
        # Official input layout. Never write into the downloaded evaluator tree.
        if config["benchmark"] == drb2.NAME and (trial / "report.md").is_file():
            target = (
                output
                / "export"
                / drb2.NAME
                / "report"
                / f"bcg-{config['pipeline']}"
                / f"idx-{task_id}.md"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((trial / "report.md").read_text(), encoding="utf-8")
        summary = summarize(output, config["task_ids"])
        print(
            f"Finished {summary['finished']}/{summary['selected']}: {summary['status_counts']}",
            flush=True,
        )


def build_config(args: argparse.Namespace, tasks: list[BenchmarkTask]) -> dict[str, Any]:
    from workflows.stateful_react_agent.bcg.runtime import BCGSettings

    resolved = resolve_profile(args.pipeline, args.profile)
    settings = BCGSettings.from_agent(resolved.get("agent") or {})
    profile = Path(args.profile)
    # Only non-secret model identifiers are persisted; never keys or full env.
    model = (resolved.get("llm") or {}).get("model", "")
    serialized = [asdict(task) for task in tasks]
    source_hashes = {
        item["source"]: hashlib.sha256(Path(item["source"]).read_bytes()).hexdigest()
        for task in tasks
        for item in task.metadata.get("workspace_files", [])
    }
    return {
        "adapter_revision": _ADAPTER_REVISION,
        "benchmark": args.benchmark,
        "pipeline": args.pipeline,
        "profile": str(profile.resolve()) if profile.is_file() else args.profile,
        "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest()
        if profile.is_file()
        else None,
        "effective_profile_sha256": _digest(public_profile(resolved)),
        "agent_model": model,
        "graph_url": settings.url,
        "recent_turns": settings.recent_turns,
        "graph_chars": settings.graph_chars,
        "cross_graph_chars": (resolved.get("agent", {}).get("bcg") or {}).get(
            "cross_graph_chars", 2000
        ),
        "timeout": args.timeout
        if args.timeout is not None
        else (3600 if args.benchmark == drb2.NAME else 2400),
        "max_turns": args.max_turns,
        "search_calls": args.search_calls,
        "fetch_calls": args.fetch_calls,
        "fetch_chars": args.fetch_chars,
        "task_ids": [task.task_id for task in tasks],
        "dataset_sha256": _digest([serialized, source_hashes]),
        "seed": args.seed,
        "official_judge": "not_implemented_upstream",
    }


def public_profile(value: Any) -> Any:
    """Hash reproducibility settings without depending on secrets/key rotation."""
    if isinstance(value, dict):
        return {
            key: public_profile(item)
            for key, item in value.items()
            if not any(
                word in key.casefold()
                for word in ("api_key", "secret", "password", "headers", "credential")
            )
        }
    if isinstance(value, list):
        return [public_profile(item) for item in value]
    return value


def resolve_profile(pipeline: str, name: str) -> dict[str, Any]:
    import yaml

    from workflows.agent_team.profile import load_swarm_profile
    from workflows.stateful_react_agent.profile import load_react_profile

    inline = yaml.safe_load(Path(name).read_text()) if Path(name).is_file() else None
    loader = load_swarm_profile if pipeline == "agent_team" else load_react_profile
    return loader(name, inline=inline)


async def run(args: argparse.Namespace, tasks: list[BenchmarkTask]) -> None:
    config = build_config(args, tasks)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    output = args.out.resolve()
    config["fingerprint"] = _digest(config)
    if (output / "run.json").exists():
        previous = json.loads((output / "run.json").read_text())
        if previous.get("fingerprint") != config["fingerprint"]:
            raise ValueError(
                "Output directory belongs to different settings/data/tasks. Use a new --out."
            )
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is nonempty and has no matching run.json.")
    if all((output / "trials" / task.task_id / "result.json").exists() for task in tasks):
        print(json.dumps(summarize(output, config["task_ids"]), ensure_ascii=False, indent=2))
        return
    # Health only: no prompt/model requests and no charges during preflight.
    import httpx

    from workflows.stateful_react_agent.bcg.runtime import BCGSettings

    settings = BCGSettings.from_agent(
        resolve_profile(args.pipeline, args.profile).get("agent") or {}
    )
    headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
    async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
        response = await client.get(config["graph_url"].rstrip("/") + "/health", headers=headers)
        response.raise_for_status()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", config)
    cases = output / "_cases"
    cases.mkdir(exist_ok=True, mode=0o700)
    for task in tasks:
        write_json(cases / f"{task.task_id}.json", asdict(task))
    sem = asyncio.Semaphore(args.workers)
    await asyncio.gather(*(run_worker(task.task_id, config, output, sem) for task in tasks))
    print(json.dumps(summarize(output, config["task_ids"]), ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    default_root = Path(
        os.environ.get("FRONTIER_AGENT_DATASETS_DIR") or _ROOT / "benchmarks/public/datasets"
    )
    prep = commands.add_parser(
        "prepare", help="Download/import and validate datasets; no model calls."
    )
    prep.add_argument("benchmarks", nargs="+", type=canonicalize)
    prep.add_argument("--data-root", type=Path, default=default_root)
    prep.add_argument("--from", dest="source", type=Path)
    prep.add_argument("--revision")
    prep.add_argument("--no-download", action="store_true")
    prep.add_argument("--skip-render", action="store_true")
    for name in ("list", "run"):
        child = commands.add_parser(name)
        child.add_argument("benchmark", type=canonicalize)
        child.add_argument("--data-root", type=Path, default=default_root)
        child.add_argument(
            "--data-file", type=Path, help="DRB2 JSONL file or DRBench repository/data directory."
        )
        child.add_argument("--language", choices=("zh", "en"))
        child.add_argument("--theme")
        child.add_argument("--subset", choices=drbench.SUBSETS, default="val")
        child.add_argument("--difficulty", choices=drbench.DIFFICULTIES)
        child.add_argument("--task-ids")
        child.add_argument("--limit", type=int)
        child.add_argument(
            "--seed",
            type=int,
            help="Deterministic shuffle before --limit; explicit IDs retain order.",
        )
        if name == "run":
            child.add_argument("--pipeline", choices=_PIPELINES, default="agent_team")
            child.add_argument("--profile", default="benchmark")
            child.add_argument("--workers", type=int, default=1)
            child.add_argument(
                "--timeout",
                type=float,
                help="Outer per-task seconds; 0 disables only this outer limit.",
            )
            child.add_argument("--max-turns", type=int)
            child.add_argument("--search-calls", type=int, default=20)
            child.add_argument("--fetch-calls", type=int, default=30)
            child.add_argument("--fetch-chars", type=int, default=20_000)
            child.add_argument("--out", type=Path, required=True)
            child.add_argument("--dry-run", action="store_true")
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--task-id", required=True)
    return result


def main() -> None:
    # Use FrontierAgent's normal .env loader, not credentials from the old BCG CLI.
    import frontier_agent.infra.config  # noqa: F401

    cli = parser()
    args = cli.parse_args()
    try:
        if args.command == "_worker":
            _worker(args.out.resolve(), args.task_id)
            return
        if args.command == "prepare":
            if args.source and len(args.benchmarks) != 1:
                raise ValueError("--from requires exactly one benchmark.")
            for name in args.benchmarks:
                prepared = prepare(
                    name,
                    args.data_root,
                    source=args.source,
                    revision=args.revision,
                    download=not args.no_download,
                    render=not args.skip_render,
                )
                print(f"{name}: {prepared.tasks} tasks, {prepared.action}: {prepared.directory}")
            return
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive.")
        tasks = load_tasks(args)
        if args.command == "list":
            for task in tasks:
                print(
                    f"{task.task_id}\t{task.metadata.get('language') or task.metadata.get('difficulty', '')}\t{task.question[:120]}"
                )
            print(f"Total: {len(tasks)}")
            return
        for key in ("workers", "max_turns", "search_calls", "fetch_calls", "fetch_chars"):
            if getattr(args, key) is not None and getattr(args, key) <= 0:
                raise ValueError(f"--{key.replace('_', '-')} must be positive.")
        if args.timeout is not None and args.timeout < 0:
            raise ValueError("--timeout must be nonnegative.")
        asyncio.run(run(args, tasks))
    except (ValueError, OSError, RuntimeError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    main()
