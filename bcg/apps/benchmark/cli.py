"""Rich command-line interface for benchmark adapters."""

from __future__ import annotations

import os
import random
import shlex
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from bcg.apps.agent_runtime import AgentLaunchError, _normalize_agent_api
from bcg.apps.benchmark import drb2, drbench
from bcg.apps.benchmark.loaders import (
    BENCHMARKS,
    REPORT_BENCHMARKS,
    BenchmarkDataError,
    canonical_name,
    load_benchmark,
)
from bcg.apps.benchmark.models import BenchmarkTask
from bcg.apps.benchmark.prepare import (
    PREPARABLE,
    PrepareError,
    dataset_manifest,
)
from bcg.apps.benchmark.prepare import prepare as prepare_benchmark
from bcg.apps.benchmark.profiles import (
    ProfileError,
    parse_bench_opts,
    resolve_profiles,
)
from bcg.apps.benchmark.runner import (
    RunConfig,
    _system_prompt,
    _task_prompt,
    _tools_for,
    run_benchmarks,
)
from bcg.apps.benchmark.scoring import JudgeConfig, LLMJudge
from bcg.apps.benchmark.selection import (
    SelectionError,
    parse_task_ids,
    select_tasks,
)


def _bootstrap_env() -> None:
    """Load the project root .env explicitly (import no longer does this)."""
    from bcg.core.env import load_project_env

    load_project_env()


app = typer.Typer(
    name="bcg benchmark",
    help="Evaluate the reference Agent across supported context modes.",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
)
console = Console()


@app.callback()
def _root() -> None:
    """Evaluate the reference Agent across supported context modes."""


@app.command("run")
def run(
    benchmarks: Annotated[
        list[str],
        typer.Argument(help=f"Benchmarks: {', '.join(BENCHMARKS)}."),
    ],
    data_root: Annotated[
        Path,
        typer.Option("--data-root", help="Root containing benchmark directories."),
    ] = Path("datasets"),
    data_file: Annotated[
        list[str] | None,
        typer.Option(
            "--data-file",
            help="Override a dataset file as BENCHMARK=PATH; repeat as needed.",
        ),
    ] = None,
    modes: Annotated[
        str,
        typer.Option(
            help=("Comma-separated context modes: default,bcg,summary,recent-only,rag.")
        ),
    ] = "default,bcg",
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Run artifact directory."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(help="Agent model (default: BCG_AGENT_MODEL/OPENAI_MODEL/MODEL)."),
    ] = None,
    thinking: Annotated[
        str,
        typer.Option(
            help="Agent thinking level: off, minimal, low, medium, high, xhigh, max."
        ),
    ] = "off",
    base_url: Annotated[
        str | None,
        typer.Option(help="Agent OpenAI-compatible API base URL."),
    ] = None,
    agent_api: Annotated[
        str | None,
        typer.Option(
            "--agent-api",
            help=(
                "Agent protocol: chat/openai-completions or "
                "responses/openai-responses (default: BCG_AGENT_API or chat)."
            ),
        ),
    ] = None,
    api_key_env: Annotated[
        str,
        typer.Option(help="Environment variable holding the Agent API key."),
    ] = "OPENAI_API_KEY",
    max_problems: Annotated[
        int,
        typer.Option(min=0, help="Examples per benchmark; 0 means all."),
    ] = 0,
    task_ids: Annotated[
        str | None,
        typer.Option(
            help=(
                "Task IDs to run, kept in the given order: `1,5`, ranges "
                "`DR0001-DR0003`, per-benchmark `drb2:1,2;drbench:DR0001`, "
                "or `@file.txt`. Unknown IDs are an error."
            )
        ),
    ] = None,
    shuffle: Annotated[
        bool,
        typer.Option("--shuffle/--no-shuffle", help="Shuffle before truncation."),
    ] = True,
    seed: Annotated[int, typer.Option(help="Selection shuffle seed.")] = 42,
    workers: Annotated[
        int,
        typer.Option(min=1, help="Concurrent Agent task processes."),
    ] = 4,
    timeout: Annotated[
        float | None,
        typer.Option(
            min=0,
            help=(
                "Per-task Agent timeout in seconds for every benchmark; 0 disables "
                "the limit (default: per-benchmark profile, 900 for QA benchmarks)."
            ),
        ),
    ] = None,
    graph_url: Annotated[
        str,
        typer.Option(help="Graph Construction server URL for BCG mode."),
    ] = "http://127.0.0.1:8848",
    graph_timeout_ms: Annotated[
        int,
        typer.Option(min=1, help="Graph request timeout in milliseconds."),
    ] = 300_000,
    graph_finalization_timeout_ms: Annotated[
        int,
        typer.Option(
            min=1,
            help="Final Graph supplement/finalize timeout in milliseconds.",
        ),
    ] = 900_000,
    graph_max_turns: Annotated[
        int | None,
        typer.Option(
            min=1,
            help=(
                "Maximum Graph messages per BCG task, including the initial "
                "system and user messages (default: per-benchmark profile, 160 "
                "for QA benchmarks)."
            ),
        ),
    ] = None,
    search_max_calls: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Hard web_search calls per task (overrides SERPER_MAX_CALLS).",
        ),
    ] = None,
    fetch_max_calls: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Hard web_fetch calls per task (overrides WEB_FETCH_MAX_CALLS).",
        ),
    ] = None,
    fetch_max_chars: Annotated[
        int | None,
        typer.Option(
            min=500,
            help="Characters returned per web_fetch page (overrides WEB_FETCH_MAX_CHARS).",
        ),
    ] = None,
    bench_opt: Annotated[
        list[str] | None,
        typer.Option(
            "--bench-opt",
            help=(
                "Per-benchmark override BENCH.KEY=VALUE; repeatable. Keys: timeout, "
                "graph_max_turns, search_max_calls, fetch_max_calls, "
                "fetch_max_chars, web."
            ),
        ),
    ] = None,
    web_cache: Annotated[
        Path | None,
        typer.Option(
            "--web-cache",
            help="Directory caching web_search/web_fetch responses across modes and runs.",
        ),
    ] = None,
    drb2_language: Annotated[
        str | None,
        typer.Option(help="DeepResearch Bench II: keep only `zh` or `en` tasks."),
    ] = None,
    drb2_theme: Annotated[
        str | None,
        typer.Option(help="DeepResearch Bench II: keep one theme, e.g. 'Health'."),
    ] = None,
    drbench_subset: Annotated[
        str,
        typer.Option(help="DRBench subset: val (100 tasks), minival (15), or sanity."),
    ] = "val",
    drbench_difficulty: Annotated[
        str | None,
        typer.Option(help="DRBench: keep `easy`, `medium`, or `hard` tasks."),
    ] = None,
    drbench_web: Annotated[
        bool | None,
        typer.Option(
            "--drbench-web/--no-drbench-web",
            help="DRBench: also expose web_search/web_fetch (default: off).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Print the selected tasks, effective budgets, and first prompt; run nothing."
        ),
    ] = False,
    recent_turns: Annotated[
        int,
        typer.Option(
            min=-1,
            help="Completed turns retained verbatim in bounded-context modes.",
        ),
    ] = 2,
    rag_top_k: Annotated[
        int,
        typer.Option(min=1, help="Maximum historical turns retrieved in RAG mode."),
    ] = 6,
    rag_max_chars: Annotated[
        int,
        typer.Option(
            min=256, help="Maximum retrieved-history characters injected in RAG mode."
        ),
    ] = 12_000,
    summary_model: Annotated[
        str | None,
        typer.Option(help="Rolling-summary model; defaults to the Agent model."),
    ] = None,
    summary_base_url: Annotated[
        str | None,
        typer.Option(help="Summary API base URL; defaults to the Agent base URL."),
    ] = None,
    summary_api_key_env: Annotated[
        str,
        typer.Option(help="Environment variable holding the Summary API key."),
    ] = "BCG_SUMMARY_API_KEY",
    summary_thinking: Annotated[
        str,
        typer.Option(
            help="Summary model thinking level: off, minimal, low, medium, high, xhigh, max."
        ),
    ] = "off",
    summary_timeout_ms: Annotated[
        int,
        typer.Option(min=1, help="Summary model request timeout in milliseconds."),
    ] = 300_000,
    summary_max_tokens: Annotated[
        int,
        typer.Option(min=1, help="Maximum output tokens per summary update."),
    ] = 2048,
    graph_view: Annotated[
        str,
        typer.Option(
            help=(
                "Graph injection view: full preserves the original per-belief "
                "dialogue; compact injects a bounded search ledger and evidence view."
            )
        ),
    ] = "full",
    graph_selection: Annotated[
        str,
        typer.Option(
            help=(
                "Compact Graph selector: ranked uses the legacy independent ranking; "
                "connected uses query-aware relation paths; focused preserves its "
                "evidence while pruning redundant investigation chains."
            )
        ),
    ] = "connected",
    allow_graph_fallback: Annotated[
        bool,
        typer.Option(help="Score BCG tasks even when Graph falls back to raw context."),
    ] = False,
    allow_summary_fallback: Annotated[
        bool,
        typer.Option(
            help="Score Summary tasks even when summarization falls back to raw context."
        ),
    ] = False,
    allow_rag_fallback: Annotated[
        bool,
        typer.Option(
            help="Score RAG tasks even when retrieval falls back to Recent-Only."
        ),
    ] = False,
    allow_no_search: Annotated[
        bool,
        typer.Option(
            help=(
                "Run web benchmarks (BrowseComp, HotpotQA, DeepResearch Bench II, "
                "DRBench with --drbench-web) closed-book when SERPER_API_KEY is unset."
            )
        ),
    ] = False,
    gaia_split: Annotated[
        str,
        typer.Option(help="GAIA split: validation or test."),
    ] = "validation",
    gaia_level: Annotated[
        int | None,
        typer.Option(min=1, max=3, help="Optionally keep one GAIA difficulty level."),
    ] = None,
    gaia_text_only: Annotated[
        bool,
        typer.Option(
            help=(
                "Exclude GAIA tasks with attachments or annotator-declared "
                "image, video, audio, OCR, or other visual tooling."
            )
        ),
    ] = False,
    judge_model: Annotated[
        str | None,
        typer.Option(help="BrowseComp judge model; defaults to the Agent model."),
    ] = None,
    judge_base_url: Annotated[
        str | None,
        typer.Option(help="Judge API base URL; defaults to the Agent base URL."),
    ] = None,
    judge_api_key_env: Annotated[
        str,
        typer.Option(help="Environment variable holding the judge API key."),
    ] = "OPENAI_API_KEY",
    judge_timeout: Annotated[
        float,
        typer.Option(min=0.1, help="Judge request timeout in seconds."),
    ] = 120.0,
    judge_max_retries: Annotated[
        int,
        typer.Option(min=0, help="Judge retry count."),
    ] = 2,
    agent_command: Annotated[
        str | None,
        typer.Option(help="Override the bcg-agent executable command."),
    ] = None,
    record_model_io: Annotated[
        bool,
        typer.Option(
            "--record-model-io/--no-record-model-io",
            help="Record every Agent model request and response as per-task JSONL.",
        ),
    ] = True,
    overwrite: Annotated[
        bool,
        typer.Option(help="Rerun task artifacts that already exist."),
    ] = False,
) -> None:
    """Run one or more benchmarks and write trajectories plus summary metrics."""

    # Reuse the persistent first-run configuration just like the interactive
    # Agent does. Explicit CLI flags and already-exported environment variables
    # still take precedence.
    from bcg.apps.setup import apply_user_configuration, is_configured
    from bcg.apps.setup import load_user_configuration as load_setup_configuration

    setup_config = load_setup_configuration()
    if is_configured(setup_config):
        apply_user_configuration(setup_config)

    if not benchmarks:
        raise typer.BadParameter("At least one benchmark is required.")
    try:
        canonical = [canonical_name(name) for name in benchmarks]
        if len(set(canonical)) != len(canonical):
            raise BenchmarkDataError("Each benchmark may be specified only once.")
        file_overrides = _parse_data_files(data_file or [])
        tasks_by_benchmark = _select_tasks(
            canonical,
            data_root=data_root,
            file_overrides=file_overrides,
            task_ids=task_ids,
            shuffle=shuffle,
            seed=seed,
            max_problems=max_problems,
            loader_options={
                "gaia_split": gaia_split,
                "gaia_level": gaia_level,
                "gaia_text_only": gaia_text_only,
                "drb2_language": drb2_language,
                "drb2_theme": drb2_theme,
                "drbench_subset": drbench_subset,
                "drbench_difficulty": drbench_difficulty,
            },
        )
        bench_overrides = parse_bench_opts(bench_opt or [], canonicalize=canonical_name)
        if drbench_web is not None:
            bench_overrides.setdefault(drbench.NAME, {}).setdefault("web", drbench_web)
        profiles = resolve_profiles(
            canonical,
            global_overrides={
                "timeout": timeout,
                "graph_max_turns": graph_max_turns,
                "search_max_calls": search_max_calls,
                "fetch_max_calls": fetch_max_calls,
                "fetch_max_chars": fetch_max_chars,
            },
            bench_overrides=bench_overrides,
        )
    except (BenchmarkDataError, SelectionError, ProfileError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    for warning in _search_binary_warnings(profiles):
        console.print(f"[yellow]{warning}[/yellow]", highlight=False)
    if dry_run:
        _print_dry_run(tasks_by_benchmark, profiles, allow_no_search=allow_no_search)
        return

    resolved_model = (
        model
        or os.environ.get("BCG_AGENT_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("MODEL")
        or ""
    ).strip()
    resolved_base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "").strip()
    try:
        resolved_agent_api = _normalize_agent_api(
            agent_api or os.environ.get("BCG_AGENT_API")
        )
    except AgentLaunchError as exc:
        raise typer.BadParameter(str(exc)) from exc
    api_key = os.environ.get(api_key_env, "")
    if not resolved_model:
        raise typer.BadParameter(
            "--model is required (or set BCG_AGENT_MODEL/OPENAI_MODEL/MODEL)."
        )
    if not resolved_base_url:
        raise typer.BadParameter("--base-url is required (or set OPENAI_BASE_URL).")
    valid_thinking = {"off", "minimal", "low", "medium", "high", "xhigh", "max"}
    if thinking not in valid_thinking:
        raise typer.BadParameter(
            "--thinking must be one of: " + ", ".join(sorted(valid_thinking))
        )
    if summary_thinking not in valid_thinking:
        raise typer.BadParameter(
            "--summary-thinking must be one of: " + ", ".join(sorted(valid_thinking))
        )
    if graph_view not in {"full", "compact"}:
        raise typer.BadParameter("--graph-view must be `full` or `compact`.")
    if graph_selection not in {"ranked", "connected", "focused"}:
        raise typer.BadParameter(
            "--graph-selection must be `ranked`, `connected`, or `focused`."
        )

    resolved_modes = tuple(value.strip() for value in modes.split(",") if value.strip())
    valid_modes = {"default", "bcg", "summary", "recent-only", "rag"}
    invalid_modes = set(resolved_modes) - valid_modes
    if not resolved_modes or invalid_modes:
        raise typer.BadParameter(
            "--modes must contain only `default`, `bcg`, `summary`, "
            "`recent-only`, and/or `rag`."
        )
    resolved_summary_model = (summary_model or resolved_model).strip()
    resolved_summary_base_url = (summary_base_url or resolved_base_url).strip()
    summary_api_key = os.environ.get(summary_api_key_env, "") or api_key
    destination = output_dir or Path("results") / "benchmarks" / time.strftime(
        "%Y%m%d-%H%M%S"
    )
    uses_judge = bool({"browsecomp", "browsecomp_zh"} & set(canonical))
    judge = None
    if uses_judge:
        judge = LLMJudge(
            JudgeConfig(
                model=(judge_model or resolved_model).strip(),
                base_url=(judge_base_url or resolved_base_url).strip(),
                api_key=os.environ.get(judge_api_key_env, ""),
                timeout=judge_timeout,
                max_retries=judge_max_retries,
            )
        )

    config = RunConfig(
        output_dir=destination,
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=api_key,
        agent_api=resolved_agent_api,
        modes=resolved_modes,
        thinking=thinking,
        workers=workers,
        timeout=timeout if timeout is not None else 900.0,
        graph_url=graph_url,
        graph_timeout_ms=graph_timeout_ms,
        graph_finalization_timeout_ms=graph_finalization_timeout_ms,
        graph_max_turns=graph_max_turns or 160,
        recent_turns=recent_turns,
        rag_top_k=rag_top_k,
        rag_max_chars=rag_max_chars,
        graph_view=graph_view,
        graph_selection=graph_selection,
        summary_model=resolved_summary_model,
        summary_base_url=resolved_summary_base_url,
        summary_api_key=summary_api_key,
        summary_thinking=summary_thinking,
        summary_timeout_ms=summary_timeout_ms,
        summary_max_tokens=summary_max_tokens,
        allow_graph_fallback=allow_graph_fallback,
        allow_summary_fallback=allow_summary_fallback,
        allow_rag_fallback=allow_rag_fallback,
        allow_no_search=allow_no_search,
        record_model_io=record_model_io,
        overwrite=overwrite,
        agent_command=tuple(shlex.split(agent_command)) if agent_command else None,
        profiles=profiles,
        web_cache_dir=web_cache,
        dataset_manifests={
            name: manifest
            for name in canonical
            if name in REPORT_BENCHMARKS
            and (manifest := dataset_manifest(name, data_root)) is not None
        },
    )
    try:
        if "bcg" in resolved_modes:
            from bcg.apps.agent_runtime import ensure_graph_server

            ensure_graph_server(graph_url)
        summary = run_benchmarks(tasks_by_benchmark, config, judge=judge)
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]bcg benchmark: {exc}[/red]", highlight=False)
        raise typer.Exit(1) from exc
    _print_summary(summary, destination.resolve())


def _load_one(
    name: str,
    data_root: Path,
    data_file: Path | None,
    options: dict[str, object],
) -> list[BenchmarkTask]:
    return load_benchmark(
        name,
        data_root,
        data_file=data_file,
        split=str(options["gaia_split"]) if name == "gaia" else None,
        gaia_level=options.get("gaia_level"),  # type: ignore[arg-type]
        gaia_text_only=bool(options.get("gaia_text_only")),
        drb2_language=options.get("drb2_language"),  # type: ignore[arg-type]
        drb2_theme=options.get("drb2_theme"),  # type: ignore[arg-type]
        drbench_subset=str(options.get("drbench_subset") or "val"),
        drbench_difficulty=options.get("drbench_difficulty"),  # type: ignore[arg-type]
    )


def _select_tasks(
    benchmarks: list[str],
    *,
    data_root: Path,
    file_overrides: dict[str, Path],
    task_ids: str | None,
    shuffle: bool,
    seed: int,
    max_problems: int,
    loader_options: dict[str, object],
) -> dict[str, list[BenchmarkTask]]:
    """Load, filter, order, and truncate tasks for every requested benchmark."""

    selection = parse_task_ids(task_ids, canonicalize=canonical_name)
    unknown_scopes = set(selection.scoped) - set(benchmarks)
    if unknown_scopes:
        raise SelectionError(
            "--task-ids names benchmarks that are not being run: "
            + ", ".join(sorted(unknown_scopes))
        )
    tasks_by_benchmark: dict[str, list[BenchmarkTask]] = {}
    unscoped_found: set[str] = set()
    unscoped_missing: dict[str, set[str]] = {}
    for index, name in enumerate(benchmarks):
        tasks = _load_one(name, data_root, file_overrides.get(name), loader_options)
        scoped_ids = selection.scoped.get(name)
        if scoped_ids is not None:
            tasks, missing = select_tasks(tasks, scoped_ids)
            if missing:
                raise SelectionError(
                    f"Unknown {name} task IDs: {', '.join(missing)}. "
                    f"Use `bcg benchmark list {name}` to see valid IDs."
                )
        elif selection.unscoped:
            tasks, missing = select_tasks(tasks, selection.unscoped)
            unscoped_missing[name] = set(missing)
            unscoped_found |= set(selection.unscoped) - set(missing)
        elif shuffle:
            random.Random(seed + index).shuffle(tasks)
        if max_problems:
            tasks = tasks[:max_problems]
        tasks_by_benchmark[name] = tasks
    if selection.unscoped:
        never_found = [
            value
            for value in selection.unscoped
            if all(value in missing for missing in unscoped_missing.values())
        ]
        if never_found:
            raise SelectionError(f"Unknown task IDs: {', '.join(never_found)}.")
    for name, tasks in tasks_by_benchmark.items():
        if not tasks:
            raise BenchmarkDataError(f"No {name} tasks remain after selection filters.")
    return tasks_by_benchmark


def _print_dry_run(
    tasks_by_benchmark: dict[str, list[BenchmarkTask]],
    profiles: dict[str, object],
    *,
    allow_no_search: bool,
) -> None:
    import tempfile

    for name, tasks in tasks_by_benchmark.items():
        profile = profiles[name]
        table = Table(title=f"{name}: {len(tasks)} task(s) selected")
        table.add_column("#", justify="right")
        table.add_column("Task ID")
        table.add_column("Details")
        for position, task in enumerate(tasks, start=1):
            table.add_row(str(position), task.task_id, _task_details(task))
        console.print(table)
        tools = _tools_for(tasks[0], allow_no_search=allow_no_search, profile=profile)  # type: ignore[arg-type]
        budget = profile.as_dict()  # type: ignore[attr-defined]
        console.print(
            "Effective profile: "
            + ", ".join(
                f"{key}={budget[key]}"
                for key in (
                    "timeout",
                    "graph_max_turns",
                    "search_max_calls",
                    "fetch_max_calls",
                    "fetch_max_chars",
                    "web",
                )
                if budget[key] is not None
            )
            + f"\nTools: {', '.join(tools) if tools else '(none)'}",
            highlight=False,
        )
        task = tasks[0]
        with tempfile.TemporaryDirectory(prefix="bcg-dry-run-") as temporary:
            workspace = Path(temporary)
            if task.benchmark == drbench.NAME:
                drbench.stage_workspace(task, workspace)
                staged = sorted(
                    str(path.relative_to(workspace))
                    for path in workspace.rglob("*")
                    if path.is_file()
                )
                console.print(
                    f"Workspace for {task.task_id}: {len(staged)} files "
                    f"({', '.join(staged[:4])}{', …' if len(staged) > 4 else ''})",
                    highlight=False,
                )
            prompt = _task_prompt(task, workspace)
        console.rule(f"System prompt ({task.task_id})")
        console.print(_system_prompt(task, tools), highlight=False, markup=False)
        console.rule(f"User prompt ({task.task_id})")
        console.print(prompt, highlight=False, markup=False)
        console.rule()


def _search_binary_warnings(profiles: dict[str, object]) -> list[str]:
    """The Agent's grep/find tools need ripgrep/fd on PATH or a download."""

    import shutil

    tools = {
        tool
        for profile in profiles.values()
        for tool in getattr(profile, "file_tools", ())
    }
    missing = []
    if "grep" in tools and not shutil.which("rg"):
        missing.append(("ripgrep (rg)", "ripgrep"))
    if "find" in tools and not (shutil.which("fd") or shutil.which("fdfind")):
        missing.append(("fd", "fd-find"))
    if not missing:
        return []
    names = " and ".join(name for name, _ in missing)
    packages = " ".join(package for _, package in missing)
    return [
        f"{names} not found on PATH. The Agent's grep/find tools will try to "
        "download them into each run's .agent-config directory and fail without "
        f"network access. Install them once, e.g. `sudo apt install {packages}`."
    ]


def _task_details(task: BenchmarkTask) -> str:
    meta = task.metadata
    if task.benchmark == drb2.NAME:
        counts = meta.get("rubric_counts", {})
        return (
            f"{meta.get('language')} · {meta.get('theme')} · "
            f"rubrics {sum(counts.values())} · {str(meta.get('description'))[:70]}"
        )
    if task.benchmark == drbench.NAME:
        return (
            f"{meta.get('difficulty')} · {meta.get('industry')} · "
            f"{meta.get('n_files')} files · {meta.get('n_insight_qa')} insights "
            f"({meta.get('n_external_insight_qa')} external) · {task.question[:60]}"
        )
    return task.question.replace("\n", " ")[:90]


def _parse_data_files(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        benchmark, separator, path = value.partition("=")
        if not separator or not path.strip():
            raise BenchmarkDataError(
                f"Invalid --data-file {value!r}; expected BENCHMARK=PATH."
            )
        name = canonical_name(benchmark)
        parsed[name] = Path(path).expanduser()
    return parsed


def _print_summary(summary: dict[str, object], output_dir: Path) -> None:
    result_table = Table(title="BCG benchmark summary")
    result_table.add_column("Benchmark")
    result_table.add_column("Mode")
    result_table.add_column("Scored", justify="right")
    result_table.add_column("Accuracy", justify="right")
    result_table.add_column("Mean time", justify="right")
    result_table.add_column("Summary time", justify="right")
    token_table = Table(title="Model token usage")
    token_table.add_column("Benchmark")
    token_table.add_column("Mode")
    token_table.add_column("Model scope")
    token_table.add_column("Input (total)", justify="right")
    token_table.add_column("Reasoning", justify="right")
    token_table.add_column("Output (non-reasoning)", justify="right")
    token_table.add_column("Total cost", justify="right")
    benchmark_summaries = summary.get("benchmarks", {})
    if isinstance(benchmark_summaries, dict):
        for benchmark, mode_values in benchmark_summaries.items():
            if not isinstance(mode_values, dict):
                continue
            for mode, values in mode_values.items():
                if not isinstance(values, dict):
                    continue
                accuracy = values.get("accuracy")
                result_table.add_row(
                    str(benchmark),
                    str(mode),
                    str(values.get("report_scored", values.get("evaluated", 0))),
                    f"{float(accuracy):.2%}" if accuracy is not None else "n/a",
                    f"{float(values.get('wall_time_seconds_mean', 0)):.1f}s",
                    f"{float(values.get('summary_wall_time_seconds_mean', 0)):.1f}s",
                )
                model_tokens = values.get("model_token_usage")
                if not isinstance(model_tokens, dict):
                    continue
                model_costs = values.get("model_cost_usage")
                if not isinstance(model_costs, dict):
                    model_costs = {}
                for scope, label in (
                    ("agent_model", "Agent model"),
                    ("graph_model", "Graph model"),
                    ("summary_model", "Summary model"),
                    ("combined", "Combined"),
                ):
                    tokens = model_tokens.get(scope)
                    if not isinstance(tokens, dict):
                        continue
                    costs = model_costs.get(scope)
                    total_cost = (
                        float(costs.get("total", 0.0))
                        if isinstance(costs, dict)
                        else 0.0
                    )
                    token_table.add_row(
                        str(benchmark),
                        str(mode),
                        label,
                        f"{int(tokens.get('input_tokens', 0)):,}",
                        f"{int(tokens.get('reasoning_tokens', 0)):,}",
                        f"{int(tokens.get('output_tokens', 0)):,}",
                        f"${total_cost:.6f}" if total_cost else "—",
                    )
    overall_tokens = summary.get("model_token_usage")
    if isinstance(overall_tokens, dict):
        overall_costs = summary.get("model_cost_usage")
        if not isinstance(overall_costs, dict):
            overall_costs = {}
        token_table.add_section()
        for scope, label in (
            ("agent_model", "Agent model"),
            ("graph_model", "Graph model"),
            ("summary_model", "Summary model"),
            ("combined", "Combined"),
        ):
            tokens = overall_tokens.get(scope)
            if not isinstance(tokens, dict):
                continue
            costs = overall_costs.get(scope)
            total_cost = (
                float(costs.get("total", 0.0)) if isinstance(costs, dict) else 0.0
            )
            token_table.add_row(
                "ALL",
                "all",
                label,
                f"{int(tokens.get('input_tokens', 0)):,}",
                f"{int(tokens.get('reasoning_tokens', 0)):,}",
                f"{int(tokens.get('output_tokens', 0)):,}",
                f"${total_cost:.6f}" if total_cost else "—",
            )
    report_table = _report_table(summary)
    console.print(result_table)
    if report_table is not None:
        console.print(report_table)
    console.print(token_table)
    console.print(f"Artifacts: [bold]{output_dir}[/bold]")


@app.command("list")
def list_tasks(
    benchmarks: Annotated[
        list[str],
        typer.Argument(help=f"Benchmarks: {', '.join(BENCHMARKS)}."),
    ],
    data_root: Annotated[
        Path,
        typer.Option("--data-root", help="Root containing benchmark directories."),
    ] = Path("datasets"),
    data_file: Annotated[
        list[str] | None,
        typer.Option("--data-file", help="Override a dataset file as BENCHMARK=PATH."),
    ] = None,
    gaia_split: Annotated[str, typer.Option(help="GAIA split.")] = "validation",
    drb2_language: Annotated[str | None, typer.Option(help="zh or en.")] = None,
    drb2_theme: Annotated[str | None, typer.Option(help="One DRB2 theme.")] = None,
    drbench_subset: Annotated[
        str, typer.Option(help="val, minival, or sanity.")
    ] = "val",
    drbench_difficulty: Annotated[
        str | None, typer.Option(help="easy, medium, or hard.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON lines.")] = False,
) -> None:
    """List task IDs and metadata, to choose tasks for --task-ids."""

    import json

    try:
        overrides = _parse_data_files(data_file or [])
        for name in [canonical_name(value) for value in benchmarks]:
            tasks = _load_one(
                name,
                data_root,
                overrides.get(name),
                {
                    "gaia_split": gaia_split,
                    "drb2_language": drb2_language,
                    "drb2_theme": drb2_theme,
                    "drbench_subset": drbench_subset,
                    "drbench_difficulty": drbench_difficulty,
                },
            )
            if as_json:
                for task in tasks:
                    print(
                        json.dumps(
                            {
                                "benchmark": name,
                                "task_id": task.task_id,
                                "aliases": list(task.aliases),
                                "details": _task_details(task),
                            },
                            ensure_ascii=False,
                        )
                    )
                continue
            table = Table(title=f"{name}: {len(tasks)} task(s)")
            table.add_column("Task ID")
            table.add_column("Aliases")
            table.add_column("Details")
            for task in tasks:
                table.add_row(
                    task.task_id, ", ".join(task.aliases), _task_details(task)
                )
            console.print(table)
    except BenchmarkDataError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("prepare")
def prepare(
    benchmarks: Annotated[
        list[str],
        typer.Argument(help=f"Benchmarks to prepare: {', '.join(PREPARABLE)}."),
    ],
    data_root: Annotated[
        Path,
        typer.Option("--data-root", help="Root containing benchmark directories."),
    ] = Path("datasets"),
    revision: Annotated[
        str | None,
        typer.Option(help="Git branch, tag, or commit to download (default: main)."),
    ] = None,
    source: Annotated[
        Path | None,
        typer.Option("--from", help="Import from a local repository clone instead."),
    ] = None,
    download: Annotated[
        bool,
        typer.Option("--download/--no-download", help="Download when data is missing."),
    ] = True,
    force: Annotated[
        bool,
        typer.Option(help="Re-download or re-import even when data already exists."),
    ] = False,
    render: Annotated[
        bool,
        typer.Option(
            "--render/--no-render", help="DRBench: pre-render workspace text."
        ),
    ] = True,
) -> None:
    """Locate, download, verify, and pre-process benchmark data."""

    try:
        names = [canonical_name(value) for value in benchmarks]
    except BenchmarkDataError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if source is not None and len(names) != 1:
        raise typer.BadParameter("--from imports exactly one benchmark at a time.")
    table = Table(title="Prepared benchmarks")
    table.add_column("Benchmark")
    table.add_column("Action")
    table.add_column("Tasks", justify="right")
    table.add_column("Directory")
    table.add_column("Commit")
    for name in names:
        try:
            result = prepare_benchmark(
                name,
                data_root,
                revision=revision,
                source=source,
                download=download,
                force=force,
                render=render,
                log=lambda message: console.print(message, highlight=False),
            )
        except PrepareError as exc:
            console.print(f"[red]bcg benchmark prepare: {exc}[/red]", highlight=False)
            raise typer.Exit(1) from exc
        origin = result.manifest.get("source", {})
        table.add_row(
            name,
            result.action,
            str(result.tasks),
            str(result.directory),
            str(origin.get("commit") or "—")[:12],
        )
        for note in result.notes:
            console.print(f"[yellow]{name}: {note}[/yellow]", highlight=False)
    console.print(table)


def _report_table(summary: dict[str, object]) -> Table | None:
    benchmark_summaries = summary.get("benchmarks", {})
    if not isinstance(benchmark_summaries, dict):
        return None
    table = Table(title="Report metrics (deterministic; LLM judges not run)")
    # Short headers keep the table readable at 80 columns; see CHANGELOG.
    for column in (
        "Bench",
        "Mode",
        "Ready",
        "Chars",
        "URLs",
        "Fetch",
        "Read",
        "Cite",
        "Prec",
        "Distr",
        "Blocked",
    ):
        table.add_column(
            column, justify="left" if column in {"Benchmark", "Mode"} else "right"
        )
    rows = 0
    for benchmark, mode_values in benchmark_summaries.items():
        if not isinstance(mode_values, dict):
            continue
        for mode, values in mode_values.items():
            if not isinstance(values, dict) or "report_metrics" not in values:
                continue
            metrics = values.get("report_metrics") or {}

            def number(key: str, *, percent: bool = False) -> str:
                value = metrics.get(key)  # noqa: B023
                if value is None:
                    return "—"
                return f"{float(value):.0%}" if percent else f"{float(value):,.0f}"

            table.add_row(
                {drb2.NAME: "drb2"}.get(str(benchmark), str(benchmark)),
                str(mode),
                f"{values.get('reports_ready', 0)}/{values.get('total', 0)}",
                number("report_chars"),
                number("urls_cited"),
                f"{float(values.get('fetch_calls_mean', 0)):.1f}",
                number("file_read_recall", percent=True),
                number("citation_recall", percent=True),
                number("citation_precision", percent=True),
                number("distractor_citation_rate", percent=True),
                number("blocked_citation_hits"),
            )
            rows += 1
    return table if rows else None


def main(argv: list[str] | None = None) -> None:
    _bootstrap_env()
    standalone = argv is None
    args = list(sys.argv[1:] if argv is None else argv)
    app(args=args, prog_name="bcg benchmark", standalone_mode=standalone)


if __name__ == "__main__":
    main()


__all__ = [
    "app",
    "console",
    "run",
    "main",
]
