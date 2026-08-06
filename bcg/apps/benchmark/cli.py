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

from bcg.apps.benchmark.loaders import (
    BENCHMARKS,
    BenchmarkDataError,
    canonical_name,
    load_benchmark,
)
from bcg.apps.benchmark.runner import RunConfig, run_benchmarks
from bcg.apps.benchmark.scoring import JudgeConfig, LLMJudge


def _bootstrap_env() -> None:
    """Load the project root .env explicitly (import no longer does this)."""
    from bcg.core.env import load_project_env

    load_project_env()


app = typer.Typer(
    name="bcg benchmark",
    help="Evaluate the reference Agent in Default and BCG context modes.",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
)
console = Console()


@app.callback()
def _root() -> None:
    """Evaluate the reference Agent in Default and BCG context modes."""


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
        typer.Option(help="Comma-separated context modes: default,bcg."),
    ] = "default,bcg",
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Run artifact directory."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(help="Agent model (default: BCG_AGENT_MODEL/OPENAI_MODEL/MODEL)."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(help="Agent OpenAI-compatible API base URL."),
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
        typer.Option(help="Comma-separated task IDs to include."),
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
        float,
        typer.Option(
            min=0,
            help="Per-task Agent timeout in seconds; 0 disables the time limit.",
        ),
    ] = 900.0,
    graph_url: Annotated[
        str,
        typer.Option(help="Graph Construction server URL for BCG mode."),
    ] = "http://127.0.0.1:8848",
    graph_timeout_ms: Annotated[
        int,
        typer.Option(min=1, help="Graph request timeout in milliseconds."),
    ] = 300_000,
    graph_max_turns: Annotated[
        int,
        typer.Option(
            min=1,
            help=(
                "Maximum Graph messages per BCG task, including the initial "
                "system and user messages."
            ),
        ),
    ] = 300,
    recent_turns: Annotated[
        int,
        typer.Option(min=-1, help="Completed turns retained verbatim in BCG mode."),
    ] = 2,
    allow_graph_fallback: Annotated[
        bool,
        typer.Option(help="Score BCG tasks even when Graph falls back to raw context."),
    ] = False,
    allow_no_search: Annotated[
        bool,
        typer.Option(help="Allow BrowseComp/HotpotQA without SERPER_API_KEY."),
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
    overwrite: Annotated[
        bool,
        typer.Option(help="Rerun task artifacts that already exist."),
    ] = False,
) -> None:
    """Run one or more benchmarks and write trajectories plus summary metrics."""

    if not benchmarks:
        raise typer.BadParameter("At least one benchmark is required.")
    try:
        canonical = [canonical_name(name) for name in benchmarks]
        if len(set(canonical)) != len(canonical):
            raise BenchmarkDataError("Each benchmark may be specified only once.")
        file_overrides = _parse_data_files(data_file or [])
        selected_ids = {
            value.strip() for value in (task_ids or "").split(",") if value.strip()
        }
        tasks_by_benchmark = {}
        for index, name in enumerate(canonical):
            tasks = load_benchmark(
                name,
                data_root,
                data_file=file_overrides.get(name),
                split=gaia_split if name == "gaia" else None,
                gaia_level=gaia_level,
                gaia_text_only=gaia_text_only,
            )
            if selected_ids:
                tasks = [task for task in tasks if task.task_id in selected_ids]
            if shuffle:
                random.Random(seed + index).shuffle(tasks)
            if max_problems:
                tasks = tasks[:max_problems]
            if not tasks:
                raise BenchmarkDataError(
                    f"No {name} tasks remain after selection filters."
                )
            tasks_by_benchmark[name] = tasks
    except BenchmarkDataError as exc:
        raise typer.BadParameter(str(exc)) from exc

    resolved_model = (
        model
        or os.environ.get("BCG_AGENT_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("MODEL")
        or ""
    ).strip()
    resolved_base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "").strip()
    api_key = os.environ.get(api_key_env, "")
    if not resolved_model:
        raise typer.BadParameter(
            "--model is required (or set BCG_AGENT_MODEL/OPENAI_MODEL/MODEL)."
        )
    if not resolved_base_url:
        raise typer.BadParameter("--base-url is required (or set OPENAI_BASE_URL).")

    resolved_modes = tuple(value.strip() for value in modes.split(",") if value.strip())
    invalid_modes = set(resolved_modes) - {"default", "bcg"}
    if not resolved_modes or invalid_modes:
        raise typer.BadParameter("--modes must contain only `default` and/or `bcg`.")
    destination = output_dir or Path("results") / "benchmarks" / time.strftime(
        "%Y%m%d-%H%M%S"
    )
    uses_judge = "browsecomp" in canonical
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
        modes=resolved_modes,
        workers=workers,
        timeout=timeout,
        graph_url=graph_url,
        graph_timeout_ms=graph_timeout_ms,
        graph_max_turns=graph_max_turns,
        recent_turns=recent_turns,
        allow_graph_fallback=allow_graph_fallback,
        allow_no_search=allow_no_search,
        overwrite=overwrite,
        agent_command=tuple(shlex.split(agent_command)) if agent_command else None,
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
    table = Table(title="BCG benchmark summary")
    table.add_column("Benchmark")
    table.add_column("Mode")
    table.add_column("Scored", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Input tokens", justify="right")
    table.add_column("Output tokens", justify="right")
    table.add_column("Mean time", justify="right")
    benchmark_summaries = summary.get("benchmarks", {})
    if isinstance(benchmark_summaries, dict):
        for benchmark, mode_values in benchmark_summaries.items():
            if not isinstance(mode_values, dict):
                continue
            for mode, values in mode_values.items():
                if not isinstance(values, dict):
                    continue
                accuracy = values.get("accuracy")
                tokens = values.get("tokens", {})
                table.add_row(
                    str(benchmark),
                    str(mode),
                    str(values.get("evaluated", 0)),
                    f"{float(accuracy):.2%}" if accuracy is not None else "n/a",
                    str(tokens.get("input", 0) if isinstance(tokens, dict) else 0),
                    str(tokens.get("output", 0) if isinstance(tokens, dict) else 0),
                    f"{float(values.get('wall_time_seconds_mean', 0)):.1f}s",
                )
    console.print(table)
    console.print(f"Artifacts: [bold]{output_dir}[/bold]")


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
