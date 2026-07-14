"""Typer command group for the Agent application."""

from __future__ import annotations

import sys
from typing import Annotated

import typer

from bcg.agent import __version__

_FORWARD_CONTEXT = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}

app = typer.Typer(
    name="bcg agent",
    help="BCG agent workflow tools.",
    invoke_without_command=True,
    no_args_is_help=False,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
)


def _print_task_list() -> None:
    from bcg.agent.benchmark_loader import AVAILABLE_BENCHMARKS

    for name in AVAILABLE_BENCHMARKS:
        print(name)


def _run(argv: list[str]) -> None:
    tasks: list[str] = []
    rest = list(argv)
    while rest and not rest[0].startswith("-"):
        tasks.append(rest.pop(0))

    if tasks and "--tasks" not in rest:
        rest = ["--tasks", *tasks, *rest]

    from bcg.agent.rollout import main as rollout_main

    rollout_main(rest, prog="bcg agent run")


def _ui(argv: list[str]) -> None:
    from bcg.agent.ui import main as ui_main

    ui_main(argv, prog="bcg agent ui")


def _check_thinking(argv: list[str]) -> None:
    from bcg.agent.check_qwen_thinking import main as check_main

    check_main(argv, prog="bcg agent check-thinking")


def _tonggraph_sync(argv: list[str]) -> None:
    from bcg.agent.tonggraph_sync import main as sync_main

    sync_main(argv)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"bcg agent {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    ctx: typer.Context,
    version_requested: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the Agent version and exit.",
        ),
    ] = False,
) -> None:
    """BCG agent workflow tools."""

    del version_requested
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command(
    "run",
    help="Run an agent workflow benchmark.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _run_command(ctx: typer.Context) -> None:
    _run(list(ctx.args))


@app.command(
    "ui",
    help="Serve the trajectory monitor web UI.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _ui_command(ctx: typer.Context) -> None:
    _ui(list(ctx.args))


@app.command(
    "check-thinking",
    help="Smoke test thinking-token behavior with a local backend.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _check_thinking_command(ctx: typer.Context) -> None:
    _check_thinking(list(ctx.args))


@app.command(
    "tonggraph-sync",
    help="Ingest a saved belief graph into TongGraph Server.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _tonggraph_sync_command(ctx: typer.Context) -> None:
    _tonggraph_sync(list(ctx.args))


@app.command("tasks", help="List supported benchmark task names.")
def _tasks_command() -> None:
    _print_task_list()


def main(argv: list[str] | None = None) -> None:
    standalone = argv is None
    args = list(sys.argv[1:] if argv is None else argv)
    app(args=args, prog_name="bcg agent", standalone_mode=standalone)


if __name__ == "__main__":
    main()
