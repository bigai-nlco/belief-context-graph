"""Unified Typer command-line entry point for the BCG package."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated

import typer


def _bootstrap_env() -> None:
    """Load the project root .env explicitly (import no longer does this)."""
    from bcg.core.env import load_project_env

    load_project_env()




def _package_version() -> str:
    try:
        return version("bcg")
    except PackageNotFoundError:
        return "0.1.0"


_FORWARD_CONTEXT = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}

app = typer.Typer(
    name="bcg",
    help="Belief Context Graph construction and agent tools.",
    invoke_without_command=True,
    no_args_is_help=False,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"bcg {_package_version()}")
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
            help="Show the installed BCG version and exit.",
        ),
    ] = False,
) -> None:
    """Belief Context Graph construction and agent tools."""

    del version_requested
    if ctx.invoked_subcommand is None:
        from bcg.apps.agent_runtime import main as agent_runtime_main

        raise typer.Exit(agent_runtime_main([]))


@app.command(
    "agent",
    help="Open the interactive BCG terminal agent.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _agent(ctx: typer.Context) -> None:
    arguments = list(ctx.args)
    from bcg.apps.agent_runtime import main as agent_runtime_main

    raise typer.Exit(agent_runtime_main(arguments))


@app.command(
    "construct",
    help="Construct, serve, replay, or visualize belief graphs.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _construct(ctx: typer.Context) -> None:
    from bcg.construct.cli import main as construct_main

    construct_main(list(ctx.args))


@app.command(
    "benchmark",
    help="Evaluate the Agent in Default and BCG context modes.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _benchmark(ctx: typer.Context) -> None:
    from bcg.apps.benchmark.cli import main as benchmark_main

    benchmark_main(list(ctx.args))


@app.command(
    "setup",
    help="Configure global Agent, model, context, and Graph settings.",
)
def _setup() -> None:
    from bcg.apps.setup import SetupError, run_setup

    try:
        run_setup()
    except SetupError as exc:
        typer.echo(f"bcg: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command(
    "config",
    help="Inspect the unified configuration or migrate legacy JSON files.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _config(ctx: typer.Context) -> None:
    from bcg.config.cli import app as config_app

    arguments = list(ctx.args)
    if not arguments:
        typer.echo(config_app.get_help())
        raise typer.Exit(0)
    config_app(args=arguments, prog_name="bcg config", standalone_mode=False)


def main(argv: list[str] | None = None) -> None:
    _bootstrap_env()
    """Launch the Agent TUI or dispatch an explicit command family."""

    standalone = argv is None
    args = list(sys.argv[1:] if argv is None else argv)
    app(args=args, prog_name="bcg", standalone_mode=standalone)


if __name__ == "__main__":
    main()


__all__ = [
    "app",
    "main",
]
