"""Unified Typer command-line entry point for the BCG package."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated

import typer


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
        typer.echo(ctx.get_help())


@app.command(
    "agent",
    help="Run agent benchmarks, UI, and graph synchronization tools.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _agent(ctx: typer.Context) -> None:
    from bcg.agent.cli import main as agent_main

    agent_main(list(ctx.args))


@app.command(
    "construct",
    help="Construct, serve, replay, or visualize belief graphs.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _construct(ctx: typer.Context) -> None:
    from bcg.construct.cli import main as construct_main

    construct_main(list(ctx.args))


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the ``bcg.agent`` or ``bcg.construct`` command family."""

    standalone = argv is None
    args = list(sys.argv[1:] if argv is None else argv)
    app(args=args, prog_name="bcg", standalone_mode=standalone)


if __name__ == "__main__":
    main()
