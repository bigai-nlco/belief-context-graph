"""Typer command group for belief-graph construction workflows."""

from __future__ import annotations

import sys

import typer

_FORWARD_CONTEXT = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}

app = typer.Typer(
    name="bcg construct",
    help="Construct and inspect Belief Context Graphs.",
    invoke_without_command=True,
    no_args_is_help=False,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
)


@app.callback()
def _root(ctx: typer.Context) -> None:
    """Construct and inspect Belief Context Graphs."""

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command(
    "run",
    help="Build belief graphs from a trajectory or dataset.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _run(ctx: typer.Context) -> None:
    from bcg.run import main as run_main

    run_main(list(ctx.args))


@app.command(
    "server",
    help="Start the streaming belief-graph HTTP server.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _server(ctx: typer.Context) -> None:
    from bcg.online_server import main as server_main

    server_main(list(ctx.args))


@app.command(
    "replay",
    help="Replay a JSONL turn stream through a construction backend.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _replay(ctx: typer.Context) -> None:
    from bcg.online_driver import main as replay_main

    replay_main(list(ctx.args))


@app.command(
    "visualize",
    help="Render a belief-graph result as an HTML visualization.",
    context_settings=_FORWARD_CONTEXT,
    add_help_option=False,
)
def _visualize(ctx: typer.Context) -> None:
    from bcg.visualize_beliefs_graph import main as visualize_main

    visualize_main(list(ctx.args))


def main(argv: list[str] | None = None) -> None:
    standalone = argv is None
    args = list(sys.argv[1:] if argv is None else argv)
    app(args=args, prog_name="bcg construct", standalone_mode=standalone)


if __name__ == "__main__":
    main()
