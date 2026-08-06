"""CLI adapter used by ``python -m bcg.construct.<backend>``."""

from __future__ import annotations

import sys

from bcg.apps.cli_help import RichArgumentParser


def backend_main(backend: str, argv: list[str] | None = None) -> None:
    """Forward one backend-scoped command to the unified entry points."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        parser = RichArgumentParser(
            prog=f"python -m bcg.construct.{backend}",
            description=f"Belief-graph construction commands for {backend!r}.",
        )
        parser.add_argument("command", choices=["run", "server", "replay"])
        parser.print_help()
        raise SystemExit(0)

    command, rest = args[0], args[1:]
    handlers = {
        "run": ("bcg.apps.run", "main"),
        "server": ("bcg.apps.online_server", "main"),
        "replay": ("bcg.apps.online_driver", "main"),
    }
    if command not in handlers:
        choices = ", ".join(handlers)
        print(
            f"error: unknown command {command!r}; choose one of: {choices}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    module_name, attribute = handlers[command]
    module = __import__(module_name, fromlist=[attribute])
    getattr(module, attribute)([backend, *rest])


__all__ = ["backend_main"]
