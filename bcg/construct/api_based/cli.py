"""API-based construction backend CLI."""

from __future__ import annotations

from bcg.construct._backend_cli import backend_main


def main(argv: list[str] | None = None) -> None:
    backend_main("api_based", argv)


__all__ = ["main"]
