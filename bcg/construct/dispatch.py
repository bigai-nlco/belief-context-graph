"""Shared command-line backend dispatch helpers."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from bcg.config.loader import defaults_dict

DEFAULT_BACKEND = str(defaults_dict()["backend"])
BACKENDS = ("unified", "hybrid")


def split_backend_args(
    argv: Sequence[str],
    *,
    backends: Collection[str] = BACKENDS,
    default: str = DEFAULT_BACKEND,
) -> tuple[str, list[str]]:
    """Return ``(backend, remaining_args)`` with legacy flag-only fallback.

    Before the two construction implementations were combined, command lines
    started directly with flags. Preserve those invocations by selecting the
    unified implementation when the first token is an option. A positional
    token still has to be an explicit valid backend, so misspellings fail
    instead of being silently treated as backend arguments.
    """

    args = list(argv)
    if not args or args[0].startswith("-"):
        return default, args
    if args[0] not in backends:
        choices = ", ".join(backends)
        raise ValueError(f"unknown backend {args[0]!r}; choose one of: {choices}")
    return args[0], args[1:]


__all__ = ["BACKENDS", "DEFAULT_BACKEND", "split_backend_args"]
