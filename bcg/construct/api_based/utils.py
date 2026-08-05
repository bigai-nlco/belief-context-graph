"""Shared utilities for belief graph construction internals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bcg.utils import new_run_id, save_json




def count_by(
    items: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
) -> dict[str, int]:
    """Count dictionaries by a derived key."""

    counts: dict[str, int] = {}
    for item in items:
        key = str(key_fn(item) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


from .._shared.spans import trim_span

__all__ = ["count_by", "new_run_id", "save_json", "trim_span"]
