"""Shared utilities for belief graph construction internals."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bcg.utils import get_random_uuid, utc_now


def save_json(payload: Any, path: str | Path) -> None:
    """Write a JSON file with stable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def new_run_id() -> str:
    """Create a compact timestamped run id."""

    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{get_random_uuid()[:8]}"


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim whitespace around a span while preserving source offsets."""

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


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
