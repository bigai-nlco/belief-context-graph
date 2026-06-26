"""
link.py
=======
Relation extraction is now handled inside the per-turn update prompt and
`extract.update_graph`. The old final backward-linking pass has been removed.

This module is intentionally kept as a tiny compatibility shim for external
imports. It performs no work and should not be used by the current pipeline.
"""

from __future__ import annotations

from typing import Any


def link_backward_all(
    client,
    model: str,
    beliefs: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    max_chars: int = 28000,
) -> dict[str, Any]:
    return {
        "relations": [],
        "raw_output": None,
        "skipped": True,
        "skip_reason": "final backward-linking pass removed; relations are extracted per turn",
    }
