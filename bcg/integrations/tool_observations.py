"""Adapt native tool text to the existing bounded Graph observation extractor."""

from __future__ import annotations

import re


def observation_body(body: str, tool_name: str, *, is_error: bool = False) -> str:
    """Give unranked output one snippet without inventing a URL or a fact.

    This only changes Graph serialization, never the original model input.
    The Graph engine still owns snippet limits, semantic extraction, and its
    bounded fallback. Ranked search results and tool errors keep their format.
    """
    if body.strip() and not is_error and not re.search(r"(?m)^\[\d+\] .+", body):
        return f"[1] Observation from {tool_name}\nSnippet: " + " ".join(body.split())
    return body
