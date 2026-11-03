"""Span trimming shared by both construct backends."""

from __future__ import annotations


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim whitespace around a span while preserving source offsets."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


__all__ = ["trim_span"]
