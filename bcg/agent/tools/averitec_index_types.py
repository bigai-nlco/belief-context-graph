"""Stable pickle types for AVeriTeC search indices."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class _Chunk:
    chunk_id: str
    source_type: str
    source_query: str
    url: str
    text: str
    tokens: tuple[str, ...]
    term_counts: Counter[str]


@dataclass(frozen=True)
class _ClaimIndex:
    claim_id: str
    chunks: tuple[_Chunk, ...]
    idf: dict[str, float]
    avg_doc_len: float


__all__ = ["_Chunk", "_ClaimIndex"]
