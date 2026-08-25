"""Deterministic parsing and compact rendering of agent tool results."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import urlparse

_HEADER_RE = re.compile(
    r"^\s*\[Tool result:\s*([^\]]+)\]\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_RESULT_START_RE = re.compile(r"(?m)^\[(\d+)\]\s+")
_TOOL_RESULT_RE = re.compile(
    r"<tool_result>\s*(.*?)\s*</tool_result>", re.DOTALL | re.IGNORECASE
)


@dataclass(frozen=True)
class SearchResult:
    rank: int
    title: str
    url: str | None
    snippet: str | None
    source_type: str | None


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    body: str
    results: tuple[SearchResult, ...]
    no_results: bool
    tool_call_id: str | None = None
    is_error: bool = False
    source_block: str | None = None


def _field(block: str, name: str) -> str | None:
    match = re.search(
        rf"(?ms)^{re.escape(name)}:\s*(.*?)(?=^[A-Za-z][A-Za-z ]*:\s|\Z)",
        block,
    )
    if match is None:
        return None
    value = " ".join(match.group(1).split()).strip()
    return value or None


def _parse_body(
    tool_name: str,
    body: str,
    *,
    tool_call_id: str | None = None,
    is_error: bool = False,
    source_block: str | None = None,
) -> ToolResult:
    starts = list(_RESULT_START_RE.finditer(body))
    results: list[SearchResult] = []
    for position, start in enumerate(starts):
        end = starts[position + 1].start() if position + 1 < len(starts) else len(body)
        block = body[start.end() : end].strip()
        url_match = re.search(r"(?m)^URL:\s*(\S.*)$", block)
        title_end = url_match.start() if url_match is not None else len(block)
        title = " ".join(block[:title_end].split()).strip()
        try:
            rank = int(start.group(1))
        except ValueError:
            rank = position + 1
        results.append(
            SearchResult(
                rank=rank,
                title=title or f"Result {rank}",
                url=(url_match.group(1).strip() if url_match is not None else None),
                snippet=_field(block, "Snippet"),
                source_type=_field(block, "Source type"),
            )
        )
    return ToolResult(
        tool_name=tool_name,
        body=body,
        results=tuple(results),
        no_results="no web results were returned" in body.lower(),
        tool_call_id=tool_call_id,
        is_error=is_error,
        source_block=source_block,
    )


def extract_tool_results(content: str) -> list[ToolResult]:
    """Parse every ID-bearing ``<tool_result>`` block in one tool turn."""

    parsed: list[ToolResult] = []
    for match in _TOOL_RESULT_RE.finditer(content or ""):
        try:
            payload = json.loads(match.group(1))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        raw_name = payload.get("name", payload.get("tool_name"))
        raw_id = payload.get("tool_call_id", payload.get("id"))
        raw_content = payload.get("content")
        parsed.append(
            _parse_body(
                raw_name.strip()
                if isinstance(raw_name, str) and raw_name.strip()
                else "tool",
                raw_content if isinstance(raw_content, str) else str(raw_content or ""),
                tool_call_id=(
                    raw_id.strip()
                    if isinstance(raw_id, str) and raw_id.strip()
                    else None
                ),
                is_error=bool(payload.get("is_error", False)),
                source_block=match.group(0),
            )
        )
    return parsed


def parse_tool_result(content: str) -> ToolResult | None:
    """Parse one canonical XML or legacy bracketed tool result."""

    grouped = extract_tool_results(content)
    if len(grouped) == 1:
        return grouped[0]
    if grouped:
        return None
    match = _HEADER_RE.match(content or "")
    if match is None:
        return None
    return _parse_body(
        match.group(1).strip() or "tool",
        match.group(2).strip(),
        source_block=content,
    )


def compact_tool_result(
    parsed: ToolResult,
    *,
    max_results: int,
    max_snippet_chars: int,
) -> tuple[str, list[dict[str, object]], list[str]]:
    """Render one bounded graph belief while retaining structured provenance."""

    max_results = max(1, int(max_results))
    max_snippet_chars = max(40, int(max_snippet_chars))
    selected = parsed.results[:max_results]
    items: list[dict[str, object]] = []
    lines: list[str] = []
    entities = [parsed.tool_name]
    for result in selected:
        snippet = (result.snippet or "").strip()
        if len(snippet) > max_snippet_chars:
            snippet = snippet[: max_snippet_chars - 1].rstrip() + "…"
        item: dict[str, object] = {
            "rank": result.rank,
            "title": result.title,
        }
        if result.url:
            item["url"] = result.url
            domain = urlparse(result.url).netloc
            if domain and domain not in entities:
                entities.append(domain)
        if snippet:
            item["snippet"] = snippet
        if result.source_type:
            item["source_type"] = result.source_type
        items.append(item)

        detail = result.title
        if snippet:
            detail += f" — {snippet}"
        if result.url:
            detail += f" ({result.url})"
        lines.append(f"{result.rank}. {detail}")

    if lines:
        omitted = max(0, len(parsed.results) - len(lines))
        suffix = (
            f" {omitted} additional result(s) are retained only in evidence."
            if omitted
            else ""
        )
        belief = (
            f"The {parsed.tool_name} tool returned {len(parsed.results)} result(s). "
            f"The bounded graph summary contains:\n" + "\n".join(lines) + suffix
        )
    elif parsed.no_results:
        belief = f"The {parsed.tool_name} tool returned no results."
    else:
        compact = " ".join(parsed.body.split())
        limit = max_results * max_snippet_chars
        if len(compact) > limit:
            compact = compact[: limit - 1].rstrip() + "…"
        belief = f"The {parsed.tool_name} tool returned: {compact or '(empty result)'}"

    return belief, items, entities


__all__ = [
    "SearchResult",
    "ToolResult",
    "compact_tool_result",
    "extract_tool_results",
    "parse_tool_result",
]
