"""Deterministic extraction of agent tool calls.

The graph's natural-language node is still produced by the configured extractor,
but the exact query value is execution metadata and must not depend on an LLM
copying JSON correctly.  This module parses the canonical ``<tool_call>`` wire
format used by the bundled agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)


@dataclass(frozen=True)
class ToolCall:
    """One valid tool invocation parsed from the agent wire format."""

    name: str
    arguments: dict[str, Any]
    excerpt: str
    query: str | None = None
    tool_call_id: str | None = None


QueryToolCall = ToolCall


def extract_tool_calls(content: str) -> list[ToolCall]:
    """Return valid tool calls in source order, preserving exact arguments."""

    calls: list[ToolCall] = []
    for match in _TOOL_CALL_RE.finditer(content or ""):
        try:
            payload: Any = json.loads(match.group(1))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        raw_query = arguments.get("query")
        if not isinstance(raw_query, str):
            raw_query = arguments.get("q")
        query = raw_query if isinstance(raw_query, str) and raw_query.strip() else None
        name = payload.get("name")
        raw_id = payload.get("id", payload.get("tool_call_id"))
        calls.append(
            ToolCall(
                name=str(name).strip()
                if isinstance(name, str) and name.strip()
                else "tool",
                arguments=dict(arguments),
                excerpt=match.group(0),
                query=query,
                tool_call_id=(
                    raw_id.strip()
                    if isinstance(raw_id, str) and raw_id.strip()
                    else None
                ),
            )
        )
    return calls


def strip_valid_tool_calls(content: str) -> str:
    """Remove valid canonical tool-call blocks, preserving reasoning/text."""

    def _replace(match: re.Match[str]) -> str:
        try:
            payload: Any = json.loads(match.group(1))
        except (TypeError, ValueError, json.JSONDecodeError):
            return match.group(0)
        return "" if isinstance(payload, dict) else match.group(0)

    return _TOOL_CALL_RE.sub(_replace, content or "").strip()


def extract_pure_tool_calls(content: str) -> list[ToolCall] | None:
    """Parse a turn made exclusively of valid ``<tool_call>`` blocks.

    ``None`` means ordinary/mixed/malformed content and must retain the model
    extraction path. A list means rule extraction is safe; whitespace between
    multiple calls is allowed.
    """

    matches = list(_TOOL_CALL_RE.finditer(content or ""))
    if not matches:
        return None
    cursor = 0
    for match in matches:
        if (content or "")[cursor : match.start()].strip():
            return None
        cursor = match.end()
    if (content or "")[cursor:].strip():
        return None
    calls = extract_tool_calls(content)
    return calls if len(calls) == len(matches) else None


def rule_tool_call_belief(call: ToolCall) -> str:
    """Build a complete, deterministic natural-language action belief."""

    if call.query is not None:
        query = json.dumps(call.query, ensure_ascii=False)
        return f"The assistant is using {call.name} to search for {query}."
    arguments = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
    return (
        f"The assistant is using {call.name} to execute the action "
        f"with arguments {arguments}."
    )


def extract_query_tool_calls(content: str) -> list[QueryToolCall]:
    """Return valid query-bearing calls in source order.

    ``query`` is preferred, while ``q`` is accepted for OpenAI-compatible
    search tools. Malformed calls and blank/non-string queries are ignored.
    The returned query is the exact JSON string value, with surrounding
    whitespace preserved because it is part of the executed argument.
    """

    return [call for call in extract_tool_calls(content) if call.query is not None]
