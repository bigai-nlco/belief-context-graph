"""Parallel tool execution for the agent loop.

Dispatches the parsed tool calls through ``asyncio.gather`` so the loop
doesn't block on any single tool. Each call runs under its own
``asyncio.wait_for``; success / timeout / exception all come back as a
:class:`ToolResult` so a raw exception never leaks into history.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol, runtime_checkable

from agent_harness.core.loop_types import ToolResult
from agent_harness.core.tool import Tool

__all__ = [
    "TOOL_RESULT_MAX_CHARS",
    "execute_tools",
    "ToolResultPostProcessor",
    "DefaultToolResultPostProcessor",
]

TOOL_RESULT_MAX_CHARS = 150_000


@runtime_checkable
class ToolResultPostProcessor(Protocol):
    """Transforms a :class:`ToolResult` into the string entering the
    ``tool``-role :class:`Message` in history. Called once per result
    after the 150 K safety cap in :func:`execute_tools` so processors
    work on bounded input. Return value replaces ``tool_result.result``
    in the message content only — the ``ToolResult`` itself stays
    unchanged for observer / evidence consumers.
    """

    def process(self, tool_result: ToolResult) -> str:
        ...


class DefaultToolResultPostProcessor:
    """Apply the configured ``tool_result_max_chars`` cap, with a tail marker
    when truncation occurs. ``max_chars=None`` is pass-through."""

    def __init__(self, max_chars: int | None = None) -> None:
        self._max_chars = max_chars

    def process(self, tool_result: ToolResult) -> str:
        content = tool_result.result
        cap = self._max_chars
        if cap and isinstance(content, str) and len(content) > cap:
            return (
                content[:cap]
                + f"\n\n[... truncated {len(content) - cap} chars past {cap}-char cap]"
            )
        return content if isinstance(content, str) else str(content)


async def execute_tools(
    tool_calls: list[dict],
    tool_map: dict[str, Tool],
    timeout: int,
    turn: int,
    count_offset: int,
) -> list[ToolResult]:
    """Execute ``tool_calls`` in parallel, returning one ``ToolResult`` each.

    Unknown tools, timeouts, and exceptions all come back as
    ``is_error=True``. Output longer than ``TOOL_RESULT_MAX_CHARS`` (150 K
    — enough for a long Wikipedia article or full academic paper) is
    truncated with a tail marker.
    """

    async def _run_one(call: dict, idx: int) -> ToolResult:
        name = call.get("name", "")
        args = call.get("args", {})
        tool_call_id = call.get("id") or f"call_{turn}_{count_offset + idx}"
        tool = tool_map.get(name)
        start = time.monotonic()

        if tool is None:
            return ToolResult(
                name=name, args=args,
                result=f"Error: unknown tool '{name}'",
                duration_ms=0, tool_call_id=tool_call_id, is_error=True,
            )

        try:
            raw = await asyncio.wait_for(
                tool.ainvoke(args), timeout=timeout,
            )
            result_str = str(raw) if raw is not None else ""
            if len(result_str) > TOOL_RESULT_MAX_CHARS:
                result_str = (
                    result_str[:TOOL_RESULT_MAX_CHARS]
                    + f"\n... [truncated, {len(result_str)} chars total]"
                )
            elapsed = int((time.monotonic() - start) * 1000)
            return ToolResult(
                name=name, args=args, result=result_str,
                duration_ms=elapsed, tool_call_id=tool_call_id, is_error=False,
            )
        except asyncio.TimeoutError:
            elapsed = int((time.monotonic() - start) * 1000)
            return ToolResult(
                name=name, args=args,
                result=f"Error: tool '{name}' timed out after {timeout}s",
                duration_ms=elapsed, tool_call_id=tool_call_id, is_error=True,
            )
        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            return ToolResult(
                name=name, args=args,
                result=f"Error: {type(exc).__name__}: {exc}",
                duration_ms=elapsed, tool_call_id=tool_call_id, is_error=True,
            )

    tasks = [_run_one(call, i) for i, call in enumerate(tool_calls)]
    return list(await asyncio.gather(*tasks))
