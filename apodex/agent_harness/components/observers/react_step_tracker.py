"""ReactStepTracker — observer that records tool calls as react_steps.

Populates ``metadata["react_steps"]`` with one dict per tool call::

    {turn, thinking, tool_name, tool_args, tool_result_preview,
     duration_ms, is_error}

Both ``tool_args`` (JSON-stringified) and ``tool_result_preview`` are
stored as untruncated strings — slicing would break downstream JSON
parsing and string-ops. Non-string tool results are JSON-encoded.

``critical=True`` because callers read ``react_steps`` from the result.
"""
from __future__ import annotations

import json
from typing import Any

from agent_harness.core.loop_types import BaseObserver, ToolResult, TurnContext


def _safe_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)


class ReactStepTracker(BaseObserver):
    """Records each tool result as a react_step dict in metadata."""

    critical: bool = True

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> None:
        steps: list[dict[str, Any]] = ctx.metadata.setdefault(
            "react_steps", [],
        )
        step: dict[str, Any] = {
            "turn": ctx.turn,
            "thinking": ctx.thinking or "",
            "tool_name": result.name,
            "tool_args": json.dumps(
                result.args, ensure_ascii=False, default=str,
            ),
            "tool_result_preview": _safe_str(result.result),
            "duration_ms": result.duration_ms,
            "is_error": result.is_error,
        }
        # Salvaged thinking from leaked tags — recorded alongside the
        # native thinking field so report-rendering / debugging tools can
        # inspect both without conflating provenance.
        if ctx.leaked_reasoning:
            step["leaked_reasoning"] = ctx.leaked_reasoning
        steps.append(step)
