"""SSEObserver — passive observer that forwards loop events to EventStore.

Emits AGENT_ACTION events so the SSE stream and frontend receive real-time
updates for every LLM turn and tool call inside the agent loop engine.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from agent_harness.core.events import EventType
from agent_harness.core.runtime.loop.model_profile import HistoryPolicy
from agent_harness.core.loop_types import TurnContext, ToolResult

logger = logging.getLogger(__name__)


class SSEObserver:
    """Passive observer that persists react_think / react_tool_call events.

    Designed to be registered with the agent loop engine and fire-and-forgot
    (critical=False).  All errors are swallowed by the base notify_observers
    helper — this observer must never crash the loop.
    """

    critical = False

    def __init__(
        self,
        event_store: Any,
        task_id: str,
        history_policy: HistoryPolicy | None = None,
        *,
        run_id: str = "",
        run_type: str = "",
    ) -> None:
        self._es = event_store
        self._task_id = task_id
        self._policy = history_policy or HistoryPolicy()
        self._run_id = run_id
        self._run_type = run_type

    def _annotate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Annotate payload with ``run_id`` / ``run_type`` when set."""
        if self._run_id:
            payload["run_id"] = self._run_id
        if self._run_type:
            payload["run_type"] = self._run_type
        return payload

    # ── LoopObserver hooks ─────────────────────────────────────────────────

    async def on_loop_start(self, config: Any) -> None:  # noqa: ARG002
        pass

    async def on_turn_end(self, ctx: TurnContext) -> None:  # noqa: ARG002
        pass

    async def on_loop_end(self, result: Any) -> None:  # noqa: ARG002
        pass

    async def on_llm_response(self, ctx: TurnContext) -> None:
        """Emit a ``react_think`` AGENT_ACTION event for the completed turn."""
        payload: dict[str, Any] = {
            "trace_type": "react_think",
            "agent": ctx.role_id,
            "turn": ctx.turn,
            "action": ctx.ai_text,
            "detail": ctx.ai_text[:200] if ctx.ai_text else "",
        }
        if self._policy.thinking_in_sse and ctx.thinking:
            payload["thinking"] = ctx.thinking
        # Reasoning recovered from leaked private tags (e.g.
        # ``<think_never_used_…>``) — gated on the same flag as the native
        # thinking stream so UIs that hide thinking also hide this.
        if self._policy.thinking_in_sse and ctx.leaked_reasoning:
            payload["leaked_reasoning"] = ctx.leaked_reasoning

        await self._es.append(
            self._task_id, EventType.AGENT_ACTION, self._annotate(payload),
        )

    async def on_tool_result(self, ctx: TurnContext, result: ToolResult) -> None:
        """Emit a ``react_tool_call`` AGENT_ACTION event for the tool execution."""
        try:
            tool_args_str = json.dumps(result.args, ensure_ascii=False)[:2000]
        except (TypeError, ValueError):
            tool_args_str = str(result.args)[:2000]
        payload: dict[str, Any] = {
            "trace_type": "react_tool_call",
            "agent": ctx.role_id,
            "turn": ctx.turn,
            "tool_name": result.name,
            "tool_args": tool_args_str,
            "detail": result.result[:200] if result.result else "",
            "duration_ms": result.duration_ms,
            "is_error": result.is_error,
        }
        await self._es.append(
            self._task_id, EventType.AGENT_ACTION, self._annotate(payload),
        )
