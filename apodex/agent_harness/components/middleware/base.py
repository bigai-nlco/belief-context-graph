"""Execution Middleware — composable hooks for phase and tool execution.

Defines ``ExecutionMiddleware`` (before/after phase, before/after tool,
on_error), the ``MiddlewareChain`` runner, and the ``PhaseContext`` /
``ToolCallContext`` dataclasses passed through each hook.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.core.protocols import (
    ExecutionMiddleware,
    PhaseContext,
    ToolCallContext,
)

logger = logging.getLogger(__name__)


# ── Middleware chain ───────────────────────────────────────────────────────


class MiddlewareChain:
    """Ordered collection of middlewares. Runs hooks sequentially."""

    def __init__(self) -> None:
        self._middlewares: list[ExecutionMiddleware] = []

    def add(self, middleware: ExecutionMiddleware) -> None:
        self._middlewares.append(middleware)

    def remove(self, middleware_type: type) -> None:
        self._middlewares = [m for m in self._middlewares if not isinstance(m, middleware_type)]

    @property
    def middlewares(self) -> list[ExecutionMiddleware]:
        return list(self._middlewares)

    async def run_before_phase(self, ctx: PhaseContext) -> PhaseContext:
        for mw in self._middlewares:
            ctx = await mw.before_phase(ctx)
        return ctx

    async def run_after_phase(self, ctx: PhaseContext, result: dict[str, Any]) -> dict[str, Any]:
        # Onion model: after hooks run in reverse order
        for mw in reversed(self._middlewares):
            result = await mw.after_phase(ctx, result)
        return result

    async def run_before_tool_call(self, ctx: ToolCallContext) -> ToolCallContext:
        for mw in self._middlewares:
            ctx = await mw.before_tool_call(ctx)
        return ctx

    async def run_after_tool_call(self, ctx: ToolCallContext, result: str) -> str:
        # Onion model: after hooks run in reverse order
        for mw in reversed(self._middlewares):
            result = await mw.after_tool_call(ctx, result)
        return result

    async def run_on_error(self, ctx: PhaseContext, error: Exception) -> Exception | None:
        """Run error handlers. If any middleware returns None, error is suppressed."""
        for mw in self._middlewares:
            result = await mw.on_error(ctx, error)
            if result is None:
                return None
            error = result
        return error
