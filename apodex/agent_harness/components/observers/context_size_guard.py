"""ContextSizeGuard — pre-empt LLM context-window overflow.

Long research loops (history growth + sub-agent reports + ``<think>``
blocks kept by ``thinking_in_history=True``) can accumulate input tokens
fast enough to approach the model's context window. When a turn's input
crosses the wall, the LLM call returns 4xx or simply hangs and the loop
exits with ``llm_error`` before the model writes a final answer.

This observer watches per-turn ``input_tokens`` usage. When it crosses
``max_input_tokens`` it requests an early stop with
``stop_reason="budget_exhausted"`` — which routes through
``_force_final_answer`` (tool-free, smaller payload) for a clean wrap-up.

Opt-in. No profile activates it by default; pick a value when you have
evidence of context-driven failures specific to your model/proxy.
"""
from __future__ import annotations

import logging
from typing import Any

from agent_harness.core.loop_types import (
    BaseObserver,
    Intervention,
    LoopConfig,
    TurnContext,
)

logger = logging.getLogger(__name__)


class ContextSizeGuard(BaseObserver):
    """Stop the loop when a turn's input tokens cross ``max_input_tokens``.

    Critical observer — its ``stop_reason`` must reach the loop driver.
    """

    critical: bool = True

    def __init__(self, max_input_tokens: int) -> None:
        if max_input_tokens <= 0:
            raise ValueError(
                f"max_input_tokens must be positive, got {max_input_tokens}",
            )
        self._limit = max_input_tokens
        self._tripped = False

    async def on_loop_start(self, config: LoopConfig) -> None:  # noqa: ARG002
        # Defensive: reset state if the same instance is ever reused.
        self._tripped = False

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        if self._tripped or ctx.usage is None:
            return None
        used = int(ctx.usage.get("input_tokens", 0))
        if used <= self._limit:
            return None
        self._tripped = True
        logger.warning(
            "ContextSizeGuard: turn=%d input_tokens=%d > limit=%d — "
            "stopping early to force a clean final answer",
            ctx.turn, used, self._limit,
        )
        return Intervention(
            stop_reason="budget_exhausted",
            inject_messages=[
                f"Context approaching the model's limit "
                f"({used:,} > {self._limit:,} tokens). Stopping now to "
                f"deliver the answer based on information gathered so far."
            ],
        )

    async def on_loop_end(self, result: Any) -> None:  # noqa: ARG002
        return None
