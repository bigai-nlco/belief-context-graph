"""LLM Middleware framework — context, protocol, chain.

Phase-level concerns live in ExecutionMiddleware. This module runs at
LLM-call granularity: every ``LLMClient.chat`` / ``stream`` call passes
through the chain when ``ResourceManager`` returns the wrapped proxy.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent_harness.core.llm import LLMResponse, StreamDelta
from agent_harness.core.messages import Message

logger = logging.getLogger(__name__)


# Keywords that indicate a transient error worth retrying. Shared with
# fallback retry logic — keep in sync if you add new transient classes.
_RETRYABLE_KEYWORDS = frozenset({
    "timeout", "timed out", "429", "500", "502", "503", "504", "529",
    "overloaded", "rate limit", "rate_limit", "server error",
    "connection reset", "connection error", "econnreset",
    "gateway timeout",
    "model_dump",
    "model_not_found",
})


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, AttributeError):
        return True
    msg = str(error).lower()
    return any(kw in msg for kw in _RETRYABLE_KEYWORDS)


@dataclass
class LLMCallContext:
    """Context for a single LLM invocation."""

    task_id: str = ""
    role_id: str = ""
    phase_id: str = ""
    call_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMMiddleware(ABC):
    """Base class for LLM-call-level middlewares.

    Override only the hooks you need; the defaults are pass-throughs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def enabled(self) -> bool:
        return True

    async def before_llm(
        self, ctx: LLMCallContext, messages: list[Message],
    ) -> list[Message]:
        """Called before each LLM invocation. May modify ``messages``."""
        return messages

    async def after_llm(
        self, ctx: LLMCallContext, response: LLMResponse,
    ) -> LLMResponse:
        """Called after each LLM invocation. May modify ``response``."""
        return response

    async def on_llm_error(
        self, ctx: LLMCallContext, error: Exception, attempt: int,
    ) -> bool:
        """Return True to retry the call, False to propagate the error."""
        return False

    async def on_chunk(
        self, ctx: LLMCallContext, delta: StreamDelta, full_text: str,
    ) -> bool:
        """Called per streaming delta. ``full_text`` is the cumulative
        visible content. Returning ``True`` aborts the stream — the proxy
        stops consuming, runs ``after_llm`` on the partial response, and
        exits cleanly. Hot path — keep hooks fast."""
        return False


class LLMMiddlewareChain:
    """Ordered collection of middlewares. ``after`` runs in reverse (onion)."""

    def __init__(self) -> None:
        self._middlewares: list[LLMMiddleware] = []

    def add(self, mw: LLMMiddleware) -> None:
        self._middlewares.append(mw)

    def remove_by_name(self, name: str) -> None:
        self._middlewares = [
            m for m in self._middlewares if m.name != name
        ]

    def get(self, name: str) -> LLMMiddleware | None:
        for m in self._middlewares:
            if m.name == name:
                return m
        return None

    @property
    def middlewares(self) -> list[LLMMiddleware]:
        return list(self._middlewares)

    def wrap_llm(self, llm: Any, *, role_id: str) -> Any:
        """Return an LLMClient proxy that runs ``llm`` through this chain."""
        from agent_harness.components.middleware.llm.proxy import LLMProxy
        return LLMProxy(inner=llm, chain=self, role_id=role_id)

    async def run_before(
        self, ctx: LLMCallContext, messages: list[Message],
    ) -> list[Message]:
        for mw in self._middlewares:
            if mw.enabled:
                try:
                    messages = await mw.before_llm(ctx, messages)
                except Exception:
                    logger.exception(
                        "LLMMiddleware %s.before_llm failed", mw.name,
                    )
        return messages

    async def run_after(
        self, ctx: LLMCallContext, response: LLMResponse,
    ) -> LLMResponse:
        for mw in reversed(self._middlewares):
            if mw.enabled:
                try:
                    response = await mw.after_llm(ctx, response)
                except Exception:
                    logger.exception(
                        "LLMMiddleware %s.after_llm failed", mw.name,
                    )
        return response

    async def run_on_llm_error(
        self, ctx: LLMCallContext, error: Exception, attempt: int,
    ) -> bool:
        for mw in self._middlewares:
            if mw.enabled:
                try:
                    if await mw.on_llm_error(ctx, error, attempt):
                        return True
                except Exception:
                    logger.exception(
                        "LLMMiddleware %s.on_llm_error failed", mw.name,
                    )
        return False

    async def run_on_chunk(
        self,
        ctx: LLMCallContext,
        delta: StreamDelta,
        full_text: str,
    ) -> bool:
        """Fan out to every enabled middleware's ``on_chunk``. Returns True
        if any middleware asks to abort. A raise here does NOT abort the
        stream — log + continue, since a flaky hook should never crash a
        good generation."""
        for mw in self._middlewares:
            if not mw.enabled:
                continue
            try:
                if await mw.on_chunk(ctx, delta, full_text):
                    return True
            except Exception:
                logger.exception(
                    "LLMMiddleware %s.on_chunk failed", mw.name,
                )
        return False
