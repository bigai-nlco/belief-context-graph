"""``LLMProxy`` — transparent :class:`LLMClient` that runs every call
through an :class:`LLMMiddlewareChain`.

Returned by ``ResourceManager.get_llm()`` when a chain is registered. To
callers (pipeline nodes, the agent loop), it's just an ``LLMClient``.
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Any, AsyncIterator

from agent_harness.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddlewareChain,
)
from agent_harness.core.execution_context import (
    ensure_trace_metadata,
    get_current_execution_scope,
)
from agent_harness.core.llm import LLMClient, LLMResponse, StreamDelta
from agent_harness.core.messages import Message

logger = logging.getLogger(__name__)

__all__ = ["LLMProxy"]


class LLMProxy(LLMClient):
    """Transparent :class:`LLMClient` wrapper that applies LLM middleware."""

    def __init__(
        self,
        inner: LLMClient,
        chain: LLMMiddlewareChain,
        role_id: str = "default",
    ) -> None:
        self.inner = inner
        self.chain = chain
        self.role_id = role_id
        self.model = getattr(inner, "model", "")
        self._counter = itertools.count(1)

    def _next_call_index(self) -> int:
        return next(self._counter)

    def _make_ctx(self, call_index: int) -> LLMCallContext:
        scope = get_current_execution_scope()
        metadata = dict(scope.metadata) if scope else {}
        if scope:
            default_step_id = f"{scope.phase_id or 'phase'}:llm:{call_index}"
            metadata = ensure_trace_metadata(
                metadata,
                default_step_id=default_step_id,
                refresh_prompt_id=True,
            )
            metadata["step_id"] = default_step_id
            scope.metadata.update(metadata)
        inner_model = getattr(self.inner, "model", None)
        if inner_model:
            metadata.setdefault("model_id", str(inner_model))
        return LLMCallContext(
            task_id=scope.task_id if scope else "",
            role_id=self.role_id,
            phase_id=scope.phase_id if scope else "",
            call_index=call_index,
            metadata=metadata,
        )

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        ctx = self._make_ctx(self._next_call_index())
        messages = await self.chain.run_before(ctx, messages)

        attempt = 0
        while True:
            try:
                response = await self.inner.chat(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers,
                    timeout=timeout,
                )
                break
            except Exception as exc:
                should_retry = await self.chain.run_on_llm_error(ctx, exc, attempt)
                if not should_retry:
                    raise
                attempt += 1
                logger.info(
                    "LLMProxy: retrying after attempt %d (%s)",
                    attempt, type(exc).__name__,
                )

        return await self.chain.run_after(ctx, response)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        ctx = self._make_ctx(self._next_call_index())
        messages = await self.chain.run_before(ctx, messages)

        attempt = 0
        while True:
            start_time = time.time()
            full_content = ""
            full_reasoning = ""
            stream_error: Exception | None = None
            any_chunk_yielded = False

            try:
                async for delta in self.inner.stream(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers,
                    timeout=timeout,
                ):
                    full_content += delta.content
                    full_reasoning += delta.reasoning_content
                    any_chunk_yielded = True
                    # Per-chunk middleware hook. ``True`` aborts the stream;
                    # the partial response still flows through ``after_llm``
                    # in the finally block.
                    yield delta
                    if await self.chain.run_on_chunk(ctx, delta, full_content):
                        ctx.metadata["stream_aborted_by_middleware"] = True
                        logger.info(
                            "LLMProxy: stream aborted by middleware after %d chars",
                            len(full_content),
                        )
                        break
                break
            except Exception as exc:
                stream_error = exc
                if not any_chunk_yielded:
                    if await self.chain.run_on_llm_error(ctx, exc, attempt):
                        attempt += 1
                        logger.info(
                            "LLMProxy: retrying stream after attempt %d (%s)",
                            attempt, type(exc).__name__,
                        )
                        continue
                raise
            finally:
                ctx.metadata["duration_ms"] = int((time.time() - start_time) * 1000)
                if stream_error:
                    ctx.metadata["error"] = str(stream_error)
                await self.chain.run_after(
                    ctx,
                    LLMResponse(
                        content=full_content,
                        reasoning_content=full_reasoning,
                    ),
                )

    def __getattr__(self, name: str) -> Any:
        # Defer unknown attributes to the inner client so callers that read
        # provider-specific fields (e.g. ``base_url``) keep working.
        return getattr(self.inner, name)
