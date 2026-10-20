"""Apply BCG only at the model boundary, for chat AND streaming clients."""

from __future__ import annotations

from collections.abc import AsyncIterator
from inspect import isawaitable
from typing import Any

from frontier_agent.core.llm import LLMClient, LLMResponse, StreamDelta
from frontier_agent.core.messages import Message

from .runtime import BCGMemory


class BCGClient:
    def __init__(self, inner: LLMClient, memory: BCGMemory) -> None:
        self.inner = inner
        self.memory = memory
        self.model = inner.model

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
        projected = await self.memory.prepare(messages)
        return await self.inner.chat(
            projected,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            timeout=timeout,
        )

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
        projected = await self.memory.prepare(messages)
        stream = self.inner.stream(
            projected,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            timeout=timeout,
        )
        try:
            async for delta in stream:
                yield delta
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                pending = close()
                if isawaitable(pending):
                    await pending
