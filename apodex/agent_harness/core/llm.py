"""LLM client contracts — provider-agnostic chat completion interface.

Implementations live in ``infra/`` (OpenAI-compatible + Anthropic). The
runtime loop and middleware code only depend on the Protocol here, so
swapping providers is a one-line factory change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from agent_harness.core.messages import Message, ToolCall


@dataclass
class LLMResponse:
    """One non-streaming completion result."""

    content: Any = ""                                # str | list[dict]
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str = ""
    finish_reason: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamDelta:
    """Incremental update during a streaming completion."""

    content: str = ""
    reasoning_content: str = ""
    tool_call_deltas: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal async chat completion client."""

    model: str

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
        """Send one non-streaming completion request. ``tools`` is a list
        of OpenAI function-schema dicts; ``None`` runs without tools."""
        ...

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Stream a completion as a sequence of incremental ``StreamDelta``s.

        The terminal ``LLMResponse`` (with assembled content + finalised
        tool_calls + usage) is accessible via :meth:`last_response` after the
        stream is exhausted.
        """
        ...


__all__ = ["LLMClient", "LLMResponse", "StreamDelta"]
