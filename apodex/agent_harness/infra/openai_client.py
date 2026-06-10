"""OpenAI-compatible LLMClient — wraps :class:`openai.AsyncOpenAI`.

Works against the OpenAI API and any OpenAI-Chat-Completions-compatible
endpoint (vLLM, SGLang, OpenRouter, etc.). For Anthropic,
use ``infra.anthropic_client.AnthropicClient`` (separate file, separate SDK).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from agent_harness.core.llm import LLMClient, LLMResponse, StreamDelta
from agent_harness.core.messages import Message, ToolCall

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """Thin async wrapper. Honors the standard OpenAI knobs (``temperature``,
    ``max_completion_tokens``, ``tools``, ``stream``); per-call ``extra_headers``
    are forwarded for proxies needing custom auth or session affinity."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        timeout: float | None = None,
        default_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_completion_tokens
        self.default_timeout = timeout
        self.extra_body = extra_body or {}
        # Per-call timeout is enforced by the agent loop's
        # ``asyncio.wait_for`` wrapper, not by the SDK.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
            default_headers=default_headers,
            max_retries=0,           # retries are owned by the runtime loop
        )

    # ── Non-streaming ────────────────────────────────────────────────────

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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = True
        eff_temp = temperature if temperature is not None else self.default_temperature
        if eff_temp is not None:
            kwargs["temperature"] = eff_temp
        eff_mt = max_tokens if max_tokens is not None else self.default_max_tokens
        if eff_mt is not None:
            kwargs["max_completion_tokens"] = eff_mt
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        # Per-call ``timeout`` triggers ``x-stainless-read-timeout``
        # header; only set when the caller explicitly passed one.
        if timeout is not None:
            kwargs["timeout"] = timeout

        raw_response = await self._client.chat.completions.with_raw_response.create(
            **kwargs,
        )
        raw = raw_response.parse()
        return _to_llm_response(raw)

    # ── Streaming ────────────────────────────────────────────────────────

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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = True
        eff_temp = temperature if temperature is not None else self.default_temperature
        if eff_temp is not None:
            kwargs["temperature"] = eff_temp
        eff_mt = max_tokens if max_tokens is not None else self.default_max_tokens
        if eff_mt is not None:
            kwargs["max_completion_tokens"] = eff_mt
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if timeout is not None:
            kwargs["timeout"] = timeout
        elif self.default_timeout is not None:
            kwargs["timeout"] = self.default_timeout

        async for chunk in await self._client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            yield StreamDelta(
                content=getattr(delta, "content", None) or "",
                reasoning_content=getattr(delta, "reasoning_content", None) or "",
                tool_call_deltas=_tool_call_deltas(getattr(delta, "tool_calls", None)),
            )


# ── Adapters ─────────────────────────────────────────────────────────────


def _to_llm_response(raw: Any) -> LLMResponse:
    """Convert an OpenAI ``ChatCompletion`` into an :class:`LLMResponse`."""
    choice = raw.choices[0]
    msg = choice.message
    tool_calls: list[ToolCall] = []
    for tc in (getattr(msg, "tool_calls", None) or []):
        tool_calls.append({
            "type": "function",
            "id": tc.id,
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments or "{}",
            },
        })
    usage = getattr(raw, "usage", None)
    usage_dict: dict[str, int] = {}
    if usage:
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = getattr(usage, k, None)
            if v is not None:
                usage_dict[k] = int(v)
        # Prompt-caching surfaced under ``usage.prompt_tokens_details.cached_tokens``
        # on modern OpenAI / OpenRouter responses.
        ptd = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(ptd, "cached_tokens", None) if ptd is not None else None
        if cached is not None:
            usage_dict["cached_tokens"] = int(cached)
    return LLMResponse(
        content=getattr(msg, "content", "") or "",
        tool_calls=tool_calls,
        reasoning_content=getattr(msg, "reasoning_content", "") or "",
        finish_reason=getattr(choice, "finish_reason", "") or "",
        model=getattr(raw, "model", "") or "",
        usage=usage_dict,
        response_metadata={"id": getattr(raw, "id", "")},
    )


def _tool_call_deltas(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for tc in raw:
        out.append({
            "index": getattr(tc, "index", 0),
            "id": getattr(tc, "id", None),
            "name": getattr(getattr(tc, "function", None), "name", None),
            "arguments": getattr(getattr(tc, "function", None), "arguments", None),
        })
    return out


__all__ = ["OpenAIClient"]
