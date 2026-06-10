"""Anthropic LLMClient — wraps :class:`anthropic.AsyncAnthropic`.

Translates between OpenAI Chat Completions message format (used everywhere
else in the runtime) and Anthropic's block-based messages API.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from agent_harness.core.llm import LLMClient, LLMResponse, StreamDelta
from agent_harness.core.messages import Message, ToolCall, text_of

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """Non-streaming-first Anthropic adapter."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = 4096,
        timeout: float | None = 300.0,
    ) -> None:
        from anthropic import AsyncAnthropic
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.default_timeout = timeout
        self._client = AsyncAnthropic(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0,
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
        system, msgs = _split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_anthropic_msg(m) for m in msgs],
            "max_tokens": max_tokens or self.default_max_tokens or 4096,
        }
        if system:
            kwargs["system"] = system
        eff_temp = temperature if temperature is not None else self.default_temperature
        if eff_temp is not None:
            kwargs["temperature"] = eff_temp
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        if timeout is not None:
            kwargs["timeout"] = timeout
        elif self.default_timeout is not None:
            kwargs["timeout"] = self.default_timeout

        raw = await self._client.messages.create(**kwargs)
        return _to_llm_response(raw)

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
        # Anthropic streaming is left non-streaming for now — yields a
        # single final delta. Plug in proper event handling when needed.
        resp = await self.chat(
            messages, tools=tools, temperature=temperature,
            max_tokens=max_tokens, extra_headers=extra_headers, timeout=timeout,
        )
        yield StreamDelta(content=text_of(resp.content), reasoning_content=resp.reasoning_content)


# ── Conversion helpers ───────────────────────────────────────────────────


def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """Pull out the (single) leading system message; Anthropic takes it
    as a top-level kwarg, not as a message."""
    if messages and messages[0].get("role") == "system":
        return text_of(messages[0].get("content", "")), messages[1:]
    return "", list(messages)


def _to_anthropic_msg(m: Message) -> dict[str, Any]:
    role = m.get("role")
    if role == "tool":
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": text_of(m.get("content", "")),
            }],
        }
    if role == "assistant":
        blocks: list[dict[str, Any]] = []
        body = text_of(m.get("content", ""))
        if body:
            blocks.append({"type": "text", "text": body})
        for tc in m.get("tool_calls", []) or []:
            blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"].get("arguments") or "{}"),
            })
        return {"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]}
    return {"role": "user", "content": text_of(m.get("content", ""))}


def _to_anthropic_tool(t: dict[str, Any]) -> dict[str, Any]:
    """OpenAI ``{type:function, function:{name,description,parameters}}`` →
    Anthropic ``{name, description, input_schema}``."""
    fn = t.get("function") or t
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {}),
    }


def _to_llm_response(raw: Any) -> LLMResponse:
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    blocks_out: list[dict[str, Any]] = []
    tool_calls: list[ToolCall] = []
    for block in (getattr(raw, "content", None) or []):
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", "") or ""
            text_parts.append(text)
            blocks_out.append({"type": "text", "text": text})
        elif btype == "thinking":
            thinking = getattr(block, "thinking", "") or ""
            thinking_parts.append(thinking)
            blocks_out.append({
                "type": "thinking",
                "thinking": thinking,
                # ``signature`` is the cryptographic token Anthropic returns
                # with each thinking block; resending it on the next turn
                # lets the model continue from the same reasoning state.
                "signature": getattr(block, "signature", "") or "",
            })
        elif btype == "tool_use":
            tool_calls.append({
                "id": getattr(block, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(block, "name", ""),
                    "arguments": json.dumps(getattr(block, "input", {}) or {}, ensure_ascii=False),
                },
            })

    # When thinking is present, keep the structured block list so the
    # ``content_block`` thinking parser can pick out reasoning vs visible
    # text. Otherwise flatten to a string for the simpler downstream path.
    if thinking_parts:
        content: Any = blocks_out
    else:
        content = "\n".join(text_parts)

    usage_dict: dict[str, int] = {}
    usage = getattr(raw, "usage", None)
    if usage:
        inp = getattr(usage, "input_tokens", None)
        out = getattr(usage, "output_tokens", None)
        if inp is not None:
            usage_dict["prompt_tokens"] = int(inp)
        if out is not None:
            usage_dict["completion_tokens"] = int(out)
        # Anthropic prompt-caching counters — surfaced under the same
        # ``cached_tokens`` key the observers consume.
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        if cache_read is not None:
            usage_dict["cached_tokens"] = int(cache_read)
        if usage_dict.get("prompt_tokens") or usage_dict.get("completion_tokens"):
            usage_dict["total_tokens"] = (
                usage_dict.get("prompt_tokens", 0)
                + usage_dict.get("completion_tokens", 0)
            )

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        reasoning_content="\n".join(thinking_parts),
        finish_reason=getattr(raw, "stop_reason", "") or "",
        model=getattr(raw, "model", "") or "",
        usage=usage_dict,
        response_metadata={"id": getattr(raw, "id", "")},
    )


__all__ = ["AnthropicClient"]
