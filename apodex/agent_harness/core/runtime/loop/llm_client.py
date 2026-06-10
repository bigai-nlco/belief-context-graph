"""LLM-client helpers for the agent loop.

Three small concerns wrap :meth:`LLMClient.chat`:

  * token estimation — tiktoken with a 4-chars-per-token fallback;
  * invocation — ``call_llm`` adds timeout + exponential backoff;
  * extraction — pulling usage, leaked reasoning, and the final visible
    answer out of an :class:`LLMResponse` / message history.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from agent_harness.core.llm import LLMClient, LLMResponse
from agent_harness.core.messages import Message, text_of
from agent_harness.infra.retriable import is_transient_network

logger = logging.getLogger(__name__)

__all__ = [
    "call_llm",
    "estimate_text_tokens",
    "estimate_message_tokens",
    "extract_final_content",
    "extract_leaked_reasoning",
    "extract_model_name",
    "extract_usage",
]


# ── Token estimation ────────────────────────────────────────────────────

_TOKEN_ENCODING: Any = None


def _get_token_encoding() -> Any:
    global _TOKEN_ENCODING
    if _TOKEN_ENCODING is not None:
        return _TOKEN_ENCODING
    try:
        import tiktoken
        try:
            _TOKEN_ENCODING = tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001
            _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001
        _TOKEN_ENCODING = False
    return _TOKEN_ENCODING


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens in *text* — tiktoken if available, ~4 chars/token otherwise."""
    if not text:
        return 0
    enc = _get_token_encoding()
    if enc is False:
        return len(text) // 4
    try:
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        return len(text) // 4


def estimate_message_tokens(message: Message) -> int:
    """Estimate tokens for a single dict message (content + tool_calls payload)."""
    tok = estimate_text_tokens(text_of(message.get("content")))
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        try:
            tok += estimate_text_tokens(
                json.dumps(tool_calls, ensure_ascii=False),
            )
        except Exception:  # noqa: BLE001
            pass
    return tok


# Inline ``<think>…</think>`` traces are carried in history (so the model
# sees prior reasoning) but must never leak into a final answer.
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*")
_DANGLING_THINK_RE = re.compile(r"<think>[\s\S]*\Z")


def _strip_thinking_blocks(text: str) -> str:
    """Strip inline ``<think>`` blocks (closed, unclosed, and SGLang's
    closer-without-opener variant) from an answer string."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _DANGLING_THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


# ── Backoff helpers ─────────────────────────────────────────────────────


def _get_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    return None


def _get_retry_after(exc: Exception) -> float | None:
    for attr in ("response", "headers"):
        obj = getattr(exc, attr, None)
        if obj is None:
            continue
        headers = getattr(obj, "headers", obj) if attr == "response" else obj
        if not hasattr(headers, "get"):
            continue
        val = headers.get("retry-after") or headers.get("Retry-After")
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def _default_backoff(attempt: int) -> int:
    return min(2 * (2 ** attempt), 60)


def _default_rate_limit_backoff(attempt: int) -> int:
    return min(30 * (2 ** attempt), 300)


# ── call_llm ────────────────────────────────────────────────────────────


async def call_llm(
    llm: LLMClient,
    messages: list[Message],
    timeout: int,
    max_retries: int,
    turn: int,
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    on_delta: Callable[[str, str, int, str], Awaitable[None]] | None = None,
    retry_wait_fixed: int | None = None,
) -> LLMResponse | None:
    """Call ``llm.chat`` (or ``llm.stream`` when ``on_delta`` is set) with
    timeout + exponential backoff on transient errors.

    Default schedule: 2/4/8/16/32s (cap 60) for timeouts/generic;
    429 honours ``Retry-After`` (cap 300s) or falls back to 30/60/120/240s.
    Non-transient 4xx (400/401/403/404) returns ``None`` immediately
    (proxy-wrapped transients on 400 still retry). ``retry_wait_fixed``
    overrides the schedule with a flat wait. Returns ``None`` when all
    retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            if on_delta is None:
                return await asyncio.wait_for(
                    llm.chat(
                        messages, tools=tools, temperature=temperature,
                        timeout=timeout,
                    ),
                    timeout=timeout,
                )
            return await _stream_llm_response(
                llm, messages, tools, temperature, timeout, on_delta,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "LLM call timed out (turn=%d, attempt=%d/%d)",
                turn, attempt + 1, max_retries,
            )
            backoff = (
                retry_wait_fixed
                if retry_wait_fixed is not None
                else _default_backoff(attempt)
            )
        except Exception as exc:
            status = _get_status_code(exc)
            if status and status in (400, 401, 403, 404):
                # Proxy-wrapped transient escape hatch — gateways occasionally
                # wrap 5xx/timeouts as a 400 envelope. Vanilla 400 falls through.
                if status == 400 and is_transient_network(exc):
                    backoff = (
                        retry_wait_fixed
                        if retry_wait_fixed is not None
                        else _default_backoff(attempt)
                    )
                    logger.warning(
                        "Proxy-wrapped transient %d (turn=%d, attempt=%d/%d): %s",
                        status, turn, attempt + 1, max_retries, exc,
                    )
                else:
                    logger.error(
                        "Non-transient LLM error %d (turn=%d): %s",
                        status, turn, exc,
                    )
                    return None
            elif status == 429:
                if retry_wait_fixed is not None:
                    backoff = retry_wait_fixed
                else:
                    retry_after = _get_retry_after(exc)
                    backoff = (
                        min(retry_after, 300)
                        if retry_after
                        else _default_rate_limit_backoff(attempt)
                    )
                logger.warning(
                    "LLM rate-limited 429 (turn=%d, attempt=%d/%d, wait=%ds): %s",
                    turn, attempt + 1, max_retries, int(backoff), exc,
                )
            else:
                backoff = (
                    retry_wait_fixed
                    if retry_wait_fixed is not None
                    else _default_backoff(attempt)
                )
                logger.warning(
                    "LLM call error (turn=%d, attempt=%d/%d): %s",
                    turn, attempt + 1, max_retries, exc,
                )

        if attempt < max_retries - 1:
            await asyncio.sleep(backoff)

    logger.error("LLM call failed after %d retries (turn=%d)", max_retries, turn)
    return None


# ── Streaming ───────────────────────────────────────────────────────────


class _ThinkTagSplitter:
    """Per-call state machine that splits inline ``<think>…</think>`` from
    a streaming text channel. ``feed(text)`` returns ``(visible, thinking)``;
    ``flush()`` drains any held-back partial tag at stream end. Up to
    ``len(CLOSE) - 1 == 7`` chars are held back to disambiguate a partial tag.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._inside = False
        self._buffer = ""

    @property
    def has_state(self) -> bool:
        return self._inside or bool(self._buffer)

    @staticmethod
    def _suffix_overlap(buf: str, tag: str) -> int:
        max_n = min(len(tag) - 1, len(buf))
        for n in range(max_n, 0, -1):
            if buf[-n:] == tag[:n]:
                return n
        return 0

    def feed(self, text: str) -> tuple[str, str]:
        visible_parts: list[str] = []
        thinking_parts: list[str] = []
        buf = self._buffer + text
        self._buffer = ""

        while buf:
            if self._inside:
                idx = buf.find(self.CLOSE)
                if idx == -1:
                    hold = self._suffix_overlap(buf, self.CLOSE)
                    safe = len(buf) - hold
                    if safe:
                        thinking_parts.append(buf[:safe])
                    self._buffer = buf[safe:]
                    break
                thinking_parts.append(buf[:idx])
                buf = buf[idx + len(self.CLOSE):]
                self._inside = False
            else:
                idx = buf.find(self.OPEN)
                if idx == -1:
                    hold = self._suffix_overlap(buf, self.OPEN)
                    safe = len(buf) - hold
                    if safe:
                        visible_parts.append(buf[:safe])
                    self._buffer = buf[safe:]
                    break
                if idx:
                    visible_parts.append(buf[:idx])
                buf = buf[idx + len(self.OPEN):]
                self._inside = True

        return ("".join(visible_parts), "".join(thinking_parts))

    def flush(self) -> tuple[str, str]:
        remainder = self._buffer
        self._buffer = ""
        if not remainder:
            return ("", "")
        if self._inside:
            return ("", remainder)
        return (remainder, "")


async def _stream_llm_response(
    llm: LLMClient,
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    timeout: int,
    on_delta: Callable[[str, str, int, str], Awaitable[None]],
) -> LLMResponse:
    """Stream chunks through ``on_delta`` and fold them into an LLMResponse."""
    accumulated = ""
    thinking_accum = ""
    delta_index = 0
    tool_call_acc: dict[int, dict[str, Any]] = {}
    finish_reason = ""
    model = ""
    usage: dict[str, int] = {}
    response_id = ""
    splitter = _ThinkTagSplitter()

    async with asyncio.timeout(timeout):
        stream = llm.stream(
            messages, tools=tools, temperature=temperature,
            timeout=timeout,
        )
        async for delta in stream:
            raw_visible = delta.content
            typed_thinking = delta.reasoning_content
            if raw_visible and (
                "<think>" in raw_visible
                or "</think>" in raw_visible
                or splitter.has_state
            ):
                visible, inline_thinking = splitter.feed(raw_visible)
            else:
                visible, inline_thinking = raw_visible, ""
            thinking = typed_thinking + inline_thinking
            if visible or thinking:
                if visible:
                    accumulated += visible
                if thinking:
                    thinking_accum += thinking
                await on_delta(visible, accumulated, delta_index, thinking)
                delta_index += 1
            for tcd in delta.tool_call_deltas:
                idx = tcd.get("index", 0)
                slot = tool_call_acc.setdefault(idx, {
                    "id": "", "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if tcd.get("id"):
                    slot["id"] = tcd["id"]
                if tcd.get("name"):
                    slot["function"]["name"] = (
                        slot["function"].get("name", "") + tcd["name"]
                    )
                if tcd.get("arguments"):
                    slot["function"]["arguments"] = (
                        slot["function"].get("arguments", "") + tcd["arguments"]
                    )
            # OpenAI streaming surfaces usage/finish_reason/model on the
            # final chunk via attributes the OpenAIClient adapter does not
            # forward into ``StreamDelta``; the consumer's accumulator stays
            # accurate without them.

    visible_flush, thinking_flush = splitter.flush()
    if visible_flush or thinking_flush:
        if visible_flush:
            accumulated += visible_flush
        if thinking_flush:
            thinking_accum += thinking_flush
        await on_delta(visible_flush, accumulated, delta_index, thinking_flush)

    return LLMResponse(
        content=accumulated,
        tool_calls=[tool_call_acc[k] for k in sorted(tool_call_acc)],
        reasoning_content=thinking_accum,
        finish_reason=finish_reason,
        model=model,
        usage=usage,
        response_metadata={"id": response_id} if response_id else {},
    )


# ── Response extraction ─────────────────────────────────────────────────


def extract_final_content(messages: list[Message]) -> str:
    """Most recent assistant message with non-empty visible text, stripped
    of any inline ``<think>…</think>`` blocks. Walks backward past empty
    assistant turns so a tool-only or empty-reply final turn doesn't
    eclipse the prior real answer.
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        cleaned = _strip_thinking_blocks(text_of(msg.get("content", "")))
        if cleaned:
            return cleaned
    return ""


# Key used by ``MultiFormatToolCallParser`` to stash salvaged reasoning on
# ``LLMResponse.response_metadata`` after stripping leaked private-tag content.
LEAKED_REASONING_KEY = "leaked_reasoning"


def extract_leaked_reasoning(response: LLMResponse | None) -> str:
    """Pull recovered reasoning that the parser stripped from leaked
    ``<think_…>`` / ``<model:thinking>`` / ``<think>`` blocks."""
    if response is None:
        return ""
    meta = response.response_metadata or {}
    value = meta.get(LEAKED_REASONING_KEY, "")
    return value if isinstance(value, str) else ""


def extract_usage(response: LLMResponse | None) -> dict | None:
    """Normalize ``response.usage`` to the
    ``{model, prompt_tokens, completion_tokens, cached_tokens}`` shape
    consumed by ProtocolStreamObserver / WorkerTraceFileObserver.
    Returns ``None`` when no token counts are present.
    """
    if response is None:
        return None
    usage = response.usage or {}
    inp = int(usage.get("prompt_tokens", 0) or 0)
    out = int(usage.get("completion_tokens", 0) or 0)
    cached = int(usage.get("cached_tokens", 0) or 0)
    if not inp and not out:
        return None
    return {
        "model": response.model or "",
        "prompt_tokens": inp,
        "completion_tokens": out,
        "cached_tokens": cached,
    }


def extract_model_name(
    llm: Any, profile: dict[str, Any] | None = None,
) -> str:
    """Best-effort model id: YAML profile first, then ``llm.model``."""
    if profile:
        name = (profile.get("llm") or {}).get("model")
        if isinstance(name, str) and name:
            return name
    for attr in ("model", "model_name", "model_id"):
        v = getattr(llm, attr, None)
        if isinstance(v, str) and v:
            return v
    return ""
