"""Summarization middleware — compress history when token count exceeds threshold."""

from __future__ import annotations

import asyncio
import logging
import re

from agent_harness.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddleware,
)
from agent_harness.core.llm import LLMClient
from agent_harness.core.messages import Message, system_msg, text_of, user_msg

logger = logging.getLogger(__name__)


class SummarizationMiddleware(LLMMiddleware):
    """Compress message history when token count exceeds the threshold.

    Token counting: tiktoken (cl100k_base) when available, else a heuristic
    that counts CJK characters as ~1 token and other text as ~4 chars/token.
    """

    # Unified Ideographs, Ext-A, radicals, strokes, Hiragana, Katakana,
    # CJK compatibility, fullwidth forms.
    _CJK_RE = re.compile(
        r"[⺀-⻿　-〿぀-ヿ㐀-䶿"
        r"一-鿿豈-﫿＀-￯]"
    )
    _MSG_OVERHEAD = 4

    def __init__(
        self,
        threshold: int = 80000,
        keep_recent: int = 6,
        summary_llm: LLMClient | None = None,
    ) -> None:
        self._threshold = threshold
        self._keep_recent = keep_recent
        self._summary_llm = summary_llm  # raw LLM, bypasses proxy

        self._tokenizer = None
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.debug("tiktoken unavailable, using heuristic token estimation")

    @property
    def name(self) -> str:
        return "summarization"

    def _estimate_tokens(self, messages: list[Message]) -> int:
        if self._tokenizer:
            total = 0
            for m in messages:
                total += len(self._tokenizer.encode(text_of(m.get("content", ""))))
                total += self._MSG_OVERHEAD
            return total
        total = 0
        for m in messages:
            text = text_of(m.get("content", ""))
            cjk = len(self._CJK_RE.findall(text))
            total += cjk + ((len(text) - cjk) // 4) + self._MSG_OVERHEAD
        return total

    async def before_llm(
        self, ctx: LLMCallContext, messages: list[Message],
    ) -> list[Message]:
        comp = ctx.metadata.get("compression") if ctx.metadata else None
        if isinstance(comp, dict):
            if not comp.get("enabled", True):
                return messages
            threshold = comp.get("threshold", self._threshold)
            keep_recent = comp.get("keep_recent", self._keep_recent)
        else:
            threshold = self._threshold
            keep_recent = self._keep_recent

        token_est = self._estimate_tokens(messages)
        if token_est <= threshold or len(messages) <= keep_recent + 1:
            return messages

        system_msgs = [m for m in messages[:1] if m.get("role") == "system"]
        rest = messages[len(system_msgs):]

        # Advance the split past leading tool messages so the kept window
        # never starts with an orphan (gateways reject "no matching tool_call_id").
        split_idx = len(rest) - keep_recent
        while split_idx < len(rest) - 1 and rest[split_idx].get("role") == "tool":
            split_idx += 1
        to_summarize = rest[:split_idx]
        keep = rest[split_idx:]

        if not to_summarize:
            return messages

        if self._summary_llm:
            summary_text = await self._generate_summary(to_summarize)
        else:
            summary_text = self._truncate_summary(to_summarize)

        summary = user_msg(
            f"[Previous conversation summary ({len(to_summarize)} messages compressed)]\n{summary_text}"
        )
        result = system_msgs + [summary] + keep
        logger.info(
            "SummarizationMiddleware: compressed %d→%d messages (est %d→%d tokens)",
            len(messages), len(result), token_est, self._estimate_tokens(result),
        )
        return result

    async def _generate_summary(self, messages: list[Message]) -> str:
        """Use a raw LLM to summarize. Falls back to truncation on failure."""
        content_parts: list[str] = []
        for m in messages:
            role = m.get("role", "?")
            text = text_of(m.get("content", ""))[:500]
            content_parts.append(f"[{role}] {text}")
        joined = "\n".join(content_parts[-20:])

        try:
            resp = await asyncio.wait_for(
                self._summary_llm.chat([
                    system_msg(
                        "Summarize the following conversation concisely, preserving "
                        "all key findings, tool results, decisions, and pending questions. "
                        "Output as 3-8 bullet points. Be specific — include names, "
                        "numbers, and URLs when present."
                    ),
                    user_msg(joined),
                ]),
                timeout=30,
            )
            return text_of(resp.content)
        except Exception as exc:
            logger.warning(
                "SummarizationMiddleware: LLM summary failed (%s), falling back to truncation",
                exc,
            )
            return self._truncate_summary(messages)

    def _truncate_summary(self, messages: list[Message]) -> str:
        parts: list[str] = []
        for m in messages[-5:]:
            role = m.get("role", "?")
            text = text_of(m.get("content", ""))[:200]
            parts.append(f"- {role}: {text}")
        return "\n".join(parts)
