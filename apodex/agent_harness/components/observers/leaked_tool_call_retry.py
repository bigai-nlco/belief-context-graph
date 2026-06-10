"""Observer that retries leaked-text tool calls with escalating temperature.

When an LLM response yields no structured tool calls but the visible
content clearly refers to a tool (free-text or exotic XML wrappers that
the parser couldn't recover), this observer re-prompts on the next turn
with a nudge message plus an escalated temperature (``[0.3, 0.6, 1.0]``).
After the sequence is exhausted, the normal ``no_tool`` handler takes over.

Disable via ``enabled=False`` or an empty ``temperatures`` sequence.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from agent_harness.core.loop_types import BaseObserver, Intervention, TurnContext

logger = logging.getLogger(__name__)

_METADATA_COUNT_KEY = "_leak_retry_count"
_METADATA_TEMP_KEY = "_llm_temp_override"

_DEFAULT_TEMPERATURES: tuple[float, ...] = (0.3, 0.6, 1.0)

# XML wrappers the parser already tries — detect-only for signalling.
_LEAK_XML_RE = re.compile(
    r"<\s*(?:tool_call\b"
    r"|function\s+name\s*="
    r"|seed:tool_call\b"
    r"|seedtool_call\b"
    r"|seed:tool[-_]name\b"
    r")",
    re.IGNORECASE,
)

_NUDGE_MESSAGE = (
    "Your previous response mentioned calling a tool in free text but no "
    "structured tool_call was emitted. Invoke the tool through the proper "
    "tool_calls interface now — do not describe the call in prose."
)


class LeakedToolCallRetryObserver(BaseObserver):
    """Detect text-only tool-call leaks and trigger a next-turn retry.

    Args:
        tool_names: allowed tool identifiers; guards the free-text mention
            check (only triggers when one of these names appears).
        temperatures: escalation sequence (default ``(0.3, 0.6, 1.0)``).
        enabled: master switch. Disables without removing from the stack.
        max_nudges_per_task: total cap on retries per loop run; prevents
            tight loops when the model ignores the nudge.
    """

    critical = True

    def __init__(
        self,
        tool_names: Iterable[str],
        *,
        temperatures: tuple[float, ...] = _DEFAULT_TEMPERATURES,
        enabled: bool = True,
        max_nudges_per_task: int | None = None,
    ) -> None:
        self._tool_names: frozenset[str] = frozenset(tool_names)
        self._temperatures = temperatures
        self._enabled = enabled and bool(temperatures)
        # Default cap = len(temperatures) so each temperature is tried once.
        self._max_nudges = (
            max_nudges_per_task
            if max_nudges_per_task is not None
            else len(temperatures)
        )
        # Pre-compile the free-text tool-name probe once.
        if self._tool_names:
            pattern = (
                r"\b("
                + "|".join(re.escape(n) for n in self._tool_names)
                + r")\b"
            )
            self._tool_name_re: re.Pattern | None = re.compile(
                pattern, re.IGNORECASE
            )
        else:
            self._tool_name_re = None

    # ------------------------------------------------------------------ API

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        if not self._enabled:
            return None

        # A structured tool call was parsed — nothing to retry. Also reset
        # the escalation counter so a future leak starts at 0.3 again.
        if ctx.tool_calls:
            ctx.metadata.pop(_METADATA_COUNT_KEY, None)
            return None

        if not self._looks_like_leak(ctx.ai_text):
            return None

        count = int(ctx.metadata.get(_METADATA_COUNT_KEY, 0))
        if count >= self._max_nudges:
            logger.info(
                "[LeakedToolCallRetry] Exhausted retry budget (%d) — "
                "letting default no_tool handler take over.",
                self._max_nudges,
            )
            return None

        temp_index = min(count, len(self._temperatures) - 1)
        next_temp = float(self._temperatures[temp_index])

        ctx.metadata[_METADATA_COUNT_KEY] = count + 1
        ctx.metadata[_METADATA_TEMP_KEY] = next_temp

        logger.warning(
            "[LeakedToolCallRetry] turn=%d | leaked content, scheduling retry "
            "with temperature=%.1f (retry %d/%d)",
            ctx.turn, next_temp, count + 1, self._max_nudges,
        )

        return Intervention(inject_messages=[_NUDGE_MESSAGE])

    # --------------------------------------------------------------- helpers

    def _looks_like_leak(self, text: str) -> bool:
        """Heuristic detector: free-text tool mention OR XML-wrapper residue."""
        if not text:
            return False
        sample = text[:4000]
        if _LEAK_XML_RE.search(sample):
            return True
        if self._tool_name_re is None:
            return False
        # Require both a tool-name mention AND either a call-shape hint
        # ("(" or "{") or a directive verb — avoids flagging plain prose
        # that happens to include the tool name in a summary.
        if not self._tool_name_re.search(sample):
            return False
        lowered = sample.lower()
        call_hints = ("(", "{", "args:", "arguments:", "parameter")
        verb_hints = (
            "i will", "let me", "i'll call", "now call", "i'm going to",
            "next, i", "calling ",
        )
        if any(hint in lowered for hint in call_hints):
            return True
        if any(verb in lowered for verb in verb_hints):
            return True
        return False
