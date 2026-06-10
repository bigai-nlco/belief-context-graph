"""Empty-search-result rollback observer.

When ``web_search`` returns the empty-result sentinel and we leave that
tool message in history, the model often keeps issuing variants of the
same dead query instead of pivoting. This observer pops the trailing
assistant + tool block on empty results so the next turn regenerates.

Fires on ``on_turn_end``. Budget caps at ``MAX_CONSECUTIVE_ROLLBACKS``
consecutive rollbacks; after that, empty results are let through.

Mutates ``ctx.messages`` in place because the ``Intervention`` contract
only supports a single ``pop_last_message`` per turn but we need to
drop a variable-length tail.
"""

from __future__ import annotations

import logging

from agent_harness.core.loop_types import (
    BaseObserver,
    Intervention,
    TurnContext,
)
from agent_harness.core.messages import text_of

logger = logging.getLogger(__name__)

__all__ = ["EmptySearchRollbackObserver"]


# Sentinel strings the tools emit on a no-result outcome.
EMPTY_SENTINELS: tuple[str, ...] = (
    "No search results found.",
    "Unknown tool:",
    "Error executing tool",
)


def _is_empty_result(content: object) -> bool:
    """True if a tool-role content text is one of the rollback sentinels."""
    s = text_of(content).strip()
    for sentinel in EMPTY_SENTINELS:
        # Accept exact match OR sentinel as the leading line — the
        # post-processor sometimes appends a newline-trailing suffix.
        if s == sentinel or s.startswith(sentinel):
            return True
    return False


class EmptySearchRollbackObserver(BaseObserver):
    """Pop the assistant message + trailing tool messages on empty search results.

    Per-trial state (``_consecutive_rollbacks``) lives on the instance.
    The loop engine constructs a fresh observer list per trial, so
    state never leaks across trials.
    """

    MAX_CONSECUTIVE_ROLLBACKS = 5

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        msgs = ctx.messages
        if not msgs:
            return None

        # Walk back over trailing tool messages; the slice
        # ``msgs[tool_start:end]`` is this turn's tool-result block.
        end = len(msgs)
        tool_start = end
        while tool_start > 0 and msgs[tool_start - 1].get("role") == "tool":
            tool_start -= 1

        if tool_start == end:
            # No trailing tool messages — refresh the budget and bail.
            self._consecutive_rollbacks = 0
            return None

        # Was there an assistant message right before the tool block? If
        # not, something earlier this turn already rewrote history — bail.
        ai_idx = tool_start - 1
        if ai_idx < 0 or msgs[ai_idx].get("role") != "assistant":
            return None

        empty_count = sum(
            1 for m in msgs[tool_start:end]
            if _is_empty_result(m.get("content"))
        )

        if empty_count == 0:
            # Clean turn — refresh budget.
            self._consecutive_rollbacks = 0
            return None

        if self._consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
            self._consecutive_rollbacks += 1
            logger.info(
                "EmptySearchRollback turn=%d: %d/%d tool result(s) empty — "
                "deleting trailing assistant message + %d tool message(s) and retrying "
                "(rollback %d/%d).",
                ctx.turn,
                empty_count,
                end - tool_start,
                end - tool_start,
                self._consecutive_rollbacks,
                self.MAX_CONSECUTIVE_ROLLBACKS,
            )
            # In-place mutation; the loop engine holds the same list
            # reference, so this propagates back without copying.
            del msgs[ai_idx:end]
            return Intervention(continue_to_next_turn=True)

        # Budget exhausted — stop popping; the empty result goes through.
        # Counter is only reset by a clean turn.
        logger.warning(
            "EmptySearchRollback turn=%d: budget (%d) exhausted on "
            "%d empty result(s) — letting them through.",
            ctx.turn,
            self.MAX_CONSECUTIVE_ROLLBACKS,
            empty_count,
        )
        return None

    def __init__(self) -> None:
        self._consecutive_rollbacks: int = 0
