"""Refusal-keyword rollback observer.

Fine-tuned chat models occasionally emit refusals (``"I'm sorry, but
I can't…"``) instead of using their tools. Without intervention the
refusal lands in history as a no-tool turn and the loop treats it as
the final answer.

Fires on ``on_llm_response``. If a refusal keyword is found in visible
text or ``<think>`` content, pop the assistant message and re-run the
turn. Budget caps at ``MAX_CONSECUTIVE_ROLLBACKS`` consecutive pops.

The smart-quote (U+2019) form of "I'm" is matched because some training
sets use ``’`` rather than ASCII ``'``.
"""

from __future__ import annotations

import logging

from agent_harness.core.loop_types import (
    BaseObserver,
    Intervention,
    TurnContext,
)

logger = logging.getLogger(__name__)

__all__ = ["RefusalRollbackObserver"]


# Both U+2019 ``’`` and ASCII ``'`` variants are matched.
REFUSAL_KEYWORDS: tuple[str, ...] = (
    "time constraint",
    "I’m sorry, but I can’t",  # smart quote
    "I’m sorry, I cannot solve",
    "I'm sorry, but I can't",            # ASCII fallback
    "I'm sorry, I cannot solve",
)


class RefusalRollbackObserver(BaseObserver):
    """Pop and re-run when the model emits a refusal keyword.

    Per-trial state (``_consecutive_rollbacks``) lives on the instance.
    The loop engine constructs a fresh observer list per trial, so
    state never leaks across trials.
    """

    MAX_CONSECUTIVE_ROLLBACKS = 5

    def __init__(
        self, keywords: tuple[str, ...] | None = None,
    ) -> None:
        self._keywords: tuple[str, ...] = (
            keywords if keywords is not None else REFUSAL_KEYWORDS
        )
        self._consecutive_rollbacks: int = 0

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        # Only scan turns with no tool calls — legitimate question text may
        # contain refusal-like substrings ("time constraints" etc.) and
        # would false-positive otherwise.
        if ctx.tool_calls:
            self._consecutive_rollbacks = 0
            return None

        # Search both visible content and thinking — refusals sometimes
        # hide inside ``<think>`` only.
        text = (ctx.ai_text or "") + "\n" + (ctx.thinking or "")
        matched: list[str] = [kw for kw in self._keywords if kw in text]

        if not matched:
            # Clean turn refreshes the budget.
            self._consecutive_rollbacks = 0
            return None

        if self._consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
            self._consecutive_rollbacks += 1
            logger.info(
                "RefusalRollback turn=%d: refusal keyword(s) %r detected — "
                "popping assistant message and retrying (rollback %d/%d)",
                ctx.turn,
                matched,
                self._consecutive_rollbacks,
                self.MAX_CONSECUTIVE_ROLLBACKS,
            )
            return Intervention(
                pop_last_message=True,
                continue_to_next_turn=True,
            )

        # Budget exhausted — let the loop exit as a no-tool turn;
        # ``_force_final_answer`` will summarize what's in history.
        logger.warning(
            "RefusalRollback turn=%d: budget (%d) exhausted on keywords "
            "%r — letting the refusal stand for the loop to handle.",
            ctx.turn,
            self.MAX_CONSECUTIVE_ROLLBACKS,
            matched,
        )
        return None
