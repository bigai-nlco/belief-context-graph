"""Duplicate-query rollback observer.

The model can get stuck re-issuing the same ``web_search`` queries.
This observer fires on ``on_llm_response`` and, when a tool call
repeats a query already seen this trial, pops the assistant message
and re-runs the turn (next sampling at T=1.0 almost always diverges).

Only ``web_search`` is deduped. ``web_fetch`` retries on the same URL
are sometimes legitimate; ``run_python_code`` args are too varied.

Dedup is at *batch* granularity: ``[a, b]`` and ``[a, c]`` are treated
as different searches, so the model can evolve a search incrementally.

Budget caps at ``MAX_CONSECUTIVE_ROLLBACKS`` consecutive pops; the
counter resets after any clean turn.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.core.loop_types import (
    BaseObserver,
    Intervention,
    TurnContext,
)

logger = logging.getLogger(__name__)

__all__ = ["DuplicateQueryRollbackObserver"]


def _dedup_key(tool_call: dict[str, Any]) -> str | None:
    """Build the per-tool-call dedup key.

    ``q: str``        → ``"<tool_name>_<q>"``
    ``q: list[str]``  → ``"<tool_name>_<q1>|<q2>|..."``

    Multi-query calls are ONE dedup unit. Tool name is prepended so
    different search tools don't collide on the same query string.
    """
    tool_name = tool_call.get("name") or ""
    if not tool_name:
        return None
    args = tool_call.get("args") or {}
    q = args.get("q")
    if isinstance(q, str):
        s = q.strip()
        return f"{tool_name}_{s}" if s else None
    if isinstance(q, list):
        parts: list[str] = []
        for item in q:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    parts.append(s)
        if not parts:
            return None
        return f"{tool_name}_" + "|".join(parts)
    return None


class DuplicateQueryRollbackObserver(BaseObserver):
    """Pop the assistant message and re-run when a duplicate web_search is detected.

    Per-trial state (``_seen_queries`` + ``_consecutive_rollbacks``) lives
    on the instance. The loop engine constructs a fresh
    ``LoopObserver`` list per trial, so state never leaks across trials.
    """

    MAX_CONSECUTIVE_ROLLBACKS = 5

    DEFAULT_TOOL_NAMES: frozenset[str] = frozenset({"web_search"})

    def __init__(self, tool_names: set[str] | None = None) -> None:
        self._tool_names: set[str] = (
            set(tool_names) if tool_names else set(self.DEFAULT_TOOL_NAMES)
        )
        # All queries executed (or that survived a rollback) in this
        # trial. Value is the execution count so the same string seen N
        # times still rolls back at the (N+1)-th attempt.
        self._seen_queries: dict[str, int] = {}
        self._consecutive_rollbacks: int = 0

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        # No tool calls → not a candidate for rollback; reset the
        # consecutive counter so the budget refreshes after any
        # think-only turn (no_tool natural finish, post-summary, etc.).
        if not ctx.tool_calls:
            self._consecutive_rollbacks = 0
            return None

        # Build the per-tool-call dedup keys for this turn — one key per
        # ``web_search`` tool call. Non-tracked tools (web_fetch /
        # run_python_code / …) contribute no keys and are skipped.
        turn_keys: list[str] = []
        for tc in ctx.tool_calls:
            if tc.get("name") not in self._tool_names:
                continue
            k = _dedup_key(tc)
            if k:
                turn_keys.append(k)

        if not turn_keys:
            # Tool call(s) were e.g. web_fetch / run_python_code — not
            # dedup-tracked. Don't touch the counter; let the next
            # tracked turn manage it.
            return None

        # A turn is duplicate if **any** of its tracked tool calls
        # repeats a key already seen this trial.
        duplicates = [k for k in turn_keys if self._seen_queries.get(k, 0) > 0]

        # Budget check: pop up to MAX-1 times, then let the next through.
        if duplicates and self._consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
            self._consecutive_rollbacks += 1
            logger.info(
                "DuplicateQueryRollback turn=%d: %d duplicate batch(es) "
                "detected — popping assistant message and retrying (rollback "
                "%d/%d). Example: %r",
                ctx.turn,
                len(duplicates),
                self._consecutive_rollbacks,
                self.MAX_CONSECUTIVE_ROLLBACKS,
                duplicates[0][:160],
            )
            return Intervention(
                pop_last_message=True,
                continue_to_next_turn=True,
            )

        # Not rolling back: record keys so future dedup sees them.
        # Reset counter only on a clean turn (no duplicates), not on a
        # budget-exhaust let-through.
        for k in turn_keys:
            self._seen_queries[k] = self._seen_queries.get(k, 0) + 1
        if not duplicates:
            self._consecutive_rollbacks = 0
        return None
