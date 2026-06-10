"""Task-scoped pause_check helper for ReAct agent loops.

``make_task_pause_check(task_id)`` returns a zero-arg async closure that
the runtime loop polls between turns. It asks the ``ProcessManager``
whether the task status has flipped to ``SUSPENDED`` / ``ABORTED``; if so
the loop exits cleanly at the turn boundary (``stop_reason="paused"``),
preserving accumulated state, and the scheduler stops the DAG at the next
node boundary.

Without this hook, long single-node pipelines never observe a pause until
``max_turns`` is exhausted.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from agent_harness.core.errors import TaskNotFoundError
from agent_harness.core.types import TaskId, TaskStatus
from agent_harness.core.runtime import registry

logger = logging.getLogger(__name__)

PauseCheckFn = Callable[[], Awaitable[bool]]

# Statuses that should stop an in-flight agent loop at the next turn.
# ``FAILED`` is omitted — a transition to FAILED usually means the
# pipeline itself set that status after the loop returned, so stopping
# on it would create a feedback cycle.
_STOP_STATUSES = {TaskStatus.SUSPENDED, TaskStatus.ABORTED}


def make_task_pause_check(task_id: str | TaskId) -> PauseCheckFn:
    """Return a ``pause_check`` closure bound to one research task.

    The closure is **safe to call from inside the kernel loop**:
    - never raises; unexpected errors log at WARNING and return False
      (i.e. keep running rather than stop on a read hiccup);
    - returns True only when the task's current status is in the
      stop set (``suspended`` or ``aborted``).
    """
    tid = str(task_id)

    async def _check() -> bool:
        # Lazy import — ``scheduling`` depends on ``core/runtime``, so
        # a top-level import here would create a circular dependency.
        from agent_harness.scheduling.process_manager import ProcessManager

        pm = registry.get_optional(ProcessManager)
        if pm is None:
            return False
        try:
            task = await pm.get_task(TaskId(tid))
        except TaskNotFoundError:
            # Sub-runs may use synthetic task ids that aren't registered
            # with ProcessManager. Treat as "no pause signal" silently
            # rather than emitting a warning every turn.
            return False
        except Exception as exc:
            logger.warning(
                "pause_check: get_task(%s) failed: %s", tid, exc,
            )
            return False
        return getattr(task, "status", None) in _STOP_STATUSES

    return _check


def pause_check_from_state(state: dict[str, Any] | None) -> PauseCheckFn | None:
    """Pull the ``pause_check`` closure out of ``state.metadata``.

    Research / agent runners stash the closure on
    ``state["metadata"]["pause_check"]`` so every node downstream can
    forward it to ``run_agent_loop`` without re-importing
    :func:`make_task_pause_check`. SDK paths inject their own closure
    the same way. ``None`` means "no pause hook wired for this call".
    """
    if not state:
        return None
    metadata = state.get("metadata") or {}
    return metadata.get("pause_check")


__all__ = ["make_task_pause_check", "pause_check_from_state", "PauseCheckFn"]
