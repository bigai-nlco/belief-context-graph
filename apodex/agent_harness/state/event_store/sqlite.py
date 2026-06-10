"""In-memory ``EventStore`` — append-only sink with no persistence.

The OSS benchmark harness needs an event-store shape so
``process_manager`` and ``scheduler`` can ``await event_store.append(...)``.
Bench runs are short enough that durable persistence is unnecessary, and
SQLite under high concurrency was a source of corruption.

The class still lives at ``state/event_store/sqlite.py`` because many
call sites import ``EventStore`` from this path. The ``sqlite`` filename
is historical — there is no SQLite here.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


class EventStore:
    """No-op event store. ``append`` records nothing; reads return [].

    Bench code routes everything through ``result.json`` instead.
    """

    async def append(
        self,
        task_id: Any = "",
        event_type: Any = None,
        payload: dict[str, Any] | None = None,
        agent_role: str = "system",
    ) -> None:
        return None

    async def replay(self, task_id: Any) -> AsyncIterator[Any]:
        return
        yield  # type: ignore[unreachable]

    async def get_events(
        self,
        task_id: Any,
        event_type: Any = None,
        after_id: int = 0,
        limit: int | None = None,
    ) -> list[Any]:
        return []

    async def get_events_for_agent(
        self,
        to_agent: str,
        after_id: int = 0,
        limit: int = 50,
        *,
        task_id: Any = None,
    ) -> list[Any]:
        return []
