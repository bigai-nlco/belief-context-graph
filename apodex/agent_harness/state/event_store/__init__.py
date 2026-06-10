"""In-memory event store used by the OSS bench harness.

``EventStore`` lives at :mod:`agent_harness.state.event_store.sqlite`. The
module name is preserved for import-path stability; the backend itself is
in-memory (see ``sqlite.py`` for details).
"""

from __future__ import annotations

__all__: list[str] = []
