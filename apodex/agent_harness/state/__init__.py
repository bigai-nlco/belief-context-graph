"""``agent_harness.state`` — optional sidecar for persistence.

- ``state/event_store/`` — in-memory ``EventStore``. Persistence is
  unnecessary for OSS bench runs (``result.json`` is the only artifact).
"""

from __future__ import annotations

__all__: list[str] = []
