"""Kernel-generic event identifiers."""

from __future__ import annotations

from enum import Enum


class EventType(str, Enum):
    """Framework-mechanical events emitted by ProcessManager / SSEObserver."""

    TASK_CREATED = "task_created"
    TASK_STATUS_CHANGED = "task_status_changed"
    AGENT_ACTION = "agent_action"
    ERROR = "error"
