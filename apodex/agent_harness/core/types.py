"""Base types for AgentHarness kernel and application layers."""

from __future__ import annotations

from enum import Enum
from typing import NewType
from uuid import uuid4

# ── Identity types ──────────────────────────────────────────────────────────

TaskId = NewType("TaskId", str)


def new_task_id() -> TaskId:
    return TaskId(uuid4().hex[:12])


def new_session_id() -> str:
    return uuid4().hex[:12]


def new_prompt_id() -> str:
    return uuid4().hex[:12]


def new_step_id() -> str:
    return uuid4().hex[:10]


# ── Enumerations ────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class PipelineEventType(str, Enum):
    """Pipeline-level events appended to the kernel EventStore."""

    REPORT_GENERATED = "report_generated"


