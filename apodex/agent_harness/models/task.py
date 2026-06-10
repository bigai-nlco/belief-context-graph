"""Task model — minimal in-memory record consumed by ProcessManager."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_harness.core.types import TaskId, TaskStatus, new_task_id


class Task(BaseModel):
    """The OS 'process' abstraction.

    Only fields actually read by ``scheduler`` / ``process_manager`` /
    ``pause_check`` remain: identity, runtime status, thread id (used
    as the DAG checkpoint key), and the bag of metadata each pipeline
    threads through ``context``.
    """

    id: TaskId = Field(default_factory=new_task_id)
    thread_id: str = ""
    status: TaskStatus = TaskStatus.CREATED
    error: str | None = None
    context: dict = Field(default_factory=dict)

    def set_status(self, status: TaskStatus) -> None:
        self.status = status
