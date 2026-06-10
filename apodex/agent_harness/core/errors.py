"""Exception hierarchy for AgentHarness."""

from __future__ import annotations


class AgentHarnessError(Exception):
    """Base exception for all AgentHarness errors."""


# ── Kernel errors ───────────────────────────────────────────────────────────


class KernelError(AgentHarnessError):
    """Errors originating from the OS kernel layer."""


class TaskNotFoundError(KernelError):
    def __init__(self, task_id: str) -> None:
        super().__init__(f"Task not found: {task_id}")
        self.task_id = task_id


class InvalidStateTransition(KernelError):
    def __init__(self, task_id: str, current: str, target: str) -> None:
        super().__init__(f"Invalid transition for {task_id}: {current} → {target}")


class ServiceNotRegistered(KernelError):
    def __init__(self, service_type: type) -> None:
        super().__init__(f"Service not registered: {service_type.__name__}")


class PermissionDenied(KernelError):
    def __init__(self, role: str, tool: str) -> None:
        super().__init__(f"Role '{role}' has no permission for tool '{tool}'")
