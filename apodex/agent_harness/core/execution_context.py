"""Shared execution context carrier for phase, LLM, and tool calls.

Execution metadata is stored durably in pipeline state under the
``execution_context`` key and exposed at runtime via a ContextVar so
LLM/tool middleware can read it without changing every call site.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from agent_harness.core.types import new_prompt_id, new_session_id, new_step_id


@dataclass
class ExecutionScope:
    """Runtime execution scope for the current phase."""

    task_id: str = ""
    phase_id: str = ""
    role_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


_CURRENT_SCOPE: ContextVar[ExecutionScope | None] = ContextVar(
    "agent_harness_execution_scope", default=None
)


def normalize_execution_context(value: Any) -> dict[str, Any]:
    """Return a mutable execution-context dict."""
    if isinstance(value, dict):
        return dict(value)
    return {}


def build_execution_scope(
    *,
    task_id: str,
    phase_id: str,
    role_id: str,
    state: dict[str, Any] | None = None,
) -> ExecutionScope:
    """Build a scope from task/phase identity plus state metadata."""
    metadata = normalize_execution_context((state or {}).get("execution_context"))
    metadata.setdefault("agent_id", role_id)
    return ExecutionScope(
        task_id=task_id,
        phase_id=phase_id,
        role_id=role_id,
        metadata=metadata,
    )


def set_current_execution_scope(scope: ExecutionScope) -> Token:
    """Set the current execution scope for this async context."""
    return _CURRENT_SCOPE.set(scope)


def get_current_execution_scope() -> ExecutionScope | None:
    """Return the current execution scope if one is active."""
    return _CURRENT_SCOPE.get()


def reset_current_execution_scope(token: Token) -> None:
    """Restore the previous execution scope."""
    _CURRENT_SCOPE.reset(token)


def ensure_trace_metadata(
    metadata: dict[str, Any],
    *,
    default_step_id: str | None = None,
    refresh_prompt_id: bool = False,
) -> dict[str, Any]:
    """Ensure trace-chain identifiers exist in execution metadata."""
    metadata.setdefault("session_id", str(new_session_id()))
    if default_step_id:
        metadata.setdefault("step_id", default_step_id)
    else:
        metadata.setdefault("step_id", str(new_step_id()))
    if refresh_prompt_id or not metadata.get("prompt_id"):
        metadata["prompt_id"] = str(new_prompt_id())
    return metadata
