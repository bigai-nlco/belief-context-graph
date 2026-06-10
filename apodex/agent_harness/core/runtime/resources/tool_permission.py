"""Tool permission context for filtering available tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolPermissionContext:
    """Permission filter for tool names.

    Supports two common patterns:
    - allowlist intersection via ``allow_names``
    - explicit denials via ``deny_names`` / ``deny_prefixes``
    """

    allow_names: frozenset[str] | None = None
    deny_names: frozenset[str] = frozenset()
    deny_prefixes: tuple[str, ...] = ()

    def blocks(self, tool_name: str) -> bool:
        lowered = tool_name.lower()
        return lowered in self.deny_names or any(
            lowered.startswith(prefix) for prefix in self.deny_prefixes
        )

    def allows(self, tool_name: str) -> bool:
        lowered = tool_name.lower()
        if lowered in self.deny_names or any(
            lowered.startswith(p) for p in self.deny_prefixes
        ):
            return False
        if self.allow_names is None:
            return True
        return lowered in self.allow_names

    def filter(self, tool_names: set[str]) -> set[str]:
        return {name for name in tool_names if self.allows(name)}

    @classmethod
    def from_iterables(
        cls,
        *,
        allow_names: set[str] | list[str] | tuple[str, ...] | None = None,
        deny_names: set[str] | list[str] | tuple[str, ...] = (),
        deny_prefixes: tuple[str, ...] | list[str] = (),
    ) -> "ToolPermissionContext":
        normalized_allow = None
        if allow_names is not None:
            normalized_allow = frozenset(name.lower() for name in allow_names)
        return cls(
            allow_names=normalized_allow,
            deny_names=frozenset(name.lower() for name in deny_names),
            deny_prefixes=tuple(prefix.lower() for prefix in deny_prefixes),
        )


def from_execution_policy(policy: Any | None) -> ToolPermissionContext:
    """Build a ToolPermissionContext from a node/workflow execution policy.

    Accepts any object exposing ``allow_tools`` / ``deny_tools`` /
    ``deny_tool_prefixes`` attributes so kernel code can consume policy data
    without importing higher-level pipeline models directly.
    """
    if policy is None:
        return ToolPermissionContext()

    allow_tools = getattr(policy, "allow_tools", None)
    deny_tools = getattr(policy, "deny_tools", ())
    deny_prefixes = getattr(policy, "deny_tool_prefixes", ())
    return ToolPermissionContext.from_iterables(
        allow_names=allow_tools,
        deny_names=deny_tools,
        deny_prefixes=deny_prefixes,
    )
