"""Resource manager — tool permissions and LLM routing.

Tool permissions are driven by AgentRegistry (dynamic roles) and per-call
``ToolPermissionContext`` (built from ``NodeExecutionPolicy`` by pipeline
nodes).  The manager itself knows nothing about workflow phases.
"""

from __future__ import annotations

from typing import Any

from agent_harness.core.errors import PermissionDenied
from agent_harness.core.llm import LLMClient
from agent_harness.core.tool import Tool
from agent_harness.core.runtime.registries import services as service_registry
from agent_harness.core.runtime.resources.llm import (
    resolve_base_llm_for_role,
    wrap_llm_with_middleware,
)
from agent_harness.core.runtime.resources.permissions import get_effective_tools_for_role
from agent_harness.core.runtime.resources.tool_permission import ToolPermissionContext

# Compatibility alias for tests and older patch sites that still point at
# ``agent_harness.core.runtime.resources.manager.service_registry``.
_ = service_registry


class ResourceManager:
    """Manages tool permissions and LLM routing per agent role."""

    def __init__(
        self,
        llm: LLMClient,
        tools: dict[str, Tool],
    ) -> None:
        self._llm = llm
        self._tools = tools  # name → tool instance
        self._role_llms: dict[str, LLMClient] = {}  # cached per-role LLMs

    @property
    def llm(self) -> LLMClient:
        return self._llm

    @property
    def all_tools(self) -> dict[str, Tool]:
        return dict(self._tools)

    def get_tools_for_role(
        self,
        role_id: str,
        *,
        permission_context: ToolPermissionContext | None = None,
    ) -> list[Tool]:
        """Return the tools permitted for a role under the given permission context."""
        allowed = get_effective_tools_for_role(
            role_id=role_id,
            permission_context=permission_context,
        )
        return [t for name, t in self._tools.items() if name in allowed]

    def get_tool_for_role(
        self,
        role_id: str,
        tool_name: str,
        *,
        permission_context: ToolPermissionContext | None = None,
    ) -> Tool | None:
        """Return a single permitted tool, or None if unavailable for the role."""
        if not self.check_permission(
            role_id,
            tool_name,
            permission_context=permission_context,
        ):
            return None
        return self._tools.get(tool_name)

    def get_tool_names_for_role(
        self,
        role_id: str,
        *,
        permission_context: ToolPermissionContext | None = None,
    ) -> list[str]:
        allowed = get_effective_tools_for_role(
            role_id=role_id,
            permission_context=permission_context,
        )
        return [name for name in self._tools if name in allowed]

    def check_permission(
        self,
        role_id: str,
        tool_name: str,
        *,
        permission_context: ToolPermissionContext | None = None,
    ) -> bool:
        """Check if a tool is permitted for a role under the given permission context."""
        allowed = get_effective_tools_for_role(
            role_id=role_id,
            permission_context=permission_context,
        )
        return tool_name in allowed

    def require_permission(
        self,
        role_id: str,
        tool_name: str,
        *,
        permission_context: ToolPermissionContext | None = None,
    ) -> None:
        if not self.check_permission(
            role_id,
            tool_name,
            permission_context=permission_context,
        ):
            raise PermissionDenied(role_id, tool_name)

    async def search_web_structured(
        self,
        role_id: str,
        query: str,
        *,
        num_results: int = 50,
        permission_context: ToolPermissionContext | None = None,
    ) -> list[dict[str, Any]]:
        """Run web search with role-aware permission checks and structured output."""
        self.require_permission(
            role_id,
            "web_search",
            permission_context=permission_context,
        )
        if "web_search" not in self._tools:
            raise KeyError("Tool 'web_search' is not registered")

        from plugins.tools.web_search import raw_web_search

        # ``raw_web_search`` returns the full Serper response; extract organic results.
        data = await raw_web_search(query, num_results=num_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("link", r.get("url", "")),
                "snippet": r.get("snippet", ""),
            }
            for r in data.get("organic", [])[:num_results]
        ]

    def get_llm(self, role_id: str | None = None) -> LLMClient:
        """Get LLM for a role. Uses per-role model if configured, else default.

        If an LLMMiddlewareChain is registered, wraps the returned LLM
        in an LLMProxy so all ainvoke/astream calls pass through the chain.
        This is transparent to callers — pipeline nodes need zero changes.
        """
        base_llm = self._resolve_base_llm(role_id)
        return wrap_llm_with_middleware(
            base_llm,
            role_id=role_id or "default",
        )

    def get_raw_llm(self, role_id: str | None = None) -> LLMClient:
        """Get LLM WITHOUT middleware wrapping. Use for internal operations
        (e.g., summarization) to avoid infinite recursion."""
        return self._resolve_base_llm(role_id)

    def _resolve_base_llm(self, role_id: str | None) -> LLMClient:
        """Resolve the base LLM for a role (no middleware)."""
        return resolve_base_llm_for_role(
            default_llm=self._llm,
            role_id=role_id,
            cache=self._role_llms,
        )
