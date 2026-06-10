"""AgentHarness built-in tools.

Exposes the three tools the ``react_base`` workflow binds to its
``react_solver`` agent role: ``web_search`` / ``web_fetch`` /
``run_python_code``.
"""

from __future__ import annotations

from agent_harness.core.tool import Tool
from plugins.tools.run_python_code import run_python_code
from plugins.tools.web_fetch import web_fetch
from plugins.tools.web_search import web_search


_BUILTIN_TOOLS: list[Tool] = [
    web_search,
    web_fetch,
    run_python_code,
]


def get_builtin_tools() -> dict[str, Tool]:
    """Return all built-in tools as a name → tool dict."""
    return {t.name: t for t in _BUILTIN_TOOLS}


__all__ = [
    "get_builtin_tools",
    "run_python_code",
    "web_fetch",
    "web_search",
]
