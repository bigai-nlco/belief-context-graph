"""react_base trajectory observer.

Wraps :class:`agent_harness.components.observers.trajectory.TrajectoryFileObserver`
so the recorded tool specs match what we actually send on the wire —
``_clean_tool_schema`` (oneOf, no title) is applied, mirroring the
profile's tool-schema cleaner.
"""

from __future__ import annotations

from typing import Any

from agent_harness.components.observers.trajectory import (
    TrajectoryFileObserver as _BaseTrajectoryFileObserver,
)

__all__ = ["TrajectoryFileObserver"]


def _clean_tools_for_log(tools: list[Any]) -> list[Any]:
    """Pre-serialize tools to OpenAI dicts with ``_clean_tool_schema`` applied."""
    from workflows.react_base.profile import _clean_tool_schema

    out: list[Any] = []
    for t in tools:
        if isinstance(t, dict):
            spec = dict(t)
        else:
            schema_fn = getattr(t, "to_openai_schema", None)
            if not callable(schema_fn):
                out.append(t)
                continue
            try:
                spec = schema_fn()
            except Exception:  # noqa: BLE001 — observer must never block a run
                out.append(t)
                continue
        fn = spec.get("function")
        if isinstance(fn, dict):
            params = fn.get("parameters")
            if isinstance(params, dict):
                fn["parameters"] = _clean_tool_schema(params)
        out.append(spec)
    return out


class TrajectoryFileObserver(_BaseTrajectoryFileObserver):
    """Profile trajectory observer — clean tool schemas before logging."""

    def __init__(
        self,
        traj_dir: Any,
        *,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        cleaned = _clean_tools_for_log(tools or [])
        super().__init__(traj_dir, tools=cleaned, **kwargs)
