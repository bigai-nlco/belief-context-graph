"""Event infrastructure — in-memory async bus only.

The durable ``EventStore`` moved to ``agent_harness.state.event_store.sqlite``.
This package no longer imports it during normal ``core/runtime`` imports so
stateless runtime users do not pull in state/sqlalchemy transitively.
"""

from agent_harness.core.runtime.events.bus import EventBus

__all__ = ["EventBus"]


def __getattr__(name: str):
    """Lazy legacy access for ``EventStore`` without eager state import."""
    if name == "EventStore":
        import importlib
        import warnings

        warnings.warn(
            "agent_harness.core.runtime.events.EventStore is deprecated; "
            "use agent_harness.state.event_store.sqlite.EventStore instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.import_module(
            "agent_harness.state.event_store.sqlite"
        ).EventStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
