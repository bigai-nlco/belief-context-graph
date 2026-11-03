"""BCG runner adapter for the Unified construct backend."""

from __future__ import annotations

from typing import Any

from bcg.construct.backends import SessionBackendAdapter
from bcg.core.contracts import RunOptions

from . import llm
from .online import StreamingTrajectorySession
from .stream import StreamingBeliefBuilder, StreamOptions


def _build_options(
    options: RunOptions,
    belief_graph_config: dict[str, Any] | None,
) -> StreamOptions:
    stream_options = _session_options(options)
    if belief_graph_config:
        stream_options.apply_belief_graph_config(belief_graph_config)
    return stream_options


def _session_options(options: Any) -> Any:
    if isinstance(options, StreamOptions):
        return options
    if not isinstance(options, RunOptions):
        return options
    return StreamOptions(
        evidence_mode=options.evidence_mode,
        incremental_merge=options.incremental_merge,
        incremental_merge_threshold=options.incremental_merge_threshold,
        verify_merge=options.verify_merge,
        context_chars=options.context_chars,
        min_content_len=options.min_content_len,
    )


def _serialize_options(options: Any) -> dict[str, Any]:
    if isinstance(options, RunOptions):
        return {
            **_session_options(options).to_dict(),
            "io_context_chars": options.io_context_chars,
        }
    return options.to_dict() if hasattr(options, "to_dict") else {}


BACKEND = SessionBackendAdapter(
    name="unified",
    session_cls=StreamingTrajectorySession,
    llm_module=llm,
    options_builder=_build_options,
    session_options_builder=_session_options,
    options_serializer=_serialize_options,
    builder_cls=StreamingBeliefBuilder,
    options_cls=StreamOptions,
)

__all__ = ["BACKEND"]
