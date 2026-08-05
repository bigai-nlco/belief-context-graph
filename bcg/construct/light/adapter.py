"""BCG runner adapter for the light construct backend."""

from __future__ import annotations

from typing import Any

from bcg.construct.backends import SessionBackendAdapter
from bcg.core.contracts import RunOptions

from . import llm
from .online import StreamingTrajectorySession
from .stream import StreamOptions


def _build_options(
    options: RunOptions,
    belief_graph_config: dict[str, Any] | None,
) -> StreamOptions:
    stream_options = StreamOptions(
        evidence_mode=options.evidence_mode,
        incremental_merge=options.incremental_merge,
        incremental_merge_threshold=options.incremental_merge_threshold,
        context_chars=options.context_chars,
        min_content_len=options.min_content_len,
    )
    if belief_graph_config:
        stream_options.apply_belief_graph_config(belief_graph_config)
    return stream_options


def _session_options(options: Any) -> Any:
    return options


def _serialize_options(options: Any) -> dict[str, Any]:
    return options.to_dict() if hasattr(options, "to_dict") else {}


BACKEND = SessionBackendAdapter(
    name="light",
    session_cls=StreamingTrajectorySession,
    llm_module=llm,
    options_builder=_build_options,
    session_options_builder=_session_options,
    options_serializer=_serialize_options,
)

__all__ = ["BACKEND"]
