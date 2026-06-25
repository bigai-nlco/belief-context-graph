"""Langfuse tracing helpers for BCG."""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeVar, overload

ObservationType = Literal[
    "span",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "guardrail",
    "generation",
    "embedding",
]

P = ParamSpec("P")
R = TypeVar("R")

try:
    from langfuse import get_client as _get_langfuse_client
    from langfuse import observe as _langfuse_observe
except Exception:  # pragma: no cover - only used if langfuse is unavailable.
    _get_langfuse_client = None
    _langfuse_observe = None


def is_tracing_enabled() -> bool:
    """Return whether Langfuse tracing should be active."""

    enabled = os.getenv("BCG_TRACING_ENABLED", "true").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@overload
def trace(func: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def trace(
    func: None = None,
    /,
    *,
    name: str | None = None,
    as_type: ObservationType | None = None,
    capture_input: bool | None = None,
    capture_output: bool | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def trace(
    func: Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
    as_type: ObservationType | None = None,
    capture_input: bool | None = None,
    capture_output: bool | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function with Langfuse tracing when configured.

    Tracing is a no-op unless Langfuse credentials are present. This keeps local
    development and tests quiet while preserving a single decorator surface for
    production instrumentation.
    """

    def decorator(inner: Callable[P, R]) -> Callable[P, R]:
        observed: Callable[P, R] | None = None

        def get_observed() -> Callable[P, R] | None:
            nonlocal observed
            if not is_tracing_enabled() or _langfuse_observe is None:
                return None
            if observed is None:
                observed = _langfuse_observe(
                    name=name,
                    as_type=as_type,
                    capture_input=capture_input,
                    capture_output=capture_output,
                )(inner)
            return observed

        if inspect.iscoroutinefunction(inner):

            @wraps(inner)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                observed_inner = get_observed()
                if observed_inner is None:
                    return await inner(*args, **kwargs)  # type: ignore[misc]
                return await observed_inner(*args, **kwargs)  # type: ignore[misc]

            return async_wrapper  # type: ignore[return-value]

        @wraps(inner)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            observed_inner = get_observed()
            if observed_inner is None:
                return inner(*args, **kwargs)
            return observed_inner(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def update_current_span(
    *,
    input: Any | None = None,
    output: Any | None = None,
    metadata: Any | None = None,
) -> None:
    """Best-effort update for the active Langfuse span."""

    if not is_tracing_enabled() or _get_langfuse_client is None:
        return
    try:
        _get_langfuse_client().update_current_span(
            input=input,
            output=output,
            metadata=metadata,
        )
    except Exception:
        return


def update_current_generation(
    *,
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
    input: Any | None = None,
    output: Any | None = None,
    usage_details: dict[str, int] | None = None,
    metadata: Any | None = None,
) -> None:
    """Best-effort update for the active Langfuse generation."""

    if not is_tracing_enabled() or _get_langfuse_client is None:
        return
    try:
        _get_langfuse_client().update_current_generation(
            model=model,
            model_parameters=model_parameters,
            input=input,
            output=output,
            usage_details=usage_details,
            metadata=metadata,
        )
    except Exception:
        return


def flush_traces() -> None:
    """Flush pending Langfuse events when tracing is configured."""

    if not is_tracing_enabled() or _get_langfuse_client is None:
        return
    try:
        _get_langfuse_client().flush()
    except Exception:
        return
