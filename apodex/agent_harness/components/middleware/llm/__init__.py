"""LLM middleware: base types + lazy-loaded concrete impls."""

from agent_harness.components.middleware.llm.base import (
    _RETRYABLE_KEYWORDS,
    LLMCallContext,
    LLMMiddleware,
    LLMMiddlewareChain,
    _is_retryable,
)
from agent_harness.components.middleware.llm.proxy import LLMProxy


def __getattr__(name: str):
    if name == "SummarizationMiddleware":
        from agent_harness.components.middleware.llm.summarization import (
            SummarizationMiddleware,
        )
        return SummarizationMiddleware
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "_RETRYABLE_KEYWORDS",
    "LLMCallContext",
    "LLMMiddleware",
    "LLMMiddlewareChain",
    "LLMProxy",
    "SummarizationMiddleware",
    "_is_retryable",
]
