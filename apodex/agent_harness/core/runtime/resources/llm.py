"""LLM-resolution helpers for ResourceManager."""

from __future__ import annotations

from agent_harness.core.llm import LLMClient


def resolve_base_llm_for_role(
    *,
    default_llm: LLMClient,
    role_id: str | None,  # noqa: ARG001
    cache: dict[str, LLMClient],  # noqa: ARG001
) -> LLMClient:
    """Return the bootstrap LLM. Per-role overrides were dropped in OSS."""
    return default_llm


def wrap_llm_with_middleware(
    llm: LLMClient,
    *,
    role_id: str,  # noqa: ARG001
) -> LLMClient:
    """Pass through. The optional ``LLMWrapper`` service has no OSS impl."""
    return llm
