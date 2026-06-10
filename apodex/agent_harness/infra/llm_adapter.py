"""LLM provider factory — returns an :class:`LLMClient` for the configured
provider."""

from __future__ import annotations

import logging

from agent_harness.core.llm import LLMClient
from agent_harness.infra.config import AgentHarnessConfig

logger = logging.getLogger(__name__)


def create_llm(config: AgentHarnessConfig) -> LLMClient:
    """Create an LLM client based on the configured provider."""
    provider = config.llm_provider.lower()
    max_tokens = config.llm_max_tokens

    if provider == "openai":
        from agent_harness.infra.openai_client import OpenAIClient
        base_url = config.openai_base_url
        default_headers: dict[str, str] | None = None
        if base_url and base_url != "https://api.openai.com/v1":
            default_headers = {
                "HTTP-Referer": "agent_harness",
                "X-Title": "AgentHarness",
            }
        else:
            base_url = None  # let SDK use its own default
        return OpenAIClient(
            model=config.openai_model,
            api_key=config.openai_api_key,
            base_url=base_url,
            temperature=0.3,
            max_completion_tokens=max_tokens,
            default_headers=default_headers,
        )

    if provider == "qwen":
        from agent_harness.infra.openai_client import OpenAIClient
        return OpenAIClient(
            model=config.qwen_model,
            api_key=config.qwen_api_key,
            base_url=config.qwen_base_url,
            temperature=0.3,
            max_completion_tokens=max_tokens,
        )

    if provider == "anthropic":
        from agent_harness.infra.anthropic_client import AnthropicClient
        return AnthropicClient(
            model=config.anthropic_model,
            api_key=config.anthropic_api_key,
            temperature=0.3,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
