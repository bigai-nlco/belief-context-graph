"""Runtime/infra configuration — Pydantic-Settings over ``.env``.

Loads from environment variables (real env + ``.env`` file). Pydantic
defaults are the lowest-priority source.

Tool secrets read directly from ``os.environ`` rather than going through
this class (see ``plugins/tools/*.py``): ``SERPER_API_KEY``,
``JINA_API_KEY``, ``E2B_API_KEY``, ``SUMMARY_LLM_*``, ``JUDGE_*``. Only
the agent-LLM provider knobs need typed Pydantic fields here, because
``create_llm`` switches on them.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ``$VAR`` and ``${VAR:-default}`` in strings.

    Used by ``workflows/react_base/profile.py`` to interpolate env vars
    into profile YAML before pydantic typing kicks in.
    """
    if isinstance(value, str):
        def _replace_braced(m: re.Match) -> str:
            var_name = m.group(1)
            default = m.group(3) if m.group(3) is not None else ""
            return os.environ.get(var_name, default)

        value = re.sub(r"\$\{(\w+)(:-([^}]*))?\}", _replace_braced, value)

        def _replace_simple(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(0))

        value = re.sub(r"\$([A-Z_][A-Z0-9_]*)", _replace_simple, value)
        return value

    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]

    return value


class AgentHarnessConfig(BaseSettings):
    """Agent-LLM provider configuration.

    Reads from environment variables (case-insensitive) + ``.env``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_provider: str = "openai"  # openai | anthropic | qwen
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-max"

    llm_max_tokens: int = 16384  # per-call max output tokens


_config: AgentHarnessConfig | None = None


def get_config(force_reload: bool = False) -> AgentHarnessConfig:
    """Singleton config accessor."""
    global _config
    if _config is None or force_reload:
        _config = AgentHarnessConfig()
    return _config
