"""Explicit configuration; no credentials are copied from the existing agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from urllib.parse import urlsplit

import yaml


def base_url(value: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "URLs must be HTTP(S) base URLs without credentials or query strings"
        )
    result = value.rstrip("/")
    if result.endswith("/chat/completions"):
        raise ValueError("This gateway requires a Responses API, not Chat Completions")
    return result.removesuffix("/responses")


@dataclass(frozen=True)
class ProxyConfig:
    upstream_base_url: str
    graph_base_url: str = "http://127.0.0.1:8848"
    upstream_api_key_env: str = "BCG_CODEX_UPSTREAM_API_KEY"
    gateway_token_env: str = "BCG_CODEX_GATEWAY_TOKEN"
    graph_api_key_env: str | None = None
    host: str = "127.0.0.1"
    port: int = 8860
    mode: str = "bcg"
    recent_turns: int = 2
    node_char_budget: int = 6600
    graph_timeout: float = 300.0
    upstream_timeout: float = 300.0
    max_sessions: int = 128
    max_request_bytes: int = 32 * 1024 * 1024
    request_reasoning_summary: bool = True
    audit_dir: str | None = "var/codex-proxy"

    def __post_init__(self) -> None:
        object.__setattr__(self, "upstream_base_url", base_url(self.upstream_base_url))
        object.__setattr__(self, "graph_base_url", base_url(self.graph_base_url))
        if self.host not in {"127.0.0.1", "localhost"}:
            raise ValueError(
                "The prototype binds only to loopback; public exposure is unsupported"
            )
        if type(self.port) is not int or not 0 <= self.port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if self.mode not in {"bcg", "default"}:
            raise ValueError("mode must be bcg or default")
        if type(self.recent_turns) is not int or self.recent_turns < 0:
            raise ValueError("recent_turns must be a non-negative integer")
        for name in ("node_char_budget", "max_sessions", "max_request_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.node_char_budget < 256:
            raise ValueError("node_char_budget must be at least 256")
        if self.graph_timeout <= 0 or self.upstream_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if type(self.request_reasoning_summary) is not bool:
            raise ValueError("request_reasoning_summary must be a boolean")

    @classmethod
    def load(cls, path: Path) -> ProxyConfig:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("The proxy config must be a YAML mapping")
        unknown = value.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(
                f"Unknown configuration fields: {', '.join(sorted(unknown))}"
            )
        return cls(**value)

    def secret(self, env_name: str) -> str:
        value = os.environ.get(env_name, "").strip()
        if not value:
            raise ValueError(
                f"Set environment variable {env_name} before starting the proxy"
            )
        return value
