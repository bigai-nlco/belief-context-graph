"""Project-bound configuration, deliberately separate from global Codex config."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path


def private_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = (
        value
        if isinstance(value, str)
        else json.dumps(value, indent=2, ensure_ascii=False)
    )
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Config:
    project: str
    state_dir: str
    graph_config: str
    env_file: str
    mode: str = "gateway"
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.mode != "gateway":
            raise ValueError("only gateway mode is available")
        for name in ("project", "state_dir", "graph_config", "env_file"):
            if not Path(getattr(self, name)).is_absolute():
                raise ValueError(f"{name} must be an absolute path")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")

    @classmethod
    def load(cls, path: str | Path) -> Config:
        value = json.loads(Path(path).expanduser().read_text())
        # Previous gateway configs included unused hook-only settings. Accept
        # those files without retaining the removed event-capture machinery.
        if isinstance(value, dict):
            for key in ("max_event_chars", "context_chars", "max_events_per_session"):
                value.pop(key, None)
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("invalid native Codex config")
        return cls(**value)

    def save(self, path: Path) -> None:
        private_write(path, asdict(self))


def default_graph_config(embedding_model: str) -> dict:
    return {
        "graph-model": {
            "model": "qwen/qwen3.8-flash",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_OPENAI_API_KEY",
            "reasoning_effort": "none",
            "max_tokens": 4096,
        },
        "embedding": {
            "provider": "local",
            "model": embedding_model,
            "device": "cpu",
            "batch_size": 8,
            "input_prefix": "Document: ",
        },
        "belief_graph": {
            "runtime": {"context_chars": 12000, "extraction_history_turns": 2},
            "edge_generation": {"max_previous_windows": 3},
            "tool_results": {
                "semantic_extraction": True,
                "max_semantic_calls": 40,
                "max_facts": 4,
                "max_search_results": 6,
                "max_snippet_chars": 1800,
            },
        },
    }
