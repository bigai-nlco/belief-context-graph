"""Compatibility bridge from unified settings to construct runtime config."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bcg.config.loader import defaults_dict, load_settings, locate_config_files
from bcg.config.schema import BCGSettings
from bcg.core.errors import BCGConfigError

LEGACY_CONFIG_PATH = "bcg/model_config.json"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Effective CLI settings and the config path passed to construct loaders."""

    settings: BCGSettings
    config_path: str | None
    uses_yaml: bool


def _explicit_config_arg(argv: Sequence[str]) -> str | None:
    args = list(argv)
    for index, value in enumerate(args):
        if value in {"--config", "-c"}:
            if index + 1 >= len(args):
                return None
            return args[index + 1]
        if value.startswith("--config="):
            return value.split("=", 1)[1]
    return None


def _is_yaml(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".yaml", ".yml"}


def resolve_runtime_config(argv: Sequence[str]) -> RuntimeConfig:
    """Resolve CLI defaults without treating a legacy JSON file as YAML."""

    explicit = _explicit_config_arg(argv)
    if explicit and not _is_yaml(explicit):
        settings = BCGSettings.model_validate(defaults_dict())
        return RuntimeConfig(settings, explicit, False)
    if explicit:
        settings, _ = load_settings(explicit=explicit)
        return RuntimeConfig(settings, explicit, True)
    if locate_config_files():
        settings, _ = load_settings()
        return RuntimeConfig(settings, None, True)
    settings = BCGSettings.model_validate(defaults_dict())
    return RuntimeConfig(settings, LEGACY_CONFIG_PATH, False)


def settings_to_construct_config(settings: BCGSettings) -> dict[str, Any]:
    """Convert typed YAML settings to the legacy shape consumed by backends."""

    raw = {
        key: value.model_dump(exclude_none=True)
        for key, value in settings.models.items()
    }
    raw["belief_graph"] = settings.pipeline.model_dump(exclude_none=True)
    return raw


def load_construct_config(
    path: str | None,
    *,
    required: bool,
) -> tuple[dict[str, Any] | None, str]:
    """Load YAML settings or a legacy JSON file into one construct shape."""

    if path is None or _is_yaml(path):
        settings, _ = load_settings(explicit=path)
        display = path or "layered YAML configuration"
        return settings_to_construct_config(settings), display

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        if required:
            raise FileNotFoundError(
                f"Missing {path}. Create a YAML configuration or the legacy "
                "model_config.json file."
            )
        return None, str(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BCGConfigError(f"invalid JSON configuration {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BCGConfigError(f"{path} must contain a JSON object")
    return raw, str(path)


__all__ = [
    "LEGACY_CONFIG_PATH",
    "RuntimeConfig",
    "load_construct_config",
    "resolve_runtime_config",
    "settings_to_construct_config",
]
