"""Layered YAML loading, validation and merge for unified BCG settings.

Precedence (highest first):

    explicit CLI fields
    > --config file
    > BCG_CONFIG file
    > project bcg.yaml
    > user ~/.bcg/config.yaml
    > packaged defaults.yaml

Deep-merge rules: mappings merge recursively; lists replace wholesale; a
``null`` value means "fall back" (the key is skipped, so the lower layer
keeps its value). ``load_settings`` returns the validated settings plus a
field-level source map for diagnostics (``bcg config show``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from bcg.config.schema import BCGSettings

DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"

# File-name candidates searched in the working directory, in order.
PROJECT_CONFIG_NAMES = ("bcg.yaml", "config.yaml")


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load one YAML file as a dict; missing files yield ``{}``."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    data = yaml.safe_load(raw)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping")
    return data


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
    *,
    source: str,
    sources: dict[str, str],
    prefix: str = "",
) -> dict[str, Any]:
    """Merge ``override`` into ``base`` and record field-level sources.

    Mappings merge recursively; lists and scalars replace wholesale; ``null``
    overrides fall back to the base value (key is skipped).
    """
    out = dict(base)
    for key, value in override.items():
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if value is None:
            continue  # null == fall back to lower layer
        current = out.get(key)
        if isinstance(value, dict):
            child_base = current if isinstance(current, dict) else {}
            out[key] = _deep_merge(
                child_base, value, source=source, sources=sources, prefix=path
            )
        else:
            out[key] = value
            sources[path] = source
    return out


def defaults_dict() -> dict[str, Any]:
    """Packaged defaults as a plain dict (never mutated by callers)."""
    return _load_yaml_file(DEFAULTS_PATH)


def locate_config_files(
    *,
    explicit: str | None = None,
    env_name: str = "BCG_CONFIG",
    project_names: tuple[str, ...] = PROJECT_CONFIG_NAMES,
    home: Path | None = None,
) -> list[Path]:
    """Resolve the config file chain from highest to lowest precedence.

    Missing files are skipped; the returned list contains only files that
    exist, ordered from highest to lowest precedence.
    """
    found: list[Path] = []
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_file():
            found.append(path)
    configured = os.environ.get(env_name)
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file() and path not in found:
            found.append(path)
    for name in project_names:
        path = (Path.cwd() / name).resolve()
        if path.is_file() and path not in found:
            found.append(path)
    user_root = home or Path.home()
    user_path = (user_root / ".bcg" / "config.yaml").resolve()
    if user_path.is_file() and user_path not in found:
        found.append(user_path)
    return found


def load_settings(
    *,
    explicit: str | None = None,
    env_name: str = "BCG_CONFIG",
    project_names: tuple[str, ...] = PROJECT_CONFIG_NAMES,
    home: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> tuple[BCGSettings, dict[str, str]]:
    """Load, merge, validate and return ``(settings, field_sources)``.

    ``cli_overrides`` are applied last (highest precedence); only keys that
    are not ``None`` override the merged layers.
    """
    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}

    chain = list(reversed(locate_config_files(
        explicit=explicit, env_name=env_name,
        project_names=project_names, home=home,
    )))
    chain.append("packaged defaults")  # lowest precedence last
    merged = _deep_merge(merged, defaults_dict(), source="packaged defaults", sources=sources)
    for path in chain:
        if path == "packaged defaults":
            continue
        merged = _deep_merge(
            merged, _load_yaml_file(path), source=str(path), sources=sources
        )
    if cli_overrides:
        merged = _deep_merge(
            merged,
            {k: v for k, v in cli_overrides.items() if v is not None},
            source="cli",
            sources=sources,
        )
    return BCGSettings.model_validate(merged), sources


__all__ = [
    "DEFAULTS_PATH",
    "PROJECT_CONFIG_NAMES",
    "defaults_dict",
    "load_settings",
    "locate_config_files",
]
