"""Legacy JSON -> unified YAML migration adapters (step 4B).

Two distinct legacy schemas exist:

1. ``model_config.json`` (project root or ``~/.bcg``): model routing
   entries (chat models + the reserved ``embedding`` key) plus an optional
   ``belief_graph`` pipeline section.
2. ``~/.bcg/config.json`` (setup wizard output): camelCase
   ``agent`` / ``context`` / ``graph`` settings.

Fallback reading is allowed only when no YAML config file exists, and every
fallback use emits a deprecation warning. ``migrate_to_yaml`` is the
explicit ``bcg config migrate`` command: it writes a new YAML file
atomically with a backup and is idempotent. Secrets are never migrated:
inline ``api_key`` values are dropped with a warning; ``api_key_env``
references are preserved.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

from bcg.config.loader import defaults_dict
from bcg.core.errors import BCGConfigError

_LEGACY_WARNING = (
    "legacy configuration detected ({paths}); the unified YAML format "
    "replaces model_config.json / ~/.bcg/config.json. Run `bcg config "
    "migrate` to convert, then remove the legacy files."
)


def _strip_comments(value: Any) -> Any:
    """Remove ``_comment`` metadata keys added because JSON has no comments."""
    if isinstance(value, dict):
        return {
            key: _strip_comments(item)
            for key, item in value.items()
            if not key.startswith("_comment")
        }
    if isinstance(value, list):
        return [_strip_comments(item) for item in value]
    return value


def _drop_inline_secrets(value: Any, *, path: str) -> Any:
    """Recursively remove inline ``api_key`` values from mappings and lists."""
    if isinstance(value, list):
        return [
            _drop_inline_secrets(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return value
    out = dict(value)
    if "api_key" in out:
        warnings.warn(
            f"dropped inline api_key in {path} (secrets belong in .env; "
            "use api_key_env instead)",
            stacklevel=3,
        )
        out.pop("api_key")
    return {
        key: _drop_inline_secrets(item, path=f"{path}.{key}")
        for key, item in out.items()
    }


def _merge_settings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge settings; mappings merge and null keeps the lower value."""
    out = dict(base)
    for key, value in override.items():
        if value is None:
            continue
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            out[key] = _merge_settings(current, value)
        else:
            out[key] = value
    return out


def migrate_model_config(path: Path) -> dict[str, Any] | None:
    """Map one legacy ``model_config.json`` onto the new settings dict."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BCGConfigError(f"invalid JSON configuration {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BCGConfigError(f"{path} must contain a JSON object")
    out: dict[str, Any] = {}
    models: dict[str, Any] = {}
    model_key: str | None = None
    embedding_key: str | None = None
    for key, value in raw.items():
        if key == "belief_graph":
            out["pipeline"] = _drop_inline_secrets(
                _strip_comments(value), path=f"{path}.belief_graph"
            )
        elif not key.startswith("_") and isinstance(value, dict):
            models[key] = _drop_inline_secrets(_strip_comments(value), path=str(path))
            if key.startswith("embedding"):
                if embedding_key is None:
                    embedding_key = key
            elif model_key is None:
                model_key = key
    if models:
        out["models"] = models
    if model_key is not None:
        out["model_key"] = model_key
    if embedding_key is not None:
        out["embedding_key"] = embedding_key
    return out or None


def migrate_user_config(
    path: Path,
    *,
    model_config_path: Path | None = None,
) -> dict[str, Any] | None:
    """Map the setup wizard's ``~/.bcg/config.json`` onto the settings dict."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BCGConfigError(f"invalid JSON configuration {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BCGConfigError(f"{path} must contain a JSON object")
    out: dict[str, Any] = {}
    graph = raw.get("graph")
    if isinstance(graph, dict):
        model_key = graph.get("modelKey")
        embedding_key = graph.get("embeddingKey")
        backend = graph.get("backend")
        if backend in {"api_based", "light"}:
            out["backend"] = backend
        if isinstance(model_key, str) and model_key.strip():
            out["model_key"] = model_key.strip()
        if isinstance(embedding_key, str) and embedding_key.strip():
            out["embedding_key"] = embedding_key.strip()
        base_url = graph.get("modelBaseUrl")
        if isinstance(base_url, str) and base_url:
            out.setdefault("models", {})
            selected_model = out.get("model_key", "graph-model")
            out["models"].setdefault(selected_model, {})
            out["models"][selected_model].setdefault("base_url", base_url)
    if model_config_path is not None and model_config_path.is_file():
        migrated = migrate_model_config(model_config_path)
        if migrated:
            out = _merge_settings(migrated, out)
    return out or None


def find_legacy_configs(
    *,
    project_root: Path | None = None,
    home: Path | None = None,
) -> list[Path]:
    """Locate legacy JSON config files (order: project, then user home)."""
    found: list[Path] = []
    root = project_root or Path.cwd()
    candidates = [
        root / "model_config.json",
        root / "bcg" / "model_config.json",
    ]
    user_root = home or Path.home()
    candidates += [
        user_root / ".bcg" / "config.json",
        user_root / ".bcg" / "model_config.json",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate not in found:
            found.append(candidate)
    return found


def legacy_settings(
    *,
    project_root: Path | None = None,
    home: Path | None = None,
) -> dict[str, Any] | None:
    """Build a settings dict from legacy JSON files, or ``None``.

    ``~/.bcg/config.json`` supplies ``backend``; ``model_config.json``
    supplies ``models`` and ``pipeline``. Emits a deprecation warning when
    any legacy file is used.
    """
    files = find_legacy_configs(project_root=project_root, home=home)
    if not files:
        return None
    user_root = (home or Path.home()).resolve()
    user_config = user_root / ".bcg" / "config.json"
    model_config = user_root / ".bcg" / "model_config.json"
    if not model_config.is_file() and project_root is not None:
        model_config = (project_root / "model_config.json").resolve()

    handled: set[Path] = set()
    merged: dict[str, Any] = {}
    if user_config.is_file():
        part = migrate_user_config(user_config, model_config_path=model_config)
        if part:
            merged = _merge_settings(merged, part)
        handled.add(user_config)
        if model_config.is_file():
            handled.add(model_config)
    for path in files:
        resolved = path.resolve()
        if resolved in handled:
            continue
        part = migrate_model_config(resolved)
        if part:
            merged = _merge_settings(merged, part)
    if not merged:
        return None
    warnings.warn(
        _LEGACY_WARNING.format(paths=", ".join(str(p) for p in files)),
        DeprecationWarning,
        stacklevel=2,
    )
    return merged


def migrate_to_yaml(
    dest: Path,
    *,
    project_root: Path | None = None,
    home: Path | None = None,
) -> Path:
    """Write a unified YAML config from legacy JSON files.

    The write is atomic (temp file + ``os.replace``); an existing ``dest``
    is backed up to ``<dest>.bak`` first. Idempotent: running it twice
    produces the same result.
    """
    settings = legacy_settings(project_root=project_root, home=home)
    if settings is None:
        raise FileNotFoundError(
            "no legacy configuration found (model_config.json / ~/.bcg/config.json)"
        )
    dest = Path(dest).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    backup = dest.with_suffix(dest.suffix + ".bak")
    if dest.exists() and not backup.exists():
        shutil.copy2(dest, backup)
    body = _merge_settings(defaults_dict(), settings)
    body["schema_version"] = 1
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            import yaml

            yaml.safe_dump(
                body,
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return dest


__all__ = [
    "find_legacy_configs",
    "legacy_settings",
    "migrate_model_config",
    "migrate_to_yaml",
    "migrate_user_config",
]
