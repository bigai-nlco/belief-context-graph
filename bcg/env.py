"""Project-wide environment loading from one user-controlled ``.env`` file."""

from __future__ import annotations

import os
from pathlib import Path

SOURCE_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_project_env() -> Path:
    """Locate the shared env file for source and ``uv tool`` installations.

    ``BCG_ENV_FILE`` has highest priority. Otherwise a ``.env`` in the current
    working directory wins, followed by the source checkout's root ``.env``.
    """

    configured = os.environ.get("BCG_ENV_FILE")
    if configured:
        return Path(configured).expanduser().resolve()

    working_env = Path.cwd() / ".env"
    source_env = SOURCE_PROJECT_ROOT / ".env"
    if working_env.is_file():
        return working_env
    if source_env.is_file():
        return source_env
    return working_env


PROJECT_ENV_FILE = find_project_env()
PROJECT_ROOT = PROJECT_ENV_FILE.parent


def read_env_file(path: str | Path = PROJECT_ENV_FILE) -> dict[str, str]:
    """Parse a small dotenv file without executing shell syntax."""

    env_path = Path(path).expanduser()
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name] = value
    return values


def load_project_env(
    path: str | Path = PROJECT_ENV_FILE,
    *,
    override: bool = False,
) -> dict[str, str]:
    """Load dotenv values into ``os.environ`` and return values applied."""

    loaded: dict[str, str] = {}
    for name, value in read_env_file(path).items():
        if override or name not in os.environ:
            os.environ[name] = value
            loaded[name] = value
    return loaded


def resolve_config_api_key(
    config: dict[str, object],
    *,
    default_env: str,
    config_path: str,
) -> None:
    """Resolve an API key referenced by a non-secret model config.

    ``api_key_env`` names the variable in the project-root ``.env``. A legacy
    inline ``api_key`` remains a fallback so existing private configs keep
    working, but templates and documentation must use ``api_key_env``.
    """

    load_project_env()
    env_name = config.get("api_key_env") or default_env
    if not isinstance(env_name, str) or not env_name.strip():
        raise ValueError(
            f"Config field 'api_key_env' must be a non-empty environment "
            f"variable name in {config_path}."
        )
    env_name = env_name.strip()
    legacy_key = config.get("api_key")
    api_key = os.environ.get(env_name) or (
        legacy_key if isinstance(legacy_key, str) else ""
    )
    if not api_key.strip():
        raise ValueError(
            f"API key environment variable {env_name!r} is empty. Add it to "
            f"the project root {PROJECT_ENV_FILE.name} file."
        )
    config["api_key_env"] = env_name
    config["api_key"] = api_key


# Importing any ``bcg`` module initializes this submodule before the rest of
# the package, so credential-backed dataclass defaults see the root .env.
load_project_env()


__all__ = [
    "PROJECT_ENV_FILE",
    "PROJECT_ROOT",
    "SOURCE_PROJECT_ROOT",
    "find_project_env",
    "load_project_env",
    "read_env_file",
    "resolve_config_api_key",
]
