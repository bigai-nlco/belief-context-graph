"""Launch the interactive BCG agent and its graph-construction service."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from bcg.core.env import PROJECT_ROOT, SOURCE_PROJECT_ROOT

DEFAULT_GRAPH_URL = "http://127.0.0.1:8848"
DEFAULT_GRAPH_BACKEND = "light"
DEFAULT_RECENT_TURNS = 2
DEFAULT_GRAPH_MAX_TURNS = 300
DEFAULT_GRAPH_TIMEOUT_MS = 300_000
GENERATED_PROVIDER = "bcg"
LEGACY_GENERATED_PROVIDER = "bcg-openai"


class AgentLaunchError(RuntimeError):
    """Raised when the interactive BCG runtime cannot be started."""


def _bootstrap_env() -> None:
    """Load the project root .env explicitly (import no longer does this)."""
    from bcg.core.env import load_project_env

    load_project_env()



def _state_root() -> Path:
    configured = os.environ.get("BCG_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".bcg"


def _agent_dir() -> Path:
    configured = os.environ.get("BCG_CODING_AGENT_DIR")
    return Path(configured).expanduser() if configured else _state_root() / "agent"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentLaunchError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AgentLaunchError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def ensure_agent_configuration(graph_url: str) -> Path:
    """Configure BCG context defaults and persistent model routing."""

    agent_dir = _agent_dir()
    settings_path = agent_dir / "settings.json"
    settings = _read_json(settings_path)

    context = settings.get("contextManagement")
    if not isinstance(context, dict):
        context = {}
    bcg_settings = context.get("bcg")
    if not isinstance(bcg_settings, dict):
        bcg_settings = {}

    try:
        recent_turns = int(
            os.environ.get("BCG_RECENT_TURNS", str(DEFAULT_RECENT_TURNS))
        )
        timeout_ms = int(
            os.environ.get(
                "BCG_GRAPH_TIMEOUT_MS",
                str(DEFAULT_GRAPH_TIMEOUT_MS),
            )
        )
        max_turns = int(
            os.environ.get(
                "BCG_GRAPH_MAX_TURNS",
                str(DEFAULT_GRAPH_MAX_TURNS),
            )
        )
    except ValueError as exc:
        raise AgentLaunchError(
            "BCG_RECENT_TURNS, BCG_GRAPH_MAX_TURNS, and "
            "BCG_GRAPH_TIMEOUT_MS must be integers."
        ) from exc

    bcg_settings.update(
        {
            "url": graph_url,
            "recentTurns": recent_turns,
            "maxTurns": max(1, max_turns),
            "timeoutMs": timeout_ms,
            "includeRelations": True,
        }
    )
    configured_context_provider = os.environ.get("BCG_CONTEXT_MODE", "").strip()
    if configured_context_provider in {"default", "bcg"}:
        context["provider"] = configured_context_provider
    elif context.get("provider") not in {"default", "bcg"}:
        context["provider"] = "bcg"
    context["bcg"] = bcg_settings
    settings["contextManagement"] = context
    settings.setdefault("defaultThinkingLevel", "off")
    settings["enableInstallTelemetry"] = False

    model_id = (
        os.environ.get("BCG_AGENT_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("MODEL")
        or ""
    ).strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    agent_provider = os.environ.get("BCG_AGENT_PROVIDER", "").strip()
    if model_id and base_url:
        models_path = agent_dir / "models.json"
        models = _read_json(models_path)
        providers = models.get("providers")
        if not isinstance(providers, dict):
            providers = {}
        providers.pop(LEGACY_GENERATED_PROVIDER, None)
        providers[GENERATED_PROVIDER] = {
            "baseUrl": base_url,
            "api": "openai-completions",
            "apiKey": "$OPENAI_API_KEY",
            "authHeader": True,
            "models": [{"id": model_id, "name": model_id}],
        }
        models["providers"] = providers
        _write_json(models_path, models)
        settings["defaultProvider"] = GENERATED_PROVIDER
        settings["defaultModel"] = model_id
    elif model_id:
        settings["defaultProvider"] = agent_provider or "openai"
        settings["defaultModel"] = model_id

    _write_json(settings_path, settings)
    return agent_dir


def _health_url(graph_url: str) -> str:
    return f"{graph_url.rstrip('/')}/health"


def graph_server_is_ready(graph_url: str, timeout: float = 1.0) -> bool:
    try:
        with urlopen(_health_url(graph_url), timeout=timeout) as response:
            if response.status != 200:
                return False
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("status") == "ok"


def _resolve_graph_config() -> Path:
    configured = os.environ.get("BCG_GRAPH_CONFIG")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path.cwd() / "bcg" / "model_config.json",
        Path.cwd() / "model_config.json",
        PROJECT_ROOT / "bcg" / "model_config.json",
        SOURCE_PROJECT_ROOT / "bcg" / "model_config.json",
        _state_root() / "model_config.json",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    searched = "\n  ".join(str(path) for path in candidates if path is not None)
    raise AgentLaunchError(
        "The Graph Construction server is not running and no model_config.json "
        "was found. Create it with\n"
        "  cp bcg/model_config.example.json bcg/model_config.json\n"
        "or set BCG_GRAPH_CONFIG to its path. Searched:\n"
        f"  {searched}"
    )


def _local_server_address(graph_url: str) -> tuple[str, int]:
    parsed = urlparse(graph_url)
    hostname = parsed.hostname
    if parsed.scheme not in {"http", "https"} or hostname is None:
        raise AgentLaunchError(
            f"BELIEF_GRAPH_URL must be an HTTP URL, got {graph_url!r}."
        )
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise AgentLaunchError(
            f"The remote Graph server at {graph_url} is unavailable. "
            "BCG only auto-starts local Graph servers."
        )
    if parsed.scheme == "https":
        raise AgentLaunchError(
            "BCG cannot auto-start a local HTTPS Graph server; use an HTTP "
            "BELIEF_GRAPH_URL or start the HTTPS service separately."
        )
    return hostname, parsed.port or 80


def ensure_graph_server(graph_url: str) -> None:
    """Reuse a healthy graph server or start a persistent local one."""

    if graph_server_is_ready(graph_url):
        return

    auto_start = os.environ.get("BCG_GRAPH_AUTOSTART", "true").strip().lower()
    if auto_start in {"0", "false", "no", "off"}:
        raise AgentLaunchError(
            f"The configured existing Graph server at {graph_url} is "
            "unavailable. Start that server or run `bcg setup` to let BCG "
            "manage a local Graph server."
        )

    host, port = _local_server_address(graph_url)
    config_path = _resolve_graph_config()
    backend = os.environ.get("BCG_GRAPH_BACKEND", DEFAULT_GRAPH_BACKEND).strip()
    if backend not in {"light", "api_based"}:
        raise AgentLaunchError(
            "BCG_GRAPH_BACKEND must be either 'light' or 'api_based'."
        )

    state_root = _state_root()
    log_dir = state_root / "logs"
    graph_dir = state_root / "graphs"
    log_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "graph-server.log"
    pid_path = state_root / "graph-server.pid"

    command = [
        sys.executable,
        "-m",
        "bcg.online_server",
        backend,
        "--host",
        host,
        "--port",
        str(port),
        "--config",
        str(config_path),
        "--output-dir",
        str(graph_dir),
        "--model-key",
        os.environ.get("BCG_GRAPH_MODEL_KEY", "gpt-5.5"),
        "--embedding-key",
        os.environ.get("BCG_GRAPH_EMBEDDING_KEY", "embedding"),
        "--quiet",
    ]
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=PROJECT_ROOT,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")

    try:
        startup_timeout = float(os.environ.get("BCG_GRAPH_START_TIMEOUT", "120"))
    except ValueError as exc:
        raise AgentLaunchError(
            "BCG_GRAPH_START_TIMEOUT must be a number of seconds."
        ) from exc
    deadline = time.monotonic() + max(1.0, startup_timeout)
    print(
        f"Starting BCG Graph Construction server at {graph_url} (log: {log_path})...",
        file=sys.stderr,
    )
    while time.monotonic() < deadline:
        if graph_server_is_ready(graph_url):
            print("BCG Graph Construction server is ready.", file=sys.stderr)
            return
        return_code = process.poll()
        if return_code is not None:
            raise AgentLaunchError(
                f"Graph Construction server exited with code {return_code}. "
                f"See {log_path}."
            )
        time.sleep(0.25)

    process.terminate()
    raise AgentLaunchError(
        f"Graph Construction server did not become ready within "
        f"{startup_timeout:g}s. See {log_path}."
    )


def _resolve_agent_command() -> list[str]:
    configured = os.environ.get("BCG_AGENT_COMMAND")
    if configured:
        command = shlex.split(configured)
        if not command:
            raise AgentLaunchError("BCG_AGENT_COMMAND is empty.")
        return command

    installed = shutil.which("bcg-agent")
    if installed:
        return [installed]

    source_candidates = [
        PROJECT_ROOT / "agent-cli" / "dist" / "cli.js",
        SOURCE_PROJECT_ROOT / "agent-cli" / "dist" / "cli.js",
    ]
    node = shutil.which("node")
    if node:
        for candidate in source_candidates:
            if candidate.is_file():
                return [node, str(candidate.resolve())]

    raise AgentLaunchError(
        "The BCG Agent runtime is not installed. Install it with\n"
        "  cd agent-cli && npm install && npm run build && npm install -g .\n"
        "or set BCG_AGENT_COMMAND to an installed BCG Agent executable."
    )


def _configured_model_arguments(arguments: list[str]) -> list[str]:
    if any(
        argument in {"--provider", "--model"}
        or argument.startswith("--provider=")
        or argument.startswith("--model=")
        for argument in arguments
    ):
        return []
    model_id = (
        os.environ.get("BCG_AGENT_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("MODEL")
        or ""
    ).strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    agent_provider = os.environ.get("BCG_AGENT_PROVIDER", "").strip()
    if model_id:
        provider = GENERATED_PROVIDER if base_url else (agent_provider or "openai")
        return ["--provider", provider, "--model", model_id]
    return []


def launch_interactive(arguments: list[str] | None = None) -> int:
    """Start/reuse the graph service, then hand the terminal to the Agent TUI."""

    agent_arguments = arguments or []
    informational = any(
        argument in {"--help", "-h", "--version", "-v"} for argument in agent_arguments
    )
    if informational:
        agent_dir = _agent_dir()
        model_arguments: list[str] = []
        graph_url = os.environ.get("BELIEF_GRAPH_URL", DEFAULT_GRAPH_URL).strip()
        setup_config: dict[str, Any] = {}
    else:
        from bcg.apps.setup import ensure_user_setup

        setup_config, _created = ensure_user_setup()
        graph_url = os.environ.get("BELIEF_GRAPH_URL", DEFAULT_GRAPH_URL).strip()
        ensure_graph_server(graph_url)
        agent_dir = ensure_agent_configuration(graph_url)
        model_arguments = _configured_model_arguments(agent_arguments)

    environment = os.environ.copy()
    environment["BELIEF_GRAPH_URL"] = graph_url
    environment["BCG_CODING_AGENT_DIR"] = str(agent_dir)
    pending_login: str | None = None
    if not informational:
        from bcg.apps.setup import pending_login_provider

        pending_login = pending_login_provider(setup_config)
        if pending_login:
            environment["BCG_START_LOGIN_PROVIDER"] = pending_login
    result = subprocess.run(
        [*_resolve_agent_command(), *model_arguments, *agent_arguments],
        env=environment,
        check=False,
    )
    if pending_login:
        from bcg.apps.setup import mark_login_prompt_consumed

        mark_login_prompt_consumed()
    return result.returncode


def main(arguments: list[str] | None = None) -> int:
    _bootstrap_env()
    from bcg.apps.setup import SetupError

    try:
        return launch_interactive(arguments)
    except (AgentLaunchError, SetupError) as exc:
        print(f"bcg: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "AgentLaunchError",
    "ensure_agent_configuration",
    "ensure_graph_server",
    "graph_server_is_ready",
    "launch_interactive",
    "main",
]
