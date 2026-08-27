"""Persistent first-run configuration for the globally installed BCG CLI."""

from __future__ import annotations

import copy
import getpass
import json
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

CONFIG_VERSION = 1
GRAPH_MODEL_KEY = "graph-model"
EMBEDDING_KEY = "embedding"
DEFAULT_AGENT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_AGENT_MODEL = "gpt-4.1-mini"
DEFAULT_SUMMARY_MAX_TOKENS = 2048
DEFAULT_SUMMARY_TIMEOUT_MS = 300_000
DEFAULT_GRAPH_URL = "http://127.0.0.1:8848"
DEFAULT_VLLM_URL = "http://127.0.0.1:8001/v1"
DEFAULT_VLLM_MODEL = "Qwen3.5-4B"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_STANCE_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
_LEGACY_GRAPH_BACKENDS = {"api_based": "unified", "light": "hybrid"}
_CONSOLE = Console()


class SetupError(RuntimeError):
    """Raised when global BCG configuration cannot be completed."""


def state_root() -> Path:
    configured = os.environ.get("BCG_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".bcg"


def config_path() -> Path:
    return state_root() / "config.json"


def credentials_path() -> Path:
    return state_root() / ".env"


def model_config_path() -> Path:
    """Legacy JSON Graph configuration path retained for fallback reads."""
    return state_root() / "model_config.json"


def graph_config_path() -> Path:
    """Unified YAML Graph configuration written by setup."""
    return state_root() / "config.yaml"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SetupError(f"Expected a JSON object in {path}.")
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


def _read_credentials() -> dict[str, str]:
    from bcg.core.env import read_env_file

    return read_env_file(credentials_path())


def _write_user_yaml_config(graph_model_config: dict[str, Any]) -> None:
    """Write setup's model/pipeline settings into ~/.bcg/config.yaml.

    The YAML settings file is the unified configuration source; the legacy
    model_config.json is no longer written (readers keep a fallback window).
    """
    from bcg.config.loader import defaults_dict
    from bcg.config.schema import BCGSettings

    def _strip_comments(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: _strip_comments(item)
                for key, item in value.items()
                if not key.startswith("_comment")
            }
        if isinstance(value, list):
            return [_strip_comments(item) for item in value]
        return value

    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Merge setup overrides without discarding nested packaged defaults."""
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    settings_dict: dict[str, Any] = {
        "schema_version": 1,
        "backend": ("hybrid" if "belief_graph" in graph_model_config else "unified"),
        # keep the default model/embedding keys aligned with what we write
        "model_key": GRAPH_MODEL_KEY,
        "embedding_key": EMBEDDING_KEY,
        "models": {
            key: _strip_comments(value)
            for key, value in graph_model_config.items()
            if key != "belief_graph"
        },
    }
    pipeline = _strip_comments(graph_model_config.get("belief_graph"))
    if isinstance(pipeline, dict) and pipeline:
        settings_dict["pipeline"] = pipeline
    merged = _deep_merge(defaults_dict(), settings_dict)
    BCGSettings.model_validate(merged)  # setup output must be consumable

    import yaml

    dest = graph_config_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(settings_dict, dest.open("w", encoding="utf-8"), sort_keys=False)


def _write_credentials(values: dict[str, str]) -> None:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BCG credentials. This file is managed by `bcg setup`.",
        "# Keep it private and do not commit it.",
    ]
    for name, value in sorted(values.items()):
        if "\n" in value or "\r" in value:
            raise SetupError(f"Credential {name} cannot contain a newline.")
        lines.append(f"{name}={value}")
    lines.append("")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _ask(
    label: str,
    *,
    default: str | None = None,
    input_fn: Callable[[str], str] = input,
) -> str:
    if _supports_rich_input(input_fn):
        while True:
            value = Prompt.ask(
                f"[bold cyan]{label}[/]",
                default=default,
                console=_CONSOLE,
            ).strip()
            if value:
                return value
            if default is not None:
                return default
            _CONSOLE.print("[red]A value is required.[/]")

    suffix = f" [{default}]" if default else ""
    while True:
        value = input_fn(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        print("A value is required.")


def _ask_secret(
    label: str,
    *,
    existing: str | None = None,
    secret_fn: Callable[[str], str] = getpass.getpass,
) -> str:
    if secret_fn is getpass.getpass and sys.stdin.isatty() and sys.stdout.isatty():
        hint = " [dim](Enter keeps the saved value)[/]" if existing else ""
        while True:
            value = Prompt.ask(
                f"[bold cyan]{label}[/]{hint}",
                password=True,
                console=_CONSOLE,
            ).strip()
            if value:
                return value
            if existing:
                return existing
            _CONSOLE.print(
                "[red]A value is required. Use EMPTY for an "
                "unauthenticated endpoint.[/]"
            )

    suffix = " [press Enter to keep the saved value]" if existing else ""
    while True:
        value = secret_fn(f"{label}{suffix}: ").strip()
        if value:
            return value
        if existing:
            return existing
        print("A value is required. Use EMPTY for an unauthenticated endpoint.")


def _choose(
    label: str,
    choices: list[tuple[str, str]],
    *,
    default: str,
    input_fn: Callable[[str], str] = input,
) -> str:
    if _supports_rich_input(input_fn):
        return _choose_interactively(label, choices, default=default)

    print(f"\n{label}")
    by_number: dict[str, str] = {}
    valid_values = {value for value, _description in choices}
    for index, (value, description) in enumerate(choices, start=1):
        marker = " (default)" if value == default else ""
        print(f"  {index}. {description}{marker}")
        by_number[str(index)] = value
    while True:
        answer = input_fn("Select: ").strip().lower()
        if not answer:
            return default
        if answer in by_number:
            return by_number[answer]
        if answer in valid_values:
            return answer
        print("Enter one of the listed numbers or names.")


def _confirm(
    label: str,
    *,
    default: bool = True,
    input_fn: Callable[[str], str] = input,
) -> bool:
    if _supports_rich_input(input_fn):
        return Confirm.ask(
            f"[bold cyan]{label}[/]",
            default=default,
            console=_CONSOLE,
        )

    hint = "Y/n" if default else "y/N"
    while True:
        answer = input_fn(f"{label} [{hint}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Enter y or n.")


def _supports_rich_input(input_fn: Callable[[str], str]) -> bool:
    return (
        os.name == "posix"
        and input_fn is input
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )


@contextmanager
def _terminal_cbreak():
    """Read individual keys while always restoring the user's terminal."""

    import termios
    import tty

    file_descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(file_descriptor)
    try:
        tty.setcbreak(file_descriptor)
        yield file_descriptor
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous)


def _read_selector_key(file_descriptor: int) -> str:
    first = os.read(file_descriptor, 1)
    if first in {b"\r", b"\n"}:
        return "enter"
    if first == b"\x03":
        raise KeyboardInterrupt
    if first == b"\x1b":
        sequence = os.read(file_descriptor, 2)
        if sequence.endswith(b"A"):
            return "up"
        if sequence.endswith(b"B"):
            return "down"
        return "unknown"
    try:
        return first.decode("utf-8").lower()
    except UnicodeDecodeError:
        return "unknown"


def _selector_panel(
    label: str,
    choices: list[tuple[str, str]],
    *,
    selected: int,
    default: str,
) -> Panel:
    content = Text()
    content.append("Use ↑/↓ to move · Enter to select\n\n", style="dim")
    for index, (value, description) in enumerate(choices):
        active = index == selected
        content.append(
            "  ❯ " if active else "    ", style="bold cyan" if active else ""
        )
        content.append(
            description,
            style="bold bright_white on rgb(30,55,75)" if active else "white",
        )
        if value == default:
            content.append("  default", style="dim cyan")
        if index < len(choices) - 1:
            content.append("\n")
    return Panel(
        content,
        title=f"[bold]{label}[/]",
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
    )


def _choose_interactively(
    label: str,
    choices: list[tuple[str, str]],
    *,
    default: str,
) -> str:
    selected = next(
        (
            index
            for index, (value, _description) in enumerate(choices)
            if value == default
        ),
        0,
    )
    with (
        Live(
            _selector_panel(label, choices, selected=selected, default=default),
            console=_CONSOLE,
            transient=True,
            auto_refresh=False,
        ) as live,
        _terminal_cbreak() as file_descriptor,
    ):
        while True:
            key = _read_selector_key(file_descriptor)
            if key in {"up", "k"}:
                selected = (selected - 1) % len(choices)
            elif key in {"down", "j"}:
                selected = (selected + 1) % len(choices)
            elif key == "enter":
                break
            elif key.isdigit() and 1 <= int(key) <= len(choices):
                selected = int(key) - 1
                break
            live.update(
                _selector_panel(
                    label,
                    choices,
                    selected=selected,
                    default=default,
                ),
                refresh=True,
            )
    value, description = choices[selected]
    _CONSOLE.print(f"[green]✓[/] [bold]{label}[/]\n  [cyan]{description}[/]\n")
    return value


def _embedding_entry(model: str) -> dict[str, Any]:
    return {
        "provider": "local",
        "model": model,
        "device": "auto",
        "dtype": "auto",
        "batch_size": 8,
        "max_length": 8192,
        "trust_remote_code": True,
        "input_prefix": "Document: ",
        "model_kwargs": {},
    }


def build_api_graph_config(
    *,
    base_url: str,
    model: str,
    api_key_env: str,
    embedding_model: str | None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        GRAPH_MODEL_KEY: {
            "base_url": base_url,
            "api_key_env": api_key_env,
            "model": model,
            "max_tokens": 65536,
            "temperature": 0,
        }
    }
    if embedding_model:
        config[EMBEDDING_KEY] = _embedding_entry(embedding_model)
    return config


def _load_hybrid_template() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "model_config.example.json"
    template = _read_json(path)
    if not template:
        raise SetupError(f"Packaged hybrid-backend template is missing: {path}")
    return template


def build_hybrid_graph_config(
    *,
    base_url: str,
    model: str,
    api_key_env: str,
    embedding_model: str,
    stance_model: str,
) -> dict[str, Any]:
    template = _load_hybrid_template()
    chat_entry = copy.deepcopy(template.get("gpt-5.5"))
    belief_graph = copy.deepcopy(template.get("belief_graph"))
    if not isinstance(chat_entry, dict) or not isinstance(belief_graph, dict):
        raise SetupError("The packaged hybrid-backend template is invalid.")

    chat_entry.update(
        {
            "base_url": base_url,
            "api_key_env": api_key_env,
            "model": model,
        }
    )
    extractor = belief_graph["extractor"]
    extractor.update(
        {
            "base_url": base_url,
            "api_key_env": api_key_env,
            "model": model,
        }
    )
    edge_generation = belief_graph["edge_generation"]
    edge_generation.update(
        {
            "base_url": base_url,
            "api_key_env": api_key_env,
            "model": model,
        }
    )
    stance = belief_graph["stance"]
    stance["model_path"] = stance_model
    stance["local_files_only"] = Path(stance_model).expanduser().exists()

    return {
        GRAPH_MODEL_KEY: chat_entry,
        EMBEDDING_KEY: _embedding_entry(embedding_model),
        "belief_graph": belief_graph,
    }


def load_user_configuration() -> dict[str, Any]:
    config = _read_json(config_path())
    graph = config.get("graph")
    if isinstance(graph, dict):
        replacement = _LEGACY_GRAPH_BACKENDS.get(graph.get("backend"))
        if replacement is not None:
            graph = dict(graph)
            graph["backend"] = replacement
            config = dict(config)
            config["graph"] = graph
    return config


def is_configured(config: dict[str, Any] | None = None) -> bool:
    value = config if config is not None else load_user_configuration()
    graph = value.get("graph")
    uses_existing_server = (
        isinstance(graph, dict) and graph.get("serverMode") == "existing"
    )
    return (
        value.get("version") == CONFIG_VERSION
        and value.get("setupComplete") is True
        and (
            uses_existing_server
            or graph_config_path().is_file()
            or model_config_path().is_file()
        )
    )


def apply_user_configuration(
    config: dict[str, Any],
    *,
    override: bool = False,
) -> None:
    """Expose persistent settings to existing environment-driven components."""

    agent = config.get("agent")
    context = config.get("context")
    graph = config.get("graph")
    summary = config.get("summary")
    if not isinstance(agent, dict) or not isinstance(context, dict):
        raise SetupError("Global BCG config is missing agent/context settings.")
    if not isinstance(graph, dict):
        raise SetupError("Global BCG config is missing graph settings.")
    if not isinstance(summary, dict):
        summary = {}

    values = {
        "BCG_AGENT_PROVIDER": str(agent.get("provider") or ""),
        "BCG_AGENT_MODEL": str(agent.get("model") or ""),
        "OPENAI_MODEL": str(agent.get("model") or ""),
        "OPENAI_BASE_URL": str(agent.get("baseUrl") or ""),
        "BELIEF_GRAPH_URL": str(graph.get("url") or DEFAULT_GRAPH_URL),
        "BCG_GRAPH_BACKEND": str(graph.get("backend") or "hybrid"),
        "BCG_GRAPH_AUTOSTART": (
            "false" if graph.get("serverMode") == "existing" else "true"
        ),
        "BCG_GRAPH_CONFIG": str(graph.get("modelConfig") or graph_config_path()),
        "BCG_GRAPH_MODEL_KEY": str(graph.get("modelKey") or GRAPH_MODEL_KEY),
        "BCG_GRAPH_EMBEDDING_KEY": str(graph.get("embeddingKey") or EMBEDDING_KEY),
        "BCG_RECENT_TURNS": str(context.get("recentTurns", 2)),
        "BCG_CONTEXT_MODE": str(context.get("mode") or "bcg"),
        "BCG_SUMMARY_MODEL": str(summary.get("model") or agent.get("model") or ""),
        "BCG_SUMMARY_BASE_URL": str(
            summary.get("baseUrl") or agent.get("baseUrl") or ""
        ),
        "BCG_SUMMARY_THINKING": str(summary.get("thinking") or "off"),
        "BCG_SUMMARY_RECENT_TURNS": str(
            summary.get("recentTurns", context.get("recentTurns", 2))
        ),
        "BCG_SUMMARY_TIMEOUT_MS": str(
            summary.get("timeoutMs", DEFAULT_SUMMARY_TIMEOUT_MS)
        ),
        "BCG_SUMMARY_MAX_TOKENS": str(
            summary.get("maxTokens", DEFAULT_SUMMARY_MAX_TOKENS)
        ),
    }
    for name, value in values.items():
        if value and (override or name not in os.environ):
            os.environ[name] = value

    credentials = _read_credentials()
    for name, value in credentials.items():
        if override or name not in os.environ:
            os.environ[name] = value


def _current_default(
    config: dict[str, Any],
    section: str,
    key: str,
    fallback: str,
) -> str:
    value = config.get(section)
    if isinstance(value, dict):
        configured = value.get(key)
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
    return fallback


def run_setup(
    *,
    input_fn: Callable[[str], str] = input,
    secret_fn: Callable[[str], str] = getpass.getpass,
) -> dict[str, Any]:
    """Run the interactive setup wizard and persist its result under BCG_HOME."""

    if not sys.stdin.isatty() and input_fn is input:
        raise SetupError(
            "First-time setup needs an interactive terminal. Run `bcg setup` "
            "from a terminal, then run `bcg` again."
        )

    current = load_user_configuration()
    credentials = _read_credentials()
    if _supports_rich_input(input_fn):
        _CONSOLE.print()
        _CONSOLE.print(
            Panel.fit(
                "[bold bright_white]BCG setup[/]\n"
                "[dim]Configure the Agent and Graph Construction runtime.[/]\n\n"
                f"[cyan]{state_root()}[/]",
                border_style="bright_cyan",
                padding=(1, 3),
            )
        )
        _CONSOLE.print()
    else:
        print("\nBCG first-time setup")
        print(f"Configuration will be stored in {state_root()}")

    auth_method = _choose(
        "How should the Agent authenticate?",
        [
            (
                "api_key",
                "OpenAI-compatible API key and base URL",
            ),
            (
                "login",
                "Use the Agent's interactive /login flow",
            ),
        ],
        default=_current_default(current, "agent", "authMethod", "api_key"),
        input_fn=input_fn,
    )
    pending_login = False
    if auth_method == "api_key":
        agent_base_url = _ask(
            "Agent API base URL",
            default=_current_default(
                current,
                "agent",
                "baseUrl",
                DEFAULT_AGENT_BASE_URL,
            ),
            input_fn=input_fn,
        )
        agent_model = _ask(
            "Agent model",
            default=_current_default(
                current,
                "agent",
                "model",
                DEFAULT_AGENT_MODEL,
            ),
            input_fn=input_fn,
        )
        credentials["OPENAI_API_KEY"] = _ask_secret(
            "Agent API key",
            existing=credentials.get("OPENAI_API_KEY"),
            secret_fn=secret_fn,
        )
        agent_provider = "bcg"
        login_provider = ""
    else:
        credentials.pop("OPENAI_API_KEY", None)
        login_provider = _ask(
            "Login provider (for example openai or anthropic)",
            default=_current_default(current, "agent", "loginProvider", "openai"),
            input_fn=input_fn,
        )
        agent_model = _ask(
            "Agent model",
            default=_current_default(
                current,
                "agent",
                "model",
                DEFAULT_AGENT_MODEL,
            ),
            input_fn=input_fn,
        )
        agent_provider = login_provider
        agent_base_url = ""
        pending_login = True

    search_provider = _choose(
        "Configure web search",
        [
            (
                "serper",
                "Serper API key (enables web_search and BrowseComp)",
            ),
            (
                "disabled",
                "Disable Serper web search",
            ),
        ],
        default="serper",
        input_fn=input_fn,
    )
    if search_provider == "serper":
        credentials["SERPER_API_KEY"] = _ask_secret(
            "Serper API key",
            existing=credentials.get("SERPER_API_KEY"),
            secret_fn=secret_fn,
        )
    else:
        credentials.pop("SERPER_API_KEY", None)

    context_mode = _choose(
        "Choose the default context mode",
        [
            ("bcg", "BCG graph-backed context"),
            ("default", "Default full-context agent with compaction"),
            ("summary", "Rolling LLM summary with recent raw turns"),
        ],
        default=_current_default(current, "context", "mode", "bcg"),
        input_fn=input_fn,
    )
    recent_turns = 2
    if context_mode in {"bcg", "summary"}:
        recent_turns_text = _ask(
            f"Recent completed turns kept verbatim in {context_mode} mode",
            default=str(
                current.get("context", {}).get("recentTurns", 2)
                if isinstance(current.get("context"), dict)
                else 2
            ),
            input_fn=input_fn,
        )
        try:
            recent_turns = max(0, int(recent_turns_text))
        except ValueError as exc:
            raise SetupError("Recent turns must be an integer.") from exc

    summary_base_url = _current_default(
        current,
        "summary",
        "baseUrl",
        agent_base_url or DEFAULT_AGENT_BASE_URL,
    )
    summary_model = _current_default(
        current,
        "summary",
        "model",
        agent_model,
    )
    summary_thinking = _current_default(
        current,
        "summary",
        "thinking",
        "off",
    )
    if context_mode == "summary":
        reuse_agent_summary = auth_method == "api_key" and _confirm(
            "Reuse the Agent API endpoint, model, and key for rolling summaries?",
            default=True,
            input_fn=input_fn,
        )
        if reuse_agent_summary:
            summary_base_url = agent_base_url
            summary_model = agent_model
            credentials["BCG_SUMMARY_API_KEY"] = credentials["OPENAI_API_KEY"]
        else:
            summary_base_url = _ask(
                "Summary model API base URL",
                default=summary_base_url,
                input_fn=input_fn,
            )
            summary_model = _ask(
                "Rolling-summary model",
                default=summary_model,
                input_fn=input_fn,
            )
            credentials["BCG_SUMMARY_API_KEY"] = _ask_secret(
                "Summary model API key",
                existing=credentials.get("BCG_SUMMARY_API_KEY"),
                secret_fn=secret_fn,
            )
        summary_thinking = _choose(
            "Summary model thinking level",
            [
                ("off", "Off (recommended for low latency and cost)"),
                ("low", "Low"),
                ("medium", "Medium"),
            ],
            default=summary_thinking
            if summary_thinking in {"off", "low", "medium"}
            else "off",
            input_fn=input_fn,
        )
    elif auth_method == "api_key" and "BCG_SUMMARY_API_KEY" not in credentials:
        credentials["BCG_SUMMARY_API_KEY"] = credentials["OPENAI_API_KEY"]

    graph_server_mode = _choose(
        "How should BCG connect to the Graph server?",
        [
            (
                "managed",
                "Start and manage a local Graph server automatically",
            ),
            (
                "existing",
                "Connect to an existing Graph server",
            ),
        ],
        default=_current_default(current, "graph", "serverMode", "managed"),
        input_fn=input_fn,
    )

    graph_model_config: dict[str, Any] | None = None
    if graph_server_mode == "existing":
        graph_url = _ask(
            "Existing Graph server URL",
            default=_current_default(
                current,
                "graph",
                "url",
                DEFAULT_GRAPH_URL,
            ),
            input_fn=input_fn,
        )
        graph_backend = "external"
        graph_base_url = ""
        graph_model = ""
    else:
        graph_url = DEFAULT_GRAPH_URL
        if _supports_rich_input(input_fn):
            _CONSOLE.print(
                Panel(
                    "BCG will start and reuse this service automatically:\n"
                    f"[bold cyan]{graph_url}[/]",
                    title="[bold]Local Graph server[/]",
                    title_align="left",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            _CONSOLE.print()
        else:
            print(
                f"\nBCG will start the local Graph server automatically at {graph_url}."
            )
        graph_backend = _choose(
            "Choose the local Graph Construction backend",
            [
                (
                    "unified",
                    "Unified: one OpenAI-compatible model builds the graph",
                ),
                (
                    "hybrid",
                    "Hybrid: OpenAI-compatible generator plus local embedding/stance models",
                ),
            ],
            default=_current_default(current, "graph", "backend", "hybrid"),
            input_fn=input_fn,
        )

    if graph_server_mode == "managed" and graph_backend == "unified":
        reuse_agent = auth_method == "api_key" and _confirm(
            "Reuse the Agent API endpoint and key for graph construction?",
            default=True,
            input_fn=input_fn,
        )
        if reuse_agent:
            graph_base_url = agent_base_url
            graph_model = agent_model
            graph_key_env = "OPENAI_API_KEY"
        else:
            graph_base_url = _ask(
                "Graph model API base URL",
                default=agent_base_url or DEFAULT_AGENT_BASE_URL,
                input_fn=input_fn,
            )
            graph_model = _ask(
                "Graph construction model",
                default=agent_model,
                input_fn=input_fn,
            )
            credentials["BCG_GRAPH_API_KEY"] = _ask_secret(
                "Graph model API key",
                existing=credentials.get("BCG_GRAPH_API_KEY"),
                secret_fn=secret_fn,
            )
            graph_key_env = "BCG_GRAPH_API_KEY"
        embedding_model_text = _ask(
            "Local embedding model (enter 'none' to disable embedding merge)",
            default=DEFAULT_EMBEDDING_MODEL,
            input_fn=input_fn,
        )
        embedding_model = (
            None
            if embedding_model_text.strip().lower() == "none"
            else embedding_model_text
        )
        graph_model_config = build_api_graph_config(
            base_url=graph_base_url,
            model=graph_model,
            api_key_env=graph_key_env,
            embedding_model=embedding_model,
        )
    elif graph_server_mode == "managed":
        graph_base_url = _ask(
            "Hybrid generator OpenAI-compatible base URL (local vLLM or remote API)",
            default=DEFAULT_VLLM_URL,
            input_fn=input_fn,
        )
        graph_model = _ask(
            "Hybrid extraction/relation model",
            default=DEFAULT_VLLM_MODEL,
            input_fn=input_fn,
        )
        credentials["BCG_GRAPH_API_KEY"] = _ask_secret(
            "Hybrid generator API key (use EMPTY when authentication is disabled)",
            existing=credentials.get("BCG_GRAPH_API_KEY") or "EMPTY",
            secret_fn=secret_fn,
        )
        embedding_model = _ask(
            "Local embedding model or path",
            default=DEFAULT_EMBEDDING_MODEL,
            input_fn=input_fn,
        )
        stance_model = _ask(
            "Local/Hugging Face stance model or path",
            default=DEFAULT_STANCE_MODEL,
            input_fn=input_fn,
        )
        graph_model_config = build_hybrid_graph_config(
            base_url=graph_base_url,
            model=graph_model,
            api_key_env="BCG_GRAPH_API_KEY",
            embedding_model=embedding_model,
            stance_model=stance_model,
        )

    result = {
        "version": CONFIG_VERSION,
        "setupComplete": True,
        "agent": {
            "authMethod": auth_method,
            "provider": agent_provider,
            "baseUrl": agent_base_url,
            "model": agent_model,
            "loginProvider": login_provider,
            "pendingLogin": pending_login,
        },
        "context": {
            "mode": context_mode,
            "recentTurns": recent_turns,
        },
        "summary": {
            "baseUrl": summary_base_url,
            "model": summary_model,
            "thinking": summary_thinking,
            "recentTurns": recent_turns,
            "timeoutMs": DEFAULT_SUMMARY_TIMEOUT_MS,
            "maxTokens": DEFAULT_SUMMARY_MAX_TOKENS,
        },
        "search": {
            "provider": search_provider,
            "configured": search_provider == "serper",
        },
        "graph": {
            "serverMode": graph_server_mode,
            "backend": graph_backend,
            "url": graph_url,
            "modelConfig": (
                str(graph_config_path()) if graph_server_mode == "managed" else ""
            ),
            "modelKey": GRAPH_MODEL_KEY,
            "embeddingKey": EMBEDDING_KEY,
            "modelBaseUrl": graph_base_url,
            "model": graph_model,
        },
    }
    _write_credentials(credentials)
    if graph_model_config is not None:
        _write_user_yaml_config(graph_model_config)
    _write_json(config_path(), result)

    if _supports_rich_input(input_fn):
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="dim")
        summary.add_column(style="cyan")
        summary.add_row("Agent", f"{agent_provider} / {agent_model}")
        summary.add_row(
            "Search",
            "Serper web search" if search_provider == "serper" else "disabled",
        )
        summary.add_row("Context", context_mode)
        if context_mode == "summary":
            summary.add_row("Summary", f"{summary_model} / {summary_thinking}")
        summary.add_row(
            "Graph",
            (
                f"managed {graph_backend} / {graph_url}"
                if graph_server_mode == "managed"
                else f"existing / {graph_url}"
            ),
        )
        summary.add_row("Runtime", str(config_path()))
        summary.add_row("Credentials", str(credentials_path()))
        _CONSOLE.print(
            Panel(
                summary,
                title="[bold green]✓ Configuration saved[/]",
                title_align="left",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        print("\nConfiguration saved.")
        print(f"  Runtime:     {config_path()}")
        print(f"  Credentials: {credentials_path()}")
        if graph_server_mode == "managed":
            print(f"  Graph config: {graph_config_path()}")
        else:
            print(f"  Graph server: {graph_url} (existing)")
    if graph_backend == "hybrid":
        print(
            "\nBefore using BCG mode, make sure the configured generator endpoint "
            f"is serving {graph_model} at {graph_base_url}."
        )
    if pending_login:
        print(
            f"\nThe Agent will now open the login flow for {login_provider}. "
            "You can run /login again later."
        )
    return result


def ensure_user_setup() -> tuple[dict[str, Any], bool]:
    config = load_user_configuration()
    created = False
    if not is_configured(config):
        config = run_setup()
        created = True
    apply_user_configuration(config, override=created)
    return config, created


def pending_login_provider(config: dict[str, Any]) -> str | None:
    agent = config.get("agent")
    if not isinstance(agent, dict) or agent.get("pendingLogin") is not True:
        return None
    provider = agent.get("loginProvider")
    return provider.strip() if isinstance(provider, str) and provider.strip() else None


def mark_login_prompt_consumed() -> None:
    config = load_user_configuration()
    agent = config.get("agent")
    if not isinstance(agent, dict) or agent.get("pendingLogin") is not True:
        return
    agent["pendingLogin"] = False
    _write_json(config_path(), config)


__all__ = [
    "CONFIG_VERSION",
    "SetupError",
    "apply_user_configuration",
    "build_api_graph_config",
    "build_hybrid_graph_config",
    "config_path",
    "credentials_path",
    "ensure_user_setup",
    "is_configured",
    "graph_config_path",
    "load_user_configuration",
    "mark_login_prompt_consumed",
    "model_config_path",
    "pending_login_provider",
    "run_setup",
    "state_root",
]
