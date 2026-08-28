from __future__ import annotations

import json
from pathlib import Path

import pytest

from bcg.apps import agent_runtime


@pytest.fixture(autouse=True)
def _clean_generated_runtime_environment(monkeypatch) -> None:
    for name in (
        "BCG_AGENT_MODEL",
        "BCG_AGENT_PROVIDER",
        "BCG_CONTEXT_MODE",
        "BCG_GRAPH_AUTOSTART",
        "BCG_SUMMARY_MODEL",
        "BCG_SUMMARY_BASE_URL",
        "BCG_SUMMARY_API_KEY",
        "BCG_RAG_DB_PATH",
        "BCG_RAG_TOP_K",
        "BCG_RAG_MAX_CHARS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_agent_configuration_enables_bcg_and_references_env_key(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-that-must-not-be-written")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "models.json").write_text(
        json.dumps(
            {"providers": {"bcg-openai": {"baseUrl": "https://legacy.invalid/v1"}}}
        )
    )

    agent_dir = agent_runtime.ensure_agent_configuration("http://127.0.0.1:8848")

    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings["contextManagement"] == {
        "provider": "bcg",
        "bcg": {
            "url": "http://127.0.0.1:8848",
            "recentTurns": 2,
            "maxTurns": 160,
            "timeoutMs": 300000,
            "finalizationTimeoutMs": 900000,
            "includeRelations": True,
            "graphView": "full",
        },
        "summary": {
            "provider": "bcg-summary",
            "model": "test-model",
            "recentTurns": 2,
            "timeoutMs": 300000,
            "maxTokens": 2048,
            "thinkingLevel": "off",
        },
        "recentOnly": {"recentTurns": 2},
        "rag": {
            "recentTurns": 2,
            "databasePath": "",
            "topK": 6,
            "maxChars": 12000,
        },
    }
    assert settings["defaultProvider"] == "bcg"
    assert settings["defaultModel"] == "test-model"

    models_text = (agent_dir / "models.json").read_text()
    models = json.loads(models_text)
    provider = models["providers"]["bcg"]
    assert "bcg-openai" not in models["providers"]
    assert provider["baseUrl"] == "https://example.test/v1"
    assert provider["apiKey"] == "$OPENAI_API_KEY"
    assert models["providers"]["bcg-summary"]["apiKey"] == "$BCG_SUMMARY_API_KEY"
    assert "secret-that-must-not-be-written" not in models_text


def test_summary_context_uses_independent_model_configuration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://agent.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "agent-model")
    monkeypatch.setenv("BCG_CONTEXT_MODE", "summary")
    monkeypatch.setenv("BCG_SUMMARY_BASE_URL", "https://summary.test/v1")
    monkeypatch.setenv("BCG_SUMMARY_MODEL", "summary-model")
    monkeypatch.setenv("BCG_SUMMARY_THINKING", "low")

    agent_dir = agent_runtime.ensure_agent_configuration("http://127.0.0.1:8848")

    settings = json.loads((agent_dir / "settings.json").read_text())
    models = json.loads((agent_dir / "models.json").read_text())
    assert settings["contextManagement"]["provider"] == "summary"
    assert settings["contextManagement"]["summary"] == {
        "provider": "bcg-summary",
        "model": "summary-model",
        "recentTurns": 2,
        "timeoutMs": 300000,
        "maxTokens": 2048,
        "thinkingLevel": "low",
    }
    assert models["providers"]["bcg-summary"]["baseUrl"] == ("https://summary.test/v1")
    assert models["providers"]["bcg-summary"]["models"][0]["id"] == ("summary-model")


@pytest.mark.parametrize("mode", ["recent-only", "rag"])
def test_bounded_context_modes_are_written_to_agent_settings(
    monkeypatch,
    tmp_path: Path,
    mode: str,
) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://agent.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "agent-model")
    monkeypatch.setenv("BCG_CONTEXT_MODE", mode)
    monkeypatch.setenv("BCG_RECENT_TURNS", "2")
    monkeypatch.setenv("BCG_RAG_DB_PATH", str(tmp_path / "rag.sqlite"))
    monkeypatch.setenv("BCG_RAG_TOP_K", "4")
    monkeypatch.setenv("BCG_RAG_MAX_CHARS", "4096")

    agent_dir = agent_runtime.ensure_agent_configuration("http://127.0.0.1:8848")
    settings = json.loads((agent_dir / "settings.json").read_text())

    assert settings["contextManagement"]["provider"] == mode
    assert settings["contextManagement"]["recentOnly"] == {"recentTurns": 2}
    assert settings["contextManagement"]["rag"] == {
        "recentTurns": 2,
        "databasePath": str(tmp_path / "rag.sqlite"),
        "topK": 4,
        "maxChars": 4096,
    }


def test_summary_provider_is_generated_when_agent_uses_login(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_MODEL", "agent-login-model")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("BCG_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("BCG_SUMMARY_BASE_URL", "https://summary.test/v1")
    monkeypatch.setenv("BCG_SUMMARY_MODEL", "summary-model")

    agent_dir = agent_runtime.ensure_agent_configuration("http://127.0.0.1:8848")

    settings = json.loads((agent_dir / "settings.json").read_text())
    models = json.loads((agent_dir / "models.json").read_text())
    assert settings["defaultProvider"] == "openai"
    assert models["providers"]["bcg-summary"]["baseUrl"] == ("https://summary.test/v1")


def test_existing_agent_settings_are_preserved(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "settings.json").write_text(
        json.dumps({"theme": "dark", "defaultModel": "existing-model"})
    )

    agent_runtime.ensure_agent_configuration("http://127.0.0.1:8848")

    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings["theme"] == "dark"
    assert settings["defaultModel"] == "existing-model"
    assert settings["contextManagement"]["provider"] == "bcg"


def test_existing_default_context_mode_is_not_reset(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "settings.json").write_text(
        json.dumps({"contextManagement": {"provider": "default"}})
    )

    agent_runtime.ensure_agent_configuration("http://127.0.0.1:8848")

    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings["contextManagement"]["provider"] == "default"
    assert settings["contextManagement"]["bcg"]["url"] == "http://127.0.0.1:8848"


def test_healthy_graph_server_is_reused(monkeypatch) -> None:
    monkeypatch.setattr(agent_runtime, "graph_server_is_ready", lambda _url: True)
    monkeypatch.setattr(
        agent_runtime,
        "_resolve_graph_config",
        lambda: (_ for _ in ()).throw(AssertionError("must not resolve config")),
    )

    agent_runtime.ensure_graph_server("http://127.0.0.1:8848")


def test_source_agent_build_precedes_an_unrelated_global_install(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    cli = source_root / "agent-cli" / "dist" / "cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("", encoding="utf-8")
    monkeypatch.setattr(agent_runtime, "PROJECT_ROOT", tmp_path / "state")
    monkeypatch.setattr(agent_runtime, "SOURCE_PROJECT_ROOT", source_root)
    monkeypatch.setattr(
        agent_runtime.shutil,
        "which",
        lambda name: "/usr/bin/node" if name == "node" else "/global/bcg-agent",
    )

    assert agent_runtime._resolve_agent_command() == ["/usr/bin/node", str(cli)]


def test_unavailable_existing_graph_server_is_not_started(monkeypatch) -> None:
    monkeypatch.setenv("BCG_GRAPH_AUTOSTART", "false")
    monkeypatch.setattr(
        agent_runtime,
        "graph_server_is_ready",
        lambda _url: False,
    )
    monkeypatch.setattr(
        agent_runtime.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not start a local server")
        ),
    )

    with pytest.raises(
        agent_runtime.AgentLaunchError,
        match="configured existing Graph server",
    ):
        agent_runtime.ensure_graph_server("https://graph.test")


def test_launch_passes_bcg_environment_to_agent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    monkeypatch.setenv("BELIEF_GRAPH_URL", "http://127.0.0.1:9999")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.setattr(agent_runtime, "ensure_graph_server", lambda _url: None)
    monkeypatch.setattr(
        "bcg.apps.setup.ensure_user_setup",
        lambda: ({}, False),
    )
    monkeypatch.setattr(
        agent_runtime,
        "_resolve_agent_command",
        lambda: ["bcg-agent"],
    )
    calls: list[tuple[list[str], dict[str, str]]] = []

    class Result:
        returncode = 7

    def fake_run(command, *, env, check):
        assert check is False
        calls.append((command, env))
        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)

    assert agent_runtime.launch_interactive(["hello"]) == 7
    assert calls[0][0] == ["bcg-agent", "hello"]
    assert calls[0][1]["BELIEF_GRAPH_URL"] == "http://127.0.0.1:9999"
    assert calls[0][1]["BCG_CODING_AGENT_DIR"] == str(tmp_path / "agent")


def test_launch_selects_generated_bcg_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(agent_runtime, "ensure_graph_server", lambda _url: None)
    monkeypatch.setattr(
        "bcg.apps.setup.ensure_user_setup",
        lambda: ({}, False),
    )
    monkeypatch.setattr(
        agent_runtime,
        "_resolve_agent_command",
        lambda: ["bcg-agent"],
    )
    commands: list[list[str]] = []

    class Result:
        returncode = 0

    def fake_run(command, **_kwargs):
        commands.append(command)
        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)

    assert agent_runtime.launch_interactive([]) == 0
    assert commands == [
        [
            "bcg-agent",
            "--provider",
            "bcg",
            "--model",
            "test-model",
        ]
    ]


def test_summary_mode_does_not_start_graph_server(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    monkeypatch.setenv("BCG_CONTEXT_MODE", "summary")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(
        agent_runtime,
        "ensure_graph_server",
        lambda _url: (_ for _ in ()).throw(
            AssertionError("Summary mode must not start Graph Construction")
        ),
    )
    monkeypatch.setattr("bcg.apps.setup.ensure_user_setup", lambda: ({}, False))
    monkeypatch.setattr(agent_runtime, "_resolve_agent_command", lambda: ["bcg-agent"])

    class Result:
        returncode = 0

    monkeypatch.setattr(
        agent_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: Result(),
    )

    assert agent_runtime.launch_interactive([]) == 0


def test_agent_help_does_not_start_graph_server(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    monkeypatch.setattr(
        agent_runtime,
        "ensure_graph_server",
        lambda _url: (_ for _ in ()).throw(
            AssertionError("help must not start Graph Construction")
        ),
    )
    monkeypatch.setattr(
        agent_runtime,
        "_resolve_agent_command",
        lambda: ["bcg-agent"],
    )

    class Result:
        returncode = 0

    monkeypatch.setattr(
        agent_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: Result(),
    )

    assert agent_runtime.launch_interactive(["--help"]) == 0
