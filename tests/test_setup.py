from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from bcg.apps import setup


def test_persisted_setup_backend_name_is_migrated(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"graph": {"backend": "light"}}), encoding="utf-8"
    )

    config = setup.load_user_configuration()

    assert config["graph"]["backend"] == "hybrid"


def test_unified_setup_persists_global_configuration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    answers = iter(
        [
            "1",  # API key authentication
            "https://agent.test/v1",
            "agent-model",
            "2",  # disable Serper web search
            "1",  # BCG context
            "",  # two recent turns
            "1",  # managed local Graph server
            "1",  # Unified Graph backend
            "",  # reuse Agent endpoint
            "none",  # no local embedding model
        ]
    )
    secrets = iter(["agent-secret"])

    config = setup.run_setup(
        input_fn=lambda _prompt: next(answers),
        secret_fn=lambda _prompt: next(secrets),
    )

    assert config["agent"]["provider"] == "bcg"
    assert config["context"] == {"mode": "bcg", "recentTurns": 2}
    assert config["graph"]["serverMode"] == "managed"
    assert config["graph"]["backend"] == "unified"
    assert config["graph"]["url"] == "http://127.0.0.1:8848"
    assert config["graph"]["modelConfig"] == str(tmp_path / "config.yaml")
    assert setup.is_configured(config)

    persisted = json.loads((tmp_path / "config.json").read_text())
    assert persisted == config
    # model settings now land in the unified YAML config (legacy JSON no longer written)
    import yaml

    yaml_config = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert yaml_config["backend"] == "unified"
    assert yaml_config["model_key"] == "graph-model"
    assert yaml_config["models"]["graph-model"] == {
        "base_url": "https://agent.test/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "agent-model",
        "max_tokens": 65536,
        "temperature": 0,
    }
    assert "embedding" not in yaml_config["models"]
    assert not (tmp_path / "model_config.json").exists()
    assert (tmp_path / ".env").read_text().endswith("OPENAI_API_KEY=agent-secret\n")
    assert stat.S_IMODE((tmp_path / ".env").stat().st_mode) == 0o600


def test_hybrid_config_points_every_graph_model_at_vllm() -> None:
    config = setup.build_hybrid_graph_config(
        base_url="http://vllm.test/v1",
        model="Qwen-test",
        api_key_env="BCG_GRAPH_API_KEY",
        embedding_model="embedding-test",
        stance_model="stance-test",
    )

    assert config["graph-model"]["model"] == "Qwen-test"
    assert config["embedding"]["model"] == "embedding-test"
    belief_graph = config["belief_graph"]
    assert belief_graph["extractor"]["base_url"] == "http://vllm.test/v1"
    assert belief_graph["extractor"]["model"] == "Qwen-test"
    assert belief_graph["edge_generation"]["model"] == "Qwen-test"
    assert belief_graph["stance"]["model_path"] == "stance-test"
    assert belief_graph["stance"]["local_files_only"] is False


def test_managed_hybrid_setup_asks_for_generator_endpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    answers = iter(
        [
            "1",
            "https://agent.test/v1",
            "agent-model",
            "2",  # disable Serper web search
            "1",
            "2",
            "1",  # managed local Graph server
            "2",  # hybrid backend
            "http://vllm.test/v1",
            "Qwen-test",
            "embedding-test",
            "stance-test",
        ]
    )
    prompts: list[str] = []
    secrets = iter(["agent-secret", "EMPTY"])

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    config = setup.run_setup(
        input_fn=answer,
        secret_fn=lambda _prompt: next(secrets),
    )

    assert config["graph"]["url"] == setup.DEFAULT_GRAPH_URL
    assert config["graph"]["modelBaseUrl"] == "http://vllm.test/v1"
    import yaml

    yaml_config = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert yaml_config["backend"] == "hybrid"
    assert any(
        "Hybrid generator OpenAI-compatible base URL" in prompt for prompt in prompts
    )
    assert not any("Graph server URL" in prompt for prompt in prompts)


def test_existing_graph_server_skips_local_model_configuration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    answers = iter(
        [
            "1",  # API key authentication
            "https://agent.test/v1",
            "agent-model",
            "2",  # disable Serper web search
            "2",  # default context
            "2",  # existing Graph server
            "https://graph.test",
        ]
    )

    config = setup.run_setup(
        input_fn=lambda _prompt: next(answers),
        secret_fn=lambda _prompt: "agent-secret",
    )

    assert config["graph"] == {
        "serverMode": "existing",
        "backend": "external",
        "url": "https://graph.test",
        "modelConfig": "",
        "modelKey": "graph-model",
        "embeddingKey": "embedding",
        "modelBaseUrl": "",
        "model": "",
    }
    assert setup.is_configured(config)
    assert not (tmp_path / "model_config.json").exists()


def test_setup_persists_serper_key_for_web_search(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    answers = iter(
        [
            "1",  # API key authentication
            "https://agent.test/v1",
            "agent-model",
            "1",  # configure Serper
            "2",  # default context
            "2",  # existing Graph server
            "https://graph.test",
        ]
    )
    secrets = iter(["agent-secret", "serper-secret"])

    config = setup.run_setup(
        input_fn=lambda _prompt: next(answers),
        secret_fn=lambda _prompt: next(secrets),
    )

    assert config["search"] == {"provider": "serper", "configured": True}
    credentials = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=agent-secret\n" in credentials
    assert "SERPER_API_KEY=serper-secret\n" in credentials


def test_setup_configures_summary_mode_with_agent_endpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    answers = iter(
        [
            "1",  # API key authentication
            "https://agent.test/v1",
            "agent-model",
            "2",  # disable Serper
            "3",  # summary context
            "",  # two recent turns
            "",  # reuse Agent endpoint/model/key
            "",  # summary thinking off
            "2",  # existing Graph server (not used by Summary mode)
            "https://graph.test",
        ]
    )

    config = setup.run_setup(
        input_fn=lambda _prompt: next(answers),
        secret_fn=lambda _prompt: "agent-secret",
    )

    assert config["context"] == {"mode": "summary", "recentTurns": 2}
    assert config["summary"] == {
        "baseUrl": "https://agent.test/v1",
        "model": "agent-model",
        "thinking": "off",
        "recentTurns": 2,
        "timeoutMs": 300000,
        "maxTokens": 2048,
    }
    credentials = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "BCG_SUMMARY_API_KEY=agent-secret\n" in credentials


def test_apply_user_configuration_uses_global_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BCG_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("OPENAI_API_KEY=global-secret\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for name in (
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
        "BCG_AGENT_PROVIDER",
        "BELIEF_GRAPH_URL",
        "BCG_GRAPH_BACKEND",
        "BCG_GRAPH_AUTOSTART",
        "BCG_GRAPH_CONFIG",
        "BCG_GRAPH_MODEL_KEY",
        "BCG_GRAPH_EMBEDDING_KEY",
        "BCG_RECENT_TURNS",
        "BCG_CONTEXT_MODE",
    ):
        monkeypatch.setenv(name, "")
    config = {
        "agent": {
            "provider": "bcg",
            "model": "agent-model",
            "baseUrl": "https://agent.test/v1",
        },
        "context": {"mode": "default", "recentTurns": 2},
        "graph": {
            "serverMode": "existing",
            "backend": "unified",
            "url": "http://127.0.0.1:8848",
            "modelConfig": str(tmp_path / "model_config.json"),
            "modelKey": "graph-model",
            "embeddingKey": "embedding",
        },
    }

    setup.apply_user_configuration(config, override=True)

    assert os.environ["OPENAI_API_KEY"] == "global-secret"
    assert os.environ["OPENAI_MODEL"] == "agent-model"
    assert os.environ["BCG_CONTEXT_MODE"] == "default"
    assert os.environ["BCG_GRAPH_BACKEND"] == "unified"
    assert os.environ["BCG_GRAPH_AUTOSTART"] == "false"
