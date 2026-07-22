from __future__ import annotations

import json

import pytest

from bcg.construct.api_based.llm import load_config, load_embedding_config
from bcg.env import find_project_env, load_project_env, read_env_file


def test_env_discovery_supports_tool_install_working_directory(
    tmp_path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")
    monkeypatch.delenv("BCG_ENV_FILE", raising=False)
    monkeypatch.chdir(tmp_path)

    assert find_project_env() == env_file


def test_env_discovery_honors_explicit_file(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / "shared.env"
    monkeypatch.setenv("BCG_ENV_FILE", str(env_file))

    assert find_project_env() == env_file


def test_project_env_parser_and_existing_environment_priority(
    tmp_path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# credentials\n"
        "export OPENAI_API_KEY='from-file'\n"
        'SERPER_API_KEY="serper-file"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "already-exported")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    assert read_env_file(env_file) == {
        "OPENAI_API_KEY": "from-file",
        "SERPER_API_KEY": "serper-file",
    }
    assert load_project_env(env_file) == {"SERPER_API_KEY": "serper-file"}
    assert load_project_env(env_file) == {}
    assert load_project_env(env_file, override=True) == {
        "OPENAI_API_KEY": "from-file",
        "SERPER_API_KEY": "serper-file",
    }


def test_construct_configs_resolve_keys_from_environment(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "model_config.json"
    config_file.write_text(
        json.dumps(
            {
                "chat-model": {
                    "base_url": "https://chat.test/v1",
                    "api_key_env": "CHAT_TEST_API_KEY",
                },
                "embedding": {
                    "base_url": "https://embedding.test/v1",
                    "api_key_env": "EMBEDDING_TEST_API_KEY",
                    "model": "embedding-model",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHAT_TEST_API_KEY", "chat-secret")
    monkeypatch.setenv("EMBEDDING_TEST_API_KEY", "embedding-secret")

    chat = load_config(str(config_file), model_key="chat-model")
    embedding = load_embedding_config(str(config_file))

    assert chat["api_key"] == "chat-secret"
    assert embedding is not None
    assert embedding["api_key"] == "embedding-secret"


def test_construct_config_reports_missing_root_env_key(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "model_config.json"
    config_file.write_text(
        json.dumps(
            {
                "chat-model": {
                    "base_url": "https://chat.test/v1",
                    "api_key_env": "MISSING_TEST_API_KEY",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("MISSING_TEST_API_KEY", raising=False)

    with pytest.raises(ValueError, match="project root .env"):
        load_config(str(config_file), model_key="chat-model")
