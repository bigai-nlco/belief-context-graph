from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bcg.integrations.codex.config import Config
from bcg.integrations.codex.gateway import SdkGraph


@pytest.fixture
def config(tmp_path):
    return Config(
        str(tmp_path),
        str(tmp_path / "state"),
        str(tmp_path / "models.json"),
        str(tmp_path / ".env"),
    )


@pytest.mark.parametrize("has_memory_mcp", [False, True])
def test_gateway_disables_only_an_installed_memory_mcp(
    config, monkeypatch, has_memory_mcp
):
    from bcg.integrations.codex import gateway

    class FakeServer:
        def __init__(self, settings):
            assert settings.host == "127.0.0.1"
            assert settings.recent_turns == 2
            assert settings.node_char_budget == 6600
            self.server_address = ("127.0.0.1", 12345)
            self.engine = SimpleNamespace()

        def serve_forever(self):
            pass

        def shutdown(self):
            pass

        def server_close(self):
            pass

    def probe(command, **kwargs):
        assert command == ["codex", "mcp", "get", "bcg_memory", "--json"]
        assert kwargs["cwd"] == config.project
        return SimpleNamespace(returncode=0 if has_memory_mcp else 1)

    def launch(command, **kwargs):
        assert ("mcp_servers.bcg_memory.enabled=false" in command) == has_memory_mcp
        assert kwargs["env"]["BCG_NATIVE_HOOKS_DISABLED"] == "1"
        assert kwargs["cwd"] == config.project
        return 0

    monkeypatch.setenv("BCG_CODEX_GATEWAY_TOKEN", "test")
    monkeypatch.setenv("BCG_ENV_FILE", "test")
    monkeypatch.setattr(gateway, "ProxyServer", FakeServer)
    monkeypatch.setattr(gateway.subprocess, "run", probe)
    monkeypatch.setattr(gateway.subprocess, "call", launch)
    assert gateway.run(config, ["--help"]) == 0


def test_gateway_sdk_is_lazy_and_preserves_batching(config, monkeypatch):
    from bcg.integrations.codex import runtime

    created, pushed, selected = [], [], []

    class Manager:
        def push_many(self, turns):
            pushed.append(turns)
            return {"latest": {"task": {"nodes": [1]}}}

        def get_session(self, problem_id, *, create):
            assert problem_id == "task"
            assert create is False
            return self

        def select_context(self, query, **kwargs):
            selected.append((query, kwargs))
            return {"selected_node_ids": [1]}

    def factory(cfg):
        assert cfg == config
        created.append(cfg)
        return Manager()

    monkeypatch.setattr(runtime, "create_graph_manager", factory)
    graph = SdkGraph(config)
    assert not created
    turns = [
        {"role": "assistant", "content": "call"},
        {"role": "tool", "content": "result"},
    ]
    assert graph.push("task", turns) == {"nodes": [1]}
    assert len(pushed) == 1
    assert len(pushed[0]) == 2
    assert all(
        t["problem_id"] == "task" and t["is_message_end"] and not t["is_trajectory_end"]
        for t in pushed[0]
    )
    assert all("problem_id" not in t for t in turns)
    assert graph.select("task", "query", "question", 6600) == {"selected_node_ids": [1]}
    assert selected == [
        (
            "query",
            {"strategy": "focused", "question": "question", "node_char_budget": 6600},
        )
    ]
    assert len(created) == 1


def test_gateway_runtime_retains_original_graph_settings(config, monkeypatch):
    from bcg.integrations.codex.runtime import create_graph_manager

    loaded, built = [], []
    sentinel = object()

    def manager(**kwargs):
        built.append(kwargs)
        return sentinel

    monkeypatch.setenv("BCG_ENV_FILE", "initial")
    monkeypatch.setattr(
        "bcg.core.env.load_project_env",
        lambda path, **kwargs: loaded.append((path, kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "bcg.construct.unified.online",
        SimpleNamespace(SessionManager=manager),
    )
    monkeypatch.setitem(
        sys.modules,
        "bcg.construct.unified.stream",
        SimpleNamespace(StreamOptions=SimpleNamespace),
    )
    assert create_graph_manager(config) is sentinel
    assert loaded == [(config.env_file, {"override": True})]
    assert built[0]["config_path"] == config.graph_config
    assert built[0]["model_key"] == "graph-model"
    assert built[0]["output_root"] == Path(config.state_dir) / "graphs"
    assert vars(built[0]["options"]) == {
        "context_chars": 12000,
        "extraction_history_turns": 2,
        "max_previous_windows": 3,
    }


@pytest.mark.parametrize(
    "module", ["events", "hooks", "mcp", "memory", "store", "worker"]
)
def test_removed_integrations_are_not_importable(module):
    assert importlib.util.find_spec(f"bcg.integrations.codex.{module}") is None
