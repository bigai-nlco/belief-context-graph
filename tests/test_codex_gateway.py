from __future__ import annotations

import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from bcg.integrations.codex.config import Config
from bcg.integrations.codex.gateway import SdkGraph
from bcg.integrations.codex_proxy.graph import GraphError


@pytest.fixture
def config(tmp_path):
    return Config(
        str(tmp_path),
        str(tmp_path / "state"),
        str(tmp_path / "models.json"),
        str(tmp_path / ".env"),
    )


@pytest.mark.parametrize("has_memory_mcp", [False, True])
@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--help"],
        ["-m", "test-model", "-c", 'model_reasoning_effort="low"'],
        ["--no-daemon", "-m", "test-model"],
        ["resume"],
        ["resume", "--last"],
        ["resume", "session-id"],
    ],
    ids=[
        "plain",
        "help",
        "model-options",
        "explicit-no-daemon",
        "resume-picker",
        "resume-last",
        "resume-id",
    ],
)
def test_gateway_disables_only_an_installed_memory_mcp(
    config, monkeypatch, has_memory_mcp, arguments
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
        expected_arguments = list(arguments)
        if "--no-daemon" not in expected_arguments:
            expected_arguments.insert(0, "--no-daemon")
        assert command[: 1 + len(expected_arguments)] == ["codex", *expected_arguments]
        assert command.count("--no-daemon") == 1
        assert 'model_provider="bcg_gateway"' in command
        assert ("mcp_servers.bcg_memory.enabled=false" in command) == has_memory_mcp
        assert kwargs["env"]["BCG_NATIVE_HOOKS_DISABLED"] == "1"
        assert kwargs["cwd"] == config.project
        return 0

    monkeypatch.setenv("BCG_CODEX_GATEWAY_TOKEN", "test")
    monkeypatch.setenv("BCG_ENV_FILE", "test")
    monkeypatch.setattr(gateway, "ProxyServer", FakeServer)
    monkeypatch.setattr(gateway.subprocess, "run", probe)
    monkeypatch.setattr(gateway.subprocess, "call", launch)
    original_arguments = list(arguments)
    assert gateway.run(config, arguments) == 0
    assert arguments == original_arguments


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


def test_gateway_sdk_initializes_once_across_concurrent_threads(config, monkeypatch):
    from bcg.integrations.codex import runtime

    factory_started = threading.Event()
    contender_arrived = threading.Event()
    release_factory = threading.Event()
    created = []

    class ObservedLock:
        """Observe contention without relying on sleeps or thread scheduling."""

        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if factory_started.is_set():
                contender_arrived.set()
            self.lock.acquire()

        def __exit__(self, *exc):
            self.lock.release()

    def factory(cfg):
        assert cfg == config
        manager = object()
        created.append(manager)
        if len(created) > 1:
            # Also expose the old, unlocked implementation's second factory
            # call so this test fails deterministically on that version.
            contender_arrived.set()
        factory_started.set()
        assert release_factory.wait(5)
        return manager

    monkeypatch.setattr(runtime, "create_graph_manager", factory)
    graph = SdkGraph(config)
    graph._manager_lock = ObservedLock()
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            first = pool.submit(graph._get)
            assert factory_started.wait(5)
            second = pool.submit(graph._get)
            assert contender_arrived.wait(5)
            assert not first.done() and not second.done()
        finally:
            release_factory.set()
        managers = [first.result(timeout=5), second.result(timeout=5)]

    assert len(created) == 1
    assert managers[0] is managers[1] is graph.manager
    assert graph._get() is graph.manager
    assert len(created) == 1


def test_gateway_sdk_can_retry_failed_initialization(config, monkeypatch):
    from bcg.integrations.codex import runtime

    attempts = []
    manager = SimpleNamespace(
        push_many=lambda turns: {"latest": {"task": {"nodes": [1]}}}
    )

    def factory(cfg):
        attempts.append(cfg)
        if len(attempts) == 1:
            raise RuntimeError("temporary initialization failure")
        return manager

    monkeypatch.setattr(runtime, "create_graph_manager", factory)
    graph = SdkGraph(config)
    with pytest.raises(GraphError, match="native_graph_write_failed"):
        graph.push("task", [{"role": "user", "content": "Test request"}])
    assert graph.manager is None
    assert graph.push("task", [{"role": "user", "content": "Test request"}]) == {
        "nodes": [1]
    }
    assert graph._get() is manager
    assert attempts == [config, config]


def test_gateway_sdk_instances_can_initialize_independently(config, monkeypatch):
    from bcg.integrations.codex import runtime

    both_initializing = threading.Barrier(2)

    def factory(cfg):
        assert cfg == config
        both_initializing.wait(timeout=5)
        return object()

    monkeypatch.setattr(runtime, "create_graph_manager", factory)
    first, second = SdkGraph(config), SdkGraph(config)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(graph._get) for graph in (first, second)]
        managers = [future.result(timeout=5) for future in pending]
    assert first.manager is managers[0]
    assert second.manager is managers[1]
    assert managers[0] is not managers[1]


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
