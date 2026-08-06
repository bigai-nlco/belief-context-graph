from __future__ import annotations

import pytest
from click.utils import strip_ansi

from bcg import cli
from bcg.construct import cli as construct_cli


def test_root_without_arguments_launches_interactive_agent(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "bcg.apps.agent_runtime.main", lambda args: calls.append(args) or 0
    )

    cli.main([])

    assert calls == [[]]


def test_root_help_lists_public_command_families(capsys) -> None:
    cli.main(["--help"])
    output = strip_ansi(capsys.readouterr().out)
    assert "Usage: bcg [OPTIONS] COMMAND [ARGS]..." in output
    assert "Commands" in output
    assert "agent" in output
    assert "benchmark" in output
    assert "construct" in output
    assert "setup" in output
    assert "belief_tracer" not in output


def test_root_routes_interactive_agent_arguments(monkeypatch) -> None:
    received: list[list[str]] = []
    monkeypatch.setattr(
        "bcg.apps.agent_runtime.main",
        lambda args: received.append(args) or 0,
    )

    cli.main(["agent", "--model", "gpt-5.6-terra"])

    assert received == [["--model", "gpt-5.6-terra"]]


def test_root_routes_construct_arguments(monkeypatch) -> None:
    received: list[str] = []
    monkeypatch.setattr("bcg.construct.cli.main", received.extend)

    cli.main(["construct", "run", "--input", "data.json"])

    assert received == ["run", "--input", "data.json"]


def test_root_routes_benchmark_arguments(monkeypatch) -> None:
    received: list[str] = []
    monkeypatch.setattr("bcg.apps.benchmark.cli.main", received.extend)

    cli.main(["benchmark", "run", "gaia", "--max-problems", "2"])

    assert received == ["run", "gaia", "--max-problems", "2"]


def test_root_routes_setup(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr("bcg.apps.setup.run_setup", lambda: calls.append(True))

    cli.main(["setup"])

    assert calls == [True]


@pytest.mark.parametrize("command", ["run", "server", "replay", "visualize"])
def test_construct_commands_expose_rich_help(command, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        construct_cli.main([command, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert f"Usage: bcg construct {command} [OPTIONS]" in output
    assert "╭─" in output


@pytest.mark.parametrize(
    "module_name",
    [
        "bcg.run",
        "bcg.online_server",
        "bcg.online_driver",
    ],
)
def test_construct_legacy_flag_only_invocations_default_to_api_based(
    module_name, monkeypatch
) -> None:
    module = __import__(f"bcg.apps.{module_name.rsplit('.', 1)[-1]}", fromlist=["main"])
    received: list[str] = []
    monkeypatch.setitem(module._BACKENDS, "api_based", received.extend)

    module.main(["--config", "model_config.json"])

    assert received == ["--config", "model_config.json"]


@pytest.mark.parametrize(
    "module_name",
    [
        "bcg.run",
        "bcg.online_server",
        "bcg.online_driver",
    ],
)
def test_construct_explicit_light_backend_is_preserved(
    module_name, monkeypatch
) -> None:
    module = __import__(f"bcg.apps.{module_name.rsplit('.', 1)[-1]}", fromlist=["main"])
    received: list[str] = []
    monkeypatch.setitem(module._BACKENDS, "light", received.extend)

    module.main(["light", "--config", "model_config.json"])

    assert received == ["--config", "model_config.json"]


def test_backend_module_cli_prepends_selected_backend(monkeypatch) -> None:
    from bcg.construct.light import cli as light_cli

    received: list[str] = []
    monkeypatch.setattr("bcg.run.main", received.extend)

    light_cli.main(["run", "--input", "data.json"])

    assert received == ["light", "--input", "data.json"]


@pytest.mark.parametrize(
    ("command", "target"),
    [
        ("run", "bcg.run.main"),
        ("server", "bcg.online_server.main"),
        ("replay", "bcg.online_driver.main"),
    ],
)
def test_api_based_module_cli_prepends_backend(command, target, monkeypatch) -> None:
    from bcg.construct.api_based import cli as api_cli

    received: list[str] = []
    monkeypatch.setattr(target, received.extend)

    api_cli.main([command, "--config", "model_config.json"])

    assert received == ["api_based", "--config", "model_config.json"]
