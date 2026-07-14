from __future__ import annotations

import pytest

from bcg import cli
from bcg.agent import cli as agent_cli
from bcg.construct import cli as construct_cli


def test_root_help_lists_only_two_command_families(capsys) -> None:
    cli.main([])

    output = capsys.readouterr().out
    assert "Usage: bcg [OPTIONS] COMMAND [ARGS]..." in output
    assert "Commands" in output
    assert "agent" in output
    assert "construct" in output
    assert "belief_tracer" not in output


def test_root_routes_agent_arguments(monkeypatch) -> None:
    received: list[str] = []
    monkeypatch.setattr("bcg.agent.cli.main", received.extend)

    cli.main(["agent", "run", "gpqa_diamond"])

    assert received == ["run", "gpqa_diamond"]


def test_root_routes_construct_arguments(monkeypatch) -> None:
    received: list[str] = []
    monkeypatch.setattr("bcg.construct.cli.main", received.extend)

    cli.main(["construct", "run", "--input", "data.json"])

    assert received == ["run", "--input", "data.json"]


@pytest.mark.parametrize("command", ["run", "server", "replay", "visualize"])
def test_construct_commands_expose_rich_help(command, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        construct_cli.main([command, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert f"Usage: bcg construct {command} [OPTIONS]" in output
    assert "╭─" in output


@pytest.mark.parametrize("command", ["run", "ui", "check-thinking", "tonggraph-sync"])
def test_agent_commands_expose_rich_help(command, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        agent_cli.main([command, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert f"Usage: bcg agent {command} [OPTIONS]" in output
    assert "╭─" in output
