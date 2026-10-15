from __future__ import annotations

import pytest

from bcg.apps import cli as public_cli
from bcg.integrations.codex import cli, gateway
from bcg.integrations.codex.config import Config


@pytest.mark.parametrize("public_entrypoint", [False, True])
@pytest.mark.parametrize(
    "arguments,forwarded,record_io",
    [
        ([], [], False),
        (["run"], [], False),
        (["--record-model-io"], [], True),
        (["run", "--record-model-io"], [], True),
        (["--", "-m", "test-model"], ["-m", "test-model"], False),
        (["-m", "test-model"], ["-m", "test-model"], False),
        (["run", "--", "-m", "test-model"], ["-m", "test-model"], False),
        (["resume"], ["resume"], False),
        (["resume", "--last"], ["resume", "--last"], False),
        (["resume", "--last", "--all"], ["resume", "--last", "--all"], False),
        (["resume", "session-id"], ["resume", "session-id"], False),
        (
            ["resume", "session-id", "Continue the previous task"],
            ["resume", "session-id", "Continue the previous task"],
            False,
        ),
        (
            ["resume", "--last", "-m", "test-model"],
            ["resume", "--last", "-m", "test-model"],
            False,
        ),
        (
            ["resume", "--record-model-io", "--last"],
            ["resume", "--last"],
            True,
        ),
        (
            ["resume", "--", "--last", "-m", "test-model"],
            ["resume", "--last", "-m", "test-model"],
            False,
        ),
        (["run", "--", "resume", "--last"], ["resume", "--last"], False),
        (
            ["--", "-c", 'model_reasoning_effort="low"'],
            ["-c", 'model_reasoning_effort="low"'],
            False,
        ),
    ],
)
def test_launch_and_resume_forward_native_options(
    tmp_path, monkeypatch, public_entrypoint, arguments, forwarded, record_io
):
    monkeypatch.chdir(tmp_path)
    config = Config(
        str(tmp_path),
        str(tmp_path / "state"),
        str(tmp_path / "graph-model.json"),
        str(tmp_path / "private.env"),
    )
    loaded, launched = [], []

    def load(path):
        loaded.append(path)
        return config

    def launch(cfg, native_arguments, *, record_io):
        launched.append((cfg, native_arguments, record_io))
        return 7

    monkeypatch.setattr(Config, "load", load)
    monkeypatch.setattr(gateway, "run", launch)
    monkeypatch.setattr(public_cli, "_bootstrap_env", lambda: None)
    original_arguments = list(arguments)
    with pytest.raises(SystemExit) as exc:
        if public_entrypoint:
            public_cli.main(["codex", *arguments])
        else:
            cli.main(arguments)
    assert exc.value.code == 7
    assert loaded == [tmp_path / ".bcg-codex/config.json"]
    assert launched == [(config, forwarded, record_io)]
    assert arguments == original_arguments
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("prefix", [[], ["run"], ["resume"]])
def test_launch_and_resume_accept_custom_bcg_config(tmp_path, monkeypatch, prefix):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "another-project/.bcg-codex/config.json"
    loaded = []
    config = object()

    def load(path):
        loaded.append(path)
        return config

    def launch(cfg, arguments, *, record_io):
        assert cfg is config
        assert arguments == (["resume"] if prefix == ["resume"] else [])
        assert not record_io
        return 0

    monkeypatch.setattr(Config, "load", load)
    monkeypatch.setattr(gateway, "run", launch)
    with pytest.raises(SystemExit) as exc:
        cli.main([*prefix, "--config", str(target)])
    assert exc.value.code == 0
    assert loaded == [target]


@pytest.mark.parametrize("arguments", [["--help"], ["resume", "--help"]])
def test_help_describes_resume_without_loading_config(monkeypatch, capsys, arguments):
    def unexpected_load(_path):
        pytest.fail("help must not load configuration or start Codex")

    monkeypatch.setattr(Config, "load", unexpected_load)
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments)
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "resume" in output
    assert "--last" in output
    assert "SESSION_ID" in output


@pytest.mark.parametrize(
    "arguments",
    [["init", "--unknown"], ["status", "--last"], ["uninstall", "--last"]],
)
def test_administrative_commands_still_reject_unknown_flags(
    tmp_path, monkeypatch, arguments
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments)
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())
