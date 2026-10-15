from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bcg.integrations.codex.cli import main
from bcg.integrations.codex.config import Config, private_write
from bcg.integrations.codex.install import install, uninstall


def initialize(project, **options):
    return install(
        project,
        **{
            "mode": "gateway",
            "env_file": project / "private.env",
            "embedding_model": "local-embedding",
            "allow_upload": True,
            **options,
        },
    )


def files_under(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_gateway_init_only_creates_private_files(tmp_path, monkeypatch):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home / "custom-codex"))
    private_write(home / "custom-codex/config.toml", 'model = "keep"\n')
    private_write(home / "custom-codex/auth.json", {"token": "keep-auth"})
    before = files_under(home)
    env_path = project / "private.env"
    private_write(env_path, "OPENROUTER_OPENAI_API_KEY=never-copy-this-secret\n")
    cfg = initialize(project)
    root = project / ".bcg-codex"
    assert cfg.mode == "gateway"
    assert cfg.enabled
    assert Config.load(root / "config.json") == cfg
    assert not (project / ".codex").exists()
    assert not Path(cfg.state_dir).exists()
    assert files_under(home) == before
    assert env_path.read_text() == "OPENROUTER_OPENAI_API_KEY=never-copy-this-secret\n"
    assert set(files_under(root)) == {
        "config.json",
        "graph-model.json",
        "install.json",
        ".gitignore",
    }
    assert "never-copy-this-secret" not in str(files_under(root))
    graph = json.loads((root / "graph-model.json").read_text())
    assert graph["embedding"]["model"] == "local-embedding"
    assert graph["graph-model"]["api_key_env"] == "OPENROUTER_OPENAI_API_KEY"
    assert (root / ".gitignore").read_text().endswith("*\n!.gitignore\n")
    assert root.stat().st_mode & 0o777 == 0o700
    for path in root.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("symlink_codex", [False, True])
def test_gateway_does_not_read_or_change_existing_codex_config(tmp_path, symlink_codex):
    project = tmp_path / "project"
    project.mkdir()
    target = tmp_path / "codex-config" if symlink_codex else project / ".codex"
    target.mkdir()
    if symlink_codex:
        (project / ".codex").symlink_to(target, target_is_directory=True)
    # Even invalid or hooks-disabled config must be left entirely alone by init.
    private_write(target / "config.toml", "not valid TOML\n[features]\nhooks=false\n")
    private_write(target / "hooks.json", "not valid JSON")
    private_write(target / "auth.json", {"secret": "unchanged"})
    before = files_under(target)
    initialize(project)
    uninstall(project)
    assert files_under(target) == before


def test_gateway_reinit_preserves_custom_settings_graph_and_records(tmp_path):
    cfg = initialize(tmp_path)
    root = tmp_path / ".bcg-codex"
    cfg = replace(cfg, state_dir=str(root / "custom-state"))
    cfg.save(root / "config.json")
    private_write(Path(cfg.graph_config), {"graph-model": {"model": "custom"}})
    private_write(Path(cfg.state_dir) / "graphs/saved.json", {"id": "keep"})
    private_write(root / ".gitignore", "# User comment\n")
    before_graph = Path(cfg.graph_config).read_bytes()
    second = initialize(tmp_path, embedding_model="should-not-overwrite")
    first_files = files_under(root)
    third = initialize(tmp_path)
    assert second == third == cfg
    assert files_under(root) == first_files
    assert Path(cfg.graph_config).read_bytes() == before_graph
    assert (Path(cfg.state_dir) / "graphs/saved.json").exists()
    assert (root / ".gitignore").read_text().startswith("# User comment\n")


def test_gateway_init_can_adopt_manual_config_without_manifest(tmp_path):
    root = tmp_path / ".bcg-codex"
    cfg = Config(
        str(tmp_path),
        str(root / "state"),
        str(root / "graph-model.json"),
        str(tmp_path / "private.env"),
    )
    cfg.save(root / "config.json")
    private_write(Path(cfg.graph_config), {"graph-model": {"model": "existing"}})
    assert initialize(tmp_path) == replace(cfg, mode="gateway")
    assert (
        json.loads(Path(cfg.graph_config).read_text())["graph-model"]["model"]
        == "existing"
    )
    assert not (tmp_path / ".codex").exists()


def test_gateway_refuses_legacy_wiring_without_mutation(tmp_path):
    private_write(
        tmp_path / ".bcg-codex/install.json",
        {"blocks": ["legacy managed block"], "hooks": {"Stop": [{"hooks": []}]}},
    )
    private_write(tmp_path / ".codex/config.toml", "legacy managed block")
    before = files_under(tmp_path)
    with pytest.raises(ValueError, match="archived installer"):
        initialize(tmp_path)
    with pytest.raises(ValueError, match="archived installer"):
        uninstall(tmp_path)
    assert files_under(tmp_path) == before


def test_gateway_uninstall_keeps_memories_without_creating_codex(tmp_path):
    cfg = initialize(tmp_path)
    private_write(Path(cfg.state_dir) / "graphs/saved.json", {"id": "keep"})
    uninstall(tmp_path)
    assert not (tmp_path / ".codex").exists()
    assert not (tmp_path / ".bcg-codex/install.json").exists()
    assert not Config.load(tmp_path / ".bcg-codex/config.json").enabled
    assert Path(cfg.graph_config).exists()
    assert (Path(cfg.state_dir) / "graphs/saved.json").exists()
    assert initialize(tmp_path).enabled


def test_gateway_loads_old_gateway_config_without_unused_capture_settings(tmp_path):
    cfg = initialize(tmp_path)
    path = tmp_path / ".bcg-codex/config.json"
    raw = json.loads(path.read_text())
    raw.update(max_event_chars=12000, context_chars=4400, max_events_per_session=500)
    private_write(path, raw)
    assert Config.load(path) == cfg
    initialize(tmp_path)
    assert "max_event_chars" not in json.loads(path.read_text())


@pytest.mark.parametrize(
    "options,match",
    [
        ({"allow_upload": False}, "--allow-graph-upload"),
        ({"mode": "assist"}, "only gateway"),
        ({"mode": "compact"}, "only gateway"),
        ({"mode": "invalid"}, "only gateway"),
    ],
)
def test_gateway_validation_does_not_write_files(tmp_path, options, match):
    with pytest.raises(ValueError, match=match):
        initialize(tmp_path, **options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "name",
    [
        ".bcg-codex",
        ".bcg-codex/config.json",
        ".bcg-codex/install.json",
        ".bcg-codex/graph-model.json",
        ".bcg-codex/.gitignore",
    ],
)
def test_gateway_rejects_symlinked_private_paths(tmp_path, name):
    project, target = tmp_path / "project", tmp_path / "outside"
    project.mkdir()
    target.mkdir()
    path = project / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked"):
        initialize(project)
    assert not list(target.iterdir())


def test_gateway_cli_init_status_and_run(tmp_path, capsys, monkeypatch):
    from bcg.integrations.codex import gateway

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    main(["init", "--mode", "gateway", "--allow-graph-upload"])
    output = capsys.readouterr().out
    assert "No hooks/MCP installed" in output
    assert "--no-daemon" in output
    assert "review BCG hooks" not in output
    cfg = Config.load(tmp_path / ".bcg-codex/config.json")
    assert cfg.env_file == str(tmp_path / ".bcg/.env")
    assert not Path(cfg.env_file).exists()
    main(["status"])
    status = json.loads(capsys.readouterr().out)
    assert status["mode"] == "gateway"
    assert status["enabled"]
    assert not Path(cfg.state_dir).exists()  # Status does not create a queue or server.
    main(["disable"])
    assert not Config.load(tmp_path / ".bcg-codex/config.json").enabled
    main(["enable"])
    assert Config.load(tmp_path / ".bcg-codex/config.json").enabled

    def launch(config, arguments, *, record_io):
        assert config == cfg
        assert arguments == ["--no-daemon", "-m", "gpt-6-luna"]
        assert record_io
        return 0

    monkeypatch.setattr(gateway, "run", launch)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--mode",
                "gateway",
                "--record-model-io",
                "--",
                "--no-daemon",
                "-m",
                "gpt-6-luna",
            ]
        )
    assert exc.value.code == 0


def test_gateway_init_through_public_cli_does_not_load_credentials(
    tmp_path, monkeypatch
):
    from bcg.apps import cli

    monkeypatch.chdir(tmp_path)

    def unexpected_env_load():
        pytest.fail("initialization must not load model credentials")

    monkeypatch.setattr(cli, "_bootstrap_env", unexpected_env_load)
    cli.main(["codex", "init", "--allow-graph-upload"])
    assert Config.load(tmp_path / ".bcg-codex/config.json").mode == "gateway"
    assert not (tmp_path / ".codex").exists()


@pytest.mark.parametrize(
    "invalid", ["other-project", "missing-graph", "graph-directory"]
)
def test_gateway_reinit_refuses_invalid_paths_before_writing(tmp_path, invalid):
    cfg = initialize(tmp_path)
    if invalid == "other-project":
        cfg = replace(cfg, project=str(tmp_path / "other"))
    elif invalid == "missing-graph":
        cfg = replace(cfg, graph_config=str(tmp_path / "missing.json"))
    else:
        cfg = replace(cfg, graph_config=str(tmp_path))
    cfg.save(tmp_path / ".bcg-codex/config.json")
    before = files_under(tmp_path)
    with pytest.raises(ValueError):
        initialize(tmp_path)
    assert files_under(tmp_path) == before


@pytest.mark.parametrize("command", ["hook", "mcp", "retry"])
def test_gateway_cli_has_no_native_queue_commands(tmp_path, monkeypatch, command):
    initialize(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main([command])
    assert exc.value.code == 2
    assert not (tmp_path / ".bcg-codex/state").exists()


@pytest.mark.parametrize(
    "arguments",
    [["--mode", "assist"], ["--mode", "compact"], ["--compact-at", "24000"]],
)
def test_gateway_cli_has_no_removed_modes(tmp_path, monkeypatch, arguments):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["init", *arguments, "--allow-graph-upload"])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())
