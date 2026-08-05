"""Step 4A: unified YAML configuration infrastructure tests."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

from bcg.config import load_settings
from bcg.config.loader import DEFAULTS_PATH


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# defaults + schema validation
# ---------------------------------------------------------------------------


def test_defaults_load_and_validate() -> None:
    settings, sources = load_settings(home=Path("/nonexistent-home"))
    assert settings.schema_version == 1
    assert settings.backend in {"api_based", "light"}
    assert settings.server.port == 8848
    assert settings.runner.incremental_merge_threshold == 0.8
    assert settings.cli_defaults.incremental_merge_threshold == 0.86
    assert settings.pipeline.runtime.context_chars == 12000
    assert sources["runner.incremental_merge_threshold"] == "packaged defaults"


def test_schema_rejects_unknown_fields(tmp_path: Path) -> None:
    bad = _write(tmp_path, "bad.yaml", "schema_version: 1\nbackend: api_based\nnope: 1\n")
    with pytest.raises(Exception, match="nope"):
        load_settings(explicit=str(bad), home=Path("/nonexistent-home"))


def test_schema_rejects_type_and_range_errors(tmp_path: Path) -> None:
    bad_port = _write(tmp_path, "port.yaml", "server:\n  port: 99999\n")
    with pytest.raises(Exception, match="port"):
        load_settings(explicit=str(bad_port), home=Path("/nonexistent-home"))

    bad_backend = _write(tmp_path, "backend.yaml", "backend: quantum\n")
    with pytest.raises(Exception, match="backend"):
        load_settings(explicit=str(bad_backend), home=Path("/nonexistent-home"))

    bad_runtime = _write(
        tmp_path, "runtime.yaml", "pipeline:\n  runtime:\n    evidence_mode: freeform\n"
    )
    with pytest.raises(Exception, match="evidence_mode"):
        load_settings(explicit=str(bad_runtime), home=Path("/nonexistent-home"))


def test_schema_version_mismatch_is_rejected(tmp_path: Path) -> None:
    bad = _write(tmp_path, "version.yaml", "schema_version: 999\n")
    with pytest.raises(Exception, match="schema_version"):
        load_settings(explicit=str(bad), home=Path("/nonexistent-home"))


# ---------------------------------------------------------------------------
# precedence and deep merge
# ---------------------------------------------------------------------------


def test_precedence_explicit_beats_env_beats_project_beats_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(
        tmp_path, "user.yaml", "server:\n  port: 1111\nbackend: light\n"
    )
    project = _write(tmp_path, "project.yaml", "server:\n  port: 2222\n")
    env_cfg = _write(tmp_path, "env.yaml", "server:\n  port: 3333\n")
    explicit = _write(tmp_path, "explicit.yaml", "server:\n  port: 4444\n")
    monkeypatch.setenv("BCG_CONFIG", str(env_cfg))
    monkeypatch.chdir(tmp_path)

    settings, sources = load_settings(
        explicit=str(explicit),
        project_names=("project.yaml",),
        home=tmp_path / "home",  # no user file -> skipped
    )
    assert settings.server.port == 4444
    assert sources["server.port"] == str(explicit)

    # user-level file under ~/.bcg/config.yaml (highest of the two local)
    user_dir = tmp_path / "home" / ".bcg"
    user_dir.mkdir(parents=True)
    _write(user_dir, "config.yaml", "server:\n  port: 1111\n")
    settings, sources = load_settings(
        explicit=str(explicit), project_names=("project.yaml",), home=tmp_path / "home"
    )
    assert settings.server.port == 4444

    settings, sources = load_settings(
        project_names=("project.yaml",), home=tmp_path / "home"
    )
    assert settings.server.port == 3333  # env beats project

    monkeypatch.delenv("BCG_CONFIG")
    settings, sources = load_settings(
        project_names=("project.yaml",), home=tmp_path / "home"
    )
    assert settings.server.port == 2222  # project beats user
    assert sources["server.port"] == str(project)

    monkeypatch.setenv("BCG_CONFIG", "")
    monkeypatch.delenv("BCG_CONFIG")
    settings, _ = load_settings(project_names=(), home=tmp_path / "home")
    assert settings.server.port == 1111  # user beats packaged defaults


def test_deep_merge_rules(tmp_path: Path) -> None:
    merged = _write(
        tmp_path,
        "merge.yaml",
        (
            "pipeline:\n"
            "  confidence:\n"
            "    source_weight: 0.9\n"
            "  chunking:\n"
            "    enabled: false\n"
            "runner:\n"
            "  incremental_merge_threshold: 0.5\n"
        ),
    )
    settings, sources = load_settings(
        explicit=str(merged), home=Path("/nonexistent-home")
    )
    # mapping merged recursively: base stance_weight survives
    assert settings.pipeline.confidence.source_weight == 0.9
    assert settings.pipeline.confidence.stance_weight == 0.5
    assert settings.pipeline.chunking.enabled is False
    assert settings.runner.incremental_merge_threshold == 0.5
    assert sources["pipeline.confidence.source_weight"] == str(merged)
    assert sources["pipeline.confidence.stance_weight"] == "packaged defaults"


def test_null_means_fallback(tmp_path: Path) -> None:
    nullable = _write(
        tmp_path, "null.yaml", "server:\n  port: null\n  host: null\n"
    )
    settings, _ = load_settings(explicit=str(nullable), home=Path("/nonexistent-home"))
    assert settings.server.port == 8848  # fell back to defaults
    assert settings.server.host == "127.0.0.1"


def test_cli_overrides_apply_last(tmp_path: Path) -> None:
    cfg = _write(tmp_path, "cli.yaml", "runner:\n  incremental_merge_threshold: 0.42\n")
    settings, sources = load_settings(
        explicit=str(cfg),
        home=Path("/nonexistent-home"),
        cli_overrides={"runner": {"incremental_merge_threshold": 0.99}},
    )
    assert settings.runner.incremental_merge_threshold == 0.99
    assert sources["runner.incremental_merge_threshold"] == "cli"

    settings, _ = load_settings(
        explicit=str(cfg),
        home=Path("/nonexistent-home"),
        cli_overrides={"runner": {"incremental_merge_threshold": None}},
    )
    assert settings.runner.incremental_merge_threshold == 0.42


def test_list_replace_wholesale(tmp_path: Path) -> None:
    listed = _write(
        tmp_path,
        "list.yaml",
        "pipeline:\n  entities:\n    fallback_methods: [rules]\n",
    )
    settings, _ = load_settings(explicit=str(listed), home=Path("/nonexistent-home"))
    assert settings.pipeline.entities.fallback_methods == ["rules"]


def test_missing_user_home_and_empty_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BCG_CONFIG", str(tmp_path / "does-not-exist.yaml"))
    settings, _ = load_settings(home=tmp_path / "missing-home")
    assert settings.backend == "api_based"


# ---------------------------------------------------------------------------
# wheel packaging
# ---------------------------------------------------------------------------


def test_wheel_contains_defaults_example_and_pytyped(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    dist = tmp_path / "dist"
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as zf:
        names = set(zf.namelist())
    assert "bcg/config/defaults.yaml" in names
    assert "bcg/config/config.example.yaml" in names
    assert "bcg/py.typed" in names
    assert DEFAULTS_PATH.read_text(encoding="utf-8") == zipfile.ZipFile(
        wheels[0]
    ).read("bcg/config/defaults.yaml").decode("utf-8")


def test_example_config_parses_and_validates() -> None:
    example = Path(__file__).parents[1] / "bcg" / "config" / "config.example.yaml"
    settings, _ = load_settings(explicit=str(example), home=Path("/nonexistent-home"))
    assert settings.models["graph-model"].base_url == "https://api.openai.com/v1"
    assert settings.pipeline.entities.method == "ml"
