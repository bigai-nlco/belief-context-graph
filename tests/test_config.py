"""Step 4A: unified YAML configuration tests."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from typing import Any

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
    assert settings.backend in {"unified", "hybrid"}
    assert settings.server.port == 8848
    assert settings.model_key == "gpt-5.5"
    assert settings.runner.incremental_merge_threshold == 0.86
    assert settings.runner.verify_merge is False
    assert settings.runner.context_chars == 100000
    assert settings.pipeline.runtime.context_chars == 12000
    assert sources["runner.incremental_merge_threshold"] == "packaged defaults"


def test_schema_rejects_unknown_fields(tmp_path: Path) -> None:
    bad = _write(tmp_path, "bad.yaml", "schema_version: 1\nbackend: unified\nnope: 1\n")
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


def test_graph_reasoning_uses_one_parameter_name(tmp_path: Path) -> None:
    old_name = _write(
        tmp_path,
        "old-thinking.yaml",
        "pipeline:\n  extractor:\n    enable_thinking: false\n",
    )
    with pytest.raises(Exception, match="enable_thinking"):
        load_settings(explicit=str(old_name), home=Path("/nonexistent-home"))

    current = _write(
        tmp_path,
        "reasoning.yaml",
        (
            "models:\n"
            "  graph-model:\n"
            "    reasoning_effort: low\n"
            "pipeline:\n"
            "  extractor:\n"
            "    reasoning_effort: minimal\n"
            "  edge_generation:\n"
            "    reasoning_effort: none\n"
        ),
    )
    settings, _ = load_settings(explicit=str(current), home=Path("/nonexistent-home"))
    assert settings.models["graph-model"].reasoning_effort == "low"
    assert settings.pipeline.extractor.reasoning_effort == "minimal"
    assert settings.pipeline.edge_generation.reasoning_effort == "none"


@pytest.mark.parametrize(
    ("legacy_name", "current_name"),
    [("api_based", "unified"), ("light", "hybrid")],
)
def test_persisted_backend_names_are_migrated(
    tmp_path: Path, legacy_name: str, current_name: str
) -> None:
    config = _write(tmp_path, "legacy.yaml", f"backend: {legacy_name}\n")
    settings, _ = load_settings(explicit=str(config), home=Path("/nonexistent-home"))
    assert settings.backend == current_name


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
    _write(tmp_path, "user.yaml", "server:\n  port: 1111\nbackend: hybrid\n")
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
    nullable = _write(tmp_path, "null.yaml", "server:\n  port: null\n  host: null\n")
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


def test_missing_explicit_or_env_config_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(Exception, match="explicit config file does not exist"):
        load_settings(explicit=str(missing), home=tmp_path / "missing-home")

    monkeypatch.setenv("BCG_CONFIG", str(missing))
    with pytest.raises(Exception, match="config file from BCG_CONFIG does not exist"):
        load_settings(home=tmp_path / "missing-home")


def test_generic_config_yaml_is_not_auto_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "config.yaml", "backend: hybrid\n")
    monkeypatch.chdir(tmp_path)
    settings, _ = load_settings(home=tmp_path / "missing-home")
    assert settings.backend == "unified"


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
    assert DEFAULTS_PATH.read_text(encoding="utf-8") == zipfile.ZipFile(wheels[0]).read(
        "bcg/config/defaults.yaml"
    ).decode("utf-8")


def test_example_config_parses_and_validates() -> None:
    example = Path(__file__).parents[1] / "bcg" / "config" / "config.example.yaml"
    settings, _ = load_settings(explicit=str(example), home=Path("/nonexistent-home"))
    assert settings.models["graph-model"].base_url == "https://api.openai.com/v1"
    assert settings.models["graph-model"].reasoning_effort == "none"
    assert settings.pipeline.extractor.reasoning_effort == "none"
    assert settings.pipeline.edge_generation.reasoning_effort == "none"
    assert settings.pipeline.entities.method == "ml"


def test_yaml_settings_are_consumed_by_both_backend_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write(
        tmp_path,
        "runtime.yaml",
        (
            "model_key: graph-model\n"
            "embedding_key: vectors\n"
            "models:\n"
            "  graph-model:\n"
            "    api_key_env: TEST_GRAPH_KEY\n"
            "    base_url: https://models.example/v1\n"
            "    model: graph-runtime\n"
            "    reasoning_effort: none\n"
            "  vectors:\n"
            "    provider: local\n"
            "    model: /models/vectors\n"
            "pipeline:\n"
            "  runtime:\n"
            "    context_chars: 4321\n"
        ),
    )
    monkeypatch.setenv("TEST_GRAPH_KEY", "secret-from-env")

    from bcg.construct.hybrid import llm as hybrid_llm
    from bcg.construct.unified import llm as unified_llm

    for module in (unified_llm, hybrid_llm):
        model = module.load_config(str(config), model_key="graph-model")
        embedding = module.load_embedding_config(str(config), embedding_key="vectors")
        pipeline = module.load_belief_graph_config(str(config))

        assert model["base_url"] == "https://models.example/v1"
        assert model["api_key"] == "secret-from-env"
        assert model["reasoning_effort"] == "none"
        assert embedding["provider"] == "local"
        assert embedding["model"] == "/models/vectors"
        assert pipeline["runtime"]["context_chars"] == 4321


def test_project_yaml_drives_construct_cli_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(
        tmp_path,
        "bcg.yaml",
        (
            "model_key: custom-model\n"
            "embedding_key: custom-embedding\n"
            "models:\n"
            "  custom-model:\n"
            "    api_key_env: OPENAI_API_KEY\n"
            "    model: custom-model\n"
            "  custom-embedding:\n"
            "    provider: local\n"
            "    model: /models/embedding\n"
            "runner:\n"
            "  incremental_merge_threshold: 0.41\n"
            "  verify_merge: true\n"
        ),
    )
    monkeypatch.chdir(tmp_path)

    from bcg.apps import run
    from bcg.construct.unified import pipeline as unified_pipeline

    captured: dict[str, Any] = {}

    def capture(*args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        captured["config_path"] = args[1]

    monkeypatch.setattr(unified_pipeline, "run_input", capture)
    run._run_unified(["--input", "input.json"])

    assert captured["config_path"] is None
    assert captured["model_key"] == "custom-model"
    assert captured["embedding_key"] == "custom-embedding"
    assert captured["options"].incremental_merge_threshold == 0.41
    assert captured["options"].verify_merge is True
