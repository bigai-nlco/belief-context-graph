"""Step 4A/4B: unified YAML configuration and legacy migration tests."""

from __future__ import annotations

import json
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
    assert settings.backend in {"api_based", "light"}
    assert settings.server.port == 8848
    assert settings.model_key == "gpt-5.5"
    assert settings.runner.incremental_merge_threshold == 0.8
    assert settings.pipeline.runtime.context_chars == 12000
    assert sources["runner.incremental_merge_threshold"] == "packaged defaults"


def test_schema_rejects_unknown_fields(tmp_path: Path) -> None:
    bad = _write(
        tmp_path, "bad.yaml", "schema_version: 1\nbackend: api_based\nnope: 1\n"
    )
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
    _write(tmp_path, "user.yaml", "server:\n  port: 1111\nbackend: light\n")
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
    _write(tmp_path, "config.yaml", "backend: light\n")
    monkeypatch.chdir(tmp_path)
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
    assert DEFAULTS_PATH.read_text(encoding="utf-8") == zipfile.ZipFile(wheels[0]).read(
        "bcg/config/defaults.yaml"
    ).decode("utf-8")


def test_example_config_parses_and_validates() -> None:
    example = Path(__file__).parents[1] / "bcg" / "config" / "config.example.yaml"
    settings, _ = load_settings(explicit=str(example), home=Path("/nonexistent-home"))
    assert settings.models["graph-model"].base_url == "https://api.openai.com/v1"
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
            "  vectors:\n"
            "    provider: local\n"
            "    model: /models/vectors\n"
            "pipeline:\n"
            "  runtime:\n"
            "    context_chars: 4321\n"
        ),
    )
    monkeypatch.setenv("TEST_GRAPH_KEY", "secret-from-env")

    from bcg.construct.api_based import llm as api_llm
    from bcg.construct.light import llm as light_llm

    for module in (api_llm, light_llm):
        model = module.load_config(str(config), model_key="graph-model")
        embedding = module.load_embedding_config(str(config), embedding_key="vectors")
        pipeline = module.load_belief_graph_config(str(config))

        assert model["base_url"] == "https://models.example/v1"
        assert model["api_key"] == "secret-from-env"
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
    from bcg.construct.api_based import pipeline as api_pipeline

    captured: dict[str, Any] = {}

    def capture(*args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        captured["config_path"] = args[1]

    monkeypatch.setattr(api_pipeline, "run_input", capture)
    run._run_api_based(["--input", "input.json"])

    assert captured["config_path"] is None
    assert captured["model_key"] == "custom-model"
    assert captured["embedding_key"] == "custom-embedding"
    assert captured["options"].incremental_merge_threshold == 0.41
    assert captured["options"].verify_merge is True


# ---------------------------------------------------------------------------
# Step 4B: legacy JSON migration
# ---------------------------------------------------------------------------

_LEGACY_MODEL_CONFIG = {
    "_comment": "legacy top-level comment",
    "gpt-5.5": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://example.test/v1",
        "max_tokens": 100000,
        "temperature": 1,
        "pricing": {"input_per_1k": 0.005, "output_per_1k": 0.03},
    },
    "embedding": {
        "provider": "local",
        "model": "/models/all-MiniLM-L6-v2",
    },
    "belief_graph": {
        "_comment": "pipeline comment",
        "runtime": {
            "evidence_mode": "chunk",
            "context_chars": 12000,
            "min_content_len": 0,
        },
        "incremental_merge": {
            "enabled": True,
            "threshold": 0.76,
            "keep_newest_text": False,
        },
    },
}

_LEGACY_USER_CONFIG = {
    "version": 1,
    "setupComplete": True,
    "agent": {
        "authMethod": "api_key",
        "baseUrl": "https://agent.test/v1",
        "model": "gpt-5.5",
    },
    "context": {"mode": "bcg", "recentTurns": 2},
    "graph": {
        "serverMode": "managed",
        "backend": "light",
        "url": "",
        "modelConfig": "",
        "modelKey": "graph-model",
        "embeddingKey": "embedding",
        "modelBaseUrl": "http://localhost:8001/v1",
        "model": "Qwen3.5-4B",
    },
}


def test_migrate_model_config_maps_sections(tmp_path: Path) -> None:
    from bcg.config import migrate_model_config

    path = tmp_path / "model_config.json"
    path.write_text(json.dumps(_LEGACY_MODEL_CONFIG), encoding="utf-8")
    out = migrate_model_config(path)

    assert "_comment" not in out
    assert out["models"]["gpt-5.5"]["base_url"] == "https://example.test/v1"
    assert out["models"]["embedding"]["model"] == "/models/all-MiniLM-L6-v2"
    assert out["model_key"] == "gpt-5.5"
    assert out["embedding_key"] == "embedding"
    assert out["pipeline"]["runtime"]["context_chars"] == 12000
    assert "_comment" not in out["pipeline"]


def test_migrate_model_config_handles_custom_embedding_and_invalid_json(
    tmp_path: Path,
) -> None:
    from bcg.config import migrate_model_config
    from bcg.core.errors import BCGConfigError

    path = tmp_path / "model_config.json"
    path.write_text(
        json.dumps({"embedding_local": {"provider": "local"}, "chat": {}}),
        encoding="utf-8",
    )
    out = migrate_model_config(path)
    assert out["model_key"] == "chat"
    assert out["embedding_key"] == "embedding_local"

    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(BCGConfigError, match="invalid JSON configuration"):
        migrate_model_config(path)


def test_migrate_model_config_drops_inline_secrets(tmp_path: Path) -> None:
    from bcg.config import migrate_model_config

    cfg = {
        "gpt-5.5": {
            "api_key": "sk-super-secret",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://example.test/v1",
            "model_kwargs": {
                "providers": [{"api_key": "nested-secret", "name": "fallback"}]
            },
        },
        "belief_graph": {
            "extractor": {
                "api_key": "pipeline-secret",
                "api_key_env": "EXTRACTOR_KEY",
            }
        },
    }
    path = tmp_path / "model_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.warns(UserWarning, match="dropped inline api_key") as caught:
        out = migrate_model_config(path)
    assert "api_key" not in out["models"]["gpt-5.5"]
    assert out["models"]["gpt-5.5"]["api_key_env"] == "OPENAI_API_KEY"
    assert len(caught) == 3
    nested = out["models"]["gpt-5.5"]["model_kwargs"]["providers"][0]
    assert nested == {"name": "fallback"}
    assert "api_key" not in out["pipeline"]["extractor"]
    assert out["pipeline"]["extractor"]["api_key_env"] == "EXTRACTOR_KEY"


def test_migrate_user_config_maps_backend_and_merges_models(tmp_path: Path) -> None:
    from bcg.config import migrate_user_config

    user_path = tmp_path / "config.json"
    user_path.write_text(json.dumps(_LEGACY_USER_CONFIG), encoding="utf-8")
    model_path = tmp_path / "model_config.json"
    model_path.write_text(json.dumps(_LEGACY_MODEL_CONFIG), encoding="utf-8")

    out = migrate_user_config(user_path, model_config_path=model_path)
    assert out["backend"] == "light"
    assert out["models"]["graph-model"]["base_url"] == "http://localhost:8001/v1"
    assert out["model_key"] == "graph-model"
    assert out["embedding_key"] == "embedding"
    assert out["models"]["gpt-5.5"]["base_url"] == "https://example.test/v1"
    assert out["pipeline"]["runtime"]["evidence_mode"] == "chunk"


def test_legacy_settings_warns_and_builds_settings(tmp_path: Path) -> None:
    from bcg.config import legacy_settings

    (tmp_path / "model_config.json").write_text(
        json.dumps(_LEGACY_MODEL_CONFIG), encoding="utf-8"
    )
    with pytest.warns(DeprecationWarning, match="legacy configuration"):
        out = legacy_settings(project_root=tmp_path, home=tmp_path / "no-home")
    assert out["models"]["gpt-5.5"]["base_url"] == "https://example.test/v1"
    assert out["pipeline"]["incremental_merge"]["threshold"] == 0.76


def test_migrate_to_yaml_is_atomic_idempotent_and_validates(tmp_path: Path) -> None:
    from bcg.config import migrate_to_yaml

    (tmp_path / "model_config.json").write_text(
        json.dumps(_LEGACY_MODEL_CONFIG), encoding="utf-8"
    )
    dest = tmp_path / "out" / "config.yaml"
    written = migrate_to_yaml(dest, project_root=tmp_path, home=tmp_path / "no-home")
    assert written == dest
    assert dest.is_file()
    assert not list(dest.parent.glob(f".{dest.name}.*.tmp"))

    # idempotent: second run succeeds and backs up the first output
    written_again = migrate_to_yaml(
        dest, project_root=tmp_path, home=tmp_path / "no-home"
    )
    assert written_again == dest
    assert dest.with_suffix(dest.suffix + ".bak").is_file()

    # migrated YAML validates against the schema
    settings, sources = load_settings(explicit=str(dest), home=tmp_path / "no-home")
    original_backup = dest.with_suffix(dest.suffix + ".bak").read_bytes()

    migrate_to_yaml(dest, project_root=tmp_path, home=tmp_path / "no-home")
    assert dest.with_suffix(dest.suffix + ".bak").read_bytes() == original_backup
    assert settings.pipeline.incremental_merge.threshold == 0.76
    assert sources["pipeline.incremental_merge.threshold"] == str(dest)


def test_migrate_to_yaml_fails_without_legacy_files(tmp_path: Path) -> None:
    from bcg.config import migrate_to_yaml

    with pytest.raises(FileNotFoundError, match="no legacy configuration"):
        migrate_to_yaml(
            tmp_path / "x.yaml", project_root=tmp_path, home=tmp_path / "no"
        )
