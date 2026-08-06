"""Runtime config bridge tests: YAML-first with a legacy JSON fallback window."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bcg.config.runtime import load_construct_config, resolve_runtime_config


def test_no_config_keeps_legacy_fallback_path(tmp_path: Path) -> None:
    """Without any YAML file the resolver points at the legacy JSON path
    (fallback window); loading it with no file is loud."""
    runtime = resolve_runtime_config([])
    assert runtime.uses_yaml is False
    assert runtime.config_path == "bcg/model_config.json"
    raw, _ = load_construct_config(runtime.config_path, required=False)
    assert raw is None


def test_yaml_config_is_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "schema_version: 1\n"
        "models:\n"
        "  graph-model:\n"
        "    base_url: https://yaml.test/v1\n"
        "    api_key_env: OPENAI_API_KEY\n"
        "    max_tokens: 12345\n",
        encoding="utf-8",
    )
    raw, display = load_construct_config(str(cfg), required=True)
    assert raw["graph-model"]["base_url"] == "https://yaml.test/v1"
    assert raw["graph-model"]["max_tokens"] == 12345
    assert "belief_graph" in raw  # pipeline defaults merged


def test_legacy_json_fallback_warns(tmp_path: Path) -> None:
    legacy = tmp_path / "model_config.json"
    legacy.write_text(
        json.dumps(
            {
                "graph-model": {
                    "base_url": "https://json.test/v1",
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "json-model",
                },
                "belief_graph": {"runtime": {"evidence_mode": "sentence"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.warns(DeprecationWarning, match="legacy JSON configuration"):
        raw, _ = load_construct_config(str(legacy), required=True)
    assert raw["graph-model"]["base_url"] == "https://json.test/v1"
    assert raw["belief_graph"]["runtime"]["evidence_mode"] == "sentence"


def test_missing_config_is_loud(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Create a YAML configuration"):
        load_construct_config(str(tmp_path / "missing.json"), required=True)


def test_explicit_json_flag_keeps_fallback(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"m": {"base_url": "x"}}), encoding="utf-8")
    runtime = resolve_runtime_config(["--config", str(legacy)])
    assert runtime.uses_yaml is False
    assert runtime.config_path == str(legacy)
