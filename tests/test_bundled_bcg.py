"""The Agent checkout owns its Graph code, configuration, and server entry point."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import bcg
from bcg.config import load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_bcg_resolves_inside_agent_checkout():
    assert Path(bcg.__file__).resolve() == ROOT / "bcg/__init__.py"
    for relative in ("config/defaults.yaml", "config/config.example.yaml", "LICENSE"):
        assert (ROOT / "bcg" / relative).is_file()


def test_studio_graph_configuration_is_unchanged(tmp_path):
    settings, _ = load_settings(
        explicit=str(ROOT / "tools/case_study_studio/profiles/graph.yaml"),
        home=tmp_path,
    )
    assert settings.backend == "unified"
    graph = settings.models[settings.model_key]
    assert graph.model == "gpt-6-luna"
    assert graph.reasoning_effort == "none"
    assert graph.api_key_env == "LITELLM_OPENAI_API_KEY"
    assert settings.runner.context_chars == 12000
    assert settings.runner.verify_merge is False
    assert settings.pipeline.runtime.extraction_history_turns == 2
    assert settings.pipeline.edge_generation.max_previous_windows == 3


def test_graph_entrypoint_works_outside_the_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "bcg.apps.online_server", "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "unified" in result.stdout
    assert "hybrid" in result.stdout
