from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import bcg
from bcg.apps import online_driver, online_server, run
from bcg.construct.api_based import online as api_online
from bcg.construct.api_based import pipeline as api_pipeline
from bcg.construct.api_based.stream import StreamOptions as ApiStreamOptions
from bcg.construct.dispatch import DEFAULT_BACKEND
from bcg.construct.light import online as light_online
from bcg.construct.light import pipeline as light_pipeline
from bcg.construct.light.stream import StreamOptions as LightStreamOptions
from bcg.core.runner import BCGRunner

FIXTURES = Path(__file__).parent / "fixtures" / "refactor"


def _capture_current_defaults(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> dict[str, Any]:
    original_parse_args = argparse.ArgumentParser.parse_args
    captured: list[dict[str, Any]] = []

    def capture_parse_args(
        parser: argparse.ArgumentParser,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = original_parse_args(parser, args, namespace)
        captured.append(vars(parsed).copy())
        return parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_parse_args)
    monkeypatch.setattr(api_pipeline, "run_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(light_pipeline, "run_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_online, "SessionManager", lambda **kwargs: object())
    monkeypatch.setattr(light_online, "SessionManager", lambda **kwargs: object())
    monkeypatch.setattr(online_server, "_serve_forever", lambda *args, **kwargs: None)
    monkeypatch.setattr(online_driver, "_run_stream", lambda *args, **kwargs: None)

    run._run_api_based(["--input", "input.json"])
    run._run_light(["--input", "input.json"])
    online_server._run_api_based([])
    online_server._run_light([])
    online_driver._run_api_based([])
    online_driver._run_light([])
    capsys.readouterr()

    entrypoint_names = (
        "run_api_based",
        "run_light",
        "server_api_based",
        "server_light",
        "replay_api_based",
        "replay_light",
    )
    defaults = dict(zip(entrypoint_names, captured, strict=True))

    observe_signature = inspect.signature(BCGRunner.observe_trajectory)
    runner_signature = inspect.signature(BCGRunner)
    defaults["sdk_runner"] = {
        "instance_backend": runner_signature.parameters["backend"].default,
        "observe_backend_override": observe_signature.parameters["backend"].default,
        **{
            name: observe_signature.parameters[name].default
            for name in (
                "evidence_mode",
                "incremental_merge",
                "incremental_merge_threshold",
                "verify_merge",
                "context_chars",
                "io_context_chars",
                "min_content_len",
            )
        },
    }

    api_options = ApiStreamOptions()
    defaults["api_stream_options"] = {
        name: getattr(api_options, name)
        for name in (
            "evidence_mode",
            "incremental_merge",
            "incremental_merge_threshold",
            "verify_merge",
            "context_chars",
            "min_content_len",
        )
    }

    light_options = LightStreamOptions()
    defaults["light_stream_options"] = {
        name: getattr(light_options, name)
        for name in (
            "evidence_mode",
            "incremental_merge",
            "incremental_merge_threshold",
            "context_chars",
            "min_content_len",
        )
    }
    defaults["construct_default_backend"] = DEFAULT_BACKEND
    return defaults


def test_current_entrypoint_defaults_match_step0_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = json.loads(
        (FIXTURES / "entrypoint_defaults.json").read_text(encoding="utf-8")
    )

    assert _capture_current_defaults(monkeypatch, capsys) == expected


def test_public_exports_match_step5_contract() -> None:
    assert sorted(bcg.__all__) == [
        "BCG",
        "BCGMemory",
        "BCGRunner",
        "BCGSettings",
        "LLMClient",
        "__version__",
        "load_settings",
    ]


def test_import_bcg_does_not_load_apps_or_concrete_backends() -> None:
    project_root = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import bcg, sys; "
                "print(any(m.startswith('bcg.apps') for m in sys.modules)); "
                "print(any(m.startswith('bcg.construct.api_based') for m in sys.modules)); "
                "print(any(m.startswith('bcg.construct.light') for m in sys.modules))"
            ),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.split() == ["False", "False", "False"]


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    [
        ("bcg.core.graph", "BCG"),
        ("bcg.core.memory", "BCGMemory"),
        ("bcg.core.runner", "BCGRunner"),
        ("bcg.core.llm", "LLMClient"),
        ("bcg.core.env", "load_project_env"),
        ("bcg.apps.cli", "main"),
        ("bcg.apps.run", "main"),
        ("bcg.apps.online_server", "main"),
        ("bcg.apps.online_driver", "main"),
        ("bcg.apps.visualize_beliefs_graph", "main"),
        ("bcg.apps.setup", "run_setup"),
        ("bcg.apps.benchmark.cli", "main"),
    ],
)
def test_public_module_paths_remain_importable(
    module_name: str,
    symbol: str,
) -> None:
    module = importlib.import_module(module_name)

    assert hasattr(module, symbol)


@pytest.mark.parametrize(
    ("module_name", "arguments", "expected_option"),
    [
        ("bcg.apps.run", ["api_based", "--help"], "--incremental-merge-threshold"),
        ("bcg.apps.online_server", ["api_based", "--help"], "--port"),
        ("bcg.apps.online_driver", ["api_based", "--help"], "--input"),
        ("bcg.apps.visualize_beliefs_graph", ["--help"], "--output"),
    ],
)
def test_module_help_contract(
    module_name: str,
    arguments: list[str],
    expected_option: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module(module_name)

    with pytest.raises(SystemExit) as exc_info:
        module.main(arguments)

    assert exc_info.value.code == 0
    assert expected_option in capsys.readouterr().out


def test_import_bcg_does_not_load_project_env(tmp_path: Path) -> None:
    """Step-5 contract: plain ``import bcg`` must not touch os.environ."""
    env_file = tmp_path / ".env"
    env_file.write_text("BCG_STEP5_IMPORT_MARKER=loaded-by-import\n", encoding="utf-8")
    child_env = os.environ.copy()
    child_env["BCG_ENV_FILE"] = str(env_file)
    child_env.pop("BCG_STEP5_IMPORT_MARKER", None)
    project_root = Path(__file__).parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import bcg; "
                "print(os.environ.get('BCG_STEP5_IMPORT_MARKER', 'missing'))"
            ),
        ],
        cwd=project_root,
        env=child_env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "missing"


def test_import_bcg_env_no_longer_loads_env(tmp_path: Path) -> None:
    """Step-7 contract: importing the env module must not touch os.environ."""
    env_file = tmp_path / ".env"
    env_file.write_text("BCG_STEP7_ENV_MARKER=loaded-by-import\n", encoding="utf-8")
    child_env = os.environ.copy()
    child_env["BCG_ENV_FILE"] = str(env_file)
    child_env.pop("BCG_STEP7_ENV_MARKER", None)
    project_root = Path(__file__).parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; from bcg.core import env; "
                "print(os.environ.get('BCG_STEP7_ENV_MARKER', 'missing'))"
            ),
        ],
        cwd=project_root,
        env=child_env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "missing"


def test_explicit_load_project_env_still_loads(tmp_path: Path) -> None:
    """The explicit ``load_project_env()`` API keeps working after the change."""
    env_file = tmp_path / ".env"
    env_file.write_text("BCG_STEP7_EXPLICIT=loaded-explicitly\n", encoding="utf-8")
    child_env = os.environ.copy()
    child_env["BCG_ENV_FILE"] = str(env_file)
    child_env.pop("BCG_STEP7_EXPLICIT", None)
    project_root = Path(__file__).parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; from bcg.core.env import load_project_env; "
                "load_project_env(); "
                "print(os.environ.get('BCG_STEP7_EXPLICIT', 'missing'))"
            ),
        ],
        cwd=project_root,
        env=child_env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "loaded-explicitly"


def test_core_env_source_root_tracks_repository_after_move() -> None:
    from bcg.core.env import SOURCE_PROJECT_ROOT

    assert Path(__file__).parents[1] == SOURCE_PROJECT_ROOT


def test_apps_do_not_mutate_sys_path() -> None:
    project_root = Path(__file__).parents[1]
    for relative in (
        "bcg/apps/run.py",
        "bcg/apps/online_server.py",
        "bcg/apps/online_driver.py",
    ):
        source = (project_root / relative).read_text(encoding="utf-8")
        assert "sys.path.insert" not in source


def test_cli_help_describes_effective_boolean_defaults(capsys: Any) -> None:
    with pytest.raises(SystemExit):
        run._run_api_based(["--help"])

    output = capsys.readouterr().out
    assert "Default: ON." in output
    assert "Default: OFF." in output


def test_unified_errors_keep_standard_exception_compatibility() -> None:
    from bcg.construct.backends import resolve_backend
    from bcg.core.errors import BCGConfigError, BCGUsageError
    from bcg.core.graph import BCG
    from bcg.core.memory import BCGMemory

    with pytest.raises(BCGConfigError):
        resolve_backend("missing")

    runner = BCGRunner(memory=BCGMemory(graph=BCG()), llm=object())
    with pytest.raises(BCGUsageError):
        runner.start_session("not-started")

    assert issubclass(BCGConfigError, ValueError)
    assert issubclass(BCGUsageError, RuntimeError)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_step1_dependency_direction_has_no_concrete_backend_imports() -> None:
    project_root = Path(__file__).parents[1]
    runner_imports: set[str] = set()
    for module_path in (project_root / "bcg" / "core").glob("*.py"):
        runner_imports.update(_imported_modules(module_path))
    registry_imports = _imported_modules(
        project_root / "bcg" / "construct" / "backends.py"
    )
    pipeline_imports = _imported_modules(
        project_root / "bcg" / "construct" / "api_based" / "pipeline.py"
    )
    core_imports: set[str] = set()
    for module_path in (project_root / "bcg" / "core").glob("*.py"):
        core_imports.update(_imported_modules(module_path))

    concrete_prefixes = (
        "bcg.construct.api_based",
        "bcg.construct.light",
    )
    assert not any(module.startswith(concrete_prefixes) for module in runner_imports)
    assert not any(module.startswith(concrete_prefixes) for module in registry_imports)
    assert not any(
        module.startswith(concrete_prefixes) or module.startswith("bcg.apps")
        for module in core_imports
    )
    assert "bcg.runner" not in pipeline_imports


def test_step1_legacy_dtos_are_core_contracts() -> None:
    from bcg.construct.api_based.pipeline import (
        BeliefGraphOptions,
        BeliefGraphRunPaths,
        BeliefGraphRunResult,
    )
    from bcg.core.contracts import RunOptions, RunPaths, RunResult

    assert issubclass(BeliefGraphOptions, RunOptions)
    assert BeliefGraphRunPaths is RunPaths
    assert BeliefGraphRunResult is RunResult


def test_step1_registry_backends_implement_protocol() -> None:
    from bcg.construct.backends import resolve_backend
    from bcg.core.contracts import ConstructBackend

    assert isinstance(resolve_backend("api_based"), ConstructBackend)
    assert isinstance(resolve_backend("light"), ConstructBackend)


# ---------------------------------------------------------------------------
# Step 3: shared session state machine and writer components
# ---------------------------------------------------------------------------


def test_step3_backends_share_one_session_class() -> None:
    from bcg.construct._shared.session import (
        StreamingTrajectorySession as SharedSession,
    )
    from bcg.construct.api_based.online import StreamingTrajectorySession as ApiSession
    from bcg.construct.light.online import StreamingTrajectorySession as LightSession

    assert ApiSession is SharedSession
    assert LightSession is SharedSession


def test_step3_resolve_dated_output_root_keeps_plain_paths(tmp_path: Path) -> None:
    from bcg.construct._shared.session import resolve_dated_output_root

    plain = tmp_path / "outputs_stream"
    assert resolve_dated_output_root(plain) == plain
    assert resolve_dated_output_root("outputs_7_6") != "outputs_7_6"


def test_step3_event_recorder_appends_jsonl(tmp_path: Path) -> None:
    from bcg.construct._shared.writers import EventRecorder

    path = tmp_path / "events.jsonl"
    recorder = EventRecorder(path)
    first = recorder.record("turn", {"index": 1})
    second = recorder.record("finalize", {"n_nodes": 3})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "turn"
    assert json.loads(lines[0])["index"] == 1
    assert json.loads(lines[1])["event"] == "finalize"
    assert first["ts"] and second["ts"]


def test_step3_artifact_writer_writes_json_atomically(tmp_path: Path) -> None:
    from bcg.construct._shared.writers import ArtifactWriter

    writer = ArtifactWriter(tmp_path)
    path = writer.write_json("result.json", {"a": [1, 2], "b": "text"})

    assert path == tmp_path / "result.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": [1, 2], "b": "text"}
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".result.json.")]
    assert leftovers == []
