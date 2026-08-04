from __future__ import annotations

import argparse
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
from bcg import online_driver, online_server, run
from bcg.construct.api_based import online as api_online
from bcg.construct.api_based import pipeline as api_pipeline
from bcg.construct.api_based.stream import StreamOptions as ApiStreamOptions
from bcg.construct.dispatch import DEFAULT_BACKEND
from bcg.construct.light import online as light_online
from bcg.construct.light import pipeline as light_pipeline
from bcg.construct.light.stream import StreamOptions as LightStreamOptions
from bcg.runner import BCGRunner

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


def test_public_exports_match_step0_contract() -> None:
    assert sorted(bcg.__all__) == [
        "BCG",
        "BCGMemory",
        "BCGRunner",
        "PROJECT_ENV_FILE",
    ]


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    [
        ("bcg.graph", "BCG"),
        ("bcg.memory", "BCGMemory"),
        ("bcg.runner", "BCGRunner"),
        ("bcg.llm", "LLMClient"),
        ("bcg.env", "load_project_env"),
        ("bcg.cli", "main"),
        ("bcg.run", "main"),
        ("bcg.online_server", "main"),
        ("bcg.online_driver", "main"),
        ("bcg.visualize_beliefs_graph", "main"),
        ("bcg.setup", "run_setup"),
        ("bcg.benchmark.cli", "main"),
    ],
)
def test_public_and_legacy_module_paths_remain_importable(
    module_name: str,
    symbol: str,
) -> None:
    module = importlib.import_module(module_name)

    assert hasattr(module, symbol)


@pytest.mark.parametrize(
    ("module_name", "arguments", "expected_option"),
    [
        ("bcg.run", ["api_based", "--help"], "--incremental-merge-threshold"),
        ("bcg.online_server", ["api_based", "--help"], "--port"),
        ("bcg.online_driver", ["api_based", "--help"], "--input"),
        ("bcg.visualize_beliefs_graph", ["--help"], "--output"),
    ],
)
def test_legacy_module_help_contract(
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


def test_import_bcg_currently_loads_explicit_project_env(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("BCG_STEP0_IMPORT_MARKER=loaded-by-import\n", encoding="utf-8")
    child_env = os.environ.copy()
    child_env["BCG_ENV_FILE"] = str(env_file)
    child_env.pop("BCG_STEP0_IMPORT_MARKER", None)
    project_root = Path(__file__).parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import bcg; "
                "print(os.environ.get('BCG_STEP0_IMPORT_MARKER', 'missing'))"
            ),
        ],
        cwd=project_root,
        env=child_env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "loaded-by-import"
