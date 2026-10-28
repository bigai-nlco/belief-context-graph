"""Worker integration with fake processes/network; never starts a model."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import dotenv
import httpx
import pytest
import yaml

from tools.case_study_studio import run_job
from tools.case_study_studio.tests.test_control import build_fixture, request_config, write


@pytest.mark.parametrize("mode", ["success", "bad_preflight", "cancel", "missing_key", "legacy_benchmark"])
def test_worker_launch_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    root = tmp_path / "agent"
    control = build_fixture(root)
    user = control.register("operator")["user"]
    run = control.create_run(request_config(profile="gpt6-luna"), user)
    directory = control.directory(control.row(run["id"]))
    directory.mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["run_job.py", "--root", str(root), "--state", str(control.state), "--run-id", str(run["id"])])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.chdir(root)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(run_job.signal, "signal", lambda *_: None)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *_: {} if mode == "missing_key" else {
        "LITELLM_OPENAI_API_KEY": "test-key-not-real", "SERPER_API_KEY": "test-serper-not-real"})
    preflights, children, stopped = [], [], []

    def preflight(cmd: list, **kwargs: object) -> SimpleNamespace:
        preflights.append(cmd)
        assert cmd[-1] == "--dry-run"
        assert cmd[cmd.index("--task-ids") + 1] == "drb2:2"
        assert cmd[cmd.index("--search-calls") + 1] == "0"
        return SimpleNamespace(returncode=1 if mode == "bad_preflight" else 0)

    def spawn(cmd: list, **kwargs: object) -> SimpleNamespace:
        graph = "-c" in cmd
        proc = SimpleNamespace(pid=90000 + len(children), stopped=False, returncode=None if graph else 0)
        proc.poll = lambda: 0 if proc.stopped or not graph else None
        children.append((cmd, proc))
        if not graph:
            write(directory, "drb2/summary.json", {"status_counts": {"completed": 1}})
        return proc

    def stop(proc: SimpleNamespace | None) -> None:
        if proc:
            proc.stopped = True
            stopped.append(proc)

    monkeypatch.setattr(subprocess, "run", preflight)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(run_job, "stop_tree", stop)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: SimpleNamespace(status_code=200))
    if mode == "cancel":
        control.cancel_run(run["id"], user)
    if mode == "legacy_benchmark":
        config = json.loads(control.row(run["id"])["config"])
        config["tasks"] = {"drbench": ["DR0001"]}
        with control.connect() as db:
            db.execute("UPDATE runs SET config=? WHERE id=?", (json.dumps(config), run["id"]))
    run_job.main()
    row = control.row(run["id"])
    batch = json.loads((directory / "batch.json").read_text())
    assert row["finished_at"] and batch["graph_stopped"]
    assert "test-key-not-real" not in (directory / "batch.json").read_text()
    if mode == "success":
        assert row["status"] == "finished"
        assert len(preflights) == 1 and len(children) == 2 and len(stopped) == 2
        graph = yaml.safe_load((directory / "graph.yaml").read_text())
        agent = yaml.safe_load((directory / "agent.yaml").read_text())
        assert graph["models"]["graph"]["reasoning_effort"] == "none"
        assert graph["models"]["graph"]["api_key_env"] == "LITELLM_OPENAI_API_KEY"
        assert agent["llm"]["extra_body"]["reasoning_effort"] == "low"
        assert agent["agent"]["bcg"]["url"] == batch["graph"]["url"]
        assert "test-key-not-real" not in (directory / "agent.yaml").read_text()
    else:
        assert row["status"] == ("cancelled" if mode == "cancel" else "failed")
        assert not children
        if mode == "legacy_benchmark":
            assert not preflights
