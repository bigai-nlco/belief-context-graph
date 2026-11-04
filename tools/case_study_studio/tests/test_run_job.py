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

from tools.case_study_studio import run_job, run_version
from tools.case_study_studio.tests.test_control import build_fixture, request_config, write


@pytest.mark.parametrize("mode", ["success", "bad_preflight", "cancel", "missing_key", "legacy_benchmark",
                                 "judge_error", "judge_partial", "judge_cancel", "legacy_no_judge", "default", "bcg_no_retrieval", "task_state"])
def test_worker_launch_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    root = tmp_path / "agent"
    control = build_fixture(root)
    user = control.register("operator")["user"]
    context_mode = mode if mode in ("default", "bcg_no_retrieval") else "bcg"
    run = control.create_run(request_config(profile="gpt6-luna", mode=context_mode, task_state=mode == "task_state"), user)
    if mode == "default":
        # Default must work without the construction profile or Graph dependencies.
        (root / "tools/case_study_studio/profiles/graph.yaml").unlink()
    else:
        write(root, "bcg/construct/unified/stream.py", "# offline Graph source fixture\n")
    directory = control.directory(control.row(run["id"]))
    directory.mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["run_job.py", "--root", str(root), "--state", str(control.state), "--run-id", str(run["id"])])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.chdir(root)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(run_job.signal, "signal", lambda *_: None)
    monkeypatch.setattr(run_version, "git_metadata", lambda *_: {
        "commit": "a" * 40, "branch": "offline-fixture", "dirty": True, "repository_url": None})
    def dotenv_fixture(path: Path) -> dict:
        if mode == "missing_key":
            return {}
        if path == root / ".env":
            return {"LITELLM_OPENAI_API_KEY": None, "JUDGE_API_KEY": ""}
        return {"LITELLM_OPENAI_API_KEY": "test-key-not-real", "SERPER_API_KEY": "test-serper-not-real"}

    monkeypatch.setattr(dotenv, "dotenv_values", dotenv_fixture)
    preflights, children, stopped = [], [], []

    def preflight(cmd: list, **kwargs: object) -> SimpleNamespace:
        assert json.loads((directory / "batch.json").read_text())["code_version"]["commit"] == "a" * 40
        preflights.append(cmd)
        assert cmd[-1] == "--dry-run"
        assert cmd[cmd.index("--task-ids") + 1] == "drb2:2"
        assert cmd[cmd.index("--search-calls") + 1] == "0"
        return SimpleNamespace(returncode=1 if mode == "bad_preflight" else 0)

    def spawn(cmd: list, **kwargs: object) -> SimpleNamespace:
        graph = "bcg.apps.online_server" in cmd
        judge = "judge" in cmd
        if graph:
            assert cmd[:4] == [sys.executable, "-u", "-m", "bcg.apps.online_server"]
            assert kwargs["cwd"] == root
            assert "belief-context-graph-gpt56-luna" not in " ".join(cmd)
        proc = SimpleNamespace(pid=90000 + len(children), stopped=False, returncode=None if graph else 0)
        proc.poll = lambda: 0 if proc.stopped or not graph else None
        children.append((cmd, proc))
        if judge:
            if mode != "default":
                assert children[0][1].stopped  # No graph service is needed to grade a report.
            assert cmd[cmd.index("--out") + 1] == str(directory / "drb2")
            assert "--force" not in cmd
            assert os.environ["JUDGE_API_KEY"] == "test-key-not-real"
            assert os.environ["JUDGE_BASE_URL"] == "https://litellm.mybigai.ac.cn/"
            if mode == "judge_error":
                proc.returncode = 2
            elif mode == "judge_cancel":
                proc.returncode = None
                proc.poll = lambda: 0 if proc.stopped else None
                control.cancel_run(run["id"], user)
            else:
                write(directory, "drb2/summary.json", {"status_counts": {"completed": 1},
                    "rubric": {"complete": mode != "judge_partial"}})
        elif not graph:
            write(directory, "drb2/summary.json", {"status_counts": {"completed": 1}})
        return proc

    def stop(proc: SimpleNamespace | None) -> None:
        if proc:
            proc.stopped = True
            stopped.append(proc)

    monkeypatch.setattr(subprocess, "run", preflight)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(run_job, "stop_tree", stop)
    def health(*args: object, **kwargs: object) -> SimpleNamespace:
        assert mode != "default", "Default must not query a Graph service"
        return SimpleNamespace(status_code=200)
    monkeypatch.setattr(httpx, "get", health)
    if mode == "cancel":
        control.cancel_run(run["id"], user)
    if mode in {"legacy_benchmark", "legacy_no_judge"}:
        config = json.loads(control.row(run["id"])["config"])
        if mode == "legacy_benchmark":
            config["tasks"] = {"drbench": ["DR0001"]}
        else:
            config.pop("judge_reports")
        with control.connect() as db:
            db.execute("UPDATE runs SET config=? WHERE id=?", (json.dumps(config), run["id"]))
    run_job.main()
    row = control.row(run["id"])
    batch = json.loads((directory / "batch.json").read_text())
    assert batch["code_version"]["commit"] == "a" * 40
    assert batch["code_version"]["branch"] == "offline-fixture"
    assert batch["code_version"]["dirty"] is True
    assert control.detail(run["id"])["code_version"] == run_version.public_version(batch["code_version"])
    assert row["finished_at"] and batch["graph_stopped"]
    assert "test-key-not-real" not in (directory / "batch.json").read_text()
    if mode == "default":
        assert row["status"] == "finished" and batch["mode"] == "default"
        assert len(children) == 2 and len(preflights) == 1
        assert not any("bcg.apps.online_server" in cmd for cmd, _ in children), "no Graph subprocess"
        assert batch["graph"] == {"enabled": False}
        assert batch["judge"]["model"] == "gpt-6-luna" and batch["judge"]["reasoning_effort"] == "high"
        assert not (directory / "graph.yaml").exists() and not (directory / "graph-server.log").exists()
        agent = yaml.safe_load((directory / "agent.yaml").read_text())
        assert agent["agent"]["context_mode"] == "default" and "bcg" not in agent["agent"]
        assert agent["llm"]["extra_body"]["reasoning_effort"] == "low"
    elif mode in {"success", "legacy_no_judge", "bcg_no_retrieval", "task_state"}:
        assert row["status"] == "finished"
        assert len(preflights) == 1
        assert len(children) == (2 if mode == "legacy_no_judge" else 3)
        assert all(child.stopped or child.returncode is not None for _, child in children)
        if mode != "legacy_no_judge":
            assert batch["judge"]["finished_at"] and batch["exit_codes"]["drb2_judge"] == 0
            assert batch["judge"]["model"] == "gpt-6-luna"
            assert batch["judge"]["reasoning_effort"] == "high"
        graph = yaml.safe_load((directory / "graph.yaml").read_text())
        agent = yaml.safe_load((directory / "agent.yaml").read_text())
        assert graph["models"]["graph"]["reasoning_effort"] == "none"
        assert bool((graph.get("pipeline", {}).get("task_state") or {}).get("enabled")) == (mode == "task_state")
        assert agent["agent"]["bcg"].get("task_state_enabled", False) == (mode == "task_state")
        assert agent["agent"]["context_mode"] == context_mode and batch["mode"] == context_mode
        if mode == "bcg_no_retrieval":
            assert agent["agent"]["bcg"]["retrieval"] == "all"
            assert batch["code_version"]["graph_sha256"]
        assert agent["agent"]["bcg"]["recent_turns"] == 2
        assert graph["models"]["graph"]["api_key_env"] == "LITELLM_OPENAI_API_KEY"
        assert "bcg/construct/unified/stream.py" in batch["source_sha256"]
        assert agent["llm"]["extra_body"]["reasoning_effort"] == "low"
        assert agent["agent"]["bcg"]["url"] == batch["graph"]["url"]
        assert "test-key-not-real" not in (directory / "agent.yaml").read_text()
    elif mode.startswith("judge_"):
        assert row["status"] == ("cancelled" if mode == "judge_cancel" else "finished_with_judge_errors")
        assert batch["phase"] == "judging"
        assert (directory / "drb2/summary.json").is_file()
        assert children[-1][1].stopped
        if mode != "judge_cancel":
            assert "报告已保留" in row["error"]
    else:
        assert row["status"] == ("cancelled" if mode == "cancel" else "failed")
        assert not children
        if mode == "legacy_benchmark":
            assert not preflights


def test_judge_configuration_does_not_inherit_local_agent_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", {"OPENAI_API_KEY": "EMPTY", "OPENAI_BASE_URL": "http://localhost:8001/v1"})
    public = run_job.configure_judge({"LITELLM_OPENAI_API_KEY": "private-judge-key"})
    assert os.environ["JUDGE_API_KEY"] == "private-judge-key"
    assert os.environ["JUDGE_BASE_URL"] == "https://litellm.mybigai.ac.cn/"
    assert public["model"] == "gpt-6-luna" and public["reasoning_effort"] == "high"
    assert "private-judge-key" not in json.dumps(public)
    public = run_job.configure_judge({"JUDGE_API_KEY": "explicit-key", "JUDGE_BASE_URL": "https://judge.example/v1",
        "JUDGE_MODEL": "custom-judge", "DRB2_JUDGE_REASONING_EFFORT": "low"})
    assert os.environ["JUDGE_API_KEY"] == "explicit-key"
    assert os.environ["JUDGE_BASE_URL"] == "https://judge.example/v1"
    assert public["model"] == "custom-judge" and public["reasoning_effort"] == "low"
