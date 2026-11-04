"""Offline provenance checks: real temporary Git repositories, no remote requests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import dotenv
import pytest

from tools.case_study_studio import run_job, run_version
from tools.case_study_studio.tests.test_control import build_fixture, request_config, write


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-b", "test-version")
    git(tmp_path, "config", "user.name", "Offline Test")
    git(tmp_path, "config", "user.email", "offline@example.invalid")
    write(tmp_path, ".gitignore", ".env\nresults/\n__pycache__/\n")
    write(tmp_path, "workflows/stateful_react_agent/agent.py", "# agent v1\n")
    write(tmp_path, "workflows/stateful_react_agent/prompts/system.md", "Prompt v1\n")
    write(tmp_path, "bcg/construct/unified/stream.py", "# graph v1\n")
    write(tmp_path, "uv.lock", "version = 1\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "test: initial version")
    git(tmp_path, "remote", "add", "origin", "git@github.com:example/project.git")
    return tmp_path


def test_capture_tracks_local_code_not_remote_or_ignored_secrets(checkout: Path) -> None:
    version, manifest = run_version.capture_version(checkout, "bcg")
    assert version["commit"] == git(checkout, "rev-parse", "HEAD")
    assert version["branch"] == "test-version" and version["dirty"] is False
    assert version["repository_url"] == "https://github.com/example/project"
    assert version["fingerprint_complete"] and all(version[key] for key in run_version.FINGERPRINTS)
    assert run_version.public_version(version) == version
    write(checkout, ".env", "SECRET=fake-not-a-key\n")
    write(checkout, "results/some-run/batch.json", {"unrelated": True})
    ignored, same = run_version.capture_version(checkout, "bcg")
    assert ignored["dirty"] is False and same == manifest
    assert "fake-not-a-key" not in json.dumps((ignored, same))
    write(checkout, "workflows/stateful_react_agent/agent.py", "# agent v2\n")
    dirty, _ = run_version.capture_version(checkout, "bcg")
    assert dirty["commit"] == version["commit"] and dirty["dirty"] is True
    assert dirty["agent_sha256"] != version["agent_sha256"]
    assert dirty["source_sha256"] != version["source_sha256"]
    assert dirty["graph_sha256"] == version["graph_sha256"]
    assert dirty["lock_sha256"] == version["lock_sha256"]


def test_untracked_graph_and_lock_changes_are_distinguishable(checkout: Path) -> None:
    before, _ = run_version.capture_version(checkout, "bcg")
    write(checkout, "bcg/new_module.py", "# untracked construction code\n")
    after, _ = run_version.capture_version(checkout, "bcg")
    assert after["dirty"] and after["agent_sha256"] == before["agent_sha256"]
    assert after["graph_sha256"] != before["graph_sha256"]
    write(checkout, "uv.lock", "version = 2\n")
    locked, _ = run_version.capture_version(checkout, "bcg")
    assert locked["lock_sha256"] != after["lock_sha256"]
    default, files = run_version.capture_version(checkout, "default")
    assert default["graph_sha256"] is None
    assert not any(name.startswith("bcg/") for name in files)
    full, files = run_version.capture_version(checkout, "bcg_no_retrieval")
    assert full["graph_sha256"] == locked["graph_sha256"]
    assert "bcg/new_module.py" in files


def test_detached_head_and_nested_non_checkout(checkout: Path) -> None:
    git(checkout, "checkout", "--detach")
    detached = run_version.git_metadata(checkout)
    assert detached["branch"] is None and detached["commit"]
    nested = checkout / "copied-project"
    write(nested, "workflows/agent.py", "# copied source\n")
    archive, _ = run_version.capture_version(nested, "bcg")
    assert archive["commit"] is None and archive["dirty"] is None
    assert archive["agent_sha256"]


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("git", 5)])
def test_missing_or_hung_git_still_fingerprints_sources(checkout: Path, monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise error
    monkeypatch.setattr(run_version.subprocess, "run", fail)
    stamp, _ = run_version.capture_version(checkout, "bcg")
    assert stamp["commit"] is None and stamp["agent_sha256"]


def test_external_symlinks_are_not_hashed(checkout: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = write(tmp_path_factory.mktemp("outside"), "secret.py", "outside checkout\n")
    (checkout / "bcg/external.py").symlink_to(outside)
    stamp, manifest = run_version.capture_version(checkout, "bcg")
    assert "bcg/external.py" not in manifest and not stamp["fingerprint_complete"]


@pytest.mark.parametrize("remote,expected", [
    ("https://user:fake-token@github.com/org/repo.git", "https://github.com/org/repo"),
    ("ssh://git@github.com/org/repo.git", "https://github.com/org/repo"),
    ("https://github.com/org/repo", "https://github.com/org/repo"),
    ("https://github.com.evil.invalid/org/repo", None),
    ("https://github.com/org/..", None), ("/tmp/private/repo", None),
])
def test_remote_is_canonical_and_credential_free(remote: str, expected: str | None) -> None:
    assert run_version.github_repository(remote) == expected


def test_api_never_backfills_or_refreshes_saved_run_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    control = build_fixture(tmp_path)
    user = control.register("version reviewer")["user"]
    monkeypatch.setattr(run_version, "git_metadata", lambda *_: pytest.fail("GET must never read current Git"))
    assert control.detail(1)["code_version"] is None
    queued = control.create_run(request_config(code_version={"commit": "f" * 40}), user)
    assert queued["code_version"] is None  # Client cannot supply the version.
    filename = control.results / "offline-smoke/batch.json"
    batch = json.loads(filename.read_text())
    batch["code_version"] = {"schema_version": 1, "commit": "a" * 40, "branch": "old-version",
                             "dirty": True, "agent_sha256": "b" * 64, "private": "do-not-expose"}
    write(filename.parent, filename.name, batch)
    saved = control.detail(1)["code_version"]
    assert saved["commit"] == "a" * 40 and "private" not in saved
    assert next(item for item in control.list_runs(user) if item["id"] == 1)["code_version"] == saved
    assert control.detail(1)["code_version"] == saved
    assert json.loads(filename.read_text()) == batch


def test_worker_records_commit_after_queueing_even_if_preflight_fails(checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    control = build_fixture(checkout)
    user = control.register("launch-time test")["user"]
    queued = control.create_run(request_config(), user)
    old_commit = git(checkout, "rev-parse", "HEAD")
    assert queued["code_version"] is None
    write(checkout, "workflows/stateful_react_agent/agent.py", "# changed while queued\n")
    git(checkout, "add", "workflows/stateful_react_agent/agent.py")
    git(checkout, "commit", "-m", "test: revision changed while queued")
    new_commit = git(checkout, "rev-parse", "HEAD")
    directory = control.directory(control.row(queued["id"]))
    directory.mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["run_job.py", "--root", str(checkout), "--state", str(control.state),
                                      "--run-id", str(queued["id"])])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(run_job.os, "umask", lambda *_: None)
    monkeypatch.setattr(run_job.signal, "signal", lambda *_: None)
    # Exit before any subprocess, service, or model call, but after stamping.
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *_: {})
    run_job.main()
    saved = control.detail(queued["id"])["code_version"]
    assert saved["commit"] == new_commit and saved["commit"] != old_commit
    assert control.row(queued["id"])["status"] == "failed"
    git(checkout, "checkout", "--detach", old_commit)
    assert control.detail(queued["id"])["code_version"] == saved


def test_public_version_rejects_unknown_and_invalid_fields() -> None:
    assert run_version.public_version(None) is None
    assert run_version.public_version({"schema_version": 2}) is None
    value = run_version.public_version({"schema_version": 1, "commit": "bad", "branch": 5,
                                       "repository_url": "javascript:alert(1)", "dirty": "false"})
    assert value and value["commit"] is None and value["dirty"] is None and value["repository_url"] is None
