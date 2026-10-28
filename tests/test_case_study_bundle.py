"""Case export is offline, task-scoped, private and directly browser-importable."""

from __future__ import annotations

import gzip
import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.public.core.kernel_adapter import BenchmarkSession
from workflows._shared import case_study as bundles


def write(root: Path, name: str, value: object) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    return path


def fixture(root: Path, task_id: str = "42", mode: str = "bcg") -> None:
    write(root, "result.json", {
        "task_id": task_id, "question": f"Question {task_id}", "benchmark": "offline",
        "mode": mode, "final_response": "# Final\n2024", "status": "completed",
    })
    write(root, "agent/trajectories/react_agent.jsonl",
          '{"t":"start","user_message":"Q"}\n{"t":"llm","content":"2024"}\n')
    write(root, "agent/trajectories/react_agent.json", {"duplicate": True})
    write(root, "agent/bcg/state.json", {
        "problem_id": "graph-42", "snapshot": {
            "nodes": [{"id": 1, "belief": "Answer 2024", "confidence": 0.8}],
            "relations": [],
        },
    })
    write(root, "agent/bcg/requests.jsonl",
          '{"call":1,"system_memory":"B1","selected_node_ids":[1]}\n')
    write(root, "agent/bcg/graph_snapshots.jsonl",
          '{"call":1,"snapshot":{"nodes":[{"id":1,"belief":"2024"}],"relations":[]}}\n')


def test_case_contains_required_evidence_but_not_configs_or_duplicates(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setenv("PRIVATE_API_KEY", "secret-value-do-not-export")
    write(tmp_path, ".env", "PRIVATE_API_KEY=secret-value-do-not-export")
    write(tmp_path, "profile.json", {"key": "secret-value-do-not-export"})
    write(tmp_path, "sandbox/outputs/private.json", {"private": True})
    write(tmp_path, "agent/trajectories/headers.jsonl",
          '{"authorization":"Bearer other-secret","content":"secret-value-do-not-export"}\n')
    packed = bundles.export_case(tmp_path)
    data = bundles.read_bundle(packed)
    files = {f["path"]: f for f in data["cases"][0]["files"]}
    assert "case.json" in files and "agent/bcg/graph_snapshots.jsonl" in files
    assert not any(p in files for p in (".env", "profile.json", "sandbox/outputs/private.json"))
    assert "agent/trajectories/react_agent.json" not in files
    assert "other-secret" not in json.dumps(data)
    assert "secret-value-do-not-export" not in json.dumps(data)
    for file in files.values():
        raw = file["text"].encode()
        assert file["size"] == len(raw)
        assert file["digest"] == hashlib.sha256(raw).hexdigest()
    assert stat.S_IMODE(packed.stat().st_mode) == 0o600
    assert data["cases"][0]["warnings"] == []
    before = packed.read_bytes()
    bundles.export_case(tmp_path)
    assert packed.read_bytes() == before


def test_case_keeps_task_mode_and_attempt_separate_and_batches_once(tmp_path):
    for folder, mode in (("a", "bcg"), ("b", "default"), ("c", "bcg")):
        trial = tmp_path / "trials" / folder
        fixture(trial, mode=mode)
        bundles.export_case(trial)
    output = bundles.export_batch(tmp_path, task_ids=["a", "b", "c", "a", "missing"])
    cases = bundles.read_bundle(output[0])["cases"]
    assert len(cases) == 3
    assert len({c["id"] for c in cases}) == 3
    assert [c["mode"] for c in cases] == ["bcg", "default", "bcg"]


def test_legacy_server_history_only_from_matching_graph_and_no_symlink_escape(tmp_path):
    root, outside = tmp_path / "trial", tmp_path / "outside"
    fixture(root)
    write(outside, "private.jsonl", '{"secret":"not included"}')
    (root / "agent/trajectories/leak.jsonl").symlink_to(outside / "private.jsonl")
    write(outside, "graphs/graph-42/belief_graph.jsonl", '{"nodes":[{"id":1}]}')
    write(outside, "graphs/another-case/belief_graph.jsonl", '{"nodes":[{"id":99}]}')
    data = bundles.read_bundle(bundles.export_case(root, graph_dir=outside / "graphs"))
    paths = [f["path"] for f in data["cases"][0]["files"]]
    assert "graphs/graph-42/belief_graph.jsonl" in paths
    assert not any("another-case" in p or "leak" in p for p in paths)


def test_missing_and_failed_cases_remain_importable_with_explicit_warnings(tmp_path):
    write(tmp_path, "result.json", {"task_id": "1", "status": "timeout", "benchmark": "offline"})
    data = bundles.read_bundle(bundles.export_case(tmp_path))
    case = data["cases"][0]
    assert len(case["warnings"]) == 3
    assert json.loads(case["files"][0]["text"])["status"] == "timeout"
    assert json.loads(case["files"][0]["text"])["final_response"] == ""


def test_oversize_export_leaves_previous_archive_intact(tmp_path, monkeypatch):
    fixture(tmp_path)
    target = bundles.export_case(tmp_path)
    before = target.read_bytes()
    monkeypatch.setattr(bundles, "MAX_BYTES", 50)
    with pytest.raises(ValueError):
        bundles.export_case(tmp_path)
    assert target.read_bytes() == before
    assert not list(tmp_path.glob(".case-study-*"))


def test_split_batch_respects_import_limit(tmp_path, monkeypatch):
    for i in range(3):
        root = tmp_path / "trials" / str(i)
        fixture(root, str(i))
        bundles.export_case(root)
    size = len(gzip.decompress((tmp_path / "trials/0" / bundles.CASE_NAME).read_bytes()))
    monkeypatch.setattr(bundles, "MAX_BYTES", size + 1500)
    paths = bundles.export_batch(tmp_path)
    assert len(paths) == 3
    assert sum(len(bundles.read_bundle(p)["cases"]) for p in paths) == 3
    monkeypatch.setattr(bundles, "MAX_BYTES", 250 * 1024 * 1024)
    outputs = bundles.export_batch(tmp_path)
    assert outputs == [tmp_path / bundles.BATCH_NAME]
    assert not list(tmp_path.glob("case-studies-*.json.gz"))


def test_sdk_task_layout_collects_only_its_own_traces(tmp_path):
    root = tmp_path / "trajectories/task-one"
    write(root, "react_agent.jsonl", '{"t":"llm","content":"Only task one"}')
    write(root, "subagents/researcher.jsonl", '{"t":"llm","content":"Specialist"}')
    write(tmp_path, "trajectories/task-two/react_agent.jsonl", '{"t":"llm","content":"Other task"}')
    data = bundles.read_bundle(bundles.export_case(root))
    files = {f["path"]:f for f in data["cases"][0]["files"]}
    assert "trajectories/react_agent.jsonl" in files
    assert "trajectories/subagents/researcher.jsonl" in files
    assert "Other task" not in json.dumps(data)


@pytest.mark.asyncio
@pytest.mark.parametrize("pipeline", ["stateful-react-agent", "agent_team", "agent_team_report"])
async def test_session_automatically_exports_after_final_pipeline_state(tmp_path, pipeline):
    fixture(tmp_path)

    class Scheduler:
        async def execute(self, *args, **kwargs):
            yield "done", {}

        async def get_state(self, task_id):
            return {"final_answer": "Delivered report", "answer_status": "complete"}

    class Manager:
        async def create_task(self, instruction):
            return SimpleNamespace(id="internal-task")

    session = BenchmarkSession()
    session._scheduler, session._pm = Scheduler(), Manager()
    result = await session.run("Full question", pipeline_id=pipeline, meta={
        "_trial_dir": str(tmp_path), "question_id": "42", "benchmark": "offline",
    })
    assert result["final_answer"] == "Delivered report"
    case = bundles.read_bundle(tmp_path / bundles.CASE_NAME)["cases"][0]
    info = json.loads(case["files"][0]["text"])
    assert info["pipeline"] == pipeline and info["question"] == "Full question"
    assert info["final_response"] == "Delivered report"


def test_cli_packages_existing_batch_without_agent_or_graph_calls(tmp_path):
    import subprocess
    import sys

    fixture(tmp_path / "trials/a")
    fixture(tmp_path / "trials/b", "43")
    subprocess.run([sys.executable, "-m", "workflows._shared.case_study", str(tmp_path)], check=True)
    assert len(bundles.read_bundle(tmp_path / bundles.BATCH_NAME)["cases"]) == 2
