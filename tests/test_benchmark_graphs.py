"""Collecting BCG Graph directories into benchmark run directories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bcg.apps.benchmark import graphs
from bcg.apps.benchmark.cli import app
from bcg.apps.benchmark.models import BenchmarkTask
from bcg.apps.benchmark.runner import RunConfig, run_benchmarks

SESSION = "01a15224-a6cb-7c03-ae85-f22a05aec538"
HEADER = json.dumps({"type": "session", "version": 3, "id": SESSION, "cwd": "/tmp/x"})


def make_store_graph(
    store: Path, session: str = SESSION, suffix: str = "0279d94d"
) -> Path:
    directory = store / f"{session}_{suffix}"
    (directory / "logs").mkdir(parents=True)
    (directory / "trajectory_stream.jsonl").write_text(
        '{"turn": {"role": "user", "content": "Q"}}\n'
    )
    (directory / "belief_graph.jsonl").write_text('{"nodes": [], "relations": []}\n')
    return directory


def test_session_id_comes_from_the_trajectory_header() -> None:
    assert (
        graphs.session_id_from_trajectory(HEADER + '\n{"type": "agent_start"}\n')
        == SESSION
    )
    assert graphs.session_id_from_trajectory('{"type": "agent_start"}\n') is None


def test_collect_moves_graph_and_links_artifacts(tmp_path: Path) -> None:
    store = tmp_path / "store"
    source = make_store_graph(store)
    make_store_graph(store, session="other-session")
    output = tmp_path / "results" / "drb2"
    model_io = output / "deepresearch_bench_ii" / "bcg" / "model-io" / "x.jsonl"
    model_io.parent.mkdir(parents=True)
    model_io.write_text("{}\n")

    outcome = graphs.collect_task_graph(
        trajectory_text=HEADER,
        output_dir=output,
        benchmark="deepresearch_bench_ii",
        mode="bcg",
        task_id="2",
        task_dir_name="2",
        graph_store=store,
        graph_url=None,
        links={
            "model_io": model_io,
            "task_result": output / "deepresearch_bench_ii/bcg/tasks/2.json",
        },
    )
    target = output / "graphs" / "deepresearch_bench_ii" / "bcg" / "2"
    assert outcome["status"] == "moved" and outcome["graph_dir"] == str(
        target.resolve()
    )
    assert not source.exists() and (target / "belief_graph.jsonl").is_file()
    link = json.loads((target / graphs.LINK_NAME).read_text())
    assert link["agent_session_id"] == SESSION
    assert link["model_io"] == "../../../../deepresearch_bench_ii/bcg/model-io/x.jsonl"
    assert (target / link["model_io"]).resolve() == model_io.resolve()
    assert (store / "other-session_0279d94d").is_dir()


def test_collect_copies_when_the_server_still_holds_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    source = make_store_graph(store)
    monkeypatch.setattr(
        graphs, "_live_problem_ids", lambda _url: [f"{SESSION}:0279d94d"]
    )
    outcome = graphs.collect_task_graph(
        trajectory_text=HEADER,
        output_dir=tmp_path / "out",
        benchmark="b",
        mode="bcg",
        task_id="1",
        task_dir_name="1",
        graph_store=store,
        graph_url="http://x",
        release_timeout=0,
    )
    assert outcome["status"] == "copied_active" and source.is_dir()


@pytest.mark.parametrize(
    ("trajectory", "status"),
    [('{"type": "agent_start"}', "no_session_id"), (HEADER, "not_found")],
)
def test_collect_reports_missing_inputs(
    tmp_path: Path, trajectory: str, status: str
) -> None:
    outcome = graphs.collect_task_graph(
        trajectory_text=trajectory,
        output_dir=tmp_path,
        benchmark="b",
        mode="bcg",
        task_id="1",
        task_dir_name="1",
        graph_store=tmp_path / "store",
        graph_url=None,
    )
    assert outcome["status"] == status and outcome["graph_dir"] is None


FAKE_AGENT = r"""
import json, os, sys
from pathlib import Path
session = "01a15224-a6cb-7c03-ae85-f22a05aec538"
print(json.dumps({"type": "session", "version": 3, "id": session}))
# Simulate the Graph server writing this session's directory.
graph = Path(os.environ["FAKE_STORE"]) / f"{session}_abcd"
graph.mkdir(parents=True, exist_ok=True)
(graph / "trajectory_stream.jsonl").write_text("{}\n")
(graph / "belief_graph.jsonl").write_text("{}\n")
Path(os.environ["BCG_MODEL_IO_TRACE_PATH"]).write_text("{}\n")
print(json.dumps({"type": "message_end", "message": {"role": "assistant",
    "content": [{"type": "text", "text": "FINAL ANSWER: A"}], "stopReason": "stop",
    "usage": {"input": 1, "output": 1, "totalTokens": 2}}}))
"""


def test_runner_collects_graphs_for_bcg_mode_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = tmp_path / "agent.py"
    agent.write_text(FAKE_AGENT)
    store = tmp_path / "store"
    monkeypatch.setenv("FAKE_STORE", str(store))
    task = BenchmarkTask("mmlu_pro", "q1", "Pick A", ("A",))
    output = tmp_path / "results"
    run_benchmarks(
        {"mmlu_pro": [task]},
        RunConfig(
            output_dir=output,
            model="fake",
            base_url="https://unused.test/v1",
            modes=("bcg",),
            workers=1,
            agent_command=(sys.executable, str(agent)),
            graph_store_dir=store,
            graph_url="http://127.0.0.1:9",
        ),
        judge=None,
    )
    result = json.loads((output / "mmlu_pro" / "bcg" / "tasks" / "q1.json").read_text())
    assert result["graph_collection"] == "moved"
    target = output / "graphs" / "mmlu_pro" / "bcg" / "q1"
    assert Path(result["graph_dir"]) == target.resolve()
    link = json.loads((target / graphs.LINK_NAME).read_text())
    assert (target / link["task_result"]).resolve() == (
        output / "mmlu_pro/bcg/tasks/q1.json"
    ).resolve()
    assert (target / link["model_io"]).is_file()


def test_collect_graphs_command_backfills_existing_runs(tmp_path: Path) -> None:
    store = tmp_path / "store"
    make_store_graph(store)
    run = tmp_path / "results" / "drb2"
    mode_dir = run / "deepresearch_bench_ii" / "bcg"
    (mode_dir / "trajectories").mkdir(parents=True)
    (mode_dir / "tasks").mkdir()
    trajectory = mode_dir / "trajectories" / "deepresearch_bench_ii-bcg-2.jsonl"
    trajectory.write_text(HEADER + "\n")
    (mode_dir / "tasks" / "2.json").write_text(
        json.dumps(
            {
                "benchmark": "deepresearch_bench_ii",
                "mode": "bcg",
                "task_id": "2",
                "trajectory": "/elsewhere/deepresearch_bench_ii-bcg-2.jsonl",
            }
        )
    )
    runner = CliRunner()
    first = runner.invoke(
        app, ["collect-graphs", str(run), "--graph-store", str(store), "--copy"]
    )
    assert first.exit_code == 0, first.output
    target = run / "graphs" / "deepresearch_bench_ii" / "bcg" / "2"
    assert (target / "belief_graph.jsonl").is_file() and (
        store / f"{SESSION}_0279d94d"
    ).is_dir()
    updated = json.loads((mode_dir / "tasks" / "2.json").read_text())
    assert updated["graph_collection"] == "copied"
    second = runner.invoke(
        app, ["collect-graphs", str(run), "--graph-store", str(store)]
    )
    assert "already_collected" in second.output
