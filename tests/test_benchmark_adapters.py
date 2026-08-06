from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from threading import Event

import pytest

from bcg.apps.benchmark.loaders import BenchmarkDataError, load_benchmark
from bcg.apps.benchmark.models import BenchmarkTask
from bcg.apps.benchmark.runner import (
    APIQuotaExhaustedError,
    RunConfig,
    _execute,
    _interleaved_work,
    parse_agent_events,
    run_benchmarks,
    summarize_results,
)
from bcg.apps.benchmark.scoring import (
    JudgeConfig,
    LLMJudge,
    extract_multiple_choice,
    gaia_match,
    score_hotpotqa,
    score_mmlu_pro,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_loads_all_four_benchmark_schemas(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "browse_comp" / "data.json",
        [{"task_id": "bc-1", "input": "Find it", "ground_truth_answer": "answer"}],
    )
    _write_json(
        tmp_path / "hotpotqa" / "data.json",
        [{"_id": "hp-1", "question": "Who?", "answer": "Ada"}],
    )
    _write_json(
        tmp_path / "mmlu_pro" / "data.json",
        [
            {
                "question_id": "mmlu-1",
                "question": "Choose",
                "options": [f"option {index}" for index in range(10)],
                "answer": "J",
                "category": "math",
            }
        ],
    )
    gaia_dir = tmp_path / "gaia" / "2023" / "validation"
    gaia_dir.mkdir(parents=True)
    (gaia_dir / "metadata.jsonl").write_text(
        json.dumps(
            {
                "task_id": "gaia-1",
                "Question": "What?",
                "Final answer": "42",
                "Level": 1,
                "file_name": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    browsecomp = load_benchmark("browsecomp", tmp_path)
    hotpot = load_benchmark("hotpotqa", tmp_path)
    mmlu = load_benchmark("mmlu_pro", tmp_path)
    gaia = load_benchmark("gaia", tmp_path, split="validation")

    assert browsecomp[0].answers == ("answer",)
    assert hotpot[0].task_id == "hp-1"
    assert mmlu[0].answers == ("J",)
    assert "J. option 9" in mmlu[0].question
    assert gaia[0].metadata["level"] == 1


def test_gaia_text_only_excludes_attachments(tmp_path: Path) -> None:
    directory = tmp_path / "gaia" / "2023" / "validation"
    directory.mkdir(parents=True)
    (directory / "evidence.txt").write_text("evidence", encoding="utf-8")
    rows = [
        {
            "task_id": "text",
            "Question": "Text",
            "Final answer": "yes",
            "file_name": "",
        },
        {
            "task_id": "file",
            "Question": "File",
            "Final answer": "yes",
            "file_name": "evidence.txt",
        },
        {
            "task_id": "hidden-image",
            "Question": "Count the visible objects on a remote museum image.",
            "Final answer": "11",
            "file_name": "",
            "Annotator Metadata": {
                "Tools": "1. Web browser\n2. Image recognition tools"
            },
        },
    ]
    (directory / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    tasks = load_benchmark(
        "gaia",
        tmp_path,
        split="validation",
        gaia_text_only=True,
    )

    assert [task.task_id for task in tasks] == ["text"]
    assert tasks[0].metadata["modality"] == "text"


def test_missing_benchmark_is_loud(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkDataError, match="was not found"):
        load_benchmark("hotpotqa", tmp_path)


@pytest.mark.parametrize(
    ("candidate", "reference", "expected"),
    [
        ("$1,234", "1234", True),
        ("25%", "25%", True),
        ("Alpha, Beta", "alpha,beta", True),
        ("The Eiffel Tower", "the eiffel tower", True),
        ("41", "42", False),
    ],
)
def test_gaia_official_normalization(
    candidate: str,
    reference: str,
    expected: bool,
) -> None:
    assert gaia_match(candidate, reference) is expected


def test_hotpotqa_reports_answer_em_and_f1() -> None:
    result = score_hotpotqa(
        "Some reasoning.\nFINAL ANSWER: the Eiffel Tower",
        ("Eiffel Tower",),
    )

    assert result.correct is True
    assert result.metrics["answer_exact_match"] == 1.0
    assert result.metrics["answer_f1"] == 1.0
    assert result.metrics["supporting_fact_metrics"] is None


def test_mmlu_pro_extracts_choices_a_through_j() -> None:
    assert extract_multiple_choice("The answer is (J).") == "J"
    result = score_mmlu_pro("FINAL ANSWER: J", ("J",))
    assert result.correct is True


def test_llm_judge_is_fail_closed_and_parses_usage() -> None:
    task = BenchmarkTask(
        benchmark="browsecomp",
        task_id="bc-1",
        question="Question",
        answers=("Reference",),
    )
    judge = LLMJudge(
        JudgeConfig(model="judge", base_url="https://unused.test/v1"),
        completion_fn=lambda _prompt: (
            "extracted_final_answer: Reference\n"
            "reasoning: matches\n"
            "correct: yes\n"
            "confidence: 100",
            {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        ),
    )

    result = judge.score(task, "FINAL ANSWER: Reference")

    assert result.correct is True
    assert result.metrics["judge_usage"]["prompt_tokens"] == 12


def test_agent_json_events_keep_input_and_output_separate() -> None:
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "working"}],
                "usage": {
                    "input": 10,
                    "output": 3,
                    "cacheRead": 4,
                    "cacheWrite": 2,
                    "reasoning": 1,
                    "totalTokens": 13,
                    "cost": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "total": 0,
                    },
                },
                "stopReason": "toolUse",
            },
        },
        {"type": "tool_execution_start", "toolName": "web_search"},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "FINAL ANSWER: 4"}],
                "usage": {
                    "input": 20,
                    "output": 5,
                    "cacheRead": 1,
                    "cacheWrite": 0,
                    "totalTokens": 25,
                    "cost": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "total": 0,
                    },
                },
                "stopReason": "stop",
            },
        },
    ]

    parsed = parse_agent_events("\n".join(json.dumps(event) for event in events))

    assert parsed["final_response"] == "FINAL ANSWER: 4"
    assert parsed["usage"].input == 30
    assert parsed["usage"].output == 8
    assert parsed["tool_calls"] == {"web_search": 1}


def test_agent_json_events_expose_provider_error_message() -> None:
    event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [],
            "usage": {},
            "stopReason": "error",
            "errorMessage": "insufficient_quota: no credit remains",
        },
    }

    parsed = parse_agent_events(json.dumps(event))

    assert parsed["error_message"] == "insufficient_quota: no credit remains"


def test_runner_stops_before_starting_more_tasks_after_quota(
    tmp_path: Path,
) -> None:
    invocations = tmp_path / "invocations.txt"
    fake_agent = tmp_path / "quota_agent.py"
    fake_agent.write_text(
        f"""
import json
from pathlib import Path

with Path({str(invocations)!r}).open("a", encoding="utf-8") as stream:
    stream.write("started\\n")
event = {{
    "type": "message_end",
    "message": {{
        "role": "assistant",
        "content": [],
        "usage": {{}},
        "stopReason": "error",
        "errorMessage": "insufficient_quota: account balance exhausted",
    }},
}}
print(json.dumps(event))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        benchmark="mmlu_pro",
        task_id="quota",
        question="Question\n\nA. yes\nB. no",
        answers=("A",),
    )
    output = tmp_path / "results"
    config = RunConfig(
        output_dir=output,
        model="fake",
        base_url="https://unused.test/v1",
        modes=("default", "bcg"),
        workers=1,
        agent_command=(sys.executable, str(fake_agent)),
    )

    with pytest.raises(APIQuotaExhaustedError, match="API quota exhausted"):
        run_benchmarks({"mmlu_pro": [task]}, config, judge=None)

    assert invocations.read_text(encoding="utf-8").splitlines() == ["started"]
    result = json.loads(
        (output / "mmlu_pro" / "default" / "tasks" / "quota.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "api_quota_exhausted"
    assert not (output / "mmlu_pro" / "bcg" / "tasks" / "quota.json").exists()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["stopped_reason"] == "api_quota_exhausted"

    with pytest.raises(APIQuotaExhaustedError, match="API quota exhausted"):
        run_benchmarks({"mmlu_pro": [task]}, config, judge=None)
    assert invocations.read_text(encoding="utf-8").splitlines() == [
        "started",
        "started",
    ]


def test_runner_retries_cached_graph_fallback(tmp_path: Path) -> None:
    invocations = tmp_path / "invocations.txt"
    fake_agent = tmp_path / "fake_agent.py"
    fake_agent.write_text(
        f"""
import json
from pathlib import Path

with Path({str(invocations)!r}).open("a", encoding="utf-8") as stream:
    stream.write("started\\n")
print(json.dumps({{
    "type": "message_end",
    "message": {{
        "role": "assistant",
        "content": [{{"type": "text", "text": "FINAL ANSWER: A"}}],
        "usage": {{}},
        "stopReason": "stop",
    }},
}}))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        benchmark="mmlu_pro",
        task_id="retry-fallback",
        question="Question\n\nA. yes\nB. no",
        answers=("A",),
    )
    output = tmp_path / "results"
    result_path = output / "mmlu_pro" / "bcg" / "tasks" / "retry-fallback.json"
    _write_json(
        result_path,
        {
            "benchmark": "mmlu_pro",
            "mode": "bcg",
            "task_id": task.task_id,
            "status": "graph_fallback",
            "correct": None,
        },
    )
    config = RunConfig(
        output_dir=output,
        model="fake",
        base_url="https://unused.test/v1",
        modes=("bcg",),
        workers=1,
        agent_command=(sys.executable, str(fake_agent)),
    )

    run_benchmarks({"mmlu_pro": [task]}, config, judge=None)

    assert invocations.read_text(encoding="utf-8").splitlines() == ["started"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["correct"] is True


def test_runner_interleaves_benchmarks_before_modes() -> None:
    browsecomp = [
        BenchmarkTask("browsecomp", f"bc-{index}", "Question", ("answer",))
        for index in range(2)
    ]
    gaia = [
        BenchmarkTask("gaia", f"gaia-{index}", "Question", ("answer",))
        for index in range(2)
    ]

    work = _interleaved_work(
        {"browsecomp": browsecomp, "gaia": gaia},
        ("default", "bcg"),
    )

    assert [(task.benchmark, task.task_id, mode) for task, mode in work] == [
        ("browsecomp", "bc-0", "default"),
        ("browsecomp", "bc-0", "bcg"),
        ("gaia", "gaia-0", "default"),
        ("gaia", "gaia-0", "bcg"),
        ("browsecomp", "bc-1", "default"),
        ("browsecomp", "bc-1", "bcg"),
        ("gaia", "gaia-1", "default"),
        ("gaia", "gaia-1", "bcg"),
    ]


def test_zero_task_timeout_waits_without_a_deadline(tmp_path: Path) -> None:
    return_code, stdout, stderr, timed_out, cancelled = _execute(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(0.2); print('finished')",
        ],
        cwd=tmp_path,
        environment=os.environ.copy(),
        timeout=0,
        stop_event=Event(),
    )

    assert return_code == 0
    assert stdout.strip() == "finished"
    assert stderr == ""
    assert timed_out is False
    assert cancelled is False


def test_summary_counts_failed_attempts_in_primary_accuracy() -> None:
    summary = summarize_results(
        [
            {
                "benchmark": "mmlu_pro",
                "mode": "default",
                "status": "completed",
                "correct": True,
                "usage": {},
                "metrics": {},
                "metadata": {"category": "math"},
            },
            {
                "benchmark": "mmlu_pro",
                "mode": "default",
                "status": "timeout",
                "correct": None,
                "usage": {},
                "metrics": {},
                "metadata": {"category": "math"},
            },
            {
                "benchmark": "mmlu_pro",
                "mode": "default",
                "status": "graph_fallback",
                "correct": None,
                "usage": {},
                "metrics": {},
                "metadata": {"category": "math"},
            },
        ]
    )["benchmarks"]["mmlu_pro"]["default"]

    assert summary["evaluated"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["completed_only_accuracy"] == 1.0
    assert summary["category_accuracy"]["math"]["accuracy"] == 0.5


def test_runner_compares_default_and_bcg_with_fake_agent(
    tmp_path: Path,
) -> None:
    fake_agent = tmp_path / "fake_agent.py"
    fake_agent.write_text(
        """
import json
event = {
    "type": "message_end",
    "message": {
        "role": "assistant",
        "content": [{"type": "text", "text": "FINAL ANSWER: A"}],
        "usage": {
            "input": 10,
            "output": 2,
            "cacheRead": 0,
            "cacheWrite": 0,
            "reasoning": 0,
            "totalTokens": 12,
            "cost": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "total": 0,
            },
        },
        "stopReason": "stop",
    },
}
print(json.dumps(event))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        benchmark="mmlu_pro",
        task_id="one",
        question="Question\n\nA. yes\nB. no",
        answers=("A",),
    )
    output = tmp_path / "results"
    config = RunConfig(
        output_dir=output,
        model="fake",
        base_url="https://unused.test/v1",
        modes=("default", "bcg"),
        workers=2,
        agent_command=(sys.executable, str(fake_agent)),
    )

    summary = run_benchmarks({"mmlu_pro": [task]}, config, judge=None)

    assert summary["benchmarks"]["mmlu_pro"]["default"]["accuracy"] == 1.0
    assert summary["benchmarks"]["mmlu_pro"]["bcg"]["accuracy"] == 1.0
    assert summary["benchmarks"]["mmlu_pro"]["default"]["tokens"]["input"] == 10
    assert (output / "mmlu_pro" / "default" / "tasks" / "one.json").is_file()


def test_runner_counts_bcg_turn_limit_as_an_incorrect_attempt(
    tmp_path: Path,
) -> None:
    fake_agent = tmp_path / "fake_turn_limit_agent.py"
    fake_agent.write_text(
        """
import sys
print(
    "BCG_TURN_LIMIT_EXCEEDED: Graph message limit 100 reached",
    file=sys.stderr,
)
raise SystemExit(1)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        benchmark="mmlu_pro",
        task_id="turn-limit",
        question="Question\n\nA. yes\nB. no",
        answers=("A",),
    )
    output = tmp_path / "results"
    config = RunConfig(
        output_dir=output,
        model="fake",
        base_url="https://unused.test/v1",
        modes=("bcg",),
        workers=1,
        agent_command=(sys.executable, str(fake_agent)),
    )

    summary = run_benchmarks({"mmlu_pro": [task]}, config, judge=None)
    result = json.loads(
        (output / "mmlu_pro" / "bcg" / "tasks" / "turn-limit.json").read_text()
    )

    assert result["status"] == "turn_limit"
    assert result["correct"] is False
    assert result["score"] == 0.0
    assert summary["benchmarks"]["mmlu_pro"]["bcg"]["evaluated"] == 1
    assert summary["benchmarks"]["mmlu_pro"]["bcg"]["accuracy"] == 0.0


def test_fixed_fixture_smoke() -> None:
    """The committed minimal fixture is loadable end to end (step 15)."""
    from bcg.apps.benchmark.loaders import load_benchmark

    fixture_dir = Path(__file__).parents[0] / "fixtures" / "benchmark"
    tasks = load_benchmark(
        "browsecomp",
        fixture_dir,
        data_file=fixture_dir / "browsecomp.jsonl",
    )
    assert [task.question for task in tasks] == [
        "Which river flows through Cairo?",
        "Who directed N!ai, the Story of a !Kung Woman?",
    ]
    assert tasks[1].answers == ("John Marshall", "Adrienne Miesmer")
