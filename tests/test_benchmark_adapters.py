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
    _write_agent_configuration,
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


def test_benchmark_agent_thinking_is_written_to_runtime_config(tmp_path: Path) -> None:
    config = RunConfig(
        output_dir=tmp_path,
        model="gpt-5.6-luna",
        base_url="https://example.test/v1",
        thinking="medium",
    )

    agent_dir = _write_agent_configuration(tmp_path, config, "bcg")
    settings = json.loads((agent_dir / "settings.json").read_text(encoding="utf-8"))
    models = json.loads((agent_dir / "models.json").read_text(encoding="utf-8"))

    assert settings["defaultThinkingLevel"] == "medium"
    assert models["providers"]["benchmark"]["models"][0]["reasoning"] is True


def test_benchmark_graph_view_is_written_to_runtime_config(tmp_path: Path) -> None:
    config = RunConfig(
        output_dir=tmp_path,
        model="gpt-5.6-luna",
        base_url="https://example.test/v1",
        graph_view="compact",
    )

    agent_dir = _write_agent_configuration(tmp_path, config, "bcg")
    settings = json.loads((agent_dir / "settings.json").read_text(encoding="utf-8"))

    assert settings["contextManagement"]["bcg"]["graphView"] == "compact"


def test_benchmark_gpt_56_off_is_sent_as_reasoning_none(tmp_path: Path) -> None:
    config = RunConfig(
        output_dir=tmp_path,
        model="gpt-5.6-luna",
        base_url="https://example.test/v1",
        thinking="off",
    )

    agent_dir = _write_agent_configuration(tmp_path, config, "bcg")
    models = json.loads((agent_dir / "models.json").read_text(encoding="utf-8"))
    definition = models["providers"]["benchmark"]["models"][0]

    assert definition["reasoning"] is True
    assert definition["thinkingLevelMap"] == {"off": "none"}


def test_benchmark_summary_uses_an_independent_model_configuration(
    tmp_path: Path,
) -> None:
    config = RunConfig(
        output_dir=tmp_path,
        model="agent-model",
        base_url="https://agent.test/v1",
        summary_model="summary-model",
        summary_base_url="https://summary.test/v1",
        summary_thinking="low",
        summary_max_tokens=1024,
    )

    agent_dir = _write_agent_configuration(tmp_path, config, "summary")
    settings = json.loads((agent_dir / "settings.json").read_text(encoding="utf-8"))
    models = json.loads((agent_dir / "models.json").read_text(encoding="utf-8"))

    assert settings["contextManagement"]["provider"] == "summary"
    assert settings["contextManagement"]["summary"] == {
        "provider": "summary",
        "model": "summary-model",
        "recentTurns": 2,
        "timeoutMs": 300000,
        "maxTokens": 1024,
        "thinkingLevel": "low",
    }
    assert models["providers"]["summary"]["baseUrl"] == "https://summary.test/v1"
    assert models["providers"]["summary"]["models"][0]["id"] == "summary-model"


def test_loads_all_supported_benchmark_schemas(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "browse_comp" / "data.json",
        [{"task_id": "bc-1", "input": "Find it", "ground_truth_answer": "answer"}],
    )
    _write_json(
        tmp_path / "browsecomp_zh" / "data.json",
        [{"Question": "请找到它", "Answer": "答案", "Topic": "测试"}],
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
    browsecomp_zh = load_benchmark("browsecomp_zh", tmp_path)
    hotpot = load_benchmark("hotpotqa", tmp_path)
    mmlu = load_benchmark("mmlu_pro", tmp_path)
    gaia = load_benchmark("gaia", tmp_path, split="validation")

    assert browsecomp[0].answers == ("answer",)
    assert browsecomp_zh[0].answers == ("答案",)
    assert browsecomp_zh[0].metadata["Topic"] == "测试"
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
    assert parsed["blocked_tool_calls"] == {}


def test_agent_json_events_distinguish_blocked_search_calls() -> None:
    events = [
        {"type": "tool_execution_start", "toolName": "web_search"},
        {
            "type": "message_end",
            "message": {
                "role": "toolResult",
                "toolName": "web_search",
                "details": {
                    "budget": {
                        "callsUsed": 20,
                        "maxCalls": 20,
                        "exhausted": True,
                        "blocked": True,
                    }
                },
            },
        },
    ]

    parsed = parse_agent_events("\n".join(json.dumps(event) for event in events))

    assert parsed["tool_calls"] == {"web_search": 1}
    assert parsed["blocked_tool_calls"] == {"web_search": 1}


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


def test_agent_json_events_capture_graph_model_usage() -> None:
    event = {
        "type": "graph_usage",
        "usage": {
            "totals": {
                "input_tokens": 999,
                "output_tokens": 999,
                "reasoning_tokens": 999,
                "total_tokens": 1998,
            },
            "llm_totals": {
                "input_tokens": 40,
                "output_tokens": 12,
                "reasoning_tokens": 5,
                "total_tokens": 52,
            },
        },
    }

    parsed = parse_agent_events(json.dumps(event))

    assert parsed["graph_usage"].input == 40
    assert parsed["graph_usage"].reasoning == 5
    assert parsed["graph_usage"].output == 12


def test_agent_json_events_capture_summary_model_usage_and_latency() -> None:
    event = {
        "type": "summary_usage",
        "usage": {
            "llm_totals": {
                "input_tokens": 50,
                "output_tokens": 15,
                "cache_read_tokens": 7,
                "cache_write_tokens": 2,
                "reasoning_tokens": 3,
                "total_tokens": 74,
            },
            "cost": {
                "input": 0.1,
                "output": 0.2,
                "cache_read": 0.01,
                "cache_write": 0.02,
                "total": 0.33,
            },
            "wall_time_seconds": 1.25,
            "updates": 4,
        },
    }

    parsed = parse_agent_events(json.dumps(event))

    assert parsed["summary_usage"].input == 50
    assert parsed["summary_usage"].cache_read == 7
    assert parsed["summary_usage"].reasoning == 3
    assert parsed["summary_usage"].total_cost == pytest.approx(0.33)
    assert parsed["summary_wall_time_seconds"] == pytest.approx(1.25)


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


def test_final_graph_warning_is_not_runtime_graph_fallback(tmp_path: Path) -> None:
    fake_agent = tmp_path / "final_warning_agent.py"
    fake_agent.write_text(
        """
import json
import sys

print("[BCG finalization] failed to ingest final unsent messages: timeout", file=sys.stderr)
print(json.dumps({
    "type": "message_end",
    "message": {
        "role": "assistant",
        "content": [{"type": "text", "text": "FINAL ANSWER: A"}],
        "usage": {},
        "stopReason": "stop",
    },
}))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        benchmark="mmlu_pro",
        task_id="final-warning",
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

    run_benchmarks({"mmlu_pro": [task]}, config, judge=None)

    result = json.loads(
        (output / "mmlu_pro" / "bcg" / "tasks" / "final-warning.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "completed"
    assert result["correct"] is True
    assert result["graph_fallback"] is False
    assert result["graph_finalization_warning"] is True


def test_runner_persists_per_task_model_io_trace_reference(tmp_path: Path) -> None:
    fake_agent = tmp_path / "trace_agent.py"
    fake_agent.write_text(
        """
import json
import os
from pathlib import Path

trace = Path(os.environ["BCG_MODEL_IO_TRACE_PATH"])
trace.parent.mkdir(parents=True, exist_ok=True)
records = [
    {
        "schema": "bcg.model_io.v1",
        "type": "request",
        "call_id": 1,
        "payload": {"messages": [{"role": "user", "content": "question"}]},
    },
    {
        "schema": "bcg.model_io.v1",
        "type": "response",
        "call_id": 1,
        "message": {"role": "assistant", "content": [{"type": "text", "text": "FINAL ANSWER: A"}]},
    },
]
trace.write_text("\\n".join(json.dumps(record) for record in records) + "\\n", encoding="utf-8")
print(json.dumps({
    "type": "message_end",
    "message": {
        "role": "assistant",
        "content": [{"type": "text", "text": "FINAL ANSWER: A"}],
        "usage": {},
        "stopReason": "stop",
    },
}))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        benchmark="mmlu_pro",
        task_id="trace",
        question="Question\n\nA. yes\nB. no",
        answers=("A",),
    )
    output = tmp_path / "results"
    config = RunConfig(
        output_dir=output,
        model="fake",
        base_url="https://unused.test/v1",
        modes=("default",),
        workers=1,
        agent_command=(sys.executable, str(fake_agent)),
    )

    run_benchmarks({"mmlu_pro": [task]}, config, judge=None)

    result_path = output / "mmlu_pro" / "default" / "tasks" / "trace.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    trace_path = Path(result["model_io_trace"])
    assert (
        trace_path
        == (
            output
            / "mmlu_pro"
            / "default"
            / "model-io"
            / "mmlu_pro-default-trace.jsonl"
        ).resolve()
    )
    assert trace_path.is_file()


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


def test_summary_separates_agent_graph_and_combined_model_tokens() -> None:
    run_summary = summarize_results(
        [
            {
                "benchmark": "mmlu_pro",
                "mode": "bcg",
                "status": "completed",
                "correct": True,
                "usage": {
                    "input": 100,
                    "cache_read": 20,
                    "cache_write": 5,
                    "output": 30,
                    "reasoning": 12,
                },
                "graph_usage": {
                    "input": 40,
                    "output": 10,
                    "reasoning": 4,
                },
                "summary_usage": {
                    "input": 30,
                    "cache_read": 5,
                    "output": 8,
                    "reasoning": 2,
                },
                "metrics": {},
                "metadata": {"category": "math"},
            }
        ]
    )
    summary = run_summary["benchmarks"]["mmlu_pro"]["bcg"]

    assert summary["model_token_usage"] == {
        "agent_model": {
            "input_tokens": 125,
            "reasoning_tokens": 12,
            "output_tokens": 18,
        },
        "graph_model": {
            "input_tokens": 40,
            "reasoning_tokens": 4,
            "output_tokens": 6,
        },
        "summary_model": {
            "input_tokens": 35,
            "reasoning_tokens": 2,
            "output_tokens": 6,
        },
        "combined": {
            "input_tokens": 200,
            "reasoning_tokens": 18,
            "output_tokens": 30,
        },
    }
    assert run_summary["model_token_usage"] == summary["model_token_usage"]


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
