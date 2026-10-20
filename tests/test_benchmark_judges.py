"""Official LLM judges for DeepResearch Bench II and DRBench.

Parity tests import the official helper functions straight from local clones
(``BCG_DRB2_REPO`` / ``BCG_DRBENCH_REPO``) and are skipped without them.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bcg.apps.benchmark.cli import app
from bcg.apps.benchmark.judges import drb2 as drb2_judge
from bcg.apps.benchmark.judges import drbench as drbench_judge
from bcg.apps.benchmark.judges.client import JudgeClient, JudgeEndpoint
from bcg.apps.benchmark.judges.official_sources import (
    extract_prompts,
    load_prompt,
    verify_bundled_prompts,
)
from bcg.apps.benchmark.judging import JudgeConfigError, build_specs, run_judges
from bcg.apps.benchmark.loaders import canonical_name
from bcg.apps.benchmark.models import TASK_KIND_REPORT, BenchmarkTask

DRB2_REPO = (
    Path(os.environ["BCG_DRB2_REPO"]) if os.environ.get("BCG_DRB2_REPO") else None
)
DRBENCH_REPO = (
    Path(os.environ["BCG_DRBENCH_REPO"]) if os.environ.get("BCG_DRBENCH_REPO") else None
)
needs_official = pytest.mark.skipif(
    not (DRB2_REPO and DRBENCH_REPO), reason="set BCG_DRB2_REPO and BCG_DRBENCH_REPO"
)


def official_functions(
    path: Path, names: list[str], namespace: dict | None = None
) -> dict:
    """Exec selected top-level functions (or class methods) from an official file."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    ns = {
        "re": re,
        "json": json,
        "logger": logging.getLogger("official"),
        **(namespace or {}),
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(textwrap.dedent(ast.get_source_segment(source, node)), ns)
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names
        ):
            exec(ast.get_source_segment(source, node), ns)
    return ns


# --- prompts --------------------------------------------------------------------


def test_bundled_prompts_match_recorded_hashes() -> None:
    assert verify_bundled_prompts() == []


@needs_official
def test_bundled_prompts_are_byte_identical_to_upstream() -> None:
    for name, text in extract_prompts(DRB2_REPO, DRBENCH_REPO).items():
        assert load_prompt(name) == text, name


# --- parity with official helpers ------------------------------------------------

DRB2_REPLIES = [
    '{"results": [{"rubric_item": "a", "score": 1, "reason": "r", "evidence": "e"}]}',  # official rejects this
    '{\n  "results": [\n    {\n      "rubric_item": "a",\n      "score": 1\n    }\n  ]\n}',
    '```json\n{"results": [{"rubric_item": "a", "score": 1}]}\n```',
    'Sure!\n```json\n{"results": [{"rubric_item": "a \\"q\\"", "score": 0}]}\n``` trailing',
    '{"results": [{"rubric_item": "a", "score": -1}]}',
    "not json at all",
    '```json\n{"results": [}\n```',
    '{"results": "nope"}',
]


@needs_official
def test_drb2_parsing_validation_and_aggregation_match_official() -> None:
    ns = official_functions(
        DRB2_REPO / "run_evaluation.py",
        [
            "FENCED_JSON_PATTERN",
            "_try_clean_and_load",
            "parse_model_text",
            "validate_batch_result",
        ],
        {"print": lambda *a, **k: None},
    )
    for reply in DRB2_REPLIES:
        assert drb2_judge.parse_model_text(reply) == ns["parse_model_text"](reply), (
            reply
        )
        parsed, _ = drb2_judge.parse_model_text(reply)
        for items in (["a"], ["a", "b"], ['a "q"']):
            assert drb2_judge.validate_batch_result(items, parsed) == ns[
                "validate_batch_result"
            ](items, parsed)
    agg = official_functions(
        DRB2_REPO / "aggregate_scores.py", ["compute_dimension_averages"]
    )
    payloads = [
        {
            "scores": {
                "info_recall": {
                    "x": {"score": 1},
                    "y": {"score": 0},
                    "z": {"score": -1},
                },
                "analysis": {"p": {"score": 1}},
                "presentation": {},
            }
        },
        {"error": "batch 1 failed"},
        {"scores": {"info_recall": {"x": {"score": "1"}}}},
    ]
    for payload in payloads:
        assert drb2_judge.compute_dimension_averages(payload) == agg[
            "compute_dimension_averages"
        ](payload)


DRBENCH_REPLIES = [
    '{"answer": "yes", "confidence": "high", "justification": "j", "selected_insight": "s"}',
    'Here: {"answer": "No", "confidence": 0.4, "justification": "j", "selected_insight": "s"} done',
    "[1, 2, 3]",
    "Error: HTTP 500",
]
QUALITY_REPLIES = [
    "<evaluation><depth_quality><score>8</score><justification>ok</justification></depth_quality>"
    "<relevance_to_question><score>11</score><justification>x</justification></relevance_to_question>"
    "<persona_consistency><score>0</score><justification>y</justification></persona_consistency>"
    "<coherence_conciseness><score>7.6</score></coherence_conciseness>"
    "contradictions: 9</evaluation>",
    "nothing useful",
]


@needs_official
def test_drbench_parsers_match_official() -> None:
    ns = official_functions(
        DRBENCH_REPO / "drbench/metrics/qa_similarity_v2.py",
        ["extract_json_from_response"],
    )
    for reply in DRBENCH_REPLIES:
        try:
            expected = ns["extract_json_from_response"](reply)
        except ValueError:
            with pytest.raises(ValueError):
                drbench_judge.extract_json_from_response(reply)
        else:
            assert drbench_judge.extract_json_from_response(reply) == expected
    quality = official_functions(
        DRBENCH_REPO / "drbench/metrics/report_quality.py",
        ["_parse_evaluation_response"],
    )
    for reply in QUALITY_REPLIES:
        official, official_valid = quality["_parse_evaluation_response"](None, reply)
        ours, valid = drbench_judge._parse_quality(reply)
        assert valid == official_valid
        assert ours["metric_result"] == official["metric_result"]
        assert ours["score"] == official["score"]


# --- fakes -------------------------------------------------------------------------


class FakeLLM:
    """Scripted Chat Completions transport; records every payload."""

    def __init__(self, respond):
        self.respond = respond
        self.payloads: list[dict] = []

    def __call__(self, url, headers, payload):
        self.payloads.append(payload)
        status, text = self.respond(payload["messages"][0]["content"], payload)
        if status >= 400:
            return status, {"error": {"message": text}}
        return 200, {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }


def client(fake, tmp_path=None, model="judge-model") -> JudgeClient:
    return JudgeClient(
        JudgeEndpoint(model=model, base_url="http://judge/v1"),
        cache_dir=(tmp_path / "cache") if tmp_path else None,
        transport=fake,
    )


def drb2_task(n_items: int = 60) -> BenchmarkTask:
    rubric = {
        "info_recall": [f"fact {i}" for i in range(n_items - 2)],
        "analysis": ["why"],
        "presentation": ["table"],
    }
    blocked = {"title": "Blocked Paper", "authors": ["A"], "urls": ["https://x.org/p"]}
    return BenchmarkTask(
        "deepresearch_bench_ii",
        "7",
        "prompt",
        (),
        kind=TASK_KIND_REPORT,
        reference={
            "official_content": {
                "task": "Write X",
                "rubric": rubric,
                "blocked": blocked,
            }
        },
        metadata={"language": "en"},
    )


def rubric_items(prompt: str) -> list[str]:
    block = prompt.split("<task_and_rubric>\n", 1)[1].split("\n</task_and_rubric>", 1)[
        0
    ]
    return json.loads(block)["rubric_items"]


# --- DRB2 judge ------------------------------------------------------------------------


def test_drb2_judge_batches_retries_and_scores_like_official(tmp_path: Path) -> None:
    calls = {"n": 0}

    def respond(prompt, payload):
        calls["n"] += 1
        items = rubric_items(prompt)
        if calls["n"] == 1:  # wrong count -> validation failure -> retry
            return 200, json.dumps(
                {"results": [{"rubric_item": items[0], "score": 1}]}, indent=2
            )
        results = [
            {
                "rubric_item": item,
                "score": 1 if i % 2 == 0 else (0 if i % 3 else -1),
                "reason": "r",
                "evidence": "e",
            }
            for i, item in enumerate(items)
        ]
        return 200, "```json\n" + json.dumps({"results": results}, indent=2) + "\n```"

    fake = FakeLLM(respond)
    task = drb2_task(60)
    report = "R" * 160_000
    result = drb2_judge.judge_report(task, report, client(fake, tmp_path))
    assert result["status"] == "ok" and result["report_truncated"]
    assert [b["items"] for b in result["batches"]] == [50, 10]
    assert result["batches"][0]["attempts"] == 2
    first = fake.payloads[0]
    assert (
        first["reasoning_effort"] == "medium"
        and first["max_completion_tokens"] == 32768
    )
    assert "temperature" not in first
    expected_prompt = load_prompt("drb2_rubric").format(
        paper="R" * 150_000,
        rubric=json.dumps(
            {
                "task": "Write X",
                "rubric_items": [f"fact {i}" for i in range(50)],
                "blocked": task.reference["official_content"]["blocked"],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    assert first["messages"][0]["content"] == expected_prompt
    official = drb2_judge.compute_dimension_averages(result["official_result"])
    assert result["scores"] == official and 0 < official["total"] < 1
    assert (
        result["official_result"]["usage_summary"]["total_tokens"] == 24
    )  # accepted replies only
    assert result["usage"]["calls"] == 3


def test_drb2_failed_batch_fails_the_task_and_quota_stops_early(tmp_path: Path) -> None:
    fake = FakeLLM(lambda prompt, payload: (200, "no json"))
    result = drb2_judge.judge_report(
        drb2_task(5), "report", client(fake), settings={"max_retries": 3}
    )
    assert result["status"] == "error" and result["official_result"] == {
        "error": result["error"]
    }
    assert len(fake.payloads) == 3
    quota = FakeLLM(
        lambda prompt, payload: (
            429,
            "insufficient_quota: You exceeded your current quota",
        )
    )
    result = drb2_judge.judge_report(drb2_task(5), "report", client(quota))
    assert result["status"] == "error" and len(quota.payloads) == 1


# --- DRBench judge ----------------------------------------------------------------------


def drbench_task() -> BenchmarkTask:
    qa = [
        {
            "id": "IN1",
            "type": "enterprise_fact",
            "qa_type": "insight",
            "question": "Q1",
            "answer": "250 products",
        },
        {
            "id": "IN2",
            "type": "enterprise_fact",
            "qa_type": "insight",
            "question": "Q2",
            "answer": "GOLD-BROKEN",
        },
        {
            "id": "EX1",
            "type": "external_fact",
            "qa_type": "insight",
            "question": "Q3",
            "answer": "FSMA 204",
        },
        {
            "id": "DI1",
            "type": "enterprise_fact",
            "qa_type": "distractor",
            "question": "Q4",
            "answer": "Budget 100",
        },
    ]
    persona = {"name": "Emily", "role": "Manager", "password": "pw-official"}
    return BenchmarkTask(
        "drbench",
        "DR0001",
        "How?",
        (),
        kind=TASK_KIND_REPORT,
        reference={
            "qa": qa,
            "task_config": {"persona": persona, "dr_question": "How to improve?"},
        },
        metadata={"difficulty": "easy"},
    )


QUALITY_OK = "".join(
    f"<{key}><score>8</score><justification>fine</justification></{key}>"
    for key in drbench_judge.QUALITY_KEYS
)


def drbench_respond(prompt, payload):
    if "Please break down the following report text into insight claims" in prompt:
        return 200, json.dumps(
            [
                {"claim": "Lee's Market tracks 250 products", "citations": ["a.pdf"]},
                {"claim": "short", "citations": []},
                {"claim": "FSMA 204 compliance is required", "citations": "not-a-list"},
            ]
        )
    if "Golden Insight" in prompt:
        if "GOLD-BROKEN" in prompt:
            return (
                200,
                '{"confidence": "high"}',
            )  # missing "answer": every attempt fails
        gold = prompt.split("Golden Insight:\n", 1)[1].split("\n", 1)[0]
        answer = "yes" if gold in {"250 products", "FSMA 204"} else "no"
        return 200, json.dumps(
            {
                "answer": answer,
                "confidence": "high",
                "justification": "j",
                "selected_insight": "s",
            }
        )
    if "You are a Deep Research Evaluator" in prompt:
        return 200, QUALITY_OK
    return 500, "unexpected prompt"


def test_drbench_judge_matches_official_flow(tmp_path: Path) -> None:
    fake = FakeLLM(drbench_respond)
    report = "# Report\nLee's Market tracks 250 products [a.pdf]." + "x" * 70_000
    result = drbench_judge.judge_report(drbench_task(), report, client(fake))
    assert result["report_truncated"]
    assert [i["claim"] for i in result["report_insights"]] == [
        "Lee's Market tracks 250 products",
        "FSMA 204 compliance is required",
    ]
    assert result["report_insights"][1]["citations"] == []
    scores = result["scores"]
    assert scores["insights_recall"] == pytest.approx(2 / 3)
    assert scores["insights_recall_enterprise"] == pytest.approx(1 / 2)
    assert scores["distractor_recall"] == 0.0
    assert scores["report_quality"] == pytest.approx(0.8)
    assert result["official_result"] == {
        k: scores[k] for k in drbench_judge.DEFAULT_METRICS
    }
    assert (
        result["judge_failures"] == 1
    )  # GOLD-BROKEN: scored 0 as officially, but flagged
    broken = [
        r
        for r in result["details"]["insights_recall"]["per_question_results"]
        if r["id"] == "IN2"
    ][0]
    assert broken["failed"] and broken["attempts"] == 3 and broken["score"] == 0.0

    by_kind = {}
    for payload in fake.payloads:
        text = payload["messages"][0]["content"]
        kind = (
            "extract"
            if "break down" in text
            else "score"
            if "Golden Insight" in text
            else "quality"
        )
        by_kind.setdefault(kind, payload)
    assert (
        by_kind["extract"].get("temperature") == 0
        and "max_tokens" not in by_kind["extract"]
    )
    assert (
        by_kind["score"]["temperature"] == 0.7
        and by_kind["score"]["max_tokens"] == 1000
    )
    assert by_kind["quality"]["temperature"] == 0
    quality_prompt = by_kind["quality"]["messages"][0]["content"]
    assert (
        "Lee's Market tracks 250 products" in quality_prompt
    )  # deviation: real report text
    assert (
        '"password": "pw-official"' in quality_prompt
    )  # raw persona, as officially embedded
    assert len(result["deviations"]) == 3
    scoring_prompt = by_kind["score"]["messages"][0]["content"]
    assert scoring_prompt == load_prompt("drbench_insight_scoring").format(
        claims_text="Insight 1: Lee's Market tracks 250 products\nInsight 2: FSMA 204 compliance is required",
        question="Q1",
        gold_insight="250 products",
    )


def test_drbench_quality_falls_back_to_neutral_after_failures() -> None:
    fake = FakeLLM(
        lambda prompt, payload: (200, "[]" if "break down" in prompt else "garbage")
    )
    result = drbench_judge.judge_report(
        drbench_task(), "r", client(fake), metrics=("report_quality",)
    )
    assert result["scores"] == {"report_quality": 0.5}
    assert result["judge_failures"] == 1
    quality_prompts = [
        p["messages"][0]["content"]
        for p in fake.payloads
        if "Deep Research Evaluator" in p["messages"][0]["content"]
    ]
    assert len(quality_prompts) == 3
    assert (
        "IMPORTANT: You MUST respond in the EXACT format" in quality_prompts[1]
    )  # official retry prompt


# --- client ---------------------------------------------------------------------------------


def test_client_adapts_rejected_parameters_and_caches(tmp_path: Path) -> None:
    def respond(prompt, payload):
        if "temperature" in payload:
            return (
                400,
                "Unsupported value: 'temperature' does not support 0.7 with this model.",
            )
        if "max_tokens" in payload:
            return (
                400,
                "Unsupported parameter: 'max_tokens' is not supported. Use 'max_completion_tokens' instead.",
            )
        return 200, "ok"

    fake = FakeLLM(respond)
    judge = client(fake, tmp_path)
    reply = judge.complete("hello", {"temperature": 0.7, "max_tokens": 1000})
    assert reply.params == {"max_completion_tokens": 1000}
    assert judge.adaptations == {
        "temperature": None,
        "max_tokens": "max_completion_tokens",
    }
    sent = len(fake.payloads)
    again = judge.complete("hello", {"temperature": 0.7, "max_tokens": 1000})
    assert again.cached and len(fake.payloads) == sent
    judge.complete("other", {"temperature": 0.7, "max_tokens": 1000})
    assert len(fake.payloads) == sent + 1  # learned adaptations apply up front


# --- orchestration and CLI ----------------------------------------------------------------


def make_run(tmp_path: Path) -> Path:
    run = tmp_path / "results" / "r"
    for benchmark, mode, task_id, report in (
        ("deepresearch_bench_ii", "bcg", "1", "# DRB2 report"),
        ("deepresearch_bench_ii", "default", "1", "# DRB2 baseline"),
        ("drbench", "bcg", "DR0001", "# DRBench report 250 products"),
    ):
        mode_dir = run / benchmark / mode
        (mode_dir / "tasks").mkdir(parents=True)
        (mode_dir / "reports").mkdir()
        report_path = mode_dir / "reports" / f"{benchmark}-{mode}-{task_id}.md"
        report_path.write_text(report, encoding="utf-8")
        (mode_dir / "tasks" / f"{task_id}.json").write_text(
            json.dumps(
                {
                    "benchmark": benchmark,
                    "mode": mode,
                    "task_id": task_id,
                    "status": "report_ready",
                    "report": str(report_path),
                    "metadata": {"language": "en", "difficulty": "easy"},
                }
            )
        )
    return run


@pytest.fixture
def references(monkeypatch: pytest.MonkeyPatch):
    from bcg.apps.benchmark import judging

    tasks = {
        "deepresearch_bench_ii": {"1": drb2_task(4)},
        "drbench": {"DR0001": drbench_task()},
    }
    monkeypatch.setattr(
        judging,
        "load_reference_tasks",
        lambda benchmark, root, data_file=None: tasks[benchmark],
    )
    return tasks


def universal_respond(prompt, payload):
    if "<task_and_rubric>" in prompt:
        items = rubric_items(prompt)
        return 200, json.dumps(
            {
                "results": [
                    {"rubric_item": i, "score": 1, "reason": "", "evidence": ""}
                    for i in items
                ]
            },
            indent=2,
        )
    return drbench_respond(prompt, payload)


def test_run_judges_writes_judgements_summary_exports_and_resumes(
    tmp_path: Path, references
) -> None:
    run = make_run(tmp_path)
    specs = build_specs(
        ["deepresearch_bench_ii", "drbench"],
        model_values=["drbench=mini-judge"],
        param_values=None,
        metrics=None,
        canonicalize=canonical_name,
    )
    assert specs["deepresearch_bench_ii"].model == "gpt-5.5-luna"
    assert specs["drbench"].model == "mini-judge"
    fake = FakeLLM(universal_respond)
    endpoint = lambda model: JudgeEndpoint(model=model, base_url="http://judge/v1")  # noqa: E731
    messages: list[str] = []
    result = run_judges(
        run,
        specs,
        endpoint,
        data_root=tmp_path,
        transport=fake,
        progress=messages.append,
    )
    assert result.judged == 3 and not result.errors

    drb_id = specs["deepresearch_bench_ii"].judge_id()
    judgement = json.loads(
        (run / "deepresearch_bench_ii/bcg/judgements" / f"1.{drb_id}.json").read_text()
    )
    assert (
        judgement["schema"] == "bcg.judgement.v1"
        and judgement["scores"]["total"] == 1.0
    )
    assert judgement["judge"]["deviates_from_official"] is False
    task_result = json.loads(
        (run / "deepresearch_bench_ii/bcg/tasks/1.json").read_text()
    )
    assert task_result["judgements"][drb_id]["scores"]["total"] == 1.0

    summary = json.loads((run / "judge_summary.json").read_text())
    entry = summary["benchmarks"]["deepresearch_bench_ii"]["bcg"][drb_id]
    assert (
        entry["ok"] == 1
        and entry["coverage"] == 1.0
        and entry["scores"]["total"] == 1.0
    )
    assert entry["scores_by_language"]["en"]["tasks"] == 1
    drbench_entry = summary["benchmarks"]["drbench"]["bcg"][specs["drbench"].judge_id()]
    assert drbench_entry["judge_failures"] == 1

    export = [
        json.loads(line)
        for line in (run / "export/deepresearch_bench_ii" / f"result.{drb_id}.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {(line["model"], line["idx"]) for line in export} == {
        ("bcg", 1),
        ("default", 1),
    }
    assert "scores" in export[0]["result"]

    calls = len(fake.payloads)
    again = run_judges(
        run,
        specs,
        endpoint,
        data_root=tmp_path,
        transport=fake,
        progress=messages.append,
    )
    assert again.judged == 0 and again.reused == 3 and len(fake.payloads) == calls
    retried = run_judges(
        run,
        specs,
        endpoint,
        data_root=tmp_path,
        transport=fake,
        progress=messages.append,
        retry_failed=True,
    )
    assert retried.judged == 1  # only the DRBench judgement with a failed call
    (
        run / "deepresearch_bench_ii/bcg/reports/deepresearch_bench_ii-bcg-1.md"
    ).write_text("# changed")
    changed = run_judges(
        run,
        specs,
        endpoint,
        data_root=tmp_path,
        transport=fake,
        progress=messages.append,
    )
    assert changed.judged == 1


@needs_official
def test_drb2_export_matches_official_aggregator(tmp_path: Path, references) -> None:
    run = make_run(tmp_path)
    specs = build_specs(
        ["deepresearch_bench_ii"],
        model_values=None,
        param_values=None,
        metrics=None,
        canonicalize=canonical_name,
    )
    run_judges(
        run,
        specs,
        lambda m: JudgeEndpoint(m, "http://judge/v1"),
        data_root=tmp_path,
        transport=FakeLLM(universal_respond),
        progress=lambda _m: None,
    )
    judge_id = specs["deepresearch_bench_ii"].judge_id()
    export = run / "export/deepresearch_bench_ii" / f"result.{judge_id}.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(DRB2_REPO / "aggregate_scores.py"),
            "--input",
            str(export),
            "--output-prefix",
            str(tmp_path / "agg"),
            "--tasks-file",
            str(tmp_path / "none.jsonl"),
        ],
        check=True,
        capture_output=True,
    )
    rows = (tmp_path / "agg_total.csv").read_text().splitlines()
    official_total = float(rows[-1].split(",")[-1])
    summary = json.loads((run / "judge_summary.json").read_text())
    ours = summary["benchmarks"]["deepresearch_bench_ii"]
    assert official_total == pytest.approx(
        sum(ours[m][judge_id]["scores"]["total"] for m in ("bcg", "default")) / 2
    )


def test_judge_specs_validate_inputs() -> None:
    specs = build_specs(
        ["deepresearch_bench_ii", "drbench"],
        model_values=["shared-judge"],
        param_values=["drb2.reasoning_effort=high", "drbench.scoring.temperature=0"],
        metrics=["insights_recall"],
        canonicalize=canonical_name,
    )
    assert specs["deepresearch_bench_ii"].request_params()["reasoning_effort"] == "high"
    assert specs["drbench"].request_params()["scoring"] == {
        "temperature": 0.0,
        "max_tokens": 1000,
    }
    assert all(spec.deviates_from_official for spec in specs.values())
    assert (
        specs["drbench"].judge_id()
        != build_specs(
            ["drbench"],
            model_values=["shared-judge"],
            param_values=None,
            metrics=None,
            canonicalize=canonical_name,
        )["drbench"].judge_id()
    )
    for bad in (["drb2.nope=1"], ["drbench.scoring.temperature=hot"], ["drb2"]):
        with pytest.raises(JudgeConfigError):
            build_specs(
                ["deepresearch_bench_ii", "drbench"],
                model_values=None,
                param_values=bad,
                metrics=None,
                canonicalize=canonical_name,
            )
    with pytest.raises(JudgeConfigError):
        build_specs(
            ["drbench"],
            model_values=None,
            param_values=None,
            metrics=["factuality"],
            canonicalize=canonical_name,
        )


def test_cli_judge_dry_run(
    tmp_path: Path, references, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path)
    result = CliRunner().invoke(
        app, ["judge", str(run), "--dry-run", "--task-ids", "drb2:1", "--modes", "bcg"]
    )
    assert result.exit_code == 0, result.output
    assert "gpt-5.5-luna" in result.output and "gpt-4o-mini" in result.output
    assert "2 report(s) to judge" in result.output  # drb2 bcg/1 + drbench bcg/DR0001
    bad = CliRunner().invoke(
        app, ["judge", str(run), "--judge-param", "drb2.bogus=1", "--dry-run"]
    )
    assert bad.exit_code != 0 and "Unknown --judge-param" in bad.output
