"""Offline coverage for report adapters, data boundaries, native BCG wiring and metrics."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from benchmarks.public.reports import drb2, drbench
from benchmarks.public.reports.prepare import prepare
from benchmarks.public.reports.report_scoring import ReportScoringContext, score_report_task
from benchmarks.public.reports.runtime import ReportPolicy, bind_report_tools, report_policy
from benchmarks.public.runner import run_reports
from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.tool import tool

BLOCKED_URL = "https://www.mdpi.com/2073-445X/11/9/1529"
BLOCKED_TITLE = "The Local Land Finance Transformation with the Synergy of Increment"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    return _write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def make_drb2(root: Path) -> Path:
    rubric = {"info_recall": ["a", "b"], "analysis": ["c"], "presentation": ["d"]}
    return _jsonl(
        root / "deepresearch_bench_ii" / "tasks_and_rubrics.jsonl",
        [
            {
                "id": "task7",
                "idx": 1,
                "language": "zh",
                "theme": "Finance & Business",
                "description": "Land finance",
                "prompt": "请撰写关于土地财政的研究报告。",
                "content": {
                    "task": "请撰写关于土地财政的研究报告。",
                    "rubric": rubric,
                    "blocked": {
                        "title": BLOCKED_TITLE,
                        "authors": ["A"],
                        "urls": [BLOCKED_URL],
                    },
                },
                "license": "CC BY 4.0",
            },
            {
                "id": "task1+",
                "idx": 2,
                "language": "en",
                "theme": "Health",
                "description": "Health systems",
                "prompt": "Write a report on health systems.",
                "content": {"task": "x", "rubric": rubric, "blocked": {}},
                "license": "CC BY 4.0",
            },
        ],
    )


def make_drbench(root: Path) -> Path:
    data = root / "drbench"
    files = data / "tasks" / "DR0001" / "files"
    _write(
        files / "IN001_md" / "insight-report.md",
        "# Food safety\n\nLee's Market tracks 250 high-risk food products.\n",
    )
    _write(files / "IN001_md" / "qa_dict.json", '{"answer": "SECRET ANSWER"}')
    _write(files / "IN001_md" / "file_dict.json", "{}")
    # The env.json below names this file `budget.csv` (case mismatch, as in DR0038).
    _write(files / "DI001_csv" / "Budget.csv", "quarter,budget\nQ3,100\n")
    _jsonl(
        files / "IN002_email" / "roundcube_email_1.jsonl",
        [
            {
                "type": "user",
                "username": "ann",
                "first_name": "Ann",
                "last_name": "Lee",
                "email": "ann@lees.com",
                "password": "hunter2",
            },
            {
                "type": "email",
                "subject": "Quarterly Supplier Audit Results",
                "from": "ann@lees.com",
                "from_name": "Ann Lee",
                "to": ["bob@lees.com"],
                "date": "2025-01-02 10:00:00",
                "body": "Audit passed for 12 suppliers.",
            },
        ],
    )
    _jsonl(
        files / "IN003_mm" / "mattermost_chat_1.jsonl",
        [
            {"type": "team", "team": {"name": "ops-team", "display_name": "Ops"}},
            {
                "type": "channel",
                "channel": {
                    "team": "ops-team",
                    "name": "cold-chain",
                    "display_name": "Cold Chain",
                },
            },
            {
                "type": "user",
                "user": {"username": "bob", "first_name": "Bob", "password": "pw"},
            },
            {
                "type": "post",
                "post": {
                    "team": "ops-team",
                    "channel": "cold-chain",
                    "user": "bob",
                    "message": "Freezer outage fixed.",
                    "create_at": 1700000000000,
                },
            },
        ],
    )
    prefix = "drbench/data/tasks/DR0001/files"
    _write(
        data / "tasks" / "DR0001" / "config" / "env.json",
        json.dumps(
            {
                "env_files": [
                    {
                        "source": f"{prefix}/IN001_md/insight-report.md",
                        "destination": "shared/insight-report.md",
                        "app": "nextcloud",
                        "qa_type": "insight",
                    },
                    {
                        "source": f"{prefix}/DI001_csv/budget.csv",
                        "destination": "shared/Budget.csv",
                        "app": "file_system",
                        "qa_type": "distractor",
                    },
                    {
                        "source": f"{prefix}/IN002_email/roundcube_email_1.jsonl",
                        "destination": "shared/roundcube_email_1.jsonl",
                        "app": "email",
                        "qa_type": "insight",
                    },
                    {
                        "source": f"{prefix}/IN003_mm/mattermost_chat_1.jsonl",
                        "destination": "shared/mattermost_chat_1.jsonl",
                        "app": "mattermost",
                        "qa_type": "insight",
                    },
                ]
            }
        ),
    )
    _write(
        data / "tasks" / "DR0001" / "config" / "task.json",
        json.dumps(
            {
                "task_id": "DR0001",
                "dr_question": "How can Lee's Market improve food safety?",
                "date": "2025-08-27",
                "company_info": {"name": "Lee's Market", "industry": "Retail"},
                "persona": {
                    "name": "Emily",
                    "role": "Manager",
                    "password": "never-show-me",
                },
            }
        ),
    )
    _write(
        data / "tasks" / "DR0001" / "config" / "eval.json",
        json.dumps(
            {
                "dr_report_evaluation_qa": [
                    {
                        "id": "IN001",
                        "type": "enterprise_fact",
                        "qa_type": "insight",
                        "answer": "250 products",
                        "supporting_file_paths": ["shared/insight-report.md"],
                    },
                    {
                        "id": "EX0000",
                        "type": "external_fact",
                        "qa_type": "insight",
                        "answer": "FSMA 204",
                        "supporting_urls": ["https://example.com"],
                    },
                    {
                        "id": "DI001",
                        "type": "enterprise_fact",
                        "qa_type": "distractor",
                        "answer": "Q3 100",
                        "supporting_file_paths": ["shared/Budget.csv"],
                    },
                ]
            }
        ),
    )
    _write(
        data / "tasks" / "DR0001" / "info.json",
        json.dumps({"difficulty": "easy", "industry": "retail"}),
    )
    _write(data / "tasks" / "DR0000" / "info.json", json.dumps({"difficulty": "easy"}))
    subset = [{"task_id": "DR0000"}, {"task_id": "DR0001"}]
    _jsonl(data / "subsets" / "val.jsonl", subset)
    _jsonl(data / "subsets" / "minival.jsonl", subset)
    _write(data / "subsets" / "sanity.jsonl", "")
    return data


@pytest.fixture
def datasets(tmp_path: Path) -> Path:
    root = tmp_path / "datasets"
    make_drb2(root)
    make_drbench(root)
    return root


def test_adapters_load_original_branch_schemas(datasets):
    tasks = drb2.load_tasks(datasets)
    assert [task.task_id for task in tasks] == ["1", "2"]
    assert tasks[0].aliases == ("task7", "idx-1")
    assert "rubric" in tasks[0].reference and "rubric" not in tasks[0].metadata
    assert [task.task_id for task in drb2.load_tasks(datasets, language="en")] == ["2"]
    enterprise = drbench.load_tasks(datasets)
    assert [task.task_id for task in enterprise] == ["DR0001"]
    assert Path(enterprise[0].reference["files"][1]["source"]).name == "Budget.csv"
    assert "password" not in enterprise[0].metadata["context"]["persona"]


def test_staging_excludes_grading_material_and_credentials(datasets, tmp_path):
    task = drbench.load_tasks(datasets)[0]
    policy = ReportPolicy(task, tmp_path / "workspace", tmp_path / "tools.jsonl")
    text = "\n".join(file.read_text() for file in policy.allowed_files)
    for secret in (
        "SECRET ANSWER",
        "hunter2",
        "never-show-me",
        "qa_type",
        "distractor",
        "FSMA 204",
    ):
        assert secret not in text
        assert secret not in policy.instruction()
        assert secret not in policy.prompt
    assert "From: Ann Lee <ann@lees.com>" in text
    assert "(User: bob, Team: ops-team, Channel: cold-chain)" in text
    assert len(policy.mapping) == 4
    assert drbench.ensure_rendered(task) == drbench.ensure_rendered(task)


@pytest.mark.parametrize(
    "source,destination",
    [
        ("/etc/passwd", "shared/passwd"),
        ("tasks/DR0001/files/IN001_md/qa_dict.json", "shared/public.json"),
        ("tasks/DR0001/files/IN001_md/insight-report.md", "../escaped.md"),
        ("tasks/DR0001/files/IN001_md/insight-report.md", "shared/eval.json"),
    ],
)
def test_bad_file_manifests_fail_closed(datasets, source, destination):
    env = datasets / "drbench/tasks/DR0001/config/env.json"
    _write(env, json.dumps({"env_files": [{"source": source, "destination": destination}]}))
    with pytest.raises(drbench.DRBenchDataError):
        drbench.load_tasks(datasets)


@pytest.mark.asyncio
async def test_file_tools_are_confined_and_failed_reads_do_not_count(datasets, tmp_path):
    task = drbench.load_tasks(datasets)[0]
    policy = ReportPolicy(task, tmp_path / "workspace", tmp_path / "trace.jsonl")
    tools = {entry.name: entry for entry in policy.bind([])}
    read = tools["read_file"]
    assert "Error" in await read.ainvoke({"path": "../eval.json"})
    assert "Error" in await read.ainvoke({"path": "/etc/passwd"})
    assert "Error" in await read.ainvoke(
        {"path": str(datasets / "drbench/tasks/DR0001/config/eval.json")}
    )
    assert "250" in await read.ainvoke({"path": "/workspace/shared/insight-report.md.md"})
    result = await tools["grep_search"].ainvoke({"pattern": "audit"})
    assert "roundcube_email_1.jsonl.md" in result
    metrics, errors = score_report_task(
        task,
        "250 products [insight-report.md].",
        ReportScoringContext(
            tool_trace=policy.trace,
            citation_keys=policy.citation_keys,
        ),
    )
    assert not errors
    assert metrics["file_read_recall"] == pytest.approx(1 / 3)
    assert metrics["file_surfaced_recall"] == pytest.approx(2 / 3)
    assert metrics["citation_recall"] == pytest.approx(1 / 3)
    assert not any("correct" in key or key == "accuracy" for key in metrics)


@pytest.mark.asyncio
async def test_policy_inherits_to_children_but_does_not_leak_to_other_tasks(datasets, tmp_path):
    @tool
    async def bash(command: str) -> str:
        """An unsafe tool that must never be exposed in report runs."""
        raise AssertionError(command)

    file_policy = ReportPolicy(
        drbench.load_tasks(datasets)[0], tmp_path / "a", tmp_path / "a.jsonl"
    )
    web_policy = ReportPolicy(drb2.load_tasks(datasets)[0], tmp_path / "b", tmp_path / "b.jsonl")

    async def child():
        await asyncio.sleep(0)
        return {entry.name for entry in bind_report_tools([bash])}

    async def parent(policy):
        with report_policy(policy):
            return await asyncio.create_task(child())

    files, web = await asyncio.gather(parent(file_policy), parent(web_policy))
    assert files == {"read_file", "grep_search", "glob_search"}
    assert web == {"web_search", "web_fetch"}
    assert bind_report_tools([bash]) == [bash]


@pytest.mark.asyncio
async def test_drb2_blocks_search_and_redirect_without_disclosing_reference(
    datasets, tmp_path, monkeypatch
):
    from plugins.tools.web_search import web_search

    policy = ReportPolicy(drb2.load_tasks(datasets)[0], tmp_path / "ws", tmp_path / "trace")
    calls = []

    async def search(**kwargs):
        calls.append(kwargs)
        return f"Secret snippet\n{BLOCKED_URL}\n{BLOCKED_TITLE}"

    monkeypatch.setattr(web_search, "func", search)
    tools = {entry.name: entry for entry in policy.bind([])}
    refused = await tools["web_search"].ainvoke({"q": "land finance"})
    assert "BLOCKED" in refused and "Secret snippet" not in refused
    assert BLOCKED_TITLE not in refused and BLOCKED_URL not in refused
    await tools["web_search"].ainvoke({"q": BLOCKED_TITLE})
    assert len(calls) == 1

    from benchmarks.public.reports import runtime

    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": BLOCKED_URL})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        runtime.httpx,
        "AsyncClient",
        lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw),
    )

    async def vetted(url):
        return "", ()

    monkeypatch.setattr(runtime, "vet_public_url", vetted)
    assert "BLOCKED" in await policy.fetch_one("https://example.org/redirect")
    assert seen == ["https://example.org/redirect"]
    assert BLOCKED_TITLE not in policy.prompt
    assert BLOCKED_URL not in policy.instruction()


@pytest.mark.asyncio
async def test_budgets_shared_across_tool_bindings(datasets, tmp_path, monkeypatch):
    from plugins.tools.web_search import web_search

    calls = []

    async def search(**kwargs):
        calls.append(kwargs)
        await asyncio.sleep(0)
        return "Public information: https://example.org/public"

    monkeypatch.setattr(web_search, "func", search)
    policy = ReportPolicy(
        drb2.load_tasks(datasets)[0], tmp_path / "ws", tmp_path / "trace", search_calls=2
    )
    main, sub = policy.bind([])[0], policy.bind([])[0]
    results = await asyncio.gather(main.ainvoke({"q": ["a", "b"]}), sub.ainvoke({"q": "c"}))
    assert len(calls) == 2 and any("budget exhausted" in result for result in results)


def test_selection_dry_run_and_prepare_do_not_call_models(datasets, tmp_path, monkeypatch, capsys):
    parsed = run_reports.parser().parse_args(
        [
            "run",
            "drb2",
            "--data-root",
            str(datasets),
            "--task-ids",
            "drb2:2,1",
            "--seed",
            "42",
            "--limit",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    tasks = run_reports.load_tasks(parsed)
    assert [task.task_id for task in tasks] == ["2"]
    asyncio.run(run_reports.run(parsed, tasks))
    assert not (tmp_path / "out").exists()
    assert "official_judge" in capsys.readouterr().out
    parsed.task_ids = "drbench:DR0001"
    with pytest.raises(ValueError, match="different benchmark"):
        run_reports.load_tasks(parsed)
    verified = prepare(drb2.NAME, datasets, download=False)
    assert verified.tasks == 2
    assert (datasets / drb2.NAME / "MANIFEST.json").exists()


def test_reimporting_prepared_data_never_removes_its_own_source(datasets):
    first = prepare(drb2.NAME, datasets, source=datasets / drb2.NAME, download=False)
    second = prepare(drbench.NAME, datasets, source=datasets / drbench.NAME, download=False)
    assert first.tasks == 2 and second.tasks == 1
    assert len(drbench.load_tasks(datasets)) == 1
    assert (datasets / "drbench/tasks/DR0001/config/eval.json").exists()


def test_usage_counts_llm_events_once_and_keeps_missing_values_unknown(tmp_path):
    root = tmp_path / "agent/trajectories"
    _jsonl(
        root / "main.jsonl",
        [
            {"t": "llm", "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
            {"t": "tool_result", "usage": {"total_tokens": 9999}},
        ],
    )
    _write(root / "main.json", json.dumps({"messages": [{"usage": {"total_tokens": 9999}}]}))
    _write(
        root / "bcg_team/g_a/state.json",
        json.dumps(
            {
                "problem_id": "one",
                "snapshot": {"token_usage": {"input_tokens": 30, "output_tokens": 10}},
            }
        ),
    )
    usage = run_reports.collect_usage(tmp_path)
    assert usage["agent"]["total_tokens"] == 120
    assert usage["graph"]["total_tokens"] == 40
    assert usage["combined"]["total_tokens"] == 160
    _jsonl(root / "sub.jsonl", [{"t": "llm"}])
    assert run_reports.collect_usage(tmp_path)["combined"]["total_tokens"] is None


@pytest.mark.asyncio
async def test_worker_preserves_whole_report_and_never_passes_reference(datasets, tmp_path):
    task = drb2.load_tasks(datasets)[0]
    report = "# 完整报告\n正文\n## References\nhttps://example.org/source"

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def run(self, instruction, **kwargs):
            assert "rubric" not in kwargs["meta"]
            assert BLOCKED_TITLE not in json.dumps(kwargs, ensure_ascii=False)
            assert instruction == task.question
            assert "COMPLETE research report" in kwargs["meta"]["_sys_prompt_addendum"]
            return {"final_answer": report}

    config = {
        "pipeline": "agent_team",
        "profile": "benchmark",
        "search_calls": 20,
        "fetch_calls": 30,
        "fetch_chars": 20000,
    }
    result = await run_reports.execute_task(
        task, config, tmp_path / "trial", session_factory=Session
    )
    assert (tmp_path / "trial/report.md").read_text() == report
    assert result["status"] == "completed"
    assert result["official_score"] is None and result["scoring_status"] == "official_pending"
    assert "reference" not in result and "ground_truth" not in result


@pytest.mark.parametrize("pipeline", ["stateful-react-agent", "agent_team"])
@pytest.mark.asyncio
async def test_report_reaches_real_bcg_nodes_and_team_specialists(
    datasets, tmp_path, monkeypatch, pipeline
):
    from test_team_bcg import Graph, call

    from frontier_agent.components.agent_bus import AgentBus
    from frontier_agent.core.runtime import registry
    from frontier_agent.core.runtime.registries.agents import AgentRegistry
    from frontier_agent.core.runtime.registries.workflows import WorkflowContext
    from frontier_agent.core.runtime.resources.manager import ResourceManager
    from frontier_agent.scheduling.pipeline_registry import PipelineRegistry
    from plugins.tools import get_builtin_tools
    from workflows.agent_team import register
    from workflows.agent_team.nodes import main_agent as team_node
    from workflows.stateful_react_agent.bcg import runtime
    from workflows.stateful_react_agent.nodes import main_agent as react_node

    class Provider:
        model = "offline"

        def __init__(self):
            self.calls = Counter()
            self.inputs = []

        async def chat(self, messages, **kwargs):
            names = [t["function"]["name"] for t in kwargs.get("tools", [])]
            self.inputs.append((names, copy.deepcopy(messages)))
            assert not set(names) & {
                "bash",
                "run_python_code",
                "web_search",
                "web_fetch",
                "download_file",
            }
            assert "read_file" in names and "glob_search" in names
            assert "Report benchmark contract" in messages[0]["content"]
            assert "SECRET ANSWER" not in str(messages)
            role = (
                "main"
                if "create_subagent" in names
                else "sub"
                if "submit_report" in names
                else "react"
            )
            self.calls[role] += 1
            n = self.calls[role]
            if role == "main":
                if n == 1:
                    return LLMResponse(
                        content="Delegate file review",
                        tool_calls=[
                            call(
                                "c",
                                "create_subagent",
                                {"agents": [{"name": "reader", "system_prompt": "Review files"}]},
                            )
                        ],
                    )
                if n == 2:
                    return LLMResponse(
                        content="Assign research",
                        tool_calls=[
                            call(
                                "a",
                                "assign_task",
                                {
                                    "tasks": [
                                        {
                                            "agent": "reader",
                                            "prompt": "Read enterprise files and report",
                                        }
                                    ]
                                },
                            )
                        ],
                    )
                if n in (3, 4):
                    return LLMResponse(
                        content="Collect evidence",
                        tool_calls=[call(f"wait{n}", "collect_reports", {"timeout": 1})],
                    )
                return LLMResponse(
                    content="# Research report\n250 products [insight-report.md].\n## References\ninsight-report.md"
                )
            if n <= 4:
                return LLMResponse(
                    content=f"Inspect evidence {n}",
                    tool_calls=[
                        call(f"read{n}", "read_file", {"path": "shared/insight-report.md.md"})
                    ],
                )
            report = "Scope: food safety\nFinding: 250 products.\nEvidence: [insight-report.md].\nConfidence: 0.8"
            return (
                LLMResponse(content=report)
                if role == "react"
                else LLMResponse(
                    content="Report ready",
                    tool_calls=[call("report", "submit_report", {"content": report})],
                )
            )

    from collections import Counter

    provider, graph = Provider(), Graph()
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    for directory in ("workspace", "outputs", "inputs"):
        (tmp_path / directory).mkdir()
        monkeypatch.setenv(f"FRONTIER_AGENT_{directory.upper()}_DIR", str(tmp_path / directory))
    profile = {
        "llm": {"max_tokens": 2048},
        "agent": {
            "planning_mode": False,
            "reporter": False,
            "main_max_turns": 20,
            "sub_max_turns": 20,
            "main_agent_tools": ["create_subagent", "assign_task", "collect_reports"],
            "sub_agent_tools": ["submit_report", "bash"],
            "sub_stuck_target_hint_after": 0,
            "reasoning_only_timeout_s": 0,
            "reasoning_only_max_tokens": 0,
            "logical_call_timeout_s": 0,
            "salvage_infra_errors": False,
        },
    }
    monkeypatch.setattr(
        team_node, "_resolve_llm_and_profile", lambda *a, **kw: (provider, profile, None)
    )
    monkeypatch.setattr(
        react_node, "_resolve_llm_and_profile", lambda *a, **kw: (provider, profile, None)
    )
    monkeypatch.setattr(runtime, "HttpGraphBackend", lambda *a: graph)
    task = drbench.load_tasks(datasets)[0]
    policy = ReportPolicy(task, tmp_path / "corpus", tmp_path / "tools.jsonl")
    saved = registry.snapshot()
    try:
        registry.clear()
        agents = AgentRegistry()
        register(WorkflowContext(PipelineRegistry(), agents))
        registry.register(AgentRegistry, agents)
        resources = ResourceManager(provider, get_builtin_tools())
        registry.register(ResourceManager, resources)
        registry.register(AgentBus, AgentBus(agent_registry=agents, resource_manager=resources))
        with report_policy(policy):
            node = (
                team_node.main_agent_node
                if pipeline == "agent_team"
                else react_node.react_agent_node
            )
            result = await asyncio.wait_for(
                node(
                    {
                        "original_question": policy.instruction(),
                        "language": "en",
                        "metadata": {
                            "_trial_dir": str(tmp_path / "trial"),
                            "_sys_prompt_addendum": policy.prompt,
                        },
                    },
                    SimpleNamespace(task_id="offline-report-bcg"),
                ),
                30,
            )
    finally:
        registry.restore(saved)
    assert "250" in result["final_answer"]
    assert policy.trace and any("250" in event["result"] for event in policy.trace)
    assert graph.turns and graph.closed
    if pipeline == "agent_team":
        assert provider.calls["sub"] == 5
        assert len(result["bcg_team"]["graphs"]) == 2
    assert any("<belief_graph>" in str(messages) for _, messages in provider.inputs)


def test_cli_subprocess_end_to_end_without_paid_apis(datasets, tmp_path):
    """Actual CLI -> subprocess -> BenchmarkSession -> BCG ReAct -> local HTTP doubles."""
    from test_team_bcg import Graph, call

    graph = Graph()
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply({"status": "ok"})
            else:
                self.reply({"error": "not found"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.split("?", 1)[0] == "/v1/chat/completions":
                calls.append(body)
                names = [item["function"]["name"] for item in body.get("tools", [])]
                assert "read_file" in names and "bash" not in names
                assert "SECRET ANSWER" not in str(body)
                number = len(calls)
                message = {"role": "assistant", "content": f"Inspecting the evidence {number}."}
                if number <= 4:
                    message["tool_calls"] = [
                        call(f"read{number}", "read_file", {"path": "shared/insight-report.md.md"})
                    ]
                else:
                    message["content"] = (
                        "# Research report\nLee's Market tracks 250 high-risk products [insight-report.md].\n## References\ninsight-report.md"
                    )
                self.reply(
                    {
                        "id": f"chatcmpl-{number}",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "offline-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": "tool_calls" if number <= 4 else "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                        },
                    }
                )
            elif self.path == "/turns":
                pid = body[0]["problem_id"]
                snapshot = asyncio.run(graph.push(pid, body))
                self.reply({"pushed": len(body), "latest": {pid: snapshot}})
            elif self.path == "/context-selection":
                pid = body["problem_id"]
                self.reply(
                    {
                        "problem_id": pid,
                        "node_ids": [n["id"] for n in graph.snapshot(pid)["nodes"]],
                        "relation_ids": [],
                    }
                )
            else:
                self.reply({"error": "not found"}, 404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    profile = _write(
        tmp_path / "offline.yaml",
        f"""
llm:
  provider: openai
  model: offline-model
  api_key: offline-placeholder
  base_url: {base}/v1
  max_tokens: 2048
agent:
  main_max_turns: 10
  reporter: false
  salvage_infra_errors: false
  reasoning_only_timeout_s: 0
  reasoning_only_max_tokens: 0
  logical_call_timeout_s: 0
  bcg:
    url: {base}
    timeout: 5
    recent_turns: 2
""",
    )
    output = tmp_path / "out"
    env = {
        **os.environ,
        "SANDBOX_BACKEND": "native",
        "OPENAI_API_KEY": "offline-placeholder",
        "OPENAI_BASE_URL": base + "/v1",
        "OPENAI_MODEL": "offline-model",
        "BCG_GRAPH_URL": base,
    }
    command = [
        sys.executable,
        "-m",
        "benchmarks.public.runner.run_reports",
        "run",
        "drbench",
        "--data-root",
        str(datasets),
        "--pipeline",
        "stateful-react-agent",
        "--profile",
        str(profile),
        "--workers",
        "1",
        "--limit",
        "1",
        "--out",
        str(output),
    ]
    try:
        completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=45)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        record = json.loads((output / "trials/DR0001/result.json").read_text())
        log = (output / "trials/DR0001/agent.log").read_text()
        assert record["status"] == "completed", log
        assert record["metrics"]["file_read_recall"] == pytest.approx(1 / 3)
        assert record["usage"]["agent"]["total_tokens"] == 600
        assert record["usage"]["graph"]["total_tokens"] > 0
        assert "250" in (output / "trials/DR0001/report.md").read_text()
        assert graph.turns
        assert any("<belief_graph>" in str(body["messages"]) for body in calls)
        before = len(calls)
        resumed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=15)
        assert resumed.returncode == 0, resumed.stderr
        assert len(calls) == before
        changed = subprocess.run(
            [*command, "--search-calls", "19"], env=env, text=True, capture_output=True, timeout=15
        )
        assert changed.returncode != 0 and "different settings" in changed.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
