"""Offline tests for the report benchmarks: DeepResearch Bench II and DRBench."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bcg.apps.benchmark import drb2, drbench
from bcg.apps.benchmark.cli import _select_tasks, app
from bcg.apps.benchmark.loaders import canonical_name, load_benchmark
from bcg.apps.benchmark.prepare import prepare
from bcg.apps.benchmark.profiles import ProfileError, parse_bench_opts, resolve_profiles
from bcg.apps.benchmark.runner import RunConfig, parse_agent_events, run_benchmarks
from bcg.apps.benchmark.selection import SelectionError, parse_task_ids, select_tasks

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
    return _write(
        path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


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


# ---------------------------------------------------------------------------
# DeepResearch Bench II
# ---------------------------------------------------------------------------


def test_drb2_loader_normalizes_tasks_and_filters(datasets: Path) -> None:
    tasks = load_benchmark("drb2", datasets)
    assert [task.task_id for task in tasks] == ["1", "2"]
    first = tasks[0]
    assert first.is_report and first.answers == ()
    assert first.aliases == ("task7", "idx-1")
    assert first.metadata["rubric_counts"] == {
        "info_recall": 2,
        "analysis": 1,
        "presentation": 1,
    }
    assert "rubric" in first.reference and "rubric" not in first.metadata
    assert [
        task.task_id for task in load_benchmark("drb2", datasets, drb2_language="en")
    ] == ["2"]
    assert canonical_name("DeepResearch-Bench-II") == "deepresearch_bench_ii"


def test_drb2_blocklist_matches_mirrors_and_audits_reports(datasets: Path) -> None:
    task = load_benchmark("drb2", datasets)[0]
    payload = drb2.blocklist_payload(task)
    assert payload["url_keys"] == ["mdpi.com/2073-445x/11/9/1529"]
    report = (
        "见 http://mdpi.com/2073-445X/11/9/1529/htm 以及 https://stats.gov.cn/x。\n"
        f"Based on '{BLOCKED_TITLE}'."
    )
    audit = drb2.audit_blocked_citations(report, payload)
    assert audit["blocked_url_hits"] == ["http://mdpi.com/2073-445X/11/9/1529/htm"]
    assert audit["blocked_citation_hits"] == 2
    assert (
        drb2.audit_blocked_citations("clean https://stats.gov.cn", payload)[
            "blocked_citation_hits"
        ]
        == 0
    )


# ---------------------------------------------------------------------------
# DRBench
# ---------------------------------------------------------------------------


def test_drbench_loader_rewrites_sources_and_skips_incomplete_tasks(
    datasets: Path,
) -> None:
    tasks = load_benchmark("drbench", datasets)
    assert [task.task_id for task in tasks] == ["DR0001"]
    files = {
        item["destination"]: Path(item["source"])
        for item in tasks[0].reference["files"]
    }
    assert files["shared/Budget.csv"].name == "Budget.csv"  # case-insensitive fallback
    assert all(path.is_file() for path in files.values())
    assert tasks[0].metadata["n_external_insight_qa"] == 1
    assert "password" not in tasks[0].metadata["context"]["persona"]
    with pytest.raises(Exception, match="remain"):
        load_benchmark("drbench", datasets, drbench_difficulty="hard")


def test_drbench_workspace_contains_only_renderings(
    datasets: Path, tmp_path: Path
) -> None:
    task = load_benchmark("drbench", datasets)[0]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mapping = drbench.stage_workspace(task, workspace)
    staged = sorted(
        str(path.relative_to(workspace))
        for path in workspace.rglob("*")
        if path.is_file()
    )
    assert staged == [
        "FILES.md",
        "shared/Budget.csv.md",
        "shared/insight-report.md.md",
        "shared/mattermost_chat_1.jsonl.md",
        "shared/roundcube_email_1.jsonl.md",
    ]
    assert mapping["shared/Budget.csv.md"] == "shared/Budget.csv"
    everything = "\n".join(
        path.read_text(encoding="utf-8") for path in workspace.rglob("*.md")
    )
    for secret in (
        "SECRET ANSWER",
        "hunter2",
        "never-show-me",
        "qa_type",
        "distractor",
    ):
        assert secret not in everything
    assert "(User: bob, Team: ops-team, Channel: cold-chain)" in everything
    assert "From: Ann Lee <ann@lees.com>" in everything
    # Renderings are cached and reused.
    index = drbench.ensure_rendered(task)
    assert index["renderer_version"] == drbench.RENDERER_VERSION


def test_drbench_deterministic_recall(datasets: Path) -> None:
    task = load_benchmark("drbench", datasets)[0]
    trace = [
        {
            "tool": "read",
            "args": {"path": "/tmp/ws/shared/insight-report.md.md"},
            "result": "...",
        },
        {
            "tool": "grep",
            "args": {"pattern": "audit"},
            "result": "shared/roundcube_email_1.jsonl.md:12: Audit passed",
        },
    ]
    report = (
        "Lee's Market tracks 250 products [insight-report.md]. Audit passed "
        '[Email from ann@lees.com, subject "Quarterly Supplier Audit Results"]. '
        "Budget was 100 [Budget.csv]."
    )
    metrics = drbench.deterministic_recall(
        task, report, tool_trace=trace, citation_keys=drbench.citation_keys_for(task)
    )
    assert metrics["insight_files"] == 3 and metrics["distractor_files"] == 1
    assert metrics["file_read_recall"] == pytest.approx(1 / 3)
    assert metrics["file_surfaced_recall"] == pytest.approx(2 / 3)
    assert metrics["citation_recall"] == pytest.approx(2 / 3)
    assert metrics["citation_precision"] == pytest.approx(2 / 3)
    assert metrics["distractor_citation_rate"] == 1.0
    assert metrics["insight_files_missed"] == ["shared/mattermost_chat_1.jsonl"]
    assert metrics["external_insights"] == 1


@pytest.mark.parametrize("suffix", [".pdf", ".docx", ".pptx", ".xlsx"])
def test_drbench_renders_office_formats(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"report{suffix}"
    if suffix == ".pdf":
        pymupdf = pytest.importorskip("pymupdf")
        document = pymupdf.open()
        document.new_page().insert_text((72, 72), "Revenue grew 12% in Q3.")
        document.save(str(path))
    elif suffix == ".docx":
        docx = pytest.importorskip("docx")
        document = docx.Document()
        document.add_heading("Summary", level=1)
        document.add_paragraph("Revenue grew 12% in Q3.")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text, table.cell(0, 1).text = "Quarter", "Growth"
        table.cell(1, 0).text, table.cell(1, 1).text = "Q3", "12%"
        document.save(str(path))
    elif suffix == ".pptx":
        pptx = pytest.importorskip("pptx")
        presentation = pptx.Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = "Summary"
        slide.placeholders[1].text = "Revenue grew 12% in Q3."
        presentation.save(str(path))
    else:
        openpyxl = pytest.importorskip("openpyxl")
        workbook = openpyxl.Workbook()
        workbook.active.append(["Quarter", "Growth"])
        workbook.active.append(["Q3", "12%"])
        workbook.save(str(path))
    text, keys = drbench.render_file(
        path, app="nextcloud", destination=f"shared/report{suffix}"
    )
    assert "12%" in text
    assert f"report{suffix}" in keys


# ---------------------------------------------------------------------------
# Selection and profiles
# ---------------------------------------------------------------------------


def test_task_id_selection_supports_scopes_ranges_files_and_aliases(
    datasets: Path, tmp_path: Path
) -> None:
    tasks = load_benchmark("drb2", datasets)
    selected, missing = select_tasks(tasks, ["2", "task7", "idx-2", "9"])
    assert [task.task_id for task in selected] == ["2", "1"] and missing == ["9"]
    assert [task.task_id for task in select_tasks(tasks, ["1-2"])[0]] == ["1", "2"]

    selection = parse_task_ids(
        "drb2:2,1;drbench:DR0001-DR0001", canonicalize=canonical_name
    )
    assert selection.scoped == {
        "deepresearch_bench_ii": ["2", "1"],
        "drbench": ["DR0001-DR0001"],
    }
    ids_file = _write(tmp_path / "ids.txt", "# smoke\ndrbench: DR0001\n")
    assert parse_task_ids(f"@{ids_file}", canonicalize=canonical_name).scoped == {
        "drbench": ["DR0001"]
    }

    options = {"gaia_split": "validation", "drbench_subset": "val"}
    chosen = _select_tasks(
        ["deepresearch_bench_ii", "drbench"],
        data_root=datasets,
        file_overrides={},
        task_ids="drb2:2,1;drbench:DR0001",
        shuffle=True,
        seed=0,
        max_problems=0,
        loader_options=options,
    )
    assert [task.task_id for task in chosen["deepresearch_bench_ii"]] == ["2", "1"]
    with pytest.raises(
        SelectionError, match="Unknown deepresearch_bench_ii task IDs: 7"
    ):
        _select_tasks(
            ["deepresearch_bench_ii"],
            data_root=datasets,
            file_overrides={},
            task_ids="drb2:7",
            shuffle=False,
            seed=0,
            max_problems=0,
            loader_options=options,
        )
    with pytest.raises(SelectionError, match="not being run"):
        _select_tasks(
            ["deepresearch_bench_ii"],
            data_root=datasets,
            file_overrides={},
            task_ids="drbench:DR0001",
            shuffle=False,
            seed=0,
            max_problems=0,
            loader_options=options,
        )


def test_profile_precedence() -> None:
    environ = {"SERPER_MAX_CALLS": "25", "WEB_FETCH_MAX_CALLS": "7"}
    profiles = resolve_profiles(
        ["deepresearch_bench_ii", "drbench", "mmlu_pro"], environ={}
    )
    assert profiles["deepresearch_bench_ii"].timeout == 3600
    assert profiles["deepresearch_bench_ii"].search_max_calls == 20
    assert profiles["drbench"].web is False
    assert (
        profiles["mmlu_pro"].timeout == 900
        and profiles["mmlu_pro"].search_max_calls is None
    )

    overrides = parse_bench_opts(
        ["drb2.timeout=7200", "drbench.web=on"], canonicalize=canonical_name
    )
    profiles = resolve_profiles(
        ["deepresearch_bench_ii", "drbench"],
        global_overrides={
            "timeout": 100.0,
            "search_max_calls": None,
            "fetch_max_calls": 9,
        },
        bench_overrides=overrides,
        environ=environ,
    )
    drb = profiles["deepresearch_bench_ii"]
    assert (drb.timeout, drb.search_max_calls, drb.fetch_max_calls) == (7200.0, 25, 9)
    assert profiles["drbench"].timeout == 100.0 and profiles["drbench"].web is True
    for bad in (
        "drb2.timeout",
        "drb2.nope=1",
        "drb2.search_max_calls=0",
        "drbench.web=maybe",
    ):
        with pytest.raises(ProfileError):
            parse_bench_opts([bad], canonicalize=canonical_name)


# ---------------------------------------------------------------------------
# Runner end to end with a fake Agent
# ---------------------------------------------------------------------------

FAKE_AGENT = r"""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
tools = args[args.index("--tools") + 1].split(",") if "--tools" in args else []
cwd = Path.cwd()
files = sorted(str(p.relative_to(cwd)) for p in cwd.rglob("*") if p.is_file())
probe = {
    "tools": tools,
    "files": files,
    "system_prompt": args[args.index("--system-prompt") + 1],
    "prompt": args[-1],
    "cwd": str(cwd.resolve()),
    "env": {k: os.environ.get(k) for k in (
        "SERPER_MAX_CALLS", "WEB_FETCH_MAX_CALLS", "WEB_FETCH_MAX_CHARS", "SERPER_LANGUAGE",
        "SERPER_COUNTRY", "BCG_TOOL_ROOT", "BCG_WEB_BLOCKLIST", "BCG_WEB_CACHE_DIR")},
}
name = "drbench" if "FILES.md" in files else "drb2"
Path(os.environ["FAKE_PROBE_DIR"], name + ".json").write_text(json.dumps(probe))
events = []
if name == "drbench":
    events += [
        {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "read",
         "args": {"path": "shared/insight-report.md.md"}},
        {"type": "message_end", "message": {"role": "toolResult", "toolCallId": "c1", "toolName": "read",
         "content": [{"type": "text", "text": "Lee's Market tracks 250"}], "details": {}}},
    ]
    report = "# Report\n\nTracks 250 products [insight-report.md].\n\n## References\n- insight-report.md"
else:
    events += [
        {"type": "tool_execution_start", "toolCallId": "s1", "toolName": "web_search", "args": {"query": "q"}},
        {"type": "message_end", "message": {"role": "toolResult", "toolCallId": "s1", "toolName": "web_search",
         "content": [{"type": "text", "text": "results"}], "details": {"policy": {"filtered": 1}}}},
        {"type": "tool_execution_start", "toolCallId": "f1", "toolName": "web_fetch", "args": {"url": "u"}},
        {"type": "message_end", "message": {"role": "toolResult", "toolCallId": "f1", "toolName": "web_fetch",
         "content": [{"type": "text", "text": "blocked"}], "details": {"policy": {"blocked": True}}}},
    ]
    report = "# 土地财政\n\n2021年收入8.71万亿元 https://stats.gov.cn/a\n\n## 参考文献\n- " + __BLOCKED_URL__
events.append({"type": "message_end", "message": {"role": "assistant",
    "content": [{"type": "text", "text": report}], "stopReason": "stop",
    "usage": {"input": 5, "output": 5, "totalTokens": 10}}})
for event in events:
    print(json.dumps(event, ensure_ascii=False))
""".replace("__BLOCKED_URL__", repr(BLOCKED_URL))


def test_runner_executes_report_benchmarks_end_to_end(
    datasets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _write(tmp_path / "fake_agent.py", FAKE_AGENT)
    probes = tmp_path / "probes"
    probes.mkdir()
    monkeypatch.setenv("FAKE_PROBE_DIR", str(probes))
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.delenv("SERPER_MAX_CALLS", raising=False)
    tasks = {
        "deepresearch_bench_ii": load_benchmark("drb2", datasets)[:1],
        "drbench": load_benchmark("drbench", datasets),
    }
    output = tmp_path / "results"
    profiles = resolve_profiles(
        tasks, global_overrides={"fetch_max_calls": 11}, environ={}
    )
    summary = run_benchmarks(
        tasks,
        RunConfig(
            output_dir=output,
            model="fake",
            base_url="https://unused.test/v1",
            modes=("default",),
            workers=1,
            agent_command=(sys.executable, str(agent)),
            profiles=profiles,
            web_cache_dir=tmp_path / "web-cache",
        ),
        judge=None,
    )

    drb_probe = json.loads((probes / "drb2.json").read_text())
    assert drb_probe["tools"] == ["web_search", "web_fetch"]
    assert drb_probe["env"]["SERPER_MAX_CALLS"] == "20"
    assert drb_probe["env"]["WEB_FETCH_MAX_CALLS"] == "11"
    assert drb_probe["env"]["SERPER_LANGUAGE"] == "zh-cn"
    assert drb_probe["env"]["BCG_TOOL_ROOT"] == drb_probe["cwd"]
    blocklist = Path(drb_probe["env"]["BCG_WEB_BLOCKLIST"])
    assert blocklist.is_file() and not str(blocklist).startswith(drb_probe["cwd"])
    assert drb_probe["files"] == []
    assert "FINAL ANSWER" not in drb_probe["prompt"]

    bench_probe = json.loads((probes / "drbench.json").read_text())
    assert bench_probe["tools"] == ["read", "grep", "find", "ls"]
    assert bench_probe["env"]["BCG_TOOL_ROOT"] == bench_probe["cwd"]
    assert bench_probe["env"]["BCG_WEB_BLOCKLIST"] is None
    assert "FILES.md" in bench_probe["files"]
    assert not any("qa_dict" in name or "eval" in name for name in bench_probe["files"])
    assert "web_search" not in bench_probe["system_prompt"]

    drb_result = json.loads(
        (output / "deepresearch_bench_ii" / "default" / "tasks" / "1.json").read_text()
    )
    assert drb_result["status"] == "report_ready" and drb_result["correct"] is None
    assert (
        Path(drb_result["report"]).read_text(encoding="utf-8").startswith("# 土地财政")
    )
    assert drb_result["report_export"].endswith(
        "export/deepresearch_bench_ii/report/default/idx-1.md"
    )
    assert drb_result["metrics"]["blocked_citation_hits"] == 1
    assert drb_result["web_policy"] == {
        "fetch_refused": 1,
        "search_results_filtered": 1,
        "cache_hits": 0,
    }
    assert not {"rubric", "blocked"} & set(drb_result["metadata"])
    assert BLOCKED_TITLE not in json.dumps(drb_result)

    bench_result = json.loads(
        (output / "drbench" / "default" / "tasks" / "DR0001.json").read_text()
    )
    assert bench_result["status"] == "report_ready"
    assert bench_result["metrics"]["file_read_recall"] == pytest.approx(1 / 3)
    assert bench_result["metrics"]["citation_recall"] == pytest.approx(1 / 3)
    assert "workspace_files" not in bench_result["metadata"]

    drb_summary = summary["benchmarks"]["deepresearch_bench_ii"]["default"]
    assert drb_summary["accuracy"] is None and drb_summary["reports_ready"] == 1
    assert drb_summary["report_metrics"]["blocked_citation_hits"] == 1
    bench_summary = summary["benchmarks"]["drbench"]["default"]
    assert bench_summary["report_metrics"]["citation_recall"] == pytest.approx(1 / 3)
    assert bench_summary["report_metrics_by_difficulty"]["easy"]["tasks"] == 1
    run = json.loads((output / "run.json").read_text())
    assert run["effective_profiles"]["drbench"]["web"] is False
    assert (
        output / ".agent-config" / "drbench" / "default" / "settings.json"
    ).is_file()


def test_event_parser_records_tool_trace_and_policy() -> None:
    stdout = "\n".join(
        json.dumps(event)
        for event in (
            {
                "type": "tool_execution_start",
                "toolCallId": "a",
                "toolName": "grep",
                "args": {"pattern": "x"},
            },
            {
                "type": "message_end",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "a",
                    "toolName": "grep",
                    "content": [{"type": "text", "text": "shared/a.md:1: x"}],
                    "details": {"cached": True},
                },
            },
        )
    )
    parsed = parse_agent_events(stdout)
    assert parsed["tool_trace"] == [
        {"tool": "grep", "args": {"pattern": "x"}, "result": "shared/a.md:1: x"}
    ]
    assert parsed["web_policy"]["cache_hits"] == 1


# ---------------------------------------------------------------------------
# prepare and CLI
# ---------------------------------------------------------------------------


def test_prepare_imports_local_clones_and_writes_manifests(
    datasets: Path, tmp_path: Path
) -> None:
    clone = tmp_path / "drbench-clone"
    (clone / "drbench").mkdir(parents=True)
    (datasets / "drbench").rename(clone / "drbench" / "data")
    target = tmp_path / "prepared"
    result = prepare(
        "drbench", target, source=clone, download=False, log=lambda _m: None
    )
    assert result.tasks == 1 and (target / "drbench" / "MANIFEST.json").is_file()
    assert result.manifest["rendered_tasks"] == 1
    assert (target / "drbench" / "_derived" / "DR0001" / "index.json").is_file()

    drb = prepare(
        "deepresearch_bench_ii",
        target,
        source=datasets / "deepresearch_bench_ii",
        download=False,
        log=lambda _m: None,
    )
    assert drb.manifest["languages"] == {"zh": 1, "en": 1}
    again = prepare(
        "deepresearch_bench_ii", target, download=False, log=lambda _m: None
    )
    assert again.action == "verified existing"
    with pytest.raises(Exception, match="no-download"):
        prepare("drbench", tmp_path / "empty", download=False, log=lambda _m: None)


def test_cli_list_and_dry_run(datasets: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "fake")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unused.test/v1")
    monkeypatch.setenv("SERPER_API_KEY", "k")
    runner = CliRunner()
    listed = runner.invoke(
        app, ["list", "drbench", "--data-root", str(datasets), "--json"]
    )
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.output.splitlines()[0])["task_id"] == "DR0001"
    dry = runner.invoke(
        app,
        [
            "run",
            "drb2",
            "drbench",
            "--data-root",
            str(datasets),
            "--task-ids",
            "drb2:2;drbench:DR0001",
            "--bench-opt",
            "drb2.timeout=60",
            "--dry-run",
        ],
    )
    assert dry.exit_code == 0, dry.output
    assert "timeout=60.0" in dry.output and "Tools: read, grep, find, ls" in dry.output
    bad = runner.invoke(
        app,
        ["run", "drb2", "--data-root", str(datasets), "--task-ids", "99", "--dry-run"],
    )
    assert bad.exit_code != 0 and "Unknown task IDs: 99" in bad.output
