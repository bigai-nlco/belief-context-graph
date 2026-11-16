"""Source tools and offline Python; real Linux isolation checks are opt-in.

RUN_REPORT_SANDBOX_TESTS=1 .venv/bin/python -m pytest tests/test_report_research_tools.py -v
No model/search API credentials or paid requests are used.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from pathlib import Path

import httpx
import pytest

from benchmarks.public.reports import drb2, runtime, sources
from benchmarks.public.reports.models import BenchmarkTask
from benchmarks.public.reports.runtime import ReportPolicy
from benchmarks.public.reports.sources import SourceStore

TOOLS = {"web_search", "web_fetch", "read_source", "find_in_source", "run_python_code"}
REAL = pytest.mark.skipif(os.environ.get("RUN_REPORT_SANDBOX_TESTS") != "1",
                          reason="opt-in real bubblewrap test")


def policy(root: Path, **kwargs) -> ReportPolicy:
    return ReportPolicy(BenchmarkTask(drb2.NAME, "smoke", "Compare the annual figures."),
                        root / "sandbox" / "worktree", root / "tools.jsonl", **kwargs)


def tools_for(p: ReportPolicy) -> dict:
    return {t.name: t for t in p.bind([])}


def test_source_identity_persistence_and_immutable_snapshots(tmp_path):
    store = SourceStore(tmp_path / "sources")
    a = store.add("https://example.org/report", "first\nsecond", title="Report")
    assert store.add(a.url, a.text).source_id == a.source_id
    b = store.add(a.url, "updated")
    assert a.source_id != b.source_id
    assert "second" in store.read(a.source_id)
    assert (store.root / f"{a.source_id}.txt").read_text() == a.text
    index = json.loads((store.root / "index.json").read_text())
    assert index[0]["file"] == f"/sources/{a.source_id}.txt"
    assert str(tmp_path) not in json.dumps(index)


def test_find_read_and_long_line_continuation(tmp_path):
    store = SourceStore(tmp_path)
    text = "Heading\n" + "a" * 25_000 + "needle.value\nLast line"
    item = store.add("https://example.org/report", text)
    preview = store.preview(item, 100)
    assert "needle.value" not in preview and "Preview truncated" in preview
    found = store.find(item.source_id, "NEEDLE.VALUE")
    assert "L2 column 25000" in found
    assert "No matching" in store.find(item.source_id, "needle.*")
    first = store.read(item.source_id, 2, 2)
    assert "start_column=" in first and "needle.value" not in first
    assert "needle.value" in store.read(item.source_id, 2, 2, 24990)
    assert "Last line" in store.read(item.source_id, 3, 3)


@pytest.mark.parametrize("source_id", ["../secret", "/etc/passwd", "https://example.org", "S999"])
def test_unknown_source_never_resolves_paths_or_urls(tmp_path, source_id):
    store = SourceStore(tmp_path)
    with pytest.raises(ValueError, match="Unknown source_id"):
        store.read(source_id)


def test_source_capacity_and_partial_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "MAX_SOURCE_CHARS", 10)
    monkeypatch.setattr(sources, "MAX_TOTAL_CHARS", 15)
    store = SourceStore(tmp_path)
    item = store.add("https://example.org/a", "a" * 20)
    assert item.truncated and len(item.text) == 10
    assert "partial" in store.read(item.source_id)
    with pytest.raises(ValueError, match="cache limit"):
        store.add("https://example.org/b", "b" * 10)
    assert len(store.sources) == 1


def test_find_pagination_and_boundaries(tmp_path):
    store = SourceStore(tmp_path)
    item = store.add("https://example.org/a", "\n".join(f"topic {i}" for i in range(30)))
    found = store.find(item.source_id, "topic", max_matches=2)
    assert "L1 column 0" in found and "L2 column 0" in found
    assert "start_line=3" in found
    assert "L3 column 0" in store.find(item.source_id, "topic", start_line=3)
    for kwargs in ({"start_line": 0}, {"start_line": 31}, {"end_line": -1},
                   {"start_column": 999}, {"start_column": -1}):
        with pytest.raises(ValueError):
            store.read(item.source_id, **kwargs)
    with pytest.raises(ValueError):
        store.find(item.source_id, "")


def test_keyword_modes_rank_and_page_without_skipping_earlier_lines(tmp_path):
    store = SourceStore(tmp_path)
    source = store.add("https://example.org/a", "targeting alone\nBISP\nBISP targeting\nBISP exclusion\n")
    assert "No matching" in store.find(source.source_id, "BISP targeting exclusion")
    page1 = store.find(source.source_id, "BISP targeting exclusion", match_mode="any", max_matches=2)
    assert "L3 column" in page1 and "L4 column" in page1 and "offset=2" in page1
    page2 = store.find(source.source_id, "BISP targeting exclusion", match_mode="any", max_matches=2, offset=2)
    assert "L1 column" in page2 and "L2 column" in page2 and "searched before" in page2
    both = store.find(source.source_id, ["BISP", "targeting"], match_mode="all")
    assert "L3 column" in both and "L1 column" not in both
    assert "No matching" in store.find(source.source_id, ["BISP targeting", "exclusion"], match_mode="all")
    assert "L3 column" in store.find(source.source_id, ["BISP targeting", "exclusion"], match_mode="any")
    for kwargs in ({"offset": -1}, {"match_mode": "regex"}, {"query": [" "]}, {"query": ["x"] * 17}):
        args = {"query": "BISP", "match_mode": "any", **kwargs}
        # Duplicate terms are safely deduplicated; 17 distinct terms are rejected.
        if isinstance(args["query"], list) and len(args["query"]) == 17:
            args["query"] = [str(i) for i in range(17)]
        with pytest.raises(ValueError):
            store.find(source.source_id, **args)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r", "\u2028"])
def test_access_tracks_only_returned_text_and_correct_newline_offsets(tmp_path, newline):
    store = SourceStore(tmp_path)
    source = store.add("https://example.org/a", newline.join(["first", "second", "third"]))
    store.preview(source, 5 + len(newline) + 2)
    before = store.seen_chars
    store.read(source.source_id, 2, 2)
    assert store.seen_chars - before == 4
    assert "0 newly returned characters" in store.read(source.source_id, 2, 2)
    before = store.seen_chars
    store.find(source.source_id, "second")
    assert store.seen_chars == before
    ledger = json.loads((store.root / "access.json").read_text())
    assert ledger[0]["returned_chars"] == store.seen_chars
    assert ledger[0]["recent_reads"][-1]["new_chars"] == 0
    assert ledger[0]["recent_queries"][-1]["matches"] == 1


def test_long_line_access_counts_only_displayed_characters(tmp_path):
    store = SourceStore(tmp_path)
    source = store.add("https://example.org/a", "a" * 40_000 + "needle")
    store.read(source.source_id, 1, 1)
    assert store.seen_chars < sources.MAX_READ_CHARS
    assert "0 newly returned characters" in store.read(source.source_id, 1, 1)
    before = store.seen_chars
    store.find(source.source_id, "needle")
    assert 0 < store.seen_chars - before < 1000


@pytest.mark.asyncio
async def test_runtime_keyword_default_line_count_and_exhausted_tools_stay_available(tmp_path):
    p = policy(tmp_path)
    p.search_remaining = p.fetch_remaining = 0
    p.sources.add("https://example.org/a", "heading\nBISP evidence\nnext\nlast")
    tools = tools_for(p)
    result = await tools["find_in_source"].ainvoke({"source_id": "S1", "query": "BISP exclusion"})
    assert "L2 column" in result and "Search mode: any" in result
    result = await tools["read_source"].ainvoke({"source_id": "S1", "start_line": 3, "line_count": 1})
    assert "L3: next" in result and "L4: last" not in result
    assert p.trace[-1]["new_chars"] == len("next")
    for _ in range(2):
        assert "Search budget exhausted" in await tools["web_search"].ainvoke({"q": "another query"})
        assert "Fetch budget exhausted" in await tools["web_fetch"].ainvoke({"url": "https://example.org/a"})
    assert all(e["outcome"] == "budget_exhausted" and not e["ok"] for e in p.trace[-4:])
    assert set(tools_for(p)) == TOOLS
    assert p.search_remaining == p.fetch_remaining == 0


@pytest.mark.asyncio
async def test_fetch_cache_read_find_share_policy_without_network_replay(tmp_path, monkeypatch):
    calls = []
    text = "# Report\n" + "Introduction\n" * 100 + "Annual revenue is 120.\n"

    def handler(req):
        calls.append(str(req.url))
        return httpx.Response(200, text=text, headers={"Content-Type": "text/plain"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(runtime.httpx, "AsyncClient", lambda **kw: real_client(
        transport=httpx.MockTransport(handler), **kw))

    async def vet(url):
        return "", ()

    monkeypatch.setattr(runtime, "vet_public_url", vet)
    p = policy(tmp_path, fetch_calls=1, fetch_chars=20)
    a, b = tools_for(p), tools_for(p)
    assert set(a) == TOOLS
    assert not a["run_python_code"].metadata["is_read_only"]
    preview = await a["web_fetch"].ainvoke({"url": "https://example.org/report"})
    assert "S1" in preview and "Annual revenue" not in preview
    found = await b["find_in_source"].ainvoke({"source_id": "S1", "query": "annual revenue"})
    assert "L102" in found
    read = await a["read_source"].ainvoke({"source_id": "S1", "start_line": 102})
    assert "120" in read and p.fetch_remaining == 0 and len(calls) == 1
    assert not p.sources.root.is_relative_to(p.workspace)
    assert {e["tool"] for e in p.trace} == {"web_fetch", "find_in_source", "read_source"}
    other = tools_for(policy(tmp_path / "other"))
    assert "Unknown source_id" in await other["read_source"].ainvoke({"source_id": "S1"})


def test_blocked_content_beyond_preview_is_not_cached(tmp_path):
    p = policy(tmp_path, fetch_chars=10)
    result = p._cache_source("https://example.org", "https://example.org",
                             "Allowed prefix" * 100 + " tasks_and_rubrics ")
    assert "BLOCKED" in result and not p.sources.sources
    assert list(p.sources.root.iterdir()) == []


def test_strict_bwrap_args_cannot_be_downgraded_by_environment(monkeypatch):
    from plugins.tools import _sandbox

    monkeypatch.setenv("SANDBOX_UNSHARE_NET", "0")
    monkeypatch.setattr(_sandbox, "_sandbox_profile", lambda: "local")
    args = _sandbox._bwrap_base_args(strict_isolation=True)
    assert all(flag in args for flag in ("--unshare-net", "--unshare-pid", "--cap-drop"))
    assert args[args.index("--proc") + 1] == "/proc"
    assert ["--ro-bind", "/proc", "/proc"] not in [args[i:i + 3] for i in range(len(args))]
    assert ["--ro-bind", "/etc", "/etc"] not in [args[i:i + 3] for i in range(len(args))]


@pytest.mark.asyncio
async def test_missing_sandbox_fails_closed(tmp_path, monkeypatch):
    from plugins.tools import _sandbox

    monkeypatch.setattr(_sandbox, "bwrap_available", lambda: False)
    p = policy(tmp_path)
    result = await tools_for(p)["run_python_code"].ainvoke({"code": "print('should not run')"})
    assert "sandbox unavailable" in result
    assert not p.trace[-1]["ok"]


@REAL
@pytest.mark.asyncio
async def test_real_sandbox_computes_persists_files_and_reads_sources(tmp_path):
    p = policy(tmp_path)
    p._cache_source("https://example.org/data", "https://example.org/data",
                    "year,value\n2023,100\n2024,120\n")
    py = tools_for(p)["run_python_code"]
    first = await py.ainvoke({"code": '''import csv, json
from pathlib import Path
import numpy as np
rows = list(csv.DictReader(Path('/sources/S1.txt').open()))
values = [int(row['value']) for row in rows]
result = {'mean': float(np.mean(values)), 'growth': values[1] / values[0] - 1}
Path('result.json').write_text(json.dumps(result))
print(json.dumps(result))
'''})
    assert '"mean": 110.0' in first, first
    second = await py.ainvoke({"code": '''import json
from pathlib import Path
p = Path('result.json')
data = json.loads(p.read_text())
data['checked'] = True
p.write_text(json.dumps(data))
print(data['checked'])
'''})
    assert second.strip() == "True", second
    assert json.loads((p.workspace / "result.json").read_text())["checked"]


@REAL
@pytest.mark.asyncio
async def test_real_sandbox_files_credentials_processes_and_network_are_isolated(tmp_path, monkeypatch):
    sentinel = tmp_path / "host-secret.txt"
    sentinel.write_text("FAKE_HOST_SECRET")
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_KEY_FOR_ISOLATION_TEST")
    monkeypatch.setenv("FRONTIER_AGENT_TOOL_ENV_ALLOWLIST", "OPENAI_API_KEY")
    monkeypatch.setenv("SANDBOX_UNSHARE_NET", "0")
    p = policy(tmp_path / "task")
    p._cache_source("https://example.org/data", "https://example.org/data", "evidence")
    sibling = policy(tmp_path / "other")
    (sibling.workspace / "answer.txt").write_text("OTHER_TASK")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        code = f'''import os, json, socket, resource
from pathlib import Path
checks = {{}}
checks['key_absent'] = 'OPENAI_API_KEY' not in os.environ
checks['host_settings_hidden'] = not Path('/etc/environment').exists()
checks['host_file_hidden'] = not Path({str(sentinel)!r}).exists()
checks['other_task_hidden'] = not Path({str(sibling.workspace / 'answer.txt')!r}).exists()
checks['host_proc_hidden'] = not Path('/proc/{os.getpid()}/environ').exists()
checks['user_env_hidden'] = not Path({str(Path.home() / '.bcg/.env')!r}).exists()
try:
    Path('/sources/S1.txt').write_text('tampered')
    checks['source_readonly'] = False
except OSError:
    checks['source_readonly'] = True
try:
    socket.create_connection(('127.0.0.1', {port}), timeout=1).close()
    checks['host_network_blocked'] = False
except OSError:
    checks['host_network_blocked'] = True
try:
    socket.create_connection(('198.51.100.1', 80), timeout=1).close()
    checks['outbound_blocked'] = False
except OSError:
    checks['outbound_blocked'] = True
checks['memory_limit'] = resource.getrlimit(resource.RLIMIT_AS)[1] <= 1024**3
checks['file_size_limit'] = resource.getrlimit(resource.RLIMIT_FSIZE)[1] <= 32 * 1024**2
checks['process_limit'] = resource.getrlimit(resource.RLIMIT_NPROC)[1] <= 256
print(json.dumps(checks))
'''
        result = await tools_for(p)["run_python_code"].ainvoke({"code": code})
    checks = json.loads(result)
    assert all(checks.values()), checks
    assert sentinel.read_text() == "FAKE_HOST_SECRET"
    assert (p.sources.root / "S1.txt").read_text() == "evidence"


@REAL
@pytest.mark.asyncio
async def test_real_sandbox_timeout_and_bounded_output(tmp_path):
    py = tools_for(policy(tmp_path))["run_python_code"]
    started = time.monotonic()
    result = await py.ainvoke({"code": "import time; time.sleep(10)", "timeout": 1})
    assert "timed out" in result, result
    assert time.monotonic() - started < 8
    output = await py.ainvoke({"code": "print('x' * 1000000); print('FINISHED')"})
    assert len(output) < 21_000
    assert "FINISHED" in output or "truncated" in output


@REAL
@pytest.mark.asyncio
async def test_real_parallel_tasks_have_independent_workspaces(tmp_path):
    policies = [policy(tmp_path / f"task-{i}") for i in range(2)]
    results = await asyncio.gather(*(
        tools_for(p)["run_python_code"].ainvoke({"code": f"from pathlib import Path; Path('value').write_text('{i}'); print({i})"})
        for i, p in enumerate(policies)
    ))
    assert [r.strip() for r in results] == ["0", "1"], results
    assert [(p.workspace / "value").read_text() for p in policies] == ["0", "1"]


@REAL
@pytest.mark.asyncio
async def test_real_cancellation_stops_python_and_allows_next_call(tmp_path, monkeypatch):
    from benchmarks.public.reports import python_exec

    instances = []
    factory = python_exec.BwrapSandbox

    def capture(**kwargs):
        instance = factory(**kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(python_exec, "BwrapSandbox", capture)
    p = policy(tmp_path)
    py = tools_for(p)["run_python_code"]
    task = asyncio.create_task(py.ainvoke({"code": (
        "from pathlib import Path; import time; "
        "Path('started').write_text('yes'); time.sleep(30); "
        "Path('late').write_text('should not execute')"
    )}))
    try:
        for _ in range(200):
            if (p.workspace / "started").exists():
                break
            await asyncio.sleep(0.025)
        assert (p.workspace / "started").exists()
        commands = instances[0].commands
        with commands._strict_lock:
            pids = set(commands._strict_pids)
        assert pids
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    for _ in range(100):
        if not any(Path(f"/proc/{pid}").exists() for pid in pids):
            break
        await asyncio.sleep(0.025)
    assert not any(Path(f"/proc/{pid}").exists() for pid in pids)
    assert not (p.workspace / "late").exists()
    assert not Path(instances[0]._tmpdir).exists()
    assert (await py.ainvoke({"code": "print(2 + 3)"})).strip() == "5"
