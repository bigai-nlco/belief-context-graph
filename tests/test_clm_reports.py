"""CLM integration checks; optional real harness + sandbox uses a fake local LLM."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from benchmarks.public.reports import clm
from benchmarks.public.reports.clm.bridge import Bridge
from benchmarks.public.reports.models import BenchmarkTask
from benchmarks.public.runner import run_reports
from tools.case_study_studio.clm_records import CLMRecords
from tools.case_study_studio.tests.test_control import build_fixture, request_config, write


def profile(base="http://127.0.0.1:1/v1", model="Qwen3.5-9B"):
    return {"llm": {"model": model, "api_key": "fake-private-provider-key", "base_url": base,
                    "temperature": 1.0, "top_p": 0.95, "max_tokens": 8192,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20}},
            "agent": {"context_mode": "clm", "max_len": 65536, "main_max_turns": 3}}


def test_settings_and_wire_generation_policy(tmp_path):
    selected = profile()
    settings = clm.settings(selected)
    assert settings["context_budget_tokens"] - settings["context_budget_reserve_tokens"] == 26624
    assert settings["lm_call_cap"] == 3
    bridge = Bridge(None, selected["llm"], tmp_path, tmp_path)
    incoming = {"model": "Qwen3.5-9B", "messages": [{"role": "user", "content": "protected"}],
                "tools": [{"type": "function"}], "max_tokens": 1, "temperature": 0.7}
    payload = bridge.payload(incoming)
    assert payload["messages"] == incoming["messages"] and payload["tools"] == incoming["tools"]
    assert payload["temperature"] == 1 and payload["max_tokens"] == 8192
    assert payload["chat_template_kwargs"]["enable_thinking"] and payload["top_k"] == 20
    bridge.llm = {**selected["llm"], "model": "gpt-6-luna", "extra_body": {"reasoning_effort": "low"}}
    payload = bridge.payload({**incoming, "model": "gpt-6-luna"})
    assert payload["reasoning_effort"] == "low" and payload["max_completion_tokens"] == 8192
    assert "max_tokens" not in payload and "chat_template_kwargs" not in payload
    selected["agent"]["max_len"] = 30000
    with pytest.raises(ValueError, match="context window"):
        clm.settings(selected)


def test_clm_run_selection_and_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(clm, "installation", lambda: {"available": True, "reason": ""})
    control = build_fixture(tmp_path)
    user = control.register("clm-test")["user"]
    run = control.create_run(request_config(mode="clm", graph_profile="not-used"), user)
    config = json.loads(control.row(run["id"])["config"])
    assert config["pipeline"] == "clm" and config["judge_reports"]
    assert not config["task_state"] and "graph_profile" not in config
    assert run["graph"] is None and run["tasks"][0]["graph_tokens"] == 0
    with pytest.raises(Exception, match="任务状态"):
        control.create_run(request_config(mode="clm", task_state=True), user)
    # No extra --clm-results flag or server restart is needed for managed runs.
    trial = control.directory(control.row(run["id"])) / "drb2/trials/2"
    write(trial, "clm/summary.json", {"status": "submitted", "task_id": "drb2:2"})
    write(trial, "clm/manifest.json", {"settings": {"model_name": "Qwen"}})
    write(trial, "clm/report.md", "# Report")
    write(trial, "result.json", {"status": "completed"})
    store = CLMRecords([], control)
    alias = f"{run['number']}-task-2"
    assert store.runs()[0]["id"] == alias and store.summary(alias)["status"] == "completed"
    assert control.detail(run["id"])["tasks"][0]["clm_recording"] == alias
    assert store.report(alias)["report"] == "# Report"
    # Unselected tasks and arbitrary sibling folders are not published.
    write(trial.parent / "99", "clm/summary.json", {"status": "submitted"})
    assert len(store.runs()) == 1
    monkeypatch.setattr(clm, "installation", lambda: {"available": False, "reason": "CLM 未安装"})
    with pytest.raises(Exception, match="未安装"):
        control.create_run(request_config(mode="clm"), user)


@pytest.mark.skipif(os.getenv("RUN_CLM_SANDBOX_TESTS") != "1", reason="opt-in local harness + bwrap test")
@pytest.mark.parametrize("outcome", ["submitted", "draft", "cancelled"])
def test_official_harness_with_fake_model(tmp_path: Path, monkeypatch, outcome):
    """Three real CLM turns, no paid LLM, no search or grading requests."""
    import yaml

    async def exercise():
        info = clm.provenance()
        seen = []
        edited = "[[CTX_TURN 1 role=assistant]]\nVerified: synthetic evidence collected. Next: submit the report.\n"
        report = "# Synthetic report\n\nThis offline report tests integration, not benchmark performance.\n\n## References\nhttps://example.org/\n"
        commands = ["python3 -c \"print('synthetic evidence ' * 500)\"",
                    f"python3 -c \"from pathlib import Path; Path('/tmp/.live_ctx/LIVE_CTX_MAIN.txt').write_text({edited!r}); Path('/workspace/report.md').write_text({report!r})\"",
                    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" if outcome == "submitted" else "echo Not submitted"]
        if outcome == "cancelled":
            commands[0] = "sleep 60"

        async def fake_llm(reader, writer):
            headers = await reader.readuntil(b"\r\n\r\n")
            fields = dict(line.split(b":", 1) for line in headers.split(b"\r\n")[1:] if b":" in line)
            length = int(next(v for k, v in fields.items() if k.lower() == b"content-length"))
            incoming = json.loads(await reader.readexactly(length))
            seen.append(incoming)
            index = len(seen)
            response = {"id": f"fake-{index}", "object": "chat.completion", "created": 1,
                        "model": "Qwen3.5-9B", "choices": [{"index": 0, "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "content": "Offline test", "reasoning_content": "Test reasoning",
                                    "tool_calls": [{"id": f"call-{index}", "type": "function", "function": {
                                        "name": "bash", "arguments": json.dumps({"command": commands[min(index-1, 2)]})}}]}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
            body = json.dumps(response).encode()
            writer.write(f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(fake_llm, "127.0.0.1", 0)
        try:
            selected = profile(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1")
            path = tmp_path / "profile.yaml"
            path.write_text(yaml.safe_dump(selected))
            task = BenchmarkTask("deepresearch_bench_ii", "5", "Write an offline synthetic report for task FIVE.",
                                 reference={"rubric": "DO NOT SHOW PRIVATE RUBRIC"})
            config = {"profile": str(path), "pipeline": "clm", "mode": "clm", "max_turns": 3,
                      "search_calls": 7, "fetch_calls": 9, "fetch_chars": 20000,
                      "clm_runtime": info, "clm_settings": clm.settings(selected, 3)}
            trial = tmp_path / "trial"
            running = asyncio.create_task(run_reports.execute_task(task, config, trial))
            if outcome == "cancelled":
                for _ in range(300):
                    if (trial / "clm/turns/0001/context.before.txt").exists():
                        break
                    if running.done():
                        raise AssertionError(await running)
                    await asyncio.sleep(0.05)
                else:
                    running.cancel()
                    raise AssertionError("CLM did not enter the first sandbox command")
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(running, 8)
                assert run_reports.collect_usage(trial, mode="clm")["combined"]["total_tokens"] == 120
                return
            result = await asyncio.wait_for(running, 90)
            assert result["status"] == ("completed" if outcome == "submitted" else "agent_error"), (result, (trial / "clm/run.log").read_text())
            assert len(seen) == 3
            assert all("PRIVATE RUBRIC" not in json.dumps(request) for request in seen)
            assert "task FIVE" in seen[0]["messages"][1]["content"]
            assert "7 searches and 9 page fetches" in seen[0]["messages"][1]["content"]
            assert seen[0]["messages"][:2] == seen[2]["messages"][:2]
            assert any("Verified: synthetic evidence" in str(m) for m in seen[2]["messages"])
            assert result["usage"]["combined"]["total_tokens"] == 360
            assert result["usage"]["graph"]["total_tokens"] == 0
            assert (trial / "report.md").read_text().strip() == report.strip()
            assert (trial / "case-study.json.gz").is_file()
            step = json.loads((trial / "clm/turns/0002/step.json").read_text())
            assert step["context_edit_applied"] and step["file_changed"]
            assert (trial / "clm/turns/0002/context.after.txt").read_text() == edited
            assert json.loads((trial / "clm/turns/0003/input.json").read_text()) == seen[2]
            assert not (trial / "agent/bcg").exists()
            assert not any("fake-private-provider-key" in p.read_text() for p in trial.rglob("*.json"))
        finally:
            server.close()
            await server.wait_closed()

    monkeypatch.setenv("SANDBOX_BACKEND", "bwrap")
    asyncio.run(exercise())
