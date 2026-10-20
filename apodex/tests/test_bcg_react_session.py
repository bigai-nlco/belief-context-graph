"""The terminal must pass typed history to ReAct, not a flattened summary."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from apodex.agent_tools import RISK_SAFE, assess_tool_risk
from apodex.config import ModelConfig
from apodex.render import Renderer
from apodex.session import TerminalSession
from frontier_agent.core.messages import assistant_msg, system_msg, user_msg


def test_bcg_recovery_is_read_only(tmp_path):
    assert assess_tool_risk("bcg_recover", {"belief_id": 42}, str(tmp_path)).level == RISK_SAFE


def test_react_passes_verbatim_history_and_checkpoints_each_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = []

    async def forbidden_compaction(*args, **kwargs):
        raise AssertionError("ReAct must not run the old session summary compactor")

    monkeypatch.setattr("apodex.task_runner.SessionHistoryCompactor.compact", forbidden_compaction)

    class Runtime:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run(self, task, *, meta, pipeline_id, extra_input):
            assert pipeline_id == "stateful-react-agent"
            assert meta["_bcg_terminal_session"]
            seen.append((task, list(extra_input["session_messages"])))
            messages = extra_input["session_messages"] or [system_msg("instructions")]
            messages = [*messages, user_msg(task), assistant_msg("done")]
            await meta["_bcg_checkpoint"](1, messages, {})
            return {
                "final_answer": "done",
                "answer_status": "complete",
                "stopped_by": "no_tool",
                "session_messages": messages,
                "turns_used": 1,
                "tool_calls_count": 0,
            }

    monkeypatch.setattr("benchmarks.public.core.kernel_adapter.BenchmarkSession", Runtime)
    session = TerminalSession(
        cfg=ModelConfig(model="offline", api_key="test-only", base_url=None),
        cwd=str(tmp_path),
        renderer=Renderer(theme="mono"),
        auto_approve=False,
        max_turns=5,
        interactive=False,
        mode="react",
    )
    session.tui_mode = True
    original = [
        system_msg("original instructions"),
        user_msg("Original request"),
        {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "id": "r1", "encrypted_content": "opaque", "summary": []}
            ],
            "tool_calls": [{"id": "c1", "name": "web_search", "args": {"query": "first"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "source evidence"},
    ]
    session.history = original[:]
    profile = SimpleNamespace(workflow="stateful-react-agent", workflow_profile="tui")

    async def run():
        await session._run_native_workflow("Follow-up request", profile)
        checkpoint = session.history[:]
        assert checkpoint[: len(original)] == original
        await session._run_native_workflow("Second follow-up", profile)
        assert seen[0] == ("Follow-up request", original)
        assert seen[1] == ("Second follow-up", checkpoint)
        assert session.display_history == session.history
        assert [m["content"] for m in session.history if m["role"] == "user"] == [
            "Original request",
            "Follow-up request",
            "Second follow-up",
        ]
        assert session._inbox is None

    asyncio.run(run())
