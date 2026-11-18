"""RUN-0027: exercise the actual node, loop, request projection and terminal state."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from frontier_agent.core.llm import LLMResponse
from frontier_agent.core.runtime import registry
from frontier_agent.core.runtime.resources.manager import ResourceManager
from frontier_agent.core.tool import tool
from tests.test_bcg_recent_steps import TaskGraph
from workflows.stateful_react_agent.bcg import runtime
from workflows.stateful_react_agent.nodes import main_agent
from workflows.stateful_react_agent.observers import NO_VISIBLE_ANSWER, VisibleAnswerGuard

PREAMBLE = "I need to continue researching this question."
GHOST = "<belief_graph>\n[B83] A search that was never executed.\n</belief_graph>"
ANSWER = "The sources support the following conclusion."


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "retry", "reporter"), [
    ("bcg_compaction", "answer", False),
    ("bcg_compaction", "tool", False),
    ("bcg_compaction", "echo", False),
    ("bcg_compaction", "empty", True),
    ("default", "empty", False),
])
async def test_terminal_turn_never_promotes_previous_tool_commentary(
    tmp_path, monkeypatch, mode, retry, reporter,
):
    @tool
    async def lookup() -> str:
        """Read offline test evidence."""
        return "Source evidence."

    def action(i):
        return LLMResponse(content=PREAMBLE, tool_calls=[{
            "id": f"lookup-{i}", "type": "function", "function": {"name": "lookup", "arguments": "{}"},
        }])

    requests = []

    class Provider:
        model = "offline"

        async def chat(self, messages, **kwargs):
            requests.append(copy.deepcopy(messages))
            n = len(requests)
            if n == 1 or (n == 3 and retry == "tool"):
                return action(n)
            if n == 2 or (n == 3 and retry == "empty"):
                return LLMResponse(content="  ", reasoning_content=GHOST, finish_reason="stop")
            if n == 3 and retry == "echo":
                return LLMResponse(content=GHOST + "<|im_end|>", finish_reason="stop")
            return LLMResponse(content=ANSWER, finish_reason="stop")

    for name in ("workspace", "outputs", "inputs"):
        (tmp_path / name).mkdir()
        monkeypatch.setenv("FRONTIER_AGENT_" + name.upper() + "_DIR", str(tmp_path / name))
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    provider, graph = Provider(), TaskGraph()
    profile = {"agent": {"context_mode": mode, "main_max_turns": 20, "reporter": reporter}}
    monkeypatch.setattr(main_agent, "is_language_detect_enabled", lambda: False)
    monkeypatch.setattr(main_agent, "_resolve_llm_and_profile", lambda *a, **kw: (provider, profile, None))
    monkeypatch.setattr(main_agent, "_tools_for_stateful_react", lambda *a: [lookup])
    monkeypatch.setattr(runtime, "HttpGraphBackend", lambda *a: graph)
    trial = tmp_path / "trial"
    saved = registry.snapshot()
    try:
        registry.register(ResourceManager, ResourceManager(provider, {"lookup": lookup}))
        result = await main_agent.react_agent_node(
            {"original_question": "Research this.", "metadata": {"_trial_dir": str(trial)}},
            SimpleNamespace(task_id="empty-final-regression"),
        )
    finally:
        registry.restore(saved)
    assert len(requests) == (4 if retry == "tool" else 3)
    hint = requests[2][-1]
    assert hint["role"] == "user" and hint.get("transient") and "original task" in hint["content"]
    assert requests[2][:-1] == requests[1]  # Exactly one request-only difference.
    assert result["tool_calls_count"] == (2 if retry == "tool" else 1)
    if retry == "tool":
        assert hint not in requests[3]
    if mode == "bcg_compaction":
        assert requests[1][-1]["role"] == "user"
        assert requests[1][-1]["content"].startswith("<belief_graph>")
    for stored in [result["session_messages"], graph.pushes]:
        assert hint["content"] not in str(stored) and "A search that was never executed." not in str(stored)
    events = [json.loads(line) for line in (trial / "agent/trajectories/react_agent.jsonl").read_text().splitlines()]
    assert len([x for x in events if x["t"] == "llm"]) == len(requests)
    if retry in {"answer", "tool"}:
        assert result["stopped_by"] == "no_tool" and result["answer_status"] == "complete"
        assert result["final_answer"] == ANSWER
    else:
        assert events[-1]["stopped_by"] == result["stopped_by"] == NO_VISIBLE_ANSWER
        assert result["final_answer"] == result["final_content"] == ""
        assert result["answer_status"] == "not_found"


@pytest.mark.asyncio
async def test_landing_retry_does_not_reenable_tools_or_reject_legitimate_graph_quotes():
    guard = VisibleAnswerGuard(graph_compaction=True)
    ctx = SimpleNamespace(ai_text="<think>private, unfinished", tool_calls=[], blocked_tool_calls=[],
                          finish_reason="stop", tool_schemas_stripped=True, turn=20, metadata={})
    assert (await guard.on_llm_response(ctx)).continue_to_next_turn
    assert ctx.metadata["_llm_strip_tools"] is True
    ctx.ai_text = GHOST + "\nThe previous observation is unverified; here is the report."
    assert await guard.on_llm_response(ctx) is None
    assert not guard.retry_instruction()
