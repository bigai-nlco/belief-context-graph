"""End-to-end coverage for the multi-tool-call protocol across
BeliefTracerAgent.update_from_model + BeliefTracerEnvironment.step:

- an assistant turn with 2+ <tool_call> blocks gets canonical call_1/call_2
  ids assigned regardless of what the model wrote,
- execution (sequential and parallel) preserves call_id <-> query pairing,
- the rendered observation has per-call <tool_result> blocks with non-
  colliding call_N_evidence_M numbering and no URLs.

Requires the real rllm package (uses the qwen3 conda env's ToolCall/Tool/
MultiTool/Action, not a stub), matching how test_api_engine_tool_call_parsing
verifies against the real dependency.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    from rllm.tools.tool_base import Tool, ToolOutput  # noqa: E402
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "multi-tool integration requires the external rllm runtime",
        allow_module_level=True,
    )

from bcg.agent.rllm_compat import (  # noqa: E402
    BeliefTracerAgent,
    BeliefTracerEnvironment,
)


class _FakeSearchTool(Tool):
    """Returns one canned evidence per query, tagged with the query text so
    assertions can check call_id <-> query <-> evidence pairing survives
    the whole pipeline."""

    NAME = "averitec_search"
    FEEDS_MEMORY = True

    def __init__(self, name: str = NAME, description: str | None = None):
        super().__init__(name=name, description=description or "fake search")

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }

    def set_task(self, task):
        return None

    def forward(self, query: str, top_k: int | None = None) -> ToolOutput:
        evidences = [
            {"text": f"evidence A for [{query}]", "url": "http://example.com/a"},
            {"text": f"evidence B for [{query}]", "url": "http://example.com/b"},
        ]
        return ToolOutput(
            name=self.name,
            output=f"formatted results for {query}",
            metadata={"query": query, "evidences": evidences},
        )


def _agent() -> BeliefTracerAgent:
    return BeliefTracerAgent(
        system_prompt="sys",
        parser_name="qwen",
        tools=["averitec_search"],
    )


def _env(max_tool_workers: int = 1) -> BeliefTracerEnvironment:
    env = BeliefTracerEnvironment(
        reward_fn=lambda task, answer: SimpleNamespace(reward=0.0, is_correct=False, metadata={}),
        max_steps=10,
        tool_map={"averitec_search": _FakeSearchTool},
        max_tool_workers=max_tool_workers,
    )
    env.reset(task={"question": "q", "extra_info": {}})
    return env


def _two_call_response() -> str:
    return (
        '<tool_call>\n{"name": "averitec_search", "arguments": {"query": "query A"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "averitec_search", "arguments": {"query": "query B"}}\n</tool_call>'
    )


def test_update_from_model_assigns_local_call_ids_and_canonicalizes():
    agent = _agent()
    action = agent.update_from_model(_two_call_response())

    assert len(action.action) == 2
    assert [c.id for c in action.action] == ["call_1", "call_2"]
    assert action.action[0].arguments == {"query": "query A"}
    assert action.action[1].arguments == {"query": "query B"}

    stored_content = agent._messages[-1]["content"]
    assert '<tool_call id="call_1">' in stored_content
    assert '<tool_call id="call_2">' in stored_content


def test_update_from_model_resets_ids_every_turn():
    agent = _agent()
    agent.update_from_model(_two_call_response())
    # A fresh turn's blocks must restart at call_1, not continue call_3/call_4.
    action2 = agent.update_from_model(
        '<tool_call>\n{"name": "averitec_search", "arguments": {"query": "query C"}}\n</tool_call>'
    )
    assert [c.id for c in action2.action] == ["call_1"]


@pytest.mark.parametrize("max_tool_workers", [1, 4])
def test_step_groups_results_by_call_id_sequential_and_parallel(max_tool_workers):
    agent = _agent()
    action = agent.update_from_model(_two_call_response())
    env = _env(max_tool_workers=max_tool_workers)

    next_obs, reward, done, info = env.step(action)

    # tool_metadata preserves call_id <-> query pairing, in input order,
    # regardless of execution mode.
    tool_metadata = info["tool_metadata"]
    assert [tm["tool_call_id"] for tm in tool_metadata] == ["call_1", "call_2"]
    assert tool_metadata[0]["metadata"]["query"] == "query A"
    assert tool_metadata[1]["metadata"]["query"] == "query B"

    # Rendered observation: two independent <tool_result> blocks, each with
    # its own call-id-prefixed evidence numbering, no collisions, no URLs.
    assert '<tool_result id="call_1" name="averitec_search">' in next_obs
    assert '<tool_result id="call_2" name="averitec_search">' in next_obs
    assert "Query: query A" in next_obs
    assert "Query: query B" in next_obs
    assert "[call_1_evidence_1]" in next_obs
    assert "[call_1_evidence_2]" in next_obs
    assert "[call_2_evidence_1]" in next_obs
    assert "[call_2_evidence_2]" in next_obs
    assert "evidence A for [query A]" in next_obs
    assert "evidence B for [query B]" in next_obs
    # No duplicate/legacy bare numbering and no URLs anywhere in the observation.
    assert "[Evidence 1]" not in next_obs
    assert "http://" not in next_obs


def test_three_calls_preserve_order_under_parallel_execution():
    agent = _agent()
    text = "\n".join(
        f'<tool_call>\n{{"name": "averitec_search", "arguments": {{"query": "q{i}"}}}}\n</tool_call>'
        for i in range(1, 4)
    )
    action = agent.update_from_model(text)
    env = _env(max_tool_workers=4)

    next_obs, _, _, info = env.step(action)

    assert [tm["tool_call_id"] for tm in info["tool_metadata"]] == ["call_1", "call_2", "call_3"]
    for i in range(1, 4):
        assert f'<tool_result id="call_{i}" name="averitec_search">' in next_obs
        assert f"Query: q{i}" in next_obs
