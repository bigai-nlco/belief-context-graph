from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass


def _stub_rllm() -> None:
    try:
        from rllm.agents.agent import Action, BaseAgent, Step, Trajectory  # noqa: F401
        from rllm.environments.base.base_env import BaseEnv  # noqa: F401
        from rllm.tools.multi_tool import MultiTool  # noqa: F401
        from rllm.tools.tool_base import ToolCall, ToolOutput  # noqa: F401
        return
    except Exception:
        pass

    rllm = types.ModuleType("rllm")
    agents_pkg = types.ModuleType("rllm.agents")
    agent_mod = types.ModuleType("rllm.agents.agent")
    parser_mod = types.ModuleType("rllm.parser")
    env_pkg = types.ModuleType("rllm.environments")
    base_pkg = types.ModuleType("rllm.environments.base")
    base_env_mod = types.ModuleType("rllm.environments.base.base_env")
    tools_pkg = types.ModuleType("rllm.tools")
    multi_tool_mod = types.ModuleType("rllm.tools.multi_tool")
    tool_base_mod = types.ModuleType("rllm.tools.tool_base")

    class BaseAgent:
        pass

    class BaseEnv:
        pass

    @dataclass
    class Action:
        action: object = None

    class Step:
        model_fields = {"metadata": object}

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Trajectory:
        def __init__(self):
            self.steps = []

    @dataclass
    class ToolCall:
        name: str = ""
        arguments: dict | None = None

    class ToolOutput:
        def __init__(self, name=None, output=None, error=None, metadata=None):
            self.name = name
            self.output = output
            self.error = error
            self.metadata = metadata

    class MultiTool:
        def __init__(self, tools=None, tool_map=None):
            self.tool_map = dict(tool_map or {})

    class Parser:
        tool_call_begin = "<tool_call>"

        def parse(self, response):
            return []

    def get_tool_parser(name):
        return Parser

    agent_mod.BaseAgent = BaseAgent
    agent_mod.Action = Action
    agent_mod.Step = Step
    agent_mod.Trajectory = Trajectory
    base_env_mod.BaseEnv = BaseEnv
    multi_tool_mod.MultiTool = MultiTool
    tool_base_mod.ToolCall = ToolCall
    tool_base_mod.ToolOutput = ToolOutput
    parser_mod.get_tool_parser = get_tool_parser

    sys.modules["rllm"] = rllm
    sys.modules["rllm.agents"] = agents_pkg
    sys.modules["rllm.agents.agent"] = agent_mod
    sys.modules["rllm.environments"] = env_pkg
    sys.modules["rllm.environments.base"] = base_pkg
    sys.modules["rllm.environments.base.base_env"] = base_env_mod
    sys.modules["rllm.tools"] = tools_pkg
    sys.modules["rllm.tools.multi_tool"] = multi_tool_mod
    sys.modules["rllm.tools.tool_base"] = tool_base_mod
    sys.modules["rllm.parser"] = parser_mod


def test_cli_context_memory_defaults_preserve_graph_mode():
    from bcg.agent.rollout import _parse_args

    cfg = _parse_args(["--model", "/m/Qwen", "--no-auto-ui"])

    assert cfg.context_memory_mode == "belief_graph"
    assert cfg.belief_graph_mode == "augment"
    assert cfg.enable_archive is True
    assert cfg.recent_turns == 2
    assert cfg.layered_context is True


def test_cli_context_memory_baseline_forces_layered_context():
    from bcg.agent.rollout import _parse_args

    cfg = _parse_args([
        "--model", "/m/Qwen",
        "--no-auto-ui",
        "--context-memory-mode", "codex_handoff",
        "--context-memory-recent-observations", "5",
        "--context-memory-max-chars", "4096",
        "--context-memory-summarizer", "llm",
        "--context-memory-summarizer-max-tokens", "1024",
        "--context-memory-summarizer-timeout", "30",
        "--context-memory-summarizer-failure-limit", "2",
        "--context-memory-log-preview-chars", "300",
    ])

    assert cfg.context_memory_mode == "codex_handoff"
    assert cfg.layered_context is True
    assert cfg.context_memory_recent_observations == 5
    assert cfg.context_memory_max_chars == 4096
    assert cfg.context_memory_summarizer == "llm"
    assert cfg.context_memory_summarizer_max_tokens == 1024
    assert cfg.context_memory_summarizer_timeout == 30
    assert cfg.context_memory_summarizer_failure_limit == 2
    assert cfg.context_memory_log_preview_chars == 300


def test_agent_graph_mode_still_renders_belief_graph_slot():
    _stub_rllm()
    from bcg.agent.rllm_compat import BeliefTracerAgent

    class Client:
        def format_graph_for_prompt(
            self, snapshot, fmt="structured", include_relations=True, **kwargs
        ):
            return "graph text"

    agent = BeliefTracerAgent(system_prompt="sys", layered_context=True)
    agent.update_from_env({"question": "claim"}, 0, False, {})
    agent.inject_belief_graph({"beliefs": [{"id": 1}]}, Client())

    messages = agent.chat_completions
    assert any("<belief_graph>" in m["content"] for m in messages)
    assert not any("<context_memory" in m["content"] for m in messages)


def test_recent_turns_zero_keeps_no_raw_turns_and_minus_one_keeps_all():
    _stub_rllm()
    from bcg.agent.rllm_compat import BeliefTracerAgent

    history = [
        {"role": "assistant", "content": "a0"},
        {"role": "tool", "content": "t0"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
    ]

    assert BeliefTracerAgent._keep_recent_turns(history, 0) == []
    assert BeliefTracerAgent._keep_recent_turns(history, -1) == history


def test_agent_deepseek_v4_graph_uses_raw_role_markers_without_xml_wrapper():
    _stub_rllm()
    from bcg.agent.rllm_compat import BeliefTracerAgent

    encoded = (
        '<｜begin▁of▁sentence｜><｜User｜>{"id":1,"content":"belief"}'
        "<｜end▁of▁sentence｜>"
    )

    class Client:
        def format_graph_for_prompt(
            self, snapshot, fmt="structured", include_relations=True, **kwargs
        ):
            assert fmt == "deepseek_v4"
            assert kwargs["deepseek_v4_payload_format"] == "xml"
            return encoded

    agent = BeliefTracerAgent(
        system_prompt="sys",
        layered_context=True,
        graph_format="deepseek_v4",
        deepseek_v4_payload_format="xml",
        belief_graph_placement="user",
    )
    agent.update_from_env({"question": "claim"}, 0, False, {})
    agent.inject_belief_graph({"beliefs": [{"id": 1}]}, Client())

    messages = agent.chat_completions
    assert messages[2]["content"] == encoded
    assert not any("<belief_graph>" in message["content"] for message in messages)


def test_agent_baseline_uses_context_memory_slot_not_graph_slot():
    _stub_rllm()
    from bcg.agent.rllm_compat import BeliefTracerAgent

    class Client:
        def format_graph_for_prompt(self, snapshot, fmt="structured", include_relations=True):
            return "graph text"

    agent = BeliefTracerAgent(
        system_prompt="sys",
        context_memory_mode="codex_handoff",
    )
    agent.update_from_env({"question": "claim"}, 0, False, {})
    agent.set_context_memory_message({
        "role": "user",
        "content": '<context_memory type="codex_handoff">summary</context_memory>',
    })
    agent.inject_belief_graph({"beliefs": [{"id": 1}]}, Client())

    messages = agent.chat_completions
    assert [m["role"] for m in messages[:3]] == ["system", "user", "user"]
    assert '<context_memory type="codex_handoff">' in messages[2]["content"]
    assert not any("<belief_graph>" in m["content"] for m in messages)


def test_agent_none_mode_uses_no_memory_slot():
    _stub_rllm()
    from bcg.agent.rllm_compat import BeliefTracerAgent

    class Client:
        def format_graph_for_prompt(self, snapshot, fmt="structured", include_relations=True):
            return "graph text"

    agent = BeliefTracerAgent(
        system_prompt="sys",
        layered_context=True,
        context_memory_mode="none",
    )
    agent.update_from_env({"question": "claim"}, 0, False, {})
    agent.inject_belief_graph({"beliefs": [{"id": 1}]}, Client())

    messages = agent.chat_completions
    assert [m["role"] for m in messages[:2]] == ["system", "user"]
    assert not any("<context_memory" in m["content"] for m in messages)
    assert not any("<belief_graph>" in m["content"] for m in messages)


def test_build_context_memory_disabled_modes_return_none():
    from bcg.agent.context_memory import (
        ContextMemoryConfig,
        build_context_memory,
        uses_belief_graph_service,
    )

    assert build_context_memory(ContextMemoryConfig(mode="belief_graph")) is None
    assert build_context_memory(ContextMemoryConfig(mode="none")) is None
    assert uses_belief_graph_service("belief_graph", "augment") is True
    assert uses_belief_graph_service("belief_graph", "only") is True
    assert uses_belief_graph_service("belief_graph", "none") is False
    assert uses_belief_graph_service("none", "augment") is False
    assert uses_belief_graph_service("claude_pipeline", "augment") is False
    assert uses_belief_graph_service("codex_handoff", "augment") is False
    assert uses_belief_graph_service("opencode_marker", "augment") is False


def test_context_memory_modes_render_expected_blocks():
    from bcg.agent.context_memory import (
        ContextMemoryConfig,
        build_context_memory,
        context_memory_prompt_templates,
    )

    for mode, marker in [
        ("claude_pipeline", "<recent_evidence>"),
        ("codex_handoff", "<handoff_summary>"),
        ("opencode_marker", "<anchored_summary>"),
    ]:
        memory = build_context_memory(ContextMemoryConfig(mode=mode, max_chars=5000))
        assert memory is not None
        memory.observe_initial(system="sys", question="Claim to verify")
        memory.observe_assistant(content="I searched for the claim.")
        memory.observe_tool(
            content="tool output",
            tool_metadata=[{
                "name": "averitec_search",
                "tool_call_id": "call_1",
                "arguments": {"query": "claim evidence"},
                "output": "evidence output",
                "metadata": {"query": "claim evidence", "evidences": [
                    {"summary": "important evidence", "text": "long evidence"}
                ]},
                "archive_entries": [{"raw_url": "file://archives/x/raw/t1_e001.json", "summary": "raw summary"}],
                "feeds_memory": True,
            }],
        )
        asyncio.run(memory.maybe_compact())
        message = memory.render_message()
        assert message is not None
        assert f'type="{mode}"' in message["content"]
        assert marker in message["content"]
        assert "important evidence" in message["content"]
        state = memory.export_state()
        assert state["config"]["summarizer"] == "local"
        assert state["config"]["effective_summarizer"] == "local"
        assert state["prompt_templates"] == context_memory_prompt_templates(mode)
        assert state["prompt_templates"]


def test_context_memory_llm_summarizer_uses_main_agent_model_config(monkeypatch):
    from bcg.agent import context_memory as cm
    from bcg.agent.context_memory import ContextMemoryConfig, build_context_memory

    calls = []

    async def fake_call_chat_completion(**kwargs):
        calls.append(kwargs)
        if "claude" in kwargs["model"]:
            return "<summary>LLM summary from DeepSeek</summary>"
        return "LLM summary from DeepSeek"

    monkeypatch.setattr(cm, "_call_chat_completion", fake_call_chat_completion)

    for mode in ["claude_pipeline", "codex_handoff", "opencode_marker"]:
        memory = build_context_memory(
            ContextMemoryConfig(
                mode=mode,
                summarizer="llm",
                summarizer_model="deepseek-v4-pro-260425",
                summarizer_base_url="https://ark.cn-beijing.volces.com/api/v3",
                summarizer_api_key="test-key",
                max_chars=5000,
            )
        )
        assert memory is not None
        memory.observe_initial(system="sys", question="Claim to verify")
        memory.observe_assistant(content="I need evidence.")
        memory.observe_tool(
            content="tool output",
            tool_metadata=[{
                "name": "averitec_search",
                "tool_call_id": "call_1",
                "arguments": {"query": "claim evidence"},
                "output": "evidence output",
                "metadata": {"query": "claim evidence", "evidences": [
                    {"summary": "important evidence"}
                ]},
                "feeds_memory": True,
            }],
        )
        asyncio.run(memory.maybe_compact())
        message = memory.render_message()
        assert message is not None
        assert "LLM summary from DeepSeek" in message["content"]
        assert memory.export_state()["config"]["effective_summarizer"] == "llm"

    assert len(calls) == 3
    assert {call["model"] for call in calls} == {"deepseek-v4-pro-260425"}
    assert {call["base_url"] for call in calls} == {"https://ark.cn-beijing.volces.com/api/v3"}
    assert {call["api_key"] for call in calls} == {"test-key"}


def test_codex_handoff_truncation_preserves_xml_closing_tags():
    from bcg.agent.context_memory import ContextMemoryConfig, build_context_memory

    memory = build_context_memory(
        ContextMemoryConfig(
            mode="codex_handoff",
            max_chars=1200,
            recent_observations=2,
            tool_summary_chars=120,
        )
    )
    assert memory is not None
    memory.observe_initial(system="sys", question="Claim to verify")
    memory.observe_assistant(content="I searched.")
    memory.observe_tool(
        content="tool output",
        tool_metadata=[{
            "name": "averitec_search",
            "tool_call_id": "call_1",
            "arguments": {"query": "claim evidence"},
            "output": "very long evidence " * 500,
            "metadata": {"query": "claim evidence", "evidences": [
                {"summary": "important evidence " * 80}
            ]},
            "feeds_memory": True,
        }],
    )
    asyncio.run(memory.maybe_compact())

    content = memory.render_message()["content"]
    assert "</handoff_summary>" in content
    assert "</verbatim_recent_observations>" in content
    assert content.endswith("</context_memory>")
    assert len(content) <= 1200 + len('<context_memory type="codex_handoff">\n\n</context_memory>')
