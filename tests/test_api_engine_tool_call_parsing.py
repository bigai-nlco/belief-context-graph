"""Tests for APIEngine's <tool_call> content parsing, incl. multi-call and
id="call_N" attribute compatibility (see tool_call_protocol.py).

Stubs the minimal rllm surface APIEngine depends on, same approach as
test_layered_archive.py, so this runs without the real rllm package.
"""

from __future__ import annotations

import sys
import types

import pytest


def _stub_rllm() -> None:
    try:
        from rllm.tools.tool_base import ToolCall  # noqa: F401
        from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine  # noqa: F401
        return
    except Exception:
        pass

    rllm = types.ModuleType("rllm")
    tools = types.ModuleType("rllm.tools")
    tool_base = types.ModuleType("rllm.tools.tool_base")
    engine_pkg = types.ModuleType("rllm.engine")
    rollout_pkg = types.ModuleType("rllm.engine.rollout")
    rollout_engine_mod = types.ModuleType("rllm.engine.rollout.rollout_engine")

    class ToolCall:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class ModelOutput:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class RolloutEngine:
        pass

    tool_base.ToolCall = ToolCall
    rollout_engine_mod.ModelOutput = ModelOutput
    rollout_engine_mod.RolloutEngine = RolloutEngine

    sys.modules["rllm"] = rllm
    sys.modules["rllm.tools"] = tools
    sys.modules["rllm.tools.tool_base"] = tool_base
    sys.modules["rllm.engine"] = engine_pkg
    sys.modules["rllm.engine.rollout"] = rollout_pkg
    sys.modules["rllm.engine.rollout.rollout_engine"] = rollout_engine_mod


@pytest.fixture()
def engine():
    _stub_rllm()
    from bcg.agent.api_engine import APIEngine

    return APIEngine(model="test-model")


def _response(content: str, finish_reason: str = "stop") -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


def test_parses_single_legacy_tool_call_no_id(engine):
    content = (
        '<tool_call>\n{"name": "averitec_search", "arguments": {"query": "q1"}}\n</tool_call>'
    )
    out = engine._parse_response(_response(content))
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "averitec_search"
    assert out.tool_calls[0].arguments == {"query": "q1"}
    assert out.finish_reason == "tool_calls"


def test_parses_multiple_tool_calls_with_id_attribute(engine):
    content = (
        '<tool_call id="call_1">\n{"name": "averitec_search", "arguments": {"query": "q1"}}\n</tool_call>\n'
        '<tool_call id="call_2">\n{"name": "averitec_search", "arguments": {"query": "q2"}}\n</tool_call>'
    )
    out = engine._parse_response(_response(content))
    assert len(out.tool_calls) == 2
    assert [tc.arguments["query"] for tc in out.tool_calls] == ["q1", "q2"]
    assert out.finish_reason == "tool_calls"


def test_parses_mixed_id_and_no_id_blocks_in_one_message(engine):
    # Defensive: a model imitating canonicalized history might only add id=
    # to some blocks. Both must still parse.
    content = (
        '<tool_call id="call_1">\n{"name": "averitec_search", "arguments": {"query": "q1"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "averitec_search", "arguments": {"query": "q2"}}\n</tool_call>'
    )
    out = engine._parse_response(_response(content))
    assert len(out.tool_calls) == 2


def test_bare_json_tool_call_without_wrapper_still_parses(engine):
    content = '{"name": "averitec_search", "arguments": {"query": "bare"}}'
    out = engine._parse_response(_response(content))
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].arguments == {"query": "bare"}
    assert out.finish_reason == "tool_calls"


def test_incomplete_tool_call_tag_with_id_reconstructs_content(engine):
    # finish_reason="length": model was cut off mid tool_call, with tool_calls
    # already present in the structured `message.tool_calls` field.
    response = {
        "choices": [{
            "message": {
                "content": '<tool_call id="call_1">\n{"name": "averitec_search", "argum',
                "tool_calls": [
                    {"function": {"name": "averitec_search", "arguments": '{"query": "q1"}'}}
                ],
            },
            "finish_reason": "length",
        }],
        "usage": {},
    }
    out = engine._parse_response(response)
    assert len(out.tool_calls) == 1
    # Reconstructed content should contain a complete tool_call block (no
    # ValueError from a literal "<tool_call>" substring search failing to
    # find the id-attributed opening tag).
    assert "<tool_call>" in out.content
    assert "</tool_call>" in out.content


def test_plain_text_response_without_tool_call_is_unaffected(engine):
    out = engine._parse_response(_response("just a plain final answer"))
    assert out.tool_calls is None
    assert out.finish_reason == "stop"
