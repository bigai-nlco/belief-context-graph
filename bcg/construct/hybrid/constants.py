"""Shared non-prompt constants for belief-graph construction."""

from __future__ import annotations

import re

IO_TYPES = {"user_input", "tool_call", "assistant_other"}
REASONING_TYPES = {"think", "tool_response"}

DEFAULT_MIN_SEGMENT_LEN: dict[str, int] = {
    "tool_response": 60,
    "assistant_other": 20,
    "user_input": 0,
    "tool_call": 0,
    "think": 0,
}

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)

SENTENCE_END_RE = re.compile(r"[.!?。！？；;…]+['\"”’\)\]）】」』]*")
MIN_SENTENCE_FRAGMENT_LEN = 4

VALID_STANCES = {"asserted", "recalled", "judged", "speculated"}
SEGMENT_SOURCE_TYPES = {
    "user_input": "user_input",
    "tool_call": "tool_call",
    "assistant_other": "assistant_other",
    "think": "llm_reasoning",
    "tool_response": "tool_result",
}


FUZZY_MIN_RATIO = 0.6
FUZZY_MAX_SPAN_FACTOR = 2.0
FUZZY_MAX_SPAN_SLACK = 80
