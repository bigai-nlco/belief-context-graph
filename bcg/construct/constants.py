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

VALID_STANCES = {"asserted", "recalled", "speculated", "judged"}
SEGMENT_SOURCE_TYPES = {
    "user_input": "user_input",
    "tool_call": "tool_call",
    "assistant_other": "assistant_other",
    "think": "llm_reasoning",
    "tool_response": "tool_result",
}

VALID_FORWARD_TYPES = {"informs"}
VALID_BACKWARD_TYPES = {"confirms", "contradicts", "extends"}

FUZZY_MIN_RATIO = 0.6
FUZZY_MAX_SPAN_FACTOR = 2.0
FUZZY_MAX_SPAN_SLACK = 80

CONFIDENCE_METHOD = "dimension_average"
DEFAULT_SOURCE_RELIABILITY = {
    "user_input": 0.88,
    "tool_call": 0.72,
    "assistant_other": 0.76,
    "llm_reasoning": 0.58,
    "tool_result": 0.90,
    "historical_retrieval": 0.70,
    "manual": 0.86,
}
DEFAULT_STANCE_CERTAINTY = {
    "asserted": 0.88,
    "recalled": 0.72,
    "judged": 0.66,
    "speculated": 0.38,
}
DEFAULT_EVIDENCE_DIRECTNESS = {
    "exact": 0.95,
    "normalized": 0.82,
    "fuzzy": 0.65,
    "manual": 0.80,
    "not_found": 0.25,
}
DEFAULT_RELATION_DIRECTNESS = {
    "confirms": 0.86,
    "contradicts": 0.88,
    "extends": 0.70,
}
