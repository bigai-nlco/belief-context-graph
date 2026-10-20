"""Project whole ReAct steps without rewriting retained provider messages.

The durable transcript and the model's context are intentionally different:
only the latter loses old steps. Signed/encrypted reasoning remains untouched
in retained messages; only exposed reasoning text is sent to graph construction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from frontier_agent.core.messages import Message


class UnsupportedHistory(ValueError):
    """The input cannot safely be represented as complete text-only steps."""


def text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise UnsupportedHistory("unsupported_content")
    parts = []
    for block in value:
        if not isinstance(block, dict) or block.get("type") not in {
            "text",
            "input_text",
            "output_text",
            "refusal",
        }:
            raise UnsupportedHistory("multimodal_or_unknown_content")
        text = block.get("text", block.get("refusal"))
        if not isinstance(text, str):
            raise UnsupportedHistory("invalid_text_block")
        parts.append(text)
    return "\n".join(parts)


def _reasoning(block: dict[str, Any]) -> str:
    content = block.get("content") or []
    exposed = (
        [
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "reasoning_text"
            and isinstance(part.get("text"), str)
        ]
        if isinstance(content, list)
        else []
    )
    if exposed:
        return "\n".join(exposed)
    summary = block.get("summary") or []
    return (
        "\n".join(
            part["text"]
            for part in summary
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        if isinstance(summary, list)
        else ""
    )


def assistant_text(message: Message) -> str:
    """Readable content only. Never send signatures/ciphertext to the graph."""
    content = message.get("content")
    thinking = message.get("reasoning_content") or ""
    parts: list[str] = []
    if thinking:
        parts.append(f"<thinking>\n{thinking}\n</thinking>")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                raise UnsupportedHistory("invalid_assistant_block")
            kind = block.get("type")
            if kind in {"thinking", "reasoning"}:
                exposed = block.get("thinking", "") if kind == "thinking" else _reasoning(block)
                if exposed and exposed != thinking:
                    parts.append(f"<thinking>\n{exposed}\n</thinking>")
            elif kind in {"redacted_thinking", "function_call", "tool_use"}:
                continue
            else:
                parts.append(text_content([block]))
    else:
        text = text_content(content)
        # Qwen-style replay uses <think>; BCG expects <thinking>.
        text = re.sub(r"<think>(.*?)</think>", r"<thinking>\1</thinking>", text, flags=re.S)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def xml_json(tag: str, value: dict[str, Any]) -> str:
    text = json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    return f"<{tag}>\n{text}\n</{tag}>"


def tool_calls(message: Message) -> dict[str, dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    raw_calls = message.get("tool_calls") or []
    content = message.get("content")
    # Native block lists sometimes carry calls without the duplicate top-level
    # tool_calls field. Read them without changing the original message.
    if not raw_calls and isinstance(content, list):
        raw_calls = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") in {"tool_use", "function_call"}
        ]
    for call in raw_calls:
        if not isinstance(call, dict):
            raise UnsupportedHistory("invalid_tool_call")
        call_id = call.get("call_id") or call.get("id")
        function = call.get("function") or call
        if not isinstance(function, dict):
            raise UnsupportedHistory("invalid_tool_function")
        name = function.get("name")
        arguments = function.get("arguments", function.get("args", function.get("input", {})))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError as exc:
                raise UnsupportedHistory("invalid_tool_arguments") from exc
        if not isinstance(call_id, str) or not call_id or call_id in calls:
            raise UnsupportedHistory("missing_or_duplicate_call_id")
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise UnsupportedHistory("invalid_tool_arguments")
        calls[call_id] = {"id": call_id, "name": name, "arguments": arguments}
    return calls


@dataclass
class Step:
    start: int
    assistant: Message
    calls: dict[str, dict[str, Any]]
    indices: list[int] = field(default_factory=list)
    results: list[Message] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return set(self.calls) == {result.get("tool_call_id") for result in self.results}

    def graph_turns(self) -> list[dict[str, str]]:
        parts = [assistant_text(self.assistant)]
        parts.extend(xml_json("tool_call", call) for call in self.calls.values())
        turns = [{"role": "assistant", "content": "\n\n".join(p for p in parts if p)}]
        if self.results:
            results = [
                xml_json(
                    "tool_result",
                    {
                        "tool_call_id": message.get("tool_call_id", ""),
                        "name": self.calls[message.get("tool_call_id", "")]["name"],
                        "content": text_content(message.get("content")),
                        "is_error": bool(message.get("is_error", False)),
                    },
                )
                for message in self.results
            ]
            # A parallel batch is ONE tool graph turn, paired by call id.
            turns.append({"role": "tool", "content": "\n\n".join(results)})
        return turns


@dataclass
class Projection:
    retained: list[Message]
    retained_indices: list[int]
    turns: list[dict[str, str]]
    question: str
    query: str
    evicted_steps: int


def partition(messages: list[Message], recent_turns: int) -> Projection:
    """Pin human inputs; only trusted host-tagged reports may age out with steps."""
    steps: list[Step] = []
    active: Step | None = None
    users: list[tuple[int, str]] = []
    reports: list[tuple[int, str]] = []
    seen_calls: set[str] = set()
    for index, message in enumerate(messages):
        role = message.get("role")
        if role in {"system", "developer"}:
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if active is None or call_id not in active.calls:
                raise UnsupportedHistory("orphan_tool_result")
            if any(item.get("tool_call_id") == call_id for item in active.results):
                raise UnsupportedHistory("duplicate_tool_result")
            text_content(message.get("content"))
            active.results.append(message)
            active.indices.append(index)
            continue
        if role not in {"user", "assistant"}:
            raise UnsupportedHistory("unsupported_role")
        if active is not None:
            if not active.complete:
                raise UnsupportedHistory("unfinished_step_before_new_input")
            steps.append(active)
            active = None
        if role == "user":
            # This is host metadata, never inferred from user-supplied text.
            target = reports if message.get("bcg_origin") == "agent_report" else users
            target.append((index, text_content(message.get("content"))))
        else:
            assistant_text(message)  # validate without touching provider blocks
            calls = tool_calls(message)
            if seen_calls.intersection(calls):
                raise UnsupportedHistory("reused_tool_call_id")
            seen_calls.update(calls)
            active = Step(index, message, calls, [index])
    if active is not None and active.complete:
        steps.append(active)
        active = None
    evicted = [] if recent_turns < 0 else steps[:-recent_turns] if recent_turns else steps[:]
    removed = {index for step in evicted for index in step.indices}
    first_step = min(
        (step.start for step in steps + ([active] if active else [])), default=len(messages)
    )
    frontier = max(
        [index for index, _ in users if index < first_step]
        + [index for step in evicted for index in step.indices],
        default=-1,
    )
    # A report after the frontier has not aged past a completed model step.
    # In particular, a newly delivered report cannot disappear before a model
    # has had an opportunity to read it, including with recent_turns=0.
    removed.update(index for index, _ in reports if index <= frontier)
    retained_indices = [index for index in range(len(messages)) if index not in removed]
    events = [
        (index, [{"role": "user", "content": text}]) for index, text in users if index <= frontier
    ]
    events.extend((step.start, step.graph_turns()) for step in evicted)
    events.extend(
        (index, [{"role": "tool", "content": text}])
        for index, text in reports if index <= frontier
    )
    turns = [turn for _, group in sorted(events) for turn in group]
    question = "\n\n".join(text for _, text in users)
    query_parts = [users[-1][1]] if users else []
    query_parts.extend(text for index, text in reports if index > frontier)
    for step in steps[len(evicted) :]:
        query_parts.extend(turn["content"] for turn in step.graph_turns())
    if active:
        query_parts.extend(turn["content"] for turn in active.graph_turns())
    return Projection(
        [messages[index] for index in retained_indices],
        retained_indices,
        turns,
        question,
        "\n\n".join(query_parts)[-8_000:] or question[-8_000:],
        len(evicted),
    )
