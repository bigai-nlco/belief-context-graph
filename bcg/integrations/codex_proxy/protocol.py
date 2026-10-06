"""Lossless Responses item partitioning and text-only graph serialization.

The original request items are never rebuilt. Only whole completed steps may
be omitted. Unknown/ambiguous item sequences disable compaction for that request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

CALL_TYPES = {"function_call", "custom_tool_call"}
RESULT_TYPES = {"function_call_output", "custom_tool_call_output"}
TEXT_TYPES = {"input_text", "output_text", "text", "refusal"}


class UnsupportedHistory(ValueError):
    """A history cannot safely be split into complete assistant/tool steps."""


def plain_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise UnsupportedHistory("non_text_content")
    text = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in TEXT_TYPES:
            raise UnsupportedHistory("multimodal_or_unknown_content")
        value = part.get("text", part.get("refusal"))
        if not isinstance(value, str):
            raise UnsupportedHistory("invalid_text_content")
        text.append(value)
    return "\n".join(text)


def reasoning_text(item: dict) -> str:
    """Prefer exposed reasoning_text, else summary; never decrypt ciphertext."""
    raw = item.get("content", [])
    text = ""
    if isinstance(raw, list):
        text = "\n".join(
            part["text"]
            for part in raw
            if isinstance(part, dict)
            and part.get("type") == "reasoning_text"
            and isinstance(part.get("text"), str)
        )
    if text:
        return text
    summary = item.get("summary", [])
    return (
        "\n".join(
            part["text"]
            for part in summary
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        if isinstance(summary, list)
        else ""
    )


def xml_json(tag: str, payload: dict) -> str:
    # JSON escapes preserve exact contents and prevent a tool's text from
    # prematurely closing the graph server's XML-style framing.
    value = (
        json.dumps(payload, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    return f"<{tag}>\n{value}\n</{tag}>"


@dataclass
class Step:
    indices: list[int] = field(default_factory=list)
    assistant: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    calls: dict[str, dict] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        if self.calls:
            return set(self.calls) == {result["call_id"] for result in self.results}
        return any(item.get("role") == "assistant" for item in self.assistant)

    def graph_turns(self) -> list[dict]:
        parts = []
        for item in self.assistant:
            kind = item.get("type", "message")
            if kind == "reasoning":
                text = reasoning_text(item)
                if text:
                    parts.append(f"<thinking>\n{text}\n</thinking>")
            elif kind in CALL_TYPES:
                arguments = (
                    json.loads(item["arguments"])
                    if kind == "function_call"
                    else {"input": item["input"]}
                )
                parts.append(
                    xml_json(
                        "tool_call",
                        {
                            "id": item["call_id"],
                            "name": item["name"],
                            "arguments": arguments,
                        },
                    )
                )
            else:
                parts.append(plain_text(item["content"]))
        turns = [{"role": "assistant", "content": "\n\n".join(parts)}]
        if self.results:
            blocks = [
                xml_json(
                    "tool_result",
                    {
                        "tool_call_id": item["call_id"],
                        "name": self.calls[item["call_id"]]["name"],
                        "content": plain_text(item["output"]),
                        "is_error": bool(item.get("is_error", False)),
                    },
                )
                for item in self.results
            ]
            # All parallel results constitute ONE graph turn.
            turns.append({"role": "tool", "content": "\n\n".join(blocks)})
        return turns


@dataclass
class Partition:
    items: list[dict]
    steps: list[Step]
    evicted: list[Step]
    retained: list[dict]
    graph_turns: list[dict]
    question: str
    query: str


def partition(items: list[dict], recent_turns: int) -> Partition:
    steps: list[Step] = []
    active = Step()
    users: list[tuple[int, dict]] = []
    seen_calls: set[str] = set()

    def finish() -> None:
        nonlocal active
        if active.indices:
            if not active.complete:
                raise UnsupportedHistory("unfinished_step_before_new_input")
            steps.append(active)
            active = Step()

    for index, item in enumerate(items):
        kind = item.get("type", "message")
        role = item.get("role")
        if not isinstance(kind, str) or not isinstance(role, (str, type(None))):
            raise UnsupportedHistory("invalid_item_type_or_role")
        if kind == "message" and role in {"system", "developer", "user"}:
            if role == "user":
                finish()
                plain_text(item.get("content"))
                users.append((index, item))
            # Instructions are always pinned, including non-text instructions.
            continue
        if kind in RESULT_TYPES:
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or call_id not in active.calls:
                raise UnsupportedHistory("orphan_tool_result")
            expected = (
                "function_call_output"
                if active.calls[call_id]["type"] == "function_call"
                else "custom_tool_call_output"
            )
            if kind != expected or any(r["call_id"] == call_id for r in active.results):
                raise UnsupportedHistory("duplicate_or_mismatched_result")
            plain_text(item.get("output"))
            active.indices.append(index)
            active.results.append(item)
            continue
        if kind not in CALL_TYPES | {"reasoning", "message"}:
            raise UnsupportedHistory("unsupported_item_type")
        if kind == "message" and role != "assistant":
            raise UnsupportedHistory("unsupported_message_role")
        if active.results:
            finish()
        if kind in CALL_TYPES:
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id or call_id in seen_calls:
                raise UnsupportedHistory("missing_or_duplicate_call_id")
            if not isinstance(item.get("name"), str) or not item["name"]:
                raise UnsupportedHistory("missing_tool_name")
            if kind == "function_call":
                try:
                    args = json.loads(item["arguments"])
                except (ValueError, TypeError, KeyError) as exc:
                    raise UnsupportedHistory("invalid_tool_arguments") from exc
                if not isinstance(args, dict):
                    raise UnsupportedHistory("non_object_tool_arguments")
            elif not isinstance(item.get("input"), str):
                raise UnsupportedHistory("invalid_custom_tool_input")
            active.calls[call_id] = item
            seen_calls.add(call_id)
        elif kind == "message":
            plain_text(item.get("content"))
        active.assistant.append(item)
        active.indices.append(index)
        if (
            kind == "message"
            and item.get("phase") == "final_answer"
            and not active.calls
        ):
            finish()

    # A partly completed parallel batch MUST remain whole, even at recent=0.
    if active.complete:
        steps.append(active)
    evicted = steps[:-recent_turns] if recent_turns else steps[:]
    removed = {index for step in evicted for index in step.indices}
    retained = [item for index, item in enumerate(items) if index not in removed]
    first_user_index = users[0][0] if users else -1
    frontier = max(
        (index for step in evicted for index in step.indices), default=first_user_index
    )
    events = [
        (index, [{"role": "user", "content": plain_text(item["content"])}])
        for index, item in users
        if index <= frontier
    ]
    events.extend((step.indices[0], step.graph_turns()) for step in evicted)
    turns = [turn for _, group in sorted(events) for turn in group]
    question = "\n".join(plain_text(item["content"]) for _, item in users)
    # Bounded textual retrieval query; ciphertext never enters Graph requests.
    recent_text = []
    for step in steps[len(evicted) :]:
        recent_text.extend(turn["content"] for turn in step.graph_turns())
    if active.indices and not active.complete:
        recent_text.extend(
            reasoning_text(item)
            for item in active.assistant
            if item.get("type") == "reasoning"
        )
    query = (question[-6000:] + "\n" + "\n".join(recent_text)[-6000:]).strip()
    return Partition(items, steps, evicted, retained, turns, question, query)
