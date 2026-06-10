"""OpenAI-compatible message types — provider-agnostic chat representation.

Messages are plain ``dict`` matching the OpenAI Chat Completions wire format,
so they round-trip through any compatible endpoint with no translation:

    {"role": "system" | "user" | "assistant" | "tool",
     "content": str | None,
     "name"?: str,
     "tool_calls"?: [ToolCall],
     "tool_call_id"?: str,
     "reasoning_content"?: str}

Anthropic content-block shape (``[{"type": "thinking", ...}, {"type": "text", ...}]``)
is preserved as the ``content`` value when the underlying client returns it;
callers that need flat text use :func:`text_of`.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(TypedDict):
    """OpenAI-style tool_call payload — ``function.arguments`` is JSON-encoded."""

    id: str
    type: Literal["function"]
    function: dict[str, str]  # {"name": str, "arguments": str}


class Message(TypedDict, total=False):
    """One chat message in OpenAI Chat Completions format."""

    role: Role
    content: Any            # str | list[dict] (Anthropic blocks) | None
    name: str
    tool_calls: list[ToolCall]
    tool_call_id: str
    reasoning_content: str


# ── Builders ─────────────────────────────────────────────────────────────


# Key insertion order: ``content`` first, then ``role``. Some served
# checkpoints are sensitive to this byte shape.


def system_msg(content: str) -> Message:
    return {"content": content, "role": "system"}


def user_msg(content: str) -> Message:
    return {"content": content, "role": "user"}


def assistant_msg(
    content: Any = "",
    *,
    tool_calls: list[ToolCall] | None = None,
    reasoning: str = "",
) -> Message:
    m: Message = {"content": content, "role": "assistant"}
    if tool_calls:
        m["tool_calls"] = tool_calls
    if reasoning:
        m["reasoning_content"] = reasoning
    return m


def tool_msg(content: str, tool_call_id: str) -> Message:
    return {"content": content, "role": "tool", "tool_call_id": tool_call_id}


# ── Helpers ──────────────────────────────────────────────────────────────


def text_of(content: Any) -> str:
    """Flatten an OpenAI/Anthropic message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                val = block.get("text") or block.get("content") or ""
                if isinstance(val, str):
                    parts.append(val)
        return "\n".join(parts)
    return str(content)


def is_tool_msg(m: Message) -> bool:
    return m.get("role") == "tool"


def is_assistant_msg(m: Message) -> bool:
    return m.get("role") == "assistant"


__all__ = [
    "Message",
    "Role",
    "ToolCall",
    "assistant_msg",
    "is_assistant_msg",
    "is_tool_msg",
    "system_msg",
    "text_of",
    "tool_msg",
    "user_msg",
]
