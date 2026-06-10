"""Message-history compaction for the agent loop.

When the loop exceeds its turn or token threshold, :class:`DefaultMessageCompactor`
squashes the middle of the conversation into a single user-role summary
:class:`Message` and keeps the recent window intact. The split point is
nudged past any leading tool message — orphan ``tool_call_id`` is HTTP 400
on Azure when the matching assistant message has been dropped.
"""

from __future__ import annotations

import re

from agent_harness.core.messages import Message, text_of

__all__ = [
    "estimate_tokens",
    "DefaultMessageCompactor",
    "DefaultCompactionPolicy",
]

_URL_RE = re.compile(r'https?://[^\s\)>"\'<]+')


def estimate_tokens(messages: list[Message]) -> int:
    """Estimate combined token count with a small per-message overhead."""
    from agent_harness.core.runtime.loop.context_budget import estimate_tokens as _est
    total = 0
    for msg in messages:
        total += _est(text_of(msg.get("content", ""))) + 4
    return total


def _compact_messages(
    messages: list[Message],
    keep_recent: int,
) -> list[Message]:
    """Squash the middle of a conversation into a short summary message."""
    system_msgs: list[Message] = []
    rest: list[Message] = []
    for msg in messages:
        if msg.get("role") == "system" and not rest:
            system_msgs.append(msg)
        else:
            rest.append(msg)

    if len(rest) <= keep_recent:
        return messages

    split_idx = len(rest) - keep_recent
    while split_idx < len(rest) - 1 and rest[split_idx].get("role") == "tool":
        split_idx += 1

    middle = rest[:split_idx]
    recent = rest[split_idx:]

    parts: list[str] = []
    for msg in middle:
        content = text_of(msg.get("content", "")).strip()
        if content.startswith("[Compacted"):
            continue
        role = msg.get("role")
        if role == "user":
            if content:
                parts.append(f"[User: {content[:400]}]")
        elif role == "assistant":
            tcs = msg.get("tool_calls") or []
            if tcs:
                tc_parts: list[str] = []
                for tc in tcs[:3]:
                    fn = tc.get("function") or {}
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "{}")
                    tc_parts.append(f"{name}({str(args)[:120]})")
                tc_line = "; ".join(tc_parts)
                if content:
                    parts.append(
                        f"[Agent: {content[:200]} | called: {tc_line}]"
                    )
                else:
                    parts.append(f"[Agent called: {tc_line}]")
            elif content:
                parts.append(f"[Agent: {content[:300]}]")
        elif role == "tool":
            urls = _URL_RE.findall(content)
            url_tag = f" url={urls[0]}" if urls else ""
            if content:
                parts.append(f"[Tool{url_tag}: {content[:300]}]")

    summary_body = "\n".join(parts[-20:]) if parts else ""
    summary_text = f"[Compacted {len(middle)} earlier messages]"
    if summary_body:
        summary_text = f"{summary_text}\n{summary_body}"

    summary_msg: Message = {"role": "user", "content": summary_text}
    return system_msgs + [summary_msg] + recent


class DefaultMessageCompactor:
    """Sync deterministic middle-squash compactor used by the agent loop."""

    def compact(
        self, messages: list[Message], keep_recent: int,
    ) -> list[Message]:
        return _compact_messages(messages, keep_recent)


class DefaultCompactionPolicy:
    """Compact when ``turn`` or ``estimated_tokens`` trips the threshold."""

    def __init__(
        self, compact_after_turns: int, context_token_limit: int,
    ) -> None:
        self._compact_after_turns = compact_after_turns
        self._context_token_limit = context_token_limit

    def should_compact(
        self,
        turn: int,
        messages: list[Message],  # noqa: ARG002
        estimated_tokens: int,
    ) -> bool:
        return (
            turn > self._compact_after_turns
            or estimated_tokens > self._context_token_limit
        )
