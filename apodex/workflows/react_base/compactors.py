"""KeepLastNToolResultsCompactor — replace older tool results with placeholders.

Keep the last ``N`` tool messages verbatim; replace the content of all
older ``tool``-role messages with a short placeholder. The system message
and the first user turn (the original question) are always left intact.

Plug into the agent loop via ``LoopConfig.compactor`` paired with
:class:`AlwaysCompactPolicy` so the filter applies on every turn.
"""

from __future__ import annotations

from agent_harness.core.messages import Message, text_of

__all__ = [
    "KeepLastNToolResultsCompactor",
    "AlwaysCompactPolicy",
    "OMITTED_PLACEHOLDER",
]

OMITTED_PLACEHOLDER = "Tool result is omitted to save tokens."


class KeepLastNToolResultsCompactor:
    """Replace older ``tool``-role contents with a short placeholder.

    Idempotent: messages already carrying the placeholder are detected by
    content match and left alone, so running this every turn is safe.
    ``keep_tool_result == -1`` disables filtering entirely.
    """

    def __init__(self, keep_tool_result: int) -> None:
        if keep_tool_result < -1:
            raise ValueError(
                f"keep_tool_result must be >= -1 (got {keep_tool_result})"
            )
        self._keep = keep_tool_result

    def compact(
        self,
        messages: list[Message],
        keep_recent: int,  # noqa: ARG002 — Protocol param; ignored
    ) -> list[Message]:
        if self._keep == -1:
            return messages

        tool_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "tool"
        ]
        if not tool_indices:
            return messages

        keep_count = min(self._keep, len(tool_indices))
        keep_set = (
            set(tool_indices[-keep_count:]) if keep_count > 0 else set()
        )
        if len(keep_set) == len(tool_indices):
            return messages

        out: list[Message] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") != "tool" or idx in keep_set:
                out.append(msg)
                continue
            if text_of(msg.get("content", "")) == OMITTED_PLACEHOLDER:
                out.append(msg)
                continue
            out.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": OMITTED_PLACEHOLDER,
            })
        return out


class AlwaysCompactPolicy:
    """Returns ``True`` every turn. Pair with :class:`KeepLastNToolResultsCompactor`
    — the compactor is idempotent and cheap when nothing needs dropping."""

    def should_compact(
        self,
        turn: int,  # noqa: ARG002
        messages: list[Message],  # noqa: ARG002
        estimated_tokens: int,  # noqa: ARG002
    ) -> bool:
        return True
