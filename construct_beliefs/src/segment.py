"""
segment.py
==========
Slice a raw trajectory into typed SEGMENTS. The rest of the pipeline operates
on segments, not on raw messages, because what matters for belief extraction
is the *kind of content* (think vs tool_call vs tool_response vs ...), not the
chat role.

Segment types
-------------
I/O layer:
    user_input        — user message text outside any <tool_response> block
    tool_call         — text inside <tool_call>...</tool_call> blocks
    assistant_other   — assistant text that is NOT inside <think> or <tool_call>
                        (this is where the model's final / boxed answer lives
                        when it is given directly to the user instead of via a
                        tool call)

Reasoning layer:
    think             — text inside <think>...</think> blocks
    tool_response     — text inside <tool_response>...</tool_response> blocks
                        (the trajectory in this project wraps tool returns in
                        user messages, so user messages carrying this tag are
                        peeled here)

Skipped:
    system messages
    empty / whitespace-only segments
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


# --- Tag layout for assistant / tool-wrapper messages ----------------------
_THINK_RE         = re.compile(r"<think>(.*?)</think>",         re.DOTALL)
_TOOL_CALL_RE     = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)


# I/O-layer types are extracted comprehensively. Reasoning-layer types are
# extracted selectively (only key reasoning nodes).
IO_TYPES        = {"user_input", "tool_call", "assistant_other"}
REASONING_TYPES = {"think", "tool_response"}


@dataclass
class Segment:
    traj_idx: int          # index of the source message in trajectory[]
    seg_idx: int           # ordinal within the source message
    role: str              # the source message's role
    type: str              # see segment-type list above
    content: str           # the segment text
    start: int             # offset within the source message
    end: int               # exclusive end offset within the source message
    is_last_assistant: bool = False   # set when the segment belongs to the
                                      # final assistant turn — useful for
                                      # treating its assistant_other as the
                                      # "final answer"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _find_blocks(text: str, regex: re.Pattern) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, inner_content) for every regex match."""
    return [(m.start(), m.end(), m.group(1)) for m in regex.finditer(text)]


def _carve_outside(text: str, occupied: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Given a list of (start, end) ranges that are 'occupied' by recognized
    blocks, return the (start, end) ranges that fall OUTSIDE all of them.
    """
    occupied = sorted(occupied)
    out: List[Tuple[int, int]] = []
    cursor = 0
    for (s, e) in occupied:
        if s > cursor:
            out.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < len(text):
        out.append((cursor, len(text)))
    return [(s, e) for (s, e) in out if text[s:e].strip()]


def _segment_assistant(traj_idx: int, content: str, is_last_assistant: bool) -> List[Segment]:
    """Split an assistant message into think / tool_call / assistant_other."""
    think_spans     = _find_blocks(content, _THINK_RE)
    tool_call_spans = _find_blocks(content, _TOOL_CALL_RE)

    raw: List[Tuple[int, int, str, str]] = []  # (start, end, type, inner)
    for (s, e, inner) in think_spans:
        raw.append((s, e, "think", inner))
    for (s, e, inner) in tool_call_spans:
        raw.append((s, e, "tool_call", inner))

    # Anything outside known blocks is assistant_other.
    occupied = [(s, e) for (s, e, _, _) in raw]
    for (s, e) in _carve_outside(content, occupied):
        raw.append((s, e, "assistant_other", content[s:e]))

    # Order segments by appearance in the original message.
    raw.sort(key=lambda x: x[0])

    segs: List[Segment] = []
    for i, (s, e, t, inner) in enumerate(raw):
        body = inner.strip()
        if not body:
            continue
        segs.append(Segment(
            traj_idx=traj_idx, seg_idx=i, role="assistant",
            type=t, content=body, start=s, end=e,
            is_last_assistant=is_last_assistant,
        ))
    return segs


def _segment_user_or_tool(traj_idx: int, role: str, content: str) -> List[Segment]:
    """User and tool messages: peel out <tool_response> wrappers."""
    tool_response_spans = _find_blocks(content, _TOOL_RESPONSE_RE)
    raw: List[Tuple[int, int, str, str]] = []
    for (s, e, inner) in tool_response_spans:
        raw.append((s, e, "tool_response", inner))

    occupied = [(s, e) for (s, e, _, _) in raw]
    for (s, e) in _carve_outside(content, occupied):
        # Outside text: user_input for role=user; otherwise (role=tool/function)
        # treat the whole content as a tool_response.
        if role == "user":
            raw.append((s, e, "user_input", content[s:e]))
        else:
            raw.append((s, e, "tool_response", content[s:e]))

    raw.sort(key=lambda x: x[0])

    # If the message has role tool/function but no <tool_response> tag, treat
    # the entire content as a single tool_response segment.
    if not raw and role in ("tool", "function") and content.strip():
        raw.append((0, len(content), "tool_response", content))

    segs: List[Segment] = []
    for i, (s, e, t, inner) in enumerate(raw):
        body = inner.strip()
        if not body:
            continue
        segs.append(Segment(
            traj_idx=traj_idx, seg_idx=i, role=role,
            type=t, content=body, start=s, end=e,
        ))
    return segs


def segment_trajectory(trajectory: List[Dict[str, Any]]) -> List[Segment]:
    """Top-level segmenter for a whole trajectory."""
    # Identify the index of the last assistant message — its assistant_other
    # segment (if any) is the "final answer".
    last_assistant_idx: Optional[int] = None
    for i, m in enumerate(trajectory):
        if m.get("role") == "assistant":
            last_assistant_idx = i

    all_segments: List[Segment] = []
    for i, msg in enumerate(trajectory):
        role = msg.get("role")
        content = msg.get("content", "") or ""
        if role == "system":
            continue
        if role == "assistant":
            all_segments.extend(_segment_assistant(i, content, i == last_assistant_idx))
        elif role in ("user", "tool", "function"):
            all_segments.extend(_segment_user_or_tool(i, role, content))
    return all_segments


def summarize_segments(segs: Iterable[Segment]) -> Dict[str, int]:
    """Tiny helper: count segments by type for logging."""
    out: Dict[str, int] = {}
    for s in segs:
        out[s.type] = out.get(s.type, 0) + 1
    return out
