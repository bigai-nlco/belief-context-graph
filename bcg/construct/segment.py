"""Trajectory and turn segmentation for belief-graph construction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from bcg.construct.constants import (
    THINK_RE,
    TOOL_CALL_RE,
    TOOL_RESPONSE_RE,
)
from bcg.construct.utils import trim_span


@dataclass(slots=True)
class Segment:
    """A typed source slice where content is exactly original[start:end]."""

    traj_idx: int
    seg_idx: int
    role: str
    type: str
    content: str
    start: int
    end: int
    is_last_assistant: bool = False
    turn_idx: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def segment_turn(
    turn_idx: int,
    role: str,
    content: str,
    *,
    trajectory_index: int | None = None,
    is_last_assistant: bool = False,
) -> list[Segment]:
    """Segment one incoming turn. System/empty turns yield no segments."""

    content = content or ""
    traj_idx = turn_idx if trajectory_index is None else trajectory_index
    if role == "system" or not content.strip():
        return []
    if role == "assistant":
        return _segment_assistant(
            turn_idx=turn_idx,
            traj_idx=traj_idx,
            content=content,
            is_last_assistant=is_last_assistant,
        )
    if role in {"user", "tool", "function"}:
        return _segment_user_or_tool(
            turn_idx=turn_idx,
            traj_idx=traj_idx,
            role=role,
            content=content,
        )
    return _segment_user_or_tool(
        turn_idx=turn_idx,
        traj_idx=traj_idx,
        role="user",
        content=content,
    )


def segment_trajectory(trajectory: list[dict[str, Any]]) -> list[Segment]:
    """Split a raw trajectory into typed belief-extraction segments."""

    last_assistant_idx = _last_assistant_index(trajectory)
    segments: list[Segment] = []
    for index, message in enumerate(trajectory):
        role = str(message.get("role") or "")
        segments.extend(
            segment_turn(
                index,
                role,
                str(message.get("content") or ""),
                trajectory_index=index,
                is_last_assistant=index == last_assistant_idx,
            )
        )
    return segments


def summarize_segments(segments: Iterable[Segment]) -> dict[str, int]:
    """Count segments by type."""

    summary: dict[str, int] = {}
    for segment in segments:
        summary[segment.type] = summary.get(segment.type, 0) + 1
    return summary


def _last_assistant_index(trajectory: list[dict[str, Any]]) -> int | None:
    last_index: int | None = None
    for index, message in enumerate(trajectory):
        if message.get("role") == "assistant":
            last_index = index
    return last_index


def _segment_assistant(
    *,
    turn_idx: int,
    traj_idx: int,
    content: str,
    is_last_assistant: bool,
) -> list[Segment]:
    raw: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for outer_start, outer_end, inner_start, inner_end in _find_inner_blocks(
        content, THINK_RE
    ):
        raw.append((inner_start, inner_end, "think"))
        occupied.append((outer_start, outer_end))
    for outer_start, outer_end, inner_start, inner_end in _find_inner_blocks(
        content, TOOL_CALL_RE
    ):
        raw.append((inner_start, inner_end, "tool_call"))
        occupied.append((outer_start, outer_end))
    for start, end in _carve_outside(content, occupied):
        raw.append((start, end, "assistant_other"))
    return _build_segments(
        turn_idx,
        traj_idx,
        "assistant",
        content,
        raw,
        is_last_assistant=is_last_assistant,
    )


def _segment_user_or_tool(
    *,
    turn_idx: int,
    traj_idx: int,
    role: str,
    content: str,
) -> list[Segment]:
    raw: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for outer_start, outer_end, inner_start, inner_end in _find_inner_blocks(
        content, TOOL_RESPONSE_RE
    ):
        raw.append((inner_start, inner_end, "tool_response"))
        occupied.append((outer_start, outer_end))
    for start, end in _carve_outside(content, occupied):
        raw.append((start, end, "user_input" if role == "user" else "tool_response"))
    if not raw and role in {"tool", "function"} and content.strip():
        raw.append((0, len(content), "tool_response"))
    return _build_segments(turn_idx, traj_idx, role, content, raw)


def _build_segments(
    turn_idx: int,
    traj_idx: int,
    role: str,
    content: str,
    raw: list[tuple[int, int, str]],
    *,
    is_last_assistant: bool = False,
) -> list[Segment]:
    raw.sort(key=lambda span: span[0])
    segments: list[Segment] = []
    for start, end, segment_type in raw:
        trimmed_start, trimmed_end = trim_span(content, start, end)
        if trimmed_start >= trimmed_end:
            continue
        segments.append(
            Segment(
                traj_idx=traj_idx,
                turn_idx=turn_idx,
                seg_idx=len(segments),
                role=role,
                type=segment_type,
                content=content[trimmed_start:trimmed_end],
                start=trimmed_start,
                end=trimmed_end,
                is_last_assistant=is_last_assistant,
            )
        )
    return segments


def _find_inner_blocks(
    text: str,
    regex: Any,
) -> list[tuple[int, int, int, int]]:
    return [
        (match.start(), match.end(), match.start(1), match.end(1))
        for match in regex.finditer(text)
    ]


def _carve_outside(text: str, occupied: list[tuple[int, int]]) -> list[tuple[int, int]]:
    occupied = sorted(occupied)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for start, end in occupied:
        if start > cursor:
            ranges.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < len(text):
        ranges.append((cursor, len(text)))
    return [(start, end) for start, end in ranges if text[start:end].strip()]
