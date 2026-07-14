"""Multi-tool-call protocol layer: parse, canonicalize, render.

Handles the XML-like ``<tool_call>...</tool_call>`` blocks an assistant turn
may contain (one or many), assigns each a local call_id (``call_1, call_2,
...``) in parse order, rewrites the assistant text so every block carries an
explicit ``id="call_N"`` attribute, and renders per-call ``<tool_result>``
blocks so multiple queries' evidence never share one unbounded, ambiguous
text blob.

Pure string/JSON handling — no dependency on rllm or any specific tool.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Matches both the legacy bare tag and the id-carrying tag. The id group is
# optional so old-format (no id) and new-format (with id) inputs share one
# pattern -- this is also what api_engine.py's content parser must use, since
# a plain literal-string / no-attribute regex silently fails to match once an
# id="..." attribute is present.
TOOL_CALL_RE = re.compile(
    r'<tool_call(?:\s+id="([^"]*)")?\s*>(.*?)</tool_call>',
    re.DOTALL,
)

# Matches just the opening tag (bare or id-carrying), for callers that need to
# detect "is there a tool_call here at all" -- including incomplete blocks
# missing a closing tag -- without requiring a full block match.
TOOL_CALL_OPEN_RE = re.compile(r'<tool_call(?:\s+id="[^"]*")?\s*>')

_CALL_ID_RE = re.compile(r"^call_(\d+)$")


@dataclass
class ParsedToolCall:
    """One ``<tool_call>`` block, after local re-numbering.

    ``id`` is always assigned by parse order (``call_1, call_2, ...``),
    regardless of what the model wrote. ``raw_id`` preserves whatever id
    attribute the model supplied (or ``None``), for logging/diagnostics only.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    raw_block: str
    raw_id: str | None = None
    format_error: str | None = None


@dataclass
class _RawMatch:
    match_id: str | None
    body: str
    full_block: str
    span: tuple[int, int]


def _extract_raw_matches(text: str) -> list[_RawMatch]:
    return [
        _RawMatch(
            match_id=m.group(1),
            body=m.group(2),
            full_block=m.group(0),
            span=m.span(),
        )
        for m in TOOL_CALL_RE.finditer(text or "")
    ]


_BOXED_BACKSLASH_RE = re.compile(r"\\(?=boxed\s*\{)")


def _parse_call_body(body: str) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Return ``(name, arguments, format_error)`` for one block's JSON body."""
    raw = body.strip()
    # The model sometimes writes \boxed{} inside a JSON argument string (e.g.
    # the finish tool's answer). \b is a valid JSON escape (backspace), so a
    # literal \boxed breaks json.loads; escape the backslash first.
    raw = _BOXED_BACKSLASH_RE.sub(r"\\\\", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, None, f"invalid JSON in <tool_call> block: {exc}"

    if not isinstance(data, dict):
        return None, None, "tool_call JSON must be an object with 'name'/'arguments'"

    name = data.get("name")
    if not isinstance(name, str) or not name:
        return None, None, "tool_call JSON missing a non-empty string 'name'"

    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        return None, None, "tool_call JSON 'arguments' must be an object"

    return name, arguments, None


def parse_tool_call_blocks(text: str) -> list[ParsedToolCall]:
    """Extract every ``<tool_call>`` block from ``text``, in order.

    Local call_ids (``call_1, call_2, ...``) are assigned by the order blocks
    are found in the text, starting fresh at 1 -- not by whatever id (if any)
    the model wrote, and not continuing any previous turn's numbering. Blocks
    whose JSON body fails to parse still appear in the returned list (with
    ``format_error`` set and ``name``/``arguments`` empty) but do NOT consume a
    call_id slot, so later valid blocks keep contiguous numbering.
    """

    calls: list[ParsedToolCall] = []
    next_index = 1
    for raw in _extract_raw_matches(text):
        name, arguments, err = _parse_call_body(raw.body)
        if err is not None:
            calls.append(
                ParsedToolCall(
                    id="",
                    name="",
                    arguments={},
                    raw_block=raw.full_block,
                    raw_id=raw.match_id,
                    format_error=err,
                )
            )
            continue
        calls.append(
            ParsedToolCall(
                id=f"call_{next_index}",
                name=name,  # type: ignore[arg-type]
                arguments=arguments or {},  # type: ignore[arg-type]
                raw_block=raw.full_block,
                raw_id=raw.match_id,
            )
        )
        next_index += 1
    return calls


def validate_raw_ids(calls: list[ParsedToolCall]) -> list[str]:
    """Return warnings for model-supplied ids that don't follow ``call_N`` or
    collide with another block in the same message. Purely diagnostic -- the
    final ``id`` assignment (parse order) is not affected either way.
    """

    warnings: list[str] = []
    seen: dict[str, int] = {}
    for call in calls:
        if call.raw_id is None:
            continue
        if not _CALL_ID_RE.match(call.raw_id):
            warnings.append(
                f"model-supplied id {call.raw_id!r} does not match ^call_\\d+$"
            )
            continue
        seen[call.raw_id] = seen.get(call.raw_id, 0) + 1
    for raw_id, count in seen.items():
        if count > 1:
            warnings.append(f"model-supplied id {raw_id!r} repeated {count} times")
    return warnings


def canonicalize_tool_call_text(text: str, calls: list[ParsedToolCall]) -> str:
    """Rewrite each successfully-parsed block's opening tag to carry its
    assigned ``id="call_N"``, leaving everything else -- JSON body, block
    count, surrounding text (e.g. ``<think>`` sections) -- byte-for-byte
    unchanged. Blocks that failed to parse (``format_error`` set, no ``id``)
    are left exactly as written.
    """

    if not text:
        return text or ""

    raw_matches = _extract_raw_matches(text)
    if not raw_matches:
        return text

    # calls and raw_matches are both in parse order and the same length
    # (parse_tool_call_blocks emits one ParsedToolCall per regex match).
    out: list[str] = []
    cursor = 0
    for raw, call in zip(raw_matches, calls):
        start, end = raw.span
        out.append(text[cursor:start])
        if call.format_error is not None:
            out.append(raw.full_block)
        else:
            out.append(f'<tool_call id="{call.id}">{raw.body}</tool_call>')
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def render_tool_result_block(
    call_id: str,
    name: str,
    query: str,
    evidence_texts: list[str],
) -> str:
    """Render one ``<tool_result id="call_N" name="...">`` block.

    Evidence entries are numbered with the call_id prefix (``call_N_evidence_M``)
    so they never collide with another call's numbering. No URLs are emitted.
    """

    lines = [f'<tool_result id="{call_id}" name="{name}">', f"Query: {query}", ""]
    if evidence_texts:
        lines.append("Evidence:")
        for i, text in enumerate(evidence_texts, 1):
            lines.append(f"[{call_id}_evidence_{i}]")
            lines.append("text:")
            lines.append(text)
            lines.append("")
        if lines[-1] == "":
            lines.pop()
    else:
        lines.append("Evidence: (none found)")
    lines.append("</tool_result>")
    return "\n".join(lines)


def render_tool_results_xml(entries: list[dict[str, Any]]) -> str:
    """Render multiple ``<tool_result>`` blocks, one per entry, joined by a
    blank line. Each entry: ``{"tool_call_id", "name", "query", "evidence"}``
    where ``evidence`` is a list of ``{"text": ...}`` dicts (or plain strings).
    """

    blocks: list[str] = []
    for entry in entries:
        evidence = entry.get("evidence") or []
        texts = [
            str(e.get("text", "")) if isinstance(e, dict) else str(e)
            for e in evidence
        ]
        blocks.append(
            render_tool_result_block(
                call_id=str(entry.get("tool_call_id", "")),
                name=str(entry.get("name", "")),
                query=str(entry.get("query", "")),
                evidence_texts=texts,
            )
        )
    return "\n\n".join(blocks)


__all__ = [
    "TOOL_CALL_RE",
    "TOOL_CALL_OPEN_RE",
    "ParsedToolCall",
    "parse_tool_call_blocks",
    "validate_raw_ids",
    "canonicalize_tool_call_text",
    "render_tool_result_block",
    "render_tool_results_xml",
]
