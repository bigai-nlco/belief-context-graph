"""Belief extraction from typed trajectory segments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from bcg.belief_graph.constants import VALID_STANCES
from bcg.belief_graph.prompts import build_extraction_prompt
from bcg.belief_graph.segment import Segment
from bcg.llm import parse_json_response


class TextGenerator(Protocol):
    async def generate_text(
        self,
        prompt: str,
        *,
        label: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...


@dataclass(slots=True)
class ExtractionResult:
    beliefs: list[dict[str, Any]]
    prompt: str | None = None
    raw_output: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


def clean_belief(
    raw: Any,
    *,
    mode: str = "excerpt",
    n_sentences: int = 0,
) -> dict[str, Any] | None:
    """Validate and normalize one raw LLM belief object."""

    if not isinstance(raw, dict):
        return None
    text = raw.get("belief")
    if not isinstance(text, str) or not text.strip():
        return None
    stance = str(raw.get("stance") or "").strip().lower()
    if stance not in VALID_STANCES:
        stance = "asserted"

    output: dict[str, Any] = {"belief": text.strip(), "stance": stance}
    event_time = _clean_optional_str(raw.get("event_time"))
    time_text = _clean_optional_str(raw.get("time_text"))
    if event_time is not None:
        output["event_time"] = event_time
    if time_text is not None:
        output["time_text"] = time_text

    if mode == "sentences":
        indices = _clean_sentence_indices(
            raw.get("supporting_sentence_indices"),
            n_sentences=n_sentences,
        )
        output["supporting_sentence_indices"] = indices
        return output

    excerpts = [
        excerpt.strip()
        for excerpt in raw.get("supporting_excerpts", []) or []
        if isinstance(excerpt, str) and excerpt.strip()
    ]
    if not excerpts:
        return None
    output["supporting_excerpts"] = excerpts
    return output


def format_graph_context(
    beliefs: list[dict[str, Any]],
    *,
    char_budget: int = 9000,
) -> str:
    """Compact current graph context for extraction/link prompts."""

    if not beliefs:
        return "[]"
    ordered = sorted(beliefs, key=lambda belief: int(belief.get("id", 0)))
    lines = [_context_line(belief) for belief in ordered]
    total = sum(len(line) + 4 for line in lines)
    omitted = 0
    while lines and total > char_budget:
        total -= len(lines[0]) + 4
        lines.pop(0)
        omitted += 1
    items = [f"  (... {omitted} earlier belief(s) omitted ...)"] if omitted else []
    items.extend("  " + line for line in lines)
    return "[\n" + ",\n".join(items) + "\n]"


def format_io_context(
    beliefs: list[dict[str, Any]],
    *,
    char_budget: int = 6000,
) -> str:
    """Compact I/O-layer-only context used by reasoning prompts."""

    return format_graph_context(
        [belief for belief in beliefs if belief.get("layer") == "io"],
        char_budget=char_budget,
    )


async def extract_segment(
    generator: TextGenerator,
    segment: Segment,
    *,
    scenario: str = "research",
    mode: str = "excerpt",
    sentences: list[str] | None = None,
    graph_context: str = "[]",
    io_context: str | list[dict[str, Any]] | None = None,
    session_date: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    label: str | None = None,
) -> ExtractionResult:
    """Run the configured extraction prompt for one segment."""

    io_context_str = (
        format_io_context(io_context)
        if isinstance(io_context, list)
        else (io_context or "[]")
    )
    prompt = build_extraction_prompt(
        scenario,
        segment.type,
        mode=mode,
        content=segment.content,
        sentences=sentences,
        graph_context=graph_context,
        io_context=io_context_str,
        session_date=session_date,
    )
    if prompt is None:
        return ExtractionResult(
            beliefs=[],
            skipped=True,
            skip_reason=f"unknown segment type {segment.type!r}",
        )

    try:
        raw = await generator.generate_text(
            prompt,
            label=label,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        return ExtractionResult(
            beliefs=[],
            prompt=prompt,
            raw_output=f"[ERROR] {exc}",
            skipped=True,
            skip_reason=str(exc),
        )

    parsed = parse_json_response(raw)
    beliefs = [
        cleaned
        for raw_belief in (
            parsed.get("beliefs", []) if isinstance(parsed, dict) else []
        )
        if (
            cleaned := clean_belief(
                raw_belief,
                mode=mode,
                n_sentences=len(sentences or []),
            )
        )
        is not None
    ]
    return ExtractionResult(
        beliefs=beliefs,
        prompt=prompt,
        raw_output=raw,
        skipped=False,
    )


def _context_line(belief: dict[str, Any]) -> str:
    source = belief.get("source") or {}
    item = {
        "id": belief.get("id"),
        "layer": belief.get("layer"),
        "source": source.get("type"),
        "session": source.get("session_index"),
        "turn": source.get("turn_index"),
        "stance": belief.get("stance"),
        "confidence": belief.get("confidence"),
    }
    if belief.get("event_time"):
        item["event_time"] = belief.get("event_time")
    text = str(belief.get("belief") or "")
    item["belief"] = text if len(text) <= 240 else text[:220] + " ..."
    return json.dumps(item, ensure_ascii=False)


def _clean_optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in {"null", "none", "n/a"}:
        return None
    return stripped


def _clean_sentence_indices(value: Any, *, n_sentences: int) -> list[int] | None:
    if not isinstance(value, list):
        return None
    cleaned = sorted(
        {
            int(item)
            for item in value
            if isinstance(item, int | float) and 0 <= int(item) < n_sentences
        }
    )
    return cleaned or None
