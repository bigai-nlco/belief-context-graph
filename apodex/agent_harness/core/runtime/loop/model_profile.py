"""Model-agnostic thinking extraction and history-normalization helpers.

Design:

- ``ModelProfile`` — model facts (thinking format is a model property).
- ``HistoryPolicy`` — agent design choices about how thinking is recorded.
- ``ThinkingParser`` — extracts thinking vs visible content from an
  :class:`LLMResponse`.
- ``MessageNormalizer`` — turns an :class:`LLMResponse` into the dict
  :class:`Message` stored in history.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol, get_args

from agent_harness.core.llm import LLMResponse
from agent_harness.core.messages import Message, assistant_msg

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>([\s\S]*?)</think>\s*", re.DOTALL)


ThinkingFormat = Literal["tag", "content_block", "reasoning_content", "none"]
_VALID_FORMATS: frozenset[str] = frozenset(get_args(ThinkingFormat))
# parents[3]: loop/ → runtime/ → core/ → agent_harness/
_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "model_registry.yaml"


@lru_cache(maxsize=1)
def _load_thinking_format_patterns() -> (
    tuple[tuple[re.Pattern[str], ThinkingFormat], ...]
):
    """Read ``thinking_formats`` from ``model_registry.yaml``. Returns an
    empty tuple on missing/malformed file; bad rows are skipped with a
    warning so one typo can't break inference."""
    if not _REGISTRY_PATH.is_file():
        return ()
    try:
        import yaml
    except ImportError:
        return ()
    try:
        raw = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to parse model_registry.yaml: %s", exc)
        return ()

    table = raw.get("thinking_formats") or []
    if not isinstance(table, list):
        return ()

    out: list[tuple[re.Pattern[str], ThinkingFormat]] = []
    for entry in table:
        if not isinstance(entry, dict):
            continue
        pattern_str = entry.get("pattern")
        fmt = entry.get("format")
        if not pattern_str or fmt not in _VALID_FORMATS:
            continue
        try:
            compiled = re.compile(pattern_str, re.IGNORECASE)
        except re.error as exc:
            logger.warning("Bad regex %r in model_registry: %s", pattern_str, exc)
            continue
        out.append((compiled, fmt))
    return tuple(out)


def reset_thinking_format_cache() -> None:
    """Clear the cached pattern table — call after editing the YAML at runtime."""
    _load_thinking_format_patterns.cache_clear()


def infer_thinking_format(
    model_id: str | None,
    *,
    default: ThinkingFormat = "tag",
) -> ThinkingFormat:
    """Pick a thinking format for a model id (case-insensitive match)."""
    if not model_id:
        return default
    needle = model_id.lower()
    for pattern, fmt in _load_thinking_format_patterns():
        if pattern.search(needle):
            return fmt
    return default


# ── Data structures ────────────────────────────────────────────────────


@dataclass
class ModelProfile:
    """Static facts about a model's capabilities."""

    model_id: str
    provider: str
    context_window: int = 128_000
    supports_native_fc: bool = True
    supports_streaming: bool = True
    supports_images: bool = False
    thinking_format: ThinkingFormat = "none"
    tool_call_format: Literal["native_fc", "text", "both"] = "native_fc"


@dataclass
class HistoryPolicy:
    """Agent design choices about how thinking output is used."""

    thinking_in_history: bool = False
    thinking_in_memory: bool = True
    thinking_in_sse: bool = True
    tool_result_max_chars: int = 15_000
    compress_thinking_after_turns: int = 0
    max_images_in_history: int = 5


@dataclass
class ThinkingResult:
    """Parsed output from a single LLM response."""

    thinking: str
    visible_content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


# ── Protocols ──────────────────────────────────────────────────────────


class ThinkingParser(Protocol):
    def extract(
        self, response: LLMResponse, profile: ModelProfile,
    ) -> ThinkingResult:
        ...


class MessageNormalizer(Protocol):
    def to_history(
        self,
        response: LLMResponse,
        thinking_result: ThinkingResult,
        policy: HistoryPolicy,
    ) -> Message:
        ...


# ── Default implementations ────────────────────────────────────────────


class DefaultThinkingParser:
    """Handles four formats:

    - ``"none"`` — visible content is the full text, no thinking.
    - ``"tag"`` — ``<think>…</think>`` regex on ``content``.
    - ``"content_block"`` — Anthropic block list (``type=thinking`` vs ``text``).
    - ``"reasoning_content"`` — DeepSeek-V4 / o-series separate field
      surfaced on :attr:`LLMResponse.reasoning_content`.
    """

    def extract(
        self, response: LLMResponse, profile: ModelProfile,
    ) -> ThinkingResult:
        tool_calls = list(response.tool_calls or [])
        fmt = profile.thinking_format
        content = response.content

        if fmt == "tag":
            raw = content if isinstance(content, str) else ""
            match = _THINK_RE.search(raw)
            if match:
                thinking = match.group(1)
                visible = _THINK_RE.sub("", raw).strip()
            else:
                thinking = ""
                visible = raw
            return ThinkingResult(
                thinking=thinking, visible_content=visible,
                tool_calls=tool_calls,
            )

        if fmt == "content_block":
            blocks = content if isinstance(content, list) else []
            thinking_parts: list[str] = []
            text_parts: list[str] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "thinking":
                    thinking_parts.append(block.get("thinking", "") or "")
                elif btype == "text":
                    text_parts.append(block.get("text", "") or "")
            return ThinkingResult(
                thinking="\n".join(thinking_parts),
                visible_content="\n".join(text_parts),
                tool_calls=tool_calls,
            )

        if fmt == "reasoning_content":
            return ThinkingResult(
                thinking=response.reasoning_content or "",
                visible_content=content if isinstance(content, str) else "",
                tool_calls=tool_calls,
            )

        # "none" or unknown — treat all visible.
        return ThinkingResult(
            thinking="",
            visible_content=content if isinstance(content, str) else "",
            tool_calls=tool_calls,
        )


class DefaultMessageNormalizer:
    """Turn an :class:`LLMResponse` into the assistant :class:`Message`
    stored in history. ``thinking_in_history=True`` snapshots the full
    content (incl. ``<think>``) so step 3.5's content mutation can't
    later strip thinking out of the recorded turn."""

    def to_history(
        self,
        response: LLMResponse,
        thinking_result: ThinkingResult,
        policy: HistoryPolicy,
    ) -> Message:
        if policy.thinking_in_history and isinstance(response.content, str):
            visible = response.content
        else:
            visible = thinking_result.visible_content
        # ``reasoning_content`` intentionally NOT carried into the history
        # Message: the reasoning-inliner wrapper already inlines it into
        # ``content`` as a ``{rc}</think>\\n\\n{visible}`` block, so
        # propagating it would add an unrecognized wire field.
        return assistant_msg(
            visible,
            tool_calls=_to_openai_tool_calls(thinking_result.tool_calls),
        )


def _to_openai_tool_calls(parsed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Echo parsed tool calls back into OpenAI wire format.

    Key order: ``{type, id, function: {name, arguments}}`` — some served
    checkpoints are sensitive to this byte shape.
    """
    import json
    out: list[dict[str, Any]] = []
    for tc in parsed:
        if "function" in tc and list(tc.keys())[:2] == ["type", "id"]:
            out.append(tc)
            continue
        args = tc.get("args") if "args" in tc else None
        if args is None and "function" in tc:
            args_str = tc["function"].get("arguments", "{}")
            name = tc["function"].get("name", "")
        else:
            args_str = (
                args if isinstance(args, str)
                else json.dumps(args or {}, ensure_ascii=False)
            )
            name = tc.get("name", "")
        out.append({
            "type": "function",
            "id": tc.get("id") or "",
            "function": {
                "name": name,
                "arguments": args_str,
            },
        })
    return out
