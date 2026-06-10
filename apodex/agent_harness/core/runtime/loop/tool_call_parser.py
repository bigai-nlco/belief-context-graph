"""Generic tool call parser: native function-calling + JSON/XML text fallback.

Strategies in priority order:

1. Native — ``LLMResponse.tool_calls`` (OpenAI format,
   ``{id, type, function: {name, arguments}}``).
2. ``<tool_call>{"tool": …}</tool_call>`` JSON.
3. ``<|FunctionCallBegin|>[…]<|FunctionCallEnd|>`` (Seed reasoning wrapper).
4. ``<tool_call><function=name>…</function></tool_call>`` (Qwen XML).
5. ``<function name="name">…</function>`` (Seed XML).
6. ``<use_mcp_tool>…</use_mcp_tool>`` MCP server/tool/arguments.

Each parsed call is returned as ``{"name": str, "args": dict, "id": str}``
— the format the agent loop expects to pair with ``tool message`` results.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, runtime_checkable

from agent_harness.core.llm import LLMResponse
from agent_harness.core.messages import text_of

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_MCP_USE_TOOL_RE = re.compile(
    r"<use_mcp_tool>\s*(.*?)\s*</use_mcp_tool>", re.DOTALL | re.IGNORECASE,
)
_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL,
)
_QWEN_PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)
_SEED_FUNCTION_RE = re.compile(r'<function\s+name="(\w+)">(.*?)</function>', re.DOTALL)
_SEED_PARAM_RE = re.compile(
    r'<parameter\s+name="(\w+)"[^>]*>(.*?)</parameter>', re.DOTALL,
)
_SEED_TAG_HINT_RE = re.compile(r"<seed[:\w]*tool")
_FC_WRAPPED_RE = re.compile(
    r"<\|FunctionCallBegin\|>\s*(\[.*?\])\s*<\|FunctionCallEnd\|>", re.DOTALL,
)

# Leaked inner-monologue tags that some models emit into ``content`` even
# though they should be private. Stripped before any parsing so they
# don't pollute history or confuse downstream parsers.
_LEAKED_TAG_RE = re.compile(
    r"<(?:think|thinking|reasoning|seed:think|seed:reasoning)>"
    r"(.*?)"
    r"</(?:think|thinking|reasoning|seed:think|seed:reasoning)>",
    re.DOTALL | re.IGNORECASE,
)
_DANGLING_LEAKED_TAG_RE = re.compile(
    r"<(?:think|thinking|reasoning|seed:think|seed:reasoning)>(.*)\Z",
    re.DOTALL | re.IGNORECASE,
)
_NEVER_USED_TAG_RE = re.compile(
    r"<think_never_used[^>]*>(.*?)</think_never_used[^>]*>",
    re.DOTALL | re.IGNORECASE,
)
_NEVER_USED_DANGLING_RE = re.compile(
    r"<think_never_used[^>]*>(.*)\Z", re.DOTALL | re.IGNORECASE,
)
_MODEL_THINKING_RE = re.compile(
    r"<model:\s*thinking>(.*?)</model:\s*thinking>",
    re.DOTALL | re.IGNORECASE,
)

# Stashed on LLMResponse.response_metadata so observers can surface it.
LEAKED_REASONING_KEY = "leaked_reasoning"


@runtime_checkable
class ToolCallParser(Protocol):
    """Protocol for tool-call parsers operating on an ``LLMResponse``."""

    def parse(
        self, response: LLMResponse, tool_names: set[str],
    ) -> list[dict]:
        ...


def _native_tool_calls(response: LLMResponse) -> list[dict]:
    """Translate OpenAI-format ``response.tool_calls`` into
    ``{name, args, id}`` dicts."""
    out: list[dict] = []
    for tc in response.tool_calls or []:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        if not name:
            continue
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, ValueError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        out.append({"name": name, "args": args, "id": tc.get("id") or ""})
    return out


class DefaultToolCallParser:
    """Native-FC first, JSON / MCP text fallback second."""

    def parse(self, response: LLMResponse, tool_names: set[str]) -> list[dict]:
        native = _native_tool_calls(response)
        if native:
            filtered = [tc for tc in native if tc["name"] in tool_names]
            logger.debug(
                "native FC: %d total, %d after filter",
                len(native), len(filtered),
            )
            return filtered

        text = text_of(response.content)
        parsed = self._parse_json_tool_calls(text, tool_names)
        if parsed:
            return parsed
        return self._parse_mcp_tool_calls(text, tool_names)

    @staticmethod
    def _parse_json_tool_calls(text: str, tool_names: set[str]) -> list[dict]:
        results: list[dict] = []
        for i, match in enumerate(_TOOL_CALL_RE.finditer(text)):
            raw = match.group(1).strip()
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                logger.debug("Skipping malformed JSON tool_call: %r", raw[:120])
                continue
            if not isinstance(payload, dict):
                continue
            name = str(payload.get("tool", "") or "").strip()
            if not name or name not in tool_names:
                continue
            args = payload.get("args", {})
            if not isinstance(args, dict):
                args = {}
            results.append({"name": name, "args": args, "id": f"json_tc_{i}"})
        logger.debug("JSON fallback: found %d tool calls", len(results))
        return results

    @staticmethod
    def _parse_mcp_tool_calls(text: str, tool_names: set[str]) -> list[dict]:
        if "use_mcp_tool" not in tool_names:
            return []
        results: list[dict] = []
        for idx, match in enumerate(_MCP_USE_TOOL_RE.finditer(text)):
            body = match.group(1)
            server_name = _extract_xml_value(body, "server_name")
            tool_name = _extract_xml_value(body, "tool_name")
            if not server_name or not tool_name:
                continue
            raw_args = _extract_xml_value(body, "arguments")
            arguments: dict[str, Any] = {}
            if raw_args:
                try:
                    parsed = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    parsed = {}
                if isinstance(parsed, dict):
                    arguments = parsed
            results.append({
                "name": "use_mcp_tool",
                "args": {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "arguments": arguments,
                },
                "id": f"mcp_tc_{idx}",
            })
        logger.debug("MCP XML fallback: found %d tool calls", len(results))
        return results


class MultiFormatToolCallParser(DefaultToolCallParser):
    """Default parser + Qwen/Seed XML + Seed FunctionCall wrapper fallbacks.

    Strips leaked private-tag content from ``response.content`` before
    parsing. Use :meth:`parse_text` to recover tool calls that ended up
    inside private reasoning channels.
    """

    def parse(self, response: LLMResponse, tool_names: set[str]) -> list[dict]:
        self._clean_leaked_content(response)

        base = super().parse(response, tool_names)
        if base:
            return base

        text = text_of(response.content)

        # FunctionCall wrapper has higher priority than the native short-
        # circuit: the marker is unambiguous, so we trust it over a possibly-
        # empty native list produced by an upstream proxy that failed to
        # repack it. Side-effect strip keeps history clean.
        if text and "<|FunctionCallBegin|>" in text:
            fc = _parse_fc_wrapped(text, tool_names)
            if fc:
                self._strip_fc_wrappers(response)
                return fc

        # Native API was used (even with unknown names) — don't second-guess
        # with XML regexes on residual content.
        if response.tool_calls:
            return base

        return self.parse_text(text, tool_names)

    def parse_text(self, text: str, tool_names: set[str]) -> list[dict]:
        """Pure-text recovery (no side effects). Used as the content fallback
        in :meth:`parse` and to recover leaks from a model's thinking block.
        """
        if not text:
            return []
        if "<use_mcp_tool>" in text:
            if (calls := self._parse_mcp_tool_calls(text, tool_names)):
                return calls
        if "<tool_call>" in text and "<function=" in text:
            if (calls := self._parse_qwen_xml(text, tool_names)):
                return calls
        if "<|FunctionCallBegin|>" in text:
            if (calls := _parse_fc_wrapped(text, tool_names)):
                return calls
        if '<function name="' in text and (
            '<parameter name="' in text or _SEED_TAG_HINT_RE.search(text)
        ):
            return self._parse_seed_xml(text, tool_names)
        return []

    @staticmethod
    def _clean_leaked_content(response: LLMResponse) -> None:
        """Strip leaked think/reasoning tags from ``response.content`` and
        accumulate the recovered inner text on
        ``response.response_metadata[LEAKED_REASONING_KEY]``."""
        content = response.content
        if content is None:
            return

        recovered_parts: list[str] = []

        if isinstance(content, str):
            cleaned, reasoning = extract_leaked_reasoning(content)
            if reasoning:
                recovered_parts.append(reasoning)
            if cleaned != content:
                response.content = cleaned
        elif isinstance(content, list):
            changed = False
            new_blocks: list[Any] = []
            for block in content:
                if isinstance(block, str):
                    clean, reasoning = extract_leaked_reasoning(block)
                    if reasoning:
                        recovered_parts.append(reasoning)
                    changed = changed or clean != block
                    new_blocks.append(clean)
                elif isinstance(block, dict) and isinstance(
                    block.get("text"), str,
                ):
                    clean, reasoning = extract_leaked_reasoning(block["text"])
                    if reasoning:
                        recovered_parts.append(reasoning)
                    cleaned_block = dict(block)
                    cleaned_block["text"] = clean
                    changed = changed or cleaned_block["text"] != block["text"]
                    new_blocks.append(cleaned_block)
                else:
                    new_blocks.append(block)
            if changed:
                response.content = new_blocks

        if recovered_parts:
            _attach_leaked_reasoning(response, "\n\n".join(recovered_parts))

    @staticmethod
    def _parse_qwen_xml(text: str, tool_names: set[str]) -> list[dict]:
        results: list[dict] = []
        for i, match in enumerate(_QWEN_TOOL_CALL_RE.finditer(text)):
            name = match.group(1)
            if name not in tool_names:
                continue
            body = match.group(2)
            args: dict = {}
            for pm in _QWEN_PARAM_RE.finditer(body):
                args[pm.group(1)] = _coerce_param_value(pm.group(2).strip())
            results.append({"name": name, "args": args, "id": f"qwen_tc_{i}"})
        logger.debug("Qwen XML fallback: found %d tool calls", len(results))
        return results

    @staticmethod
    def _parse_seed_xml(text: str, tool_names: set[str]) -> list[dict]:
        results: list[dict] = []
        for i, match in enumerate(_SEED_FUNCTION_RE.finditer(text)):
            name = match.group(1)
            if name not in tool_names:
                continue
            body = match.group(2)
            args: dict = {}
            for pm in _SEED_PARAM_RE.finditer(body):
                args[pm.group(1)] = _coerce_param_value(pm.group(2).strip())
            results.append({"name": name, "args": args, "id": f"seed_tc_{i}"})
        logger.debug("Seed XML fallback: found %d tool calls", len(results))
        return results

    @staticmethod
    def _strip_fc_wrappers(response: LLMResponse) -> None:
        content = response.content
        if not isinstance(content, str):
            return
        cleaned = _FC_WRAPPED_RE.sub("", content).strip()
        if cleaned != content:
            response.content = cleaned


def _parse_fc_wrapped(text: str, tool_names: set[str]) -> list[dict]:
    """Seed reasoning-mode ``<|FunctionCallBegin|>[…]<|FunctionCallEnd|>``."""
    results: list[dict] = []
    idx = 0
    for match in _FC_WRAPPED_RE.finditer(text):
        try:
            calls = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "FunctionCall wrapper: bad JSON (%s) body_preview=%r",
                exc, match.group(1)[:200],
            )
            continue
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name", "")
            if name not in tool_names:
                continue
            args = call.get("parameters") or call.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            results.append({"name": name, "args": args, "id": f"fc_tc_{idx}"})
            idx += 1
    logger.debug("FunctionCall wrapper: found %d tool calls", len(results))
    return results


def _extract_xml_value(body: str, tag: str) -> str:
    match = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        body, re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _coerce_param_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _attach_leaked_reasoning(response: LLMResponse, reasoning: str) -> None:
    """Accumulate recovered reasoning on ``response.response_metadata``."""
    if not reasoning:
        return
    meta = response.response_metadata
    if meta is None:
        response.response_metadata = {LEAKED_REASONING_KEY: reasoning}
        return
    prior = meta.get(LEAKED_REASONING_KEY, "")
    if prior:
        if reasoning in prior:
            return
        meta[LEAKED_REASONING_KEY] = f"{prior}\n\n{reasoning}"
    else:
        meta[LEAKED_REASONING_KEY] = reasoning


def extract_leaked_reasoning(text: str) -> tuple[str, str]:
    """Strip leaked thinking/reasoning tags and return ``(cleaned, reasoning)``.

    Recovers inner content from every leak pattern we know about:
      * ``<think>…</think>`` / ``<thinking>…`` / ``<reasoning>…`` (balanced)
      * ``<think_never_used_…>…</think_never_used_…>`` (Seed / GPT-OSS)
      * ``<model:thinking>…</model:thinking>``
      * dangling opens — the tail becomes reasoning, not silently dropped.
    """
    if not text:
        return text, ""

    reasoning_parts: list[str] = []

    def _collect(pattern: re.Pattern, src: str) -> str:
        for match in pattern.finditer(src):
            inner = match.group(1).strip() if match.groups() else ""
            if inner:
                reasoning_parts.append(inner)
        return pattern.sub("", src)

    cleaned = text
    cleaned = _collect(_NEVER_USED_TAG_RE, cleaned)
    cleaned = _collect(_MODEL_THINKING_RE, cleaned)
    cleaned = _collect(_LEAKED_TAG_RE, cleaned)

    for dangling_re in (_NEVER_USED_DANGLING_RE, _DANGLING_LEAKED_TAG_RE):
        match = dangling_re.search(cleaned)
        if match:
            inner = match.group(1).strip()
            if inner:
                reasoning_parts.append(inner)
            cleaned = dangling_re.sub("", cleaned)

    reasoning = "\n\n".join(p for p in reasoning_parts if p)
    return cleaned, reasoning
