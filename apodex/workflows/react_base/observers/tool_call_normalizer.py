"""Observer that reverses SGLang stringification of ``list[str]`` tool args.

Some SGLang deployments serialise tool args of type ``anyOf[str, list[str]]``
as a JSON-encoded **string** instead of a native list. The aligned tools'
own ``_ensure_list`` unwraps it on the way in, so dispatch is correct —
but the assistant message recorded in history still carries the
stringified form, drifting next-turn distribution. This observer
restores native lists on both the history message and the parsed
``ctx.tool_calls``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_harness.core.loop_types import BaseObserver, Intervention, TurnContext

logger = logging.getLogger(__name__)


def _normalize_tool_call_args(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restore native list form for any stringified ``list[str]`` arg.

    Conservative + idempotent + non-mutating:

    - Only unwraps string values that **parse cleanly** as a JSON list
      whose elements are all strings. A legitimate string like
      ``"[urgent] my real query"`` fails ``json.loads`` and stays a
      string; ``"[1, 2, 3]"`` parses but elements aren't strings, so
      it stays a string too.
    - Allocates fresh dicts only for tool_calls that actually need
      changing — original input is never mutated.
    - **Returns the input list itself** when no normalisation is
      needed, so callers can ``is``-compare to skip downstream work.
    """
    if not tool_calls:
        return tool_calls

    out: list[dict[str, Any]] | None = None
    for i, tc in enumerate(tool_calls):
        args = tc.get("args") if isinstance(tc, dict) else None
        if not isinstance(args, dict):
            continue
        new_args: dict[str, Any] | None = None
        for k, v in args.items():
            if not (isinstance(v, str) and v.startswith("[") and v.endswith("]")):
                continue
            try:
                parsed = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                if new_args is None:
                    new_args = dict(args)
                new_args[k] = parsed
        if new_args is not None:
            if out is None:
                out = list(tool_calls)
            out[i] = {**tc, "args": new_args}
    return out if out is not None else tool_calls


def _normalize_openai_tool_calls(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Same conservative unwrap, on the OpenAI wire form
    (``function.arguments`` JSON-encoded). Returns ``None`` if nothing changed."""
    if not tool_calls:
        return None
    out: list[dict[str, Any]] | None = None
    for i, tc in enumerate(tool_calls):
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(fn, dict):
            continue
        raw_args = fn.get("arguments")
        if not isinstance(raw_args, str):
            continue
        try:
            args = json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(args, dict):
            continue
        new_args: dict[str, Any] | None = None
        for k, v in args.items():
            if not (isinstance(v, str) and v.startswith("[") and v.endswith("]")):
                continue
            try:
                parsed = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                if new_args is None:
                    new_args = dict(args)
                new_args[k] = parsed
        if new_args is not None:
            if out is None:
                out = list(tool_calls)
            new_fn = dict(fn)
            new_fn["arguments"] = json.dumps(new_args, ensure_ascii=False)
            out[i] = {**tc, "function": new_fn}
    return out


class ToolCallArgsNormalizer(BaseObserver):
    """Restore native lists on the recorded assistant turn and on
    ``ctx.tool_calls``. Dispatch already unwraps via each tool's
    ``_ensure_list``; this fixes the *history* the model sees next turn."""

    async def on_llm_response(
        self, ctx: TurnContext,
    ) -> Intervention | None:
        if ctx.messages:
            last = ctx.messages[-1]
            if isinstance(last, dict) and last.get("role") == "assistant":
                tcs = last.get("tool_calls") or []
                normalized = _normalize_openai_tool_calls(tcs)
                if normalized is not None:
                    last["tool_calls"] = normalized

        # ``ctx.tool_calls`` is the parsed ``{name, args, id}`` form that
        # downstream observers (trajectory, SSE, tracker) read for logging.
        if ctx.tool_calls:
            normalized_parsed = _normalize_tool_call_args(ctx.tool_calls)
            if normalized_parsed is not ctx.tool_calls:
                ctx.tool_calls[:] = normalized_parsed
                logger.debug(
                    "Normalized stringified list args in turn %d (%d tool_call(s))",
                    ctx.turn, len(normalized_parsed),
                )

        return None


__all__ = ["ToolCallArgsNormalizer", "_normalize_tool_call_args"]
