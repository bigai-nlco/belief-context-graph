"""Standalone OpenAI-compatible API engine for agent rollouts.

Replaces vLLM/rllm engines when you want to call a remote API directly
(OpenAI, DashScope, DeepSeek, any OpenAI-compatible endpoint) without
needing vLLM installed locally.

Usage:
    --backend api --openai-base-url https://api.openai.com/v1 --model gpt-4o

The API key is loaded from ``OPENAI_API_KEY`` in the project-root ``.env``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import aiohttp

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine

logger = logging.getLogger(__name__)


class APIEngine(RolloutEngine):
    """Minimal OpenAI chat completions engine for remote API calls."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "EMPTY",
        max_prompt_length: int = 32768,
        max_response_length: int = 32768,
        sampling_params: dict | None = None,
        timeout: float = 300.0,
        **kwargs,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.sampling_params = dict(sampling_params or {})
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """Call the remote API and return a ModelOutput."""
        kwargs.pop("application_id", None)
        kwargs.pop("enforce_max_prompt_length", None)
        tools = kwargs.pop("tools", None)
        kwargs.pop("accumulate_reasoning", None)
        kwargs.pop("validate", None)
        kwargs.pop("precomputed_prompt_ids", None)
        kwargs.pop("reasoning_effort", None)

        # Build clean request payload
        params = self.sampling_params.copy()
        params.update(kwargs)

        # Remove non-standard fields
        params.pop("extra_body", None)
        params.pop("min_p", None)
        params.pop("repetition_penalty", None)
        params.pop("top_k", None)

        max_tokens = params.pop(
            "max_tokens", params.pop("max_new_tokens", self.max_response_length)
        )

        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": self._clean_messages(messages),
            "max_tokens": max_tokens,
        }

        # Standard OpenAI params
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "stop", "seed", "n"):
            if key in params and params[key] is not None:
                request_body[key] = params[key]

        if tools:
            tool_defs = []
            for t in tools:
                if isinstance(t, dict):
                    tool_defs.append(t)
                elif hasattr(t, "json"):
                    tool_defs.append(t.json)
            if tool_defs:
                request_body["tools"] = tool_defs

        logger.debug(
            "[APIEngine] Request: model=%s, n_messages=%d, max_tokens=%s, has_tools=%s",
            request_body.get("model"),
            len(request_body.get("messages", [])),
            request_body.get("max_tokens"),
            "tools" in request_body,
        )

        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=request_body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"API returned {resp.status}: {error_text[:500]}"
                    )
                result = await resp.json()
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc

        return self._parse_response(result)

    def _clean_messages(self, messages: list[dict]) -> list[dict]:
        """Keep only standard fields for the API request.

        Ensures every assistant tool_call has id/type/function, and every
        subsequent tool-role message carries the matching tool_call_id.
        """
        import uuid

        pending_tool_call_ids: list[str] = []

        cleaned = []
        for msg in messages:
            clean = {"role": msg["role"]}
            if msg.get("content") is not None:
                clean["content"] = msg["content"]

            if msg.get("tool_calls"):
                tool_calls = []
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        tc_clean = dict(tc)
                        if "function" not in tc_clean and "name" in tc_clean:
                            tc_clean = {
                                "type": "function",
                                "function": {
                                    "name": tc_clean.pop("name"),
                                    "arguments": tc_clean.pop("arguments", "{}"),
                                },
                            }
                    else:
                        args = tc.arguments if isinstance(tc.arguments, str) else json.dumps(tc.arguments)
                        tc_clean = {
                            "type": "function",
                            "function": {"name": tc.name, "arguments": args},
                        }
                    tc_clean.setdefault("type", "function")
                    if "id" not in tc_clean:
                        tc_clean["id"] = f"call_{uuid.uuid4().hex[:24]}"
                    if "function" in tc_clean and isinstance(tc_clean["function"].get("arguments"), dict):
                        tc_clean["function"]["arguments"] = json.dumps(tc_clean["function"]["arguments"])
                    pending_tool_call_ids.append(tc_clean["id"])
                    tool_calls.append(tc_clean)
                clean["tool_calls"] = tool_calls

            if msg["role"] == "tool":
                if msg.get("tool_call_id"):
                    clean["tool_call_id"] = msg["tool_call_id"]
                elif pending_tool_call_ids:
                    clean["tool_call_id"] = pending_tool_call_ids.pop(0)
                else:
                    clean["tool_call_id"] = f"call_{uuid.uuid4().hex[:24]}"

            if msg.get("name"):
                clean["name"] = msg["name"]
            cleaned.append(clean)
        return cleaned

    def _parse_response(self, result: dict) -> ModelOutput:
        """Convert API response to ModelOutput."""
        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError("API returned no choices")

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")

        content = message.get("content") or ""
        reasoning = ""

        # Extract thinking/reasoning if present (Qwen, DeepSeek style)
        reasoning_content = message.get("reasoning_content", "")
        if reasoning_content:
            reasoning = reasoning_content
        elif "<think>" in content:
            reasoning, content = self._split_thinking(content)

        # Parse tool calls from response
        tool_calls = []
        raw_tool_calls = message.get("tool_calls") or []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")
            try:
                arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                arguments = {}
            if name:
                from rllm.tools.tool_base import ToolCall
                tool_calls.append(ToolCall(name=name, arguments=arguments))

        from bcg.agent.tool_call_protocol import TOOL_CALL_OPEN_RE

        has_tool_call_tag = bool(TOOL_CALL_OPEN_RE.search(content))

        # Also try parsing tool calls from content (Qwen/DeepSeek <tool_call> format)
        if not tool_calls and has_tool_call_tag:
            parsed = self._parse_tool_calls_from_content(content)
            if parsed:
                tool_calls = parsed
                finish_reason = "tool_calls"

        # Fallback: some models (e.g. DeepSeek with thinking) emit a bare JSON
        # tool-call object WITHOUT the <tool_call> wrapper. Recognize a
        # {"name": ..., "arguments": {...}} shape in the content and treat it as
        # a tool call so the turn isn't mis-scored as a final answer.
        if not tool_calls and not has_tool_call_tag:
            parsed = self._parse_bare_json_tool_call(content)
            if parsed:
                tool_calls = parsed
                finish_reason = "tool_calls"

        # If content has an incomplete <tool_call> tag but structured tool_calls
        # were parsed, reconstruct the full content so trajectory logs are complete.
        if tool_calls and has_tool_call_tag and "</tool_call>" not in content:
            tc_lines = []
            for tc in tool_calls:
                tc_json = json.dumps({"name": tc.name, "arguments": tc.arguments}, ensure_ascii=False)
                tc_lines.append(f"<tool_call>\n{tc_json}\n</tool_call>")
            open_match = TOOL_CALL_OPEN_RE.search(content)
            cut = open_match.start() if open_match else len(content)
            content = content[:cut] + "\n".join(tc_lines)

        # Build the full text as the model would have produced it
        text = content
        if reasoning:
            text = f"<think>\n{reasoning}\n</think>\n{content}"

        usage = result.get("usage", {})

        return ModelOutput(
            text=text,
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls or None,
            prompt_ids=None,
            completion_ids=None,
            logprobs=[],
            prompt_logprobs=[],
            prompt_length=usage.get("prompt_tokens", 0),
            completion_length=usage.get("completion_tokens", 0),
            finish_reason=finish_reason,
        )

    def _split_thinking(self, text: str) -> tuple[str, str]:
        """Split <think>...</think> from content."""
        match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        # Unclosed think tag
        match = re.search(r"<think>(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip(), ""
        return "", text

    def _parse_tool_calls_from_content(self, content: str) -> list:
        """Parse tool calls from content text (Qwen, DeepSeek, etc.).

        Uses the shared ``TOOL_CALL_RE`` from ``tool_call_protocol`` so this
        recognizes both the legacy bare ``<tool_call>`` tag and the
        canonicalized ``<tool_call id="call_N">`` tag -- a model that has seen
        canonicalized tool calls in its own chat history may imitate that
        format in a new turn, and a literal/no-attribute regex would silently
        fail to match, dropping the tool call entirely.
        """
        from rllm.tools.tool_base import ToolCall

        from bcg.agent.tool_call_protocol import TOOL_CALL_RE

        calls = []
        for match in TOOL_CALL_RE.finditer(content):
            try:
                raw = match.group(2)
                # The model writes \boxed{} inside its JSON argument
                # strings, but \b is a valid JSON escape (backspace).
                # Protect it before json.loads by escaping the backslash.
                raw = re.sub(r"\\(?=boxed\s*\{)", r"\\\\", raw)
                data = json.loads(raw)
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                if name:
                    calls.append(ToolCall(name=name, arguments=arguments))
            except json.JSONDecodeError:
                continue
        return calls

    def _parse_bare_json_tool_call(self, content: str) -> list:
        """Parse a bare (unwrapped) JSON tool-call object from content.

        Handles models that emit ``{"name": "...", "arguments": {...}}`` without
        the ``<tool_call>`` wrapper. Scans for balanced ``{...}`` spans and
        accepts the first one that has a string ``name`` and a dict
        ``arguments`` — a shape specific enough not to misfire on prose.
        """
        from rllm.tools.tool_base import ToolCall

        text = content.strip()
        if "{" not in text or '"name"' not in text:
            return []

        # Collect candidate JSON object spans by brace balancing (handles the
        # whole-content case and a JSON object embedded in surrounding text).
        candidates: list[str] = []
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        candidates.append(text[start : i + 1])
                        start = -1

        for raw in candidates:
            # \boxed{} appears inside arguments; \b is a JSON escape, protect it.
            raw_fixed = re.sub(r"\\(?=boxed\s*\{)", r"\\\\", raw)
            try:
                data = json.loads(raw_fixed)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            name = data.get("name")
            arguments = data.get("arguments")
            if isinstance(name, str) and name and isinstance(arguments, dict):
                return [ToolCall(name=name, arguments=arguments)]
        return []

    async def wake_up(self) -> None:
        return None

    async def sleep(self) -> None:
        return None

    def shutdown(self) -> None:
        if self._session and not self._session.closed:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._session.close())
                else:
                    loop.run_until_complete(self._session.close())
            except Exception:
                pass


__all__ = ["APIEngine"]
