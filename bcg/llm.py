from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from bcg import tracing
from bcg.utils import inline_image_url, maybe_await

Messages = list[dict[str, Any]]
ToolRunner = Callable[[str, dict[str, Any]], Any]


@dataclass(frozen=True)
class _ResponseToolCall:
    call_id: str
    name: str
    arguments: str


class _ModelDumpable(Protocol):
    def model_dump(self, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _GenerationOptions:
    model: str
    temperature: float | None
    max_tokens: int | None
    response_format: dict[str, Any] | None
    reasoning_effort: str

    @property
    def model_parameters(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning_effort,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class _ParsedResponse:
    content: str
    usage: dict[str, Any]
    reasoning_summary: str
    tool_calls: list[_ResponseToolCall]


@dataclass(frozen=True)
class LLMConfig:
    model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    )
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None
    )
    timeout: float = field(
        default_factory=lambda: float(os.getenv("OPENAI_TIMEOUT", "60"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    )
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str = "medium"


@dataclass(frozen=True)
class LLMResponse:
    content: str
    messages: Messages
    usage: dict[str, Any] = field(default_factory=dict)
    tool_calls: int = 0
    reasoning_summary: str = ""


class LLMClient:
    """Async OpenAI Responses API client with retry and tool dispatch."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self.enabled = bool(self.config.api_key or self.config.base_url)
        self._client: AsyncOpenAI | None = None
        if self.enabled:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key or "not-needed",
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=0,
            )

    @tracing.trace(
        name="bcg.llm.generate",
        as_type="chain",
        capture_input=False,
        capture_output=False,
    )
    async def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_runner: ToolRunner | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = "medium",
    ) -> LLMResponse:
        if self._client is None:
            raise RuntimeError(
                "LLMClient is disabled. Set OPENAI_API_KEY or OPENAI_BASE_URL to enable it."
            )

        options = _generation_options(
            config=self.config,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        working_messages: Messages = [dict(message) for message in messages]
        tool_calls_seen = 0

        tracing.update_current_span(
            input={
                "message_count": len(working_messages),
                "tool_count": len(tools or []),
                "model": options.model,
            },
            metadata={
                "temperature": options.temperature,
                "max_tokens": options.max_tokens,
                "reasoning_effort": options.reasoning_effort,
                "response_format": options.response_format,
            },
        )

        response = await self._completion_with_retry(
            working_messages,
            tools=tools,
            options=options,
        )
        parsed_response = _parse_response(response)
        assistant_message = _assistant_message(
            parsed_response.content,
            parsed_response.tool_calls,
        )
        working_messages.append(assistant_message)

        if tool_runner is not None:
            for tool_call in parsed_response.tool_calls:
                tool_calls_seen += 1
                arguments = _parse_tool_arguments(tool_call.arguments)
                result = await _run_tool_call(
                    tool_runner=tool_runner,
                    name=tool_call.name,
                    arguments=arguments,
                )
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.call_id,
                        "content": str(result),
                    }
                )

        llm_response = LLMResponse(
            content=parsed_response.content,
            messages=working_messages,
            usage=parsed_response.usage,
            tool_calls=tool_calls_seen,
            reasoning_summary=parsed_response.reasoning_summary,
        )

        tracing.update_current_span(
            output={
                "content": parsed_response.content,
                "tool_calls": tool_calls_seen,
                "reasoning_summary": parsed_response.reasoning_summary,
            },
            metadata={"usage": parsed_response.usage},
        )
        return llm_response

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

    @tracing.trace(
        name="bcg.llm.image",
        as_type="chain",
        capture_input=False,
        capture_output=False,
    )
    async def image(
        self,
        image_bytes: bytes,
        *,
        prompt: str,
        mime_type: str = "image/png",
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        image_url = inline_image_url(image_bytes, mime_type=mime_type)
        return await self.generate(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @tracing.trace(
        name="bcg.openai.responses.create",
        as_type="generation",
        capture_input=False,
        capture_output=False,
    )
    async def _completion_with_retry(
        self,
        messages: Messages,
        *,
        tools: Sequence[dict[str, Any]] | None,
        options: _GenerationOptions,
    ) -> Any:
        if self._client is None:
            raise RuntimeError("LLMClient is disabled.")

        kwargs = _responses_create_kwargs(
            messages=messages, tools=tools, options=options
        )
        max_retries = max(0, self.config.max_retries)
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.responses.create(**kwargs)
                tracing.update_current_generation(
                    model=options.model,
                    model_parameters=options.model_parameters,
                    input=kwargs["input"],
                    output=_response_content(response),
                    usage_details=_usage_details(response),
                    metadata={
                        "response_id": _get_attr_or_key(response, "id"),
                        "attempt": attempt + 1,
                    },
                )
                return response
            except Exception as exc:
                if attempt >= max_retries or not _is_retryable(exc):
                    raise
                await asyncio.sleep(min(8.0, 0.5 * (2**attempt)))

        raise RuntimeError("unreachable retry state")


def _parse_tool_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return value if isinstance(value, dict) else {"value": value}


@tracing.trace(
    name="bcg.llm.tool_call",
    as_type="tool",
    capture_input=False,
    capture_output=True,
)
async def _run_tool_call(
    *,
    tool_runner: ToolRunner,
    name: str,
    arguments: dict[str, Any],
) -> Any:
    tracing.update_current_span(input={"name": name, "arguments": arguments})
    return await maybe_await(tool_runner(name, arguments))


def _generation_options(
    *,
    config: LLMConfig,
    model: str | None,
    temperature: float | None,
    max_tokens: int | None,
    response_format: dict[str, Any] | None,
    reasoning_effort: str | None,
) -> _GenerationOptions:
    return _GenerationOptions(
        model=model or config.model,
        temperature=config.temperature if temperature is None else temperature,
        max_tokens=max_tokens if max_tokens is not None else config.max_tokens,
        response_format=response_format,
        reasoning_effort=reasoning_effort or config.reasoning_effort,
    )


def _responses_create_kwargs(
    *,
    messages: Messages,
    tools: Sequence[dict[str, Any]] | None,
    options: _GenerationOptions,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": options.model,
        "input": _responses_input(messages),
        "reasoning": {"effort": options.reasoning_effort, "summary": "auto"},
    }
    if options.temperature is not None:
        kwargs["temperature"] = options.temperature
    if tools:
        kwargs["tools"] = [_responses_tool_schema(tool) for tool in tools]
        kwargs["tool_choice"] = "auto"
    if options.max_tokens is not None:
        kwargs["max_output_tokens"] = options.max_tokens
    if options.response_format is not None:
        kwargs["text"] = {"format": dict(options.response_format)}
    return kwargs


def _parse_response(response: Any) -> _ParsedResponse:
    return _ParsedResponse(
        content=_response_content(response),
        usage=_usage_dict(response),
        reasoning_summary=_reasoning_summary_from_response(response),
        tool_calls=_response_tool_calls(response),
    )


def _responses_input(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        message_type = str(message.get("type") or "")
        if message_type in {"function_call", "function_call_output", "message"}:
            items.append(dict(message))
            continue
        role = str(message.get("role") or "user")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": str(message.get("content") or ""),
                }
            )
            continue
        for call in list(message.get("tool_calls", []) or []):
            if not isinstance(call, dict):
                continue
            function = dict(call.get("function", {}) or {})
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(call.get("id") or ""),
                    "name": str(function.get("name") or ""),
                    "arguments": str(function.get("arguments") or "{}"),
                }
            )
        if role == "assistant" and message.get("tool_calls"):
            continue
        items.append(
            {
                "role": role,
                "content": _responses_message_content(message.get("content")),
            }
        )
    return items


def _responses_message_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "")
        if part_type == "text":
            parts.append({"type": "input_text", "text": str(part.get("text") or "")})
        elif part_type == "image_url":
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            parts.append({"type": "input_image", "image_url": str(url or "")})
        else:
            parts.append(dict(part))
    return parts


def _responses_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if isinstance(function, dict):
        return {
            "type": "function",
            "name": str(function.get("name") or ""),
            "description": str(function.get("description") or ""),
            "parameters": dict(function.get("parameters", {}) or {}),
        }
    return dict(tool)


def _response_output_items(response: Any) -> list[dict[str, Any]]:
    output = _get_attr_or_key(response, "output")
    if not isinstance(output, list):
        return []
    items: list[dict[str, Any]] = []
    for item in output:
        if isinstance(item, dict):
            items.append(dict(item))
            continue
        dumped = _model_dump_dict(item)
        if dumped is not None:
            items.append(dumped)
            continue
        payload = {
            key: _get_attr_or_key(item, key)
            for key in ("type", "id", "call_id", "name", "arguments", "content")
            if _get_attr_or_key(item, key) is not None
        }
        if payload:
            items.append(payload)
    return items


def _response_content(response: Any) -> str:
    output_text = _get_attr_or_key(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    parts: list[str] = []
    for item in _response_output_items(response):
        if str(item.get("type") or "") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict):
                continue
            text = part.get("text") or part.get("content")
            if text:
                parts.append(str(text))
    return "\n".join(part for part in parts if part.strip())


def _response_tool_calls(response: Any) -> list[_ResponseToolCall]:
    calls: list[_ResponseToolCall] = []
    for item in _response_output_items(response):
        if str(item.get("type") or "") != "function_call":
            continue
        call_id = str(item.get("call_id") or item.get("id") or "")
        name = str(item.get("name") or "")
        raw_arguments = item.get("arguments")
        arguments = (
            json.dumps(raw_arguments, sort_keys=True)
            if isinstance(raw_arguments, dict)
            else str(raw_arguments or "{}")
        )
        calls.append(_ResponseToolCall(call_id=call_id, name=name, arguments=arguments))
    return calls


def _assistant_message(
    content: str,
    tool_calls: list[_ResponseToolCall],
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in tool_calls
        ]
    return message


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    dumped = _model_dump_dict(usage)
    if dumped is not None:
        return dumped
    return cast(dict[str, Any], dict(usage))


def _model_dump_dict(value: Any) -> dict[str, Any] | None:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return None

    dumped = cast(_ModelDumpable, value).model_dump(exclude_none=True)
    return dict(dumped)


def _usage_details(response: Any) -> dict[str, int]:
    usage = _usage_dict(response)
    return {key: value for key, value in usage.items() if isinstance(value, int)}


def _reasoning_summary_from_response(response: Any) -> str:
    parts: list[str] = []

    for output in _as_list(_get_attr_or_key(response, "output")):
        if _get_attr_or_key(output, "type") != "reasoning":
            continue
        parts.extend(_reasoning_text_parts(_get_attr_or_key(output, "summary")))

    return "\n".join(part for part in parts if part.strip())


def _reasoning_text_parts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        dict_parts: list[str] = []
        for key in ("text", "summary", "content", "reasoning_content"):
            dict_parts.extend(_reasoning_text_parts(value.get(key)))
        return dict_parts
    if isinstance(value, list):
        list_parts: list[str] = []
        for item in value:
            list_parts.extend(_reasoning_text_parts(item))
        return list_parts
    object_parts: list[str] = []
    for key in ("text", "summary", "content", "reasoning_content"):
        object_parts.extend(_reasoning_text_parts(_get_attr_or_key(value, key)))
    return object_parts


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _get_attr_or_key(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    value = getattr(item, key, None)
    if value is not None:
        return value
    extra = getattr(item, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(key)
    return None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(
        exc,
        APIConnectionError
        | APITimeoutError
        | InternalServerError
        | RateLimitError
        | httpx.TimeoutException
        | httpx.TransportError,
    ):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    if isinstance(exc, APIError):
        status_code = getattr(exc, "status_code", None)
        return isinstance(status_code, int) and (
            status_code in {408, 409, 429} or status_code >= 500
        )
    return False
