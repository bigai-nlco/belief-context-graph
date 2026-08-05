"""Adapters between the public SDK LLM and construct client interfaces."""

from __future__ import annotations

import asyncio
import inspect
from types import ModuleType, SimpleNamespace
from typing import Any


class ConstructClientAdapter:
    """Expose an OpenAI chat-completions shape over the public SDK LLM client."""

    def __init__(self, llm: Any, backend_llm_module: ModuleType | None = None) -> None:
        self.llm = llm
        self._backend_llm_module = backend_llm_module
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs: Any) -> Any:
        messages = list(kwargs.get("messages") or [])
        prompt = str(messages[-1].get("content") or "") if messages else ""
        model = kwargs.get("model")
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        if hasattr(self.llm, "generate"):
            call_kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if model and _accepts_keyword(self.llm.generate, "model"):
                call_kwargs["model"] = model
            response = _resolve_sync(self.llm.generate(messages, **call_kwargs))
            content = _response_content(response)
            usage = getattr(response, "usage", {})
        elif hasattr(self.llm, "generate_text"):
            call_kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if _accepts_keyword(self.llm.generate_text, "label"):
                tracker_factory = getattr(
                    self._backend_llm_module,
                    "current_usage_tracker",
                    None,
                )
                if tracker_factory is not None:
                    tracker = tracker_factory()
                    call_kwargs["label"] = getattr(tracker, "_label", "unlabeled")
            if model and _accepts_keyword(self.llm.generate_text, "model"):
                call_kwargs["model"] = model
            content = str(_resolve_sync(self.llm.generate_text(prompt, **call_kwargs)))
            usage = {}
        else:
            raise TypeError("BCGRunner requires an LLM with generate or generate_text")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=_chat_usage(usage),
        )


def _resolve_sync(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _response_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    if isinstance(response, dict):
        return str(response.get("content") or response.get("text") or "")
    return str(response)


def _chat_usage(usage: Any) -> Any:
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    usage = usage if isinstance(usage, dict) else {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = int(prompt) + int(completion)
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _accepts_keyword(func: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


__all__ = ["ConstructClientAdapter"]
