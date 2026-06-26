from __future__ import annotations

import asyncio

from bcg.tracing import is_tracing_enabled, trace


def test_trace_is_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("BCG_TRACING_ENABLED", "false")

    @trace(name="test.sync")
    def add(left: int, right: int) -> int:
        return left + right

    assert not is_tracing_enabled()
    assert add(2, 3) == 5


def test_trace_supports_async_functions_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("BCG_TRACING_ENABLED", "false")

    @trace(name="test.async")
    async def add(left: int, right: int) -> int:
        return left + right

    assert asyncio.run(add(2, 3)) == 5


def test_llm_module_imports_with_tracing() -> None:
    from bcg.llm import LLMClient, LLMConfig, LLMResponse

    assert LLMClient is not None
    assert LLMConfig is not None
    assert LLMResponse is not None
