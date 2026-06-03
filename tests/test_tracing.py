from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from bcg.tracing import is_tracing_enabled, trace


class TracingTests(unittest.IsolatedAsyncioTestCase):
    def test_trace_is_noop_when_disabled(self) -> None:
        with patch.dict(os.environ, {"BCG_TRACING_ENABLED": "false"}):

            @trace(name="test.sync")
            def add(left: int, right: int) -> int:
                return left + right

            self.assertFalse(is_tracing_enabled())
            self.assertEqual(add(2, 3), 5)

    async def test_trace_supports_async_functions_when_disabled(self) -> None:
        with patch.dict(os.environ, {"BCG_TRACING_ENABLED": "false"}):

            @trace(name="test.async")
            async def add(left: int, right: int) -> int:
                return left + right

            self.assertEqual(await add(2, 3), 5)

    def test_llm_module_imports_with_tracing(self) -> None:
        from bcg.llm import LLMClient, LLMConfig, LLMResponse

        self.assertIsNotNone(LLMClient)
        self.assertIsNotNone(LLMConfig)
        self.assertIsNotNone(LLMResponse)


if __name__ == "__main__":
    unittest.main()
