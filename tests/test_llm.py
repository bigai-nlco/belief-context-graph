from __future__ import annotations

import unittest
from typing import Any

from bcg.llm import _response_output_items, _usage_dict


class _Dumpable:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        exclude_none = bool(kwargs.get("exclude_none"))
        if not exclude_none:
            return dict(self.payload)
        return {key: value for key, value in self.payload.items() if value is not None}


class _Response:
    def __init__(self, *, output: Any = None, usage: Any = None) -> None:
        self.output = output
        self.usage = usage


class LLMParsingTests(unittest.TestCase):
    def test_response_output_items_support_model_dump(self) -> None:
        response = _Response(
            output=[_Dumpable({"type": "message", "content": "ok", "empty": None})]
        )

        self.assertEqual(
            _response_output_items(response),
            [{"type": "message", "content": "ok"}],
        )

    def test_usage_dict_supports_model_dump(self) -> None:
        response = _Response(usage=_Dumpable({"input_tokens": 1, "output_tokens": 2}))

        self.assertEqual(
            _usage_dict(response),
            {"input_tokens": 1, "output_tokens": 2},
        )


if __name__ == "__main__":
    unittest.main()
