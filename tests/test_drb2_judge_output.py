"""Offline output/retry regressions; these fixtures never call a model."""
from __future__ import annotations

import asyncio
import json

import pytest

from benchmarks.public.reports import drb2_judge as judge


def result(item, **overrides):
    return {"rubric_item": item, "score": 0, "reason": "Not mentioned.", "evidence": "", **overrides}


@pytest.mark.parametrize("indent", [None, 2])
@pytest.mark.parametrize("fenced", [False, True])
def test_valid_json_is_not_rewritten(indent, fenced):
    items = ['Exact "quoted" field: value', "中文：精确文本与换行\n第二行"]
    expected = {"results": [result(item) for item in items]}
    text = json.dumps(expected, ensure_ascii=False, indent=indent)
    if fenced:
        text = f"```json\n{text}\n```"
    parsed, ok = judge.parse_model_text(text)
    assert ok and parsed == expected
    assert judge.validate_batch_result(items, parsed)


@pytest.mark.parametrize("value", [None, [], {"results": None}, {"results": [None]}])
def test_bad_shapes_are_validation_errors_not_request_exceptions(value):
    assert judge.batch_validation_errors(["a"], value)


def test_exact_membership_and_duplicate_counts_are_required():
    assert judge.validate_batch_result(["a", "b"], {"results": [result("b"), result("a")]})
    assert not judge.validate_batch_result(["a", "b"], {"results": [result("a"), result("a")]})
    assert not judge.validate_batch_result(["a", "a"], {"results": [result("a"), result("b")]})
    assert not judge.validate_batch_result(["a"], {"results": [result("a ")]})


@pytest.mark.parametrize("overrides", [
    {"score": True}, {"score": "1"}, {"score": 2}, {"reason": None}, {"evidence": []},
])
def test_invalid_score_fields_are_not_silently_accepted(overrides):
    assert not judge.validate_batch_result(["a"], {"results": [result("a", **overrides)]})


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    async def complete(self, prompt):
        self.prompts.append(prompt)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply, {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}


def test_retry_feedback_changes_request_and_keeps_original_scoring_prompt():
    bad = json.dumps({"results": [result("renamed a")]})
    good = json.dumps({"results": [result("a"), result("b")]})
    client = FakeClient([bad, good])
    records = []
    results, usage = asyncio.run(judge.query_rubric_batch(
        client, ["a", "b"], "Task", {}, "Report", max_retries=2, record_attempt=records.append,
    ))
    assert results == [result("a"), result("b")]
    assert usage["totalTokenCount"] == 30
    assert client.prompts[1].startswith(client.prompts[0])
    assert "<output_validation_feedback>" not in client.prompts[0]
    assert "Expected 2 results, received 1" in client.prompts[1]
    assert 'Missing exact rubric_item text (2): ["a", "b"]' in client.prompts[1]
    assert "renamed a" in client.prompts[1]
    assert records[0]["response"] == bad and records[0]["status"] == "invalid_output"
    assert records[1]["status"] == "valid" and records[1]["validation_errors"] == []


def test_exhausted_retries_have_distinct_feedback_and_keep_usage():
    client = FakeClient(["{}"] * 3)
    records = []
    results, usage = asyncio.run(judge.query_rubric_batch(
        client, ["a"], "Task", {}, "Report", max_retries=3, record_attempt=records.append,
    ))
    assert results is None and usage["totalTokenCount"] == 45
    assert len(set(client.prompts)) == 3
    assert len(records) == 3
    assert all(record["status"] == "invalid_output" for record in records)


def test_attempt_write_failure_does_not_trigger_paid_retry():
    client = FakeClient([json.dumps({"results": [result("a")]})])

    def fail(_):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        asyncio.run(judge.query_rubric_batch(
            client, ["a"], "Task", {}, "Report", max_retries=3, record_attempt=fail,
        ))
    assert len(client.prompts) == 1


def test_transport_failure_does_not_log_exception_credentials():
    client = FakeClient([ConnectionError("secret-key-in-url")])
    records = []
    asyncio.run(judge.query_rubric_batch(
        client, ["a"], "Task", {}, "Report", max_retries=1, record_attempt=records.append,
    ))
    assert records[0]["status"] == "request_error"
    assert "ConnectionError" in json.dumps(records)
    assert "secret-key-in-url" not in json.dumps(records)
