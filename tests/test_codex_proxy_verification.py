"""Verifier regression fixtures are deliberately synthetic, not live results."""

from __future__ import annotations

import json

import pytest

from bcg.integrations.codex_proxy.protocol import partition
from bcg.integrations.codex_proxy.verification import verify_bcg_run
from scripts.codex_browsecomp_smoke import save


def fixture_run(root):
    save(
        root / "result.json",
        {
            "auth_only": False,
            "codex_exit_code": 0,
            "model": "test-model",
            "agent_reasoning_effort": "low",
            "recent_turns": 2,
        },
    )
    save(root / "final-answer.txt", "Synthetic answer for verifier unit test")
    save(root / "codex-events.jsonl", '{"type":"turn.completed"}\n')
    items = [{"role": "user", "content": "Synthetic question"}]
    for i in range(3):
        items.extend(
            [
                {
                    "type": "function_call",
                    "name": "web_search",
                    "call_id": str(i),
                    "arguments": '{"query":"test"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": str(i),
                    "output": "URL: https://example.invalid\nSnippet: Test fact",
                },
            ]
        )
    original = {"model": "test-model", "reasoning": {"effort": "low"}, "input": items}
    outgoing = {
        **original,
        "input": partition(items, 2).retained,
        "instructions": "<belief_graph>\nTest fact\n</belief_graph>",
    }
    context = {
        "request_index": 1,
        "mode": "bcg",
        "fallback": None,
        "evicted_steps": 1,
        "graph_chars": 40,
        "selected_node_ids": [1],
        "graph_turns_written": 3,
        "problem_id": "codex_fixture",
    }
    save(root / "proxy/requests.jsonl", json.dumps(context) + "\n")
    records = [
        {
            "event": "request",
            "incoming": original,
            "forwarded": outgoing,
            "context": context,
        },
        {
            "event": "response",
            "status": 200,
            "terminal": "response.completed",
            "response": {"output": []},
        },
    ]
    save(
        root / "proxy/model_io/fixture.jsonl",
        "\n".join(json.dumps(r) for r in records) + "\n",
    )
    save(
        root / "graphs/codex_fixture/belief_graph_latest.json",
        {"nodes": [{"id": 1, "belief": "Test fact"}]},
    )
    save(
        root / "graphs/codex_fixture/trajectory.json",
        {"messages": [{"role": "user", "content": "Synthetic question"}]},
    )
    return records


def test_fixture_gate_accepts_complete_mechanics_but_disclaims_live_proof(tmp_path):
    fixture_run(tmp_path)
    result = verify_bcg_run(tmp_path)
    assert result["passed"], result
    assert "fixtures cannot prove live success" in result["limitations"]


@pytest.mark.parametrize(
    "broken", ["auth_only", "no_graph", "lost_user", "broken_pair", "changed_reasoning"]
)
def test_gate_rejects_false_success_despite_codex_zero_exit(tmp_path, broken):
    records = fixture_run(tmp_path)
    if broken == "auth_only":
        path = tmp_path / "result.json"
        save(path, {**json.loads(path.read_text()), "auth_only": True})
    elif broken == "no_graph":
        (tmp_path / "graphs/codex_fixture/belief_graph_latest.json").unlink()
    elif broken == "lost_user":
        records[0]["forwarded"]["input"].pop(0)
    elif broken == "broken_pair":
        records[0]["forwarded"]["input"].pop()
    elif broken == "changed_reasoning":
        records[0]["forwarded"]["reasoning"] = {"effort": "high"}
    save(
        tmp_path / "proxy/model_io/fixture.jsonl",
        "\n".join(json.dumps(r) for r in records) + "\n",
    )
    assert not verify_bcg_run(tmp_path)["passed"]


def test_gate_rejects_missing_artifacts(tmp_path):
    result = verify_bcg_run(tmp_path)
    assert not result["passed"]
    assert not result["checks"]["artifacts_readable_and_consistent"]
