"""Read-only audit plus private verification.json for completed gateway tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bcg.integrations.codex.config import private_write
from bcg.integrations.codex_proxy.protocol import partition


def check(root: Path) -> dict:
    run = json.loads((root / "run.json").read_text())
    if run["mode"] != "gateway":
        raise ValueError("This checker only supports gateway runs")
    exit_data = json.loads((root / "exit.json").read_text())
    rows = []
    for path in root.glob("sessions/**/*.jsonl"):
        rows.extend(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    state = root / "workspace/.bcg-codex/state"
    completions = [
        r["payload"]
        for r in rows
        if r.get("payload", {}).get("type") == "task_complete"
        and r["payload"].get("last_agent_message")
    ]
    compacted = [r for r in rows if r.get("type") == "compacted"]
    report = {
        "mode": run["mode"],
        "codex_exit": exit_data["exit_code"],
        "global_config_and_auth_unchanged": exit_data[
            "global_config_and_auth_unchanged"
        ],
        "completed_user_turns": len(completions),
        "native_compactions": len(compacted),
        "final_answers": [x["last_agent_message"] for x in completions],
        "agent": {"model": run["agent_model"], "reasoning_effort": run["agent_effort"]},
        "graph": {"model": run["graph_model"], "reasoning_effort": run["graph_effort"]},
    }
    assert report["codex_exit"] == 0
    assert report["global_config_and_auth_unchanged"]
    assert len(completions) >= 2
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "cost_complete": True,
        "unpriced_calls": 0,
    }
    for path in (state / "graphs").glob("*/belief_graph_latest.json"):
        graph = json.loads(path.read_text())
        totals = graph.get("token_usage", {}).get("llm_totals", {})
        for key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        ):
            usage[key] += totals.get(key, 0)
        cost = graph.get("token_usage", {}).get("cost", {})
        usage["cost_usd"] += cost.get("total", 0)
        usage["cost_complete"] = usage["cost_complete"] and bool(cost.get("complete"))
        usage["unpriced_calls"] += cost.get("unpriced_calls", 0)
    if not usage["cost_complete"]:
        # Some providers return token usage but no pricing. Do not label these
        # paid requests as free merely because the accumulator starts at zero.
        usage["cost_usd"] = None
    report["graph_llm_usage"] = usage
    audit = [
        json.loads(line)
        for line in (state / "gateway/requests.jsonl").read_text().splitlines()
    ]
    assert not any(row.get("error") for row in audit), audit
    calls = [row for row in audit if row.get("upstream_status") is not None]
    assert all(row["upstream_status"] == 200 for row in calls)
    assert any(
        row.get("native_compaction_passthrough")
        or (
            row.get("mode") == "native_auxiliary"
            and row.get("native_request_kind") == "compaction"
        )
        for row in audit
    )
    assert any(row.get("mode") == "native_auxiliary" for row in audit)
    requests = []
    for path in (state / "gateway/model_io").glob("*.jsonl"):
        for row in map(json.loads, path.read_text().splitlines()):
            if (
                row.get("event") != "request"
                or row.get("context", {}).get("mode") != "bcg"
            ):
                continue
            before, after, ctx = row["incoming"], row["forwarded"], row["context"]
            if ctx.get("evicted_steps", 0):
                assert after["input"] == partition(before["input"], 2).retained
                assert "<belief_graph>" in after["instructions"]
                assert after["instructions"].startswith(
                    before.get("instructions") or ""
                )
            for role in ("user", "system", "developer"):
                assert [i for i in before["input"] if i.get("role") == role] == [
                    i for i in after["input"] if i.get("role") == role
                ]
            requests.append(
                {
                    k: ctx.get(k)
                    for k in (
                        "input_items_before",
                        "input_items_after",
                        "evicted_steps",
                        "graph_chars",
                        "fallback",
                    )
                }
            )
    assert any(r["evicted_steps"] for r in requests)
    report["request_context_checks"] = requests
    report["auxiliary_calls_bypassed_graph"] = sum(
        row.get("mode") == "native_auxiliary" for row in audit
    )
    report["mechanical_checks_passed"] = True
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    result = check(args.directory)
    private_write(args.directory / "verification.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "final_answers"},
            ensure_ascii=False,
            indent=2,
        )
    )
