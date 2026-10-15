"""Conservative artifact gate for the explicitly authorized native Codex test.

This gate checks observed mechanics, not answer correctness or Graph quality.
It cannot turn mocked fixtures or a login-only probe into a live BCG result.
"""

from __future__ import annotations

import json
from pathlib import Path

from .protocol import CALL_TYPES, RESULT_TYPES, UnsupportedHistory, partition


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _subsequence(wanted: list[dict], actual: list[dict]) -> bool:
    iterator = iter(actual)
    return all(any(item == candidate for candidate in iterator) for item in wanted)


def verify_bcg_run(directory: Path) -> dict:
    """Require positive evidence of BCG, not merely a zero Codex exit code.

    Run after all proxy handlers have finished. Missing/malformed artifacts
    fail closed. Opaque reasoning and other retained items are compared as
    data; none of their contents are emitted in the verification report.
    """
    checks: dict[str, bool] = {}
    metrics = {"requests": 0, "graph_writes": 0, "injected_requests": 0}
    try:
        result = json.loads((directory / "result.json").read_text())
        checks["benchmark_not_auth_probe"] = not result.get("auth_only", False)
        checks["codex_exit_zero"] = result.get("codex_exit_code") == 0
        checks["final_answer_present"] = bool(
            (directory / "final-answer.txt").read_text().strip()
        )
        events = _jsonl(directory / "codex-events.jsonl")
        checks["native_codex_completed"] = any(
            e.get("type") == "turn.completed" for e in events
        ) and not any(e.get("type") in {"error", "turn.failed"} for e in events)
        audit = _jsonl(directory / "proxy/requests.jsonl")
        checks["bcg_without_fallback"] = bool(audit) and all(
            row.get("mode") == "bcg"
            and row.get("request_index")
            and not row.get("error")
            and not row.get("fallback")
            for row in audit
        )
        metrics["requests"] = len(audit)
        metrics["graph_writes"] = sum(
            row.get("graph_turns_written", 0) for row in audit
        )
        injected = [row for row in audit if row.get("evicted_steps", 0) > 0]
        metrics["injected_requests"] = len(injected)
        checks["graph_acknowledged"] = metrics["graph_writes"] > 0
        checks["eviction_and_selection_exercised"] = bool(injected) and all(
            row.get("graph_chars", 0) > 0 and row.get("selected_node_ids")
            for row in injected
        )

        request_files = list((directory / "proxy/model_io").glob("*.jsonl"))
        traces = [_jsonl(path) for path in request_files]
        requests = [r for trace in traces for r in trace if r.get("event") == "request"]
        responses = [
            r for trace in traces for r in trace if r.get("event") == "response"
        ]
        checks["complete_io_record"] = (
            bool(requests)
            and len(requests) == len(audit) == len(responses)
            and all(
                len([r for r in trace if r.get("event") == "request"]) == 1
                and len([r for r in trace if r.get("event") == "response"]) == 1
                for trace in traces
            )
        )
        checks["responses_completed"] = bool(responses) and all(
            r.get("status") == 200
            and r.get("terminal") == "response.completed"
            and isinstance(r.get("response"), dict)
            for r in responses
        )
        checks["requested_model_and_effort"] = bool(requests) and all(
            r["forwarded"].get("model") == result["model"]
            and r["forwarded"].get("reasoning", {}).get("effort")
            == result["agent_reasoning_effort"]
            for r in requests
        )
        pinned_ok = retained_ok = paired_ok = instructions_ok = True
        steps_ok = True
        seen_search_result = False
        graph_blocks = 0
        for record in requests:
            original, forwarded = record["incoming"], record["forwarded"]
            before, after = original["input"], forwarded["input"]
            try:
                expected = partition(before, result["recent_turns"])
                steps_ok &= after == expected.retained
            except UnsupportedHistory:
                steps_ok = False
            pinned = [
                i for i in before if i.get("role") in {"system", "developer", "user"}
            ]
            pinned_ok &= _subsequence(pinned, after)
            retained_ok &= _subsequence(after, before)
            instructions = original.get("instructions") or ""
            instructions_ok &= (forwarded.get("instructions") or "").startswith(
                instructions
            )
            for item in after:
                if item.get("type") in RESULT_TYPES:
                    calls = [
                        c
                        for c in after
                        if c.get("type") in CALL_TYPES
                        and c.get("call_id") == item.get("call_id")
                    ]
                    paired_ok &= len(calls) == 1
            # For each originally completed call, its result must survive if
            # the call survives. A parallel batch cannot be partially retained.
            before_results = {
                i["call_id"]: i for i in before if i.get("type") in RESULT_TYPES
            }
            for item in after:
                if (
                    item.get("type") in CALL_TYPES
                    and item.get("call_id") in before_results
                ):
                    paired_ok &= before_results[item["call_id"]] in after
            if record.get("context", {}).get("evicted_steps", 0):
                text = forwarded.get("instructions") or ""
                graph_blocks += int(
                    "<belief_graph>" in text and "</belief_graph>" in text
                )
            for item in before:
                if item.get("type") not in CALL_TYPES:
                    continue
                is_search = "web_search" in item.get("name", "") or (
                    "mcp__search__web_search" in item.get("input", "")
                )
                output = before_results.get(item.get("call_id"), {}).get("output")
                if is_search and output is not None:
                    seen_search_result |= "URL: http" in json.dumps(output)
        checks["pinned_input_unchanged"] = pinned_ok and bool(requests)
        checks["retained_items_unchanged"] = retained_ok and bool(requests)
        checks["retained_calls_have_results"] = paired_ok and bool(requests)
        checks["whole_recent_steps_retained"] = steps_ok and bool(requests)
        checks["original_instructions_preserved"] = instructions_ok and bool(requests)
        checks["graph_in_native_instructions"] = bool(injected) and graph_blocks == len(
            injected
        )
        checks["search_result_observed"] = seen_search_result
        graph_ids = {
            row.get("problem_id") for row in audit if row.get("graph_turns_written", 0)
        }
        graph_ids.discard(None)
        artifacts = []
        for graph_id in graph_ids:
            if not isinstance(graph_id, str) or Path(graph_id).name != graph_id:
                artifacts.append(False)
                continue
            root = directory / "graphs" / graph_id
            snapshot = json.loads((root / "belief_graph_latest.json").read_text())
            nodes = snapshot.get("nodes") or snapshot.get("beliefs", [])
            artifacts.append(
                bool(nodes) and (root / "trajectory.json").stat().st_size > 0
            )
        checks["nonempty_graph_records"] = bool(artifacts) and all(artifacts)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        checks["artifacts_readable_and_consistent"] = False
    return {
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "limitations": "Mechanical gate only. Review answer correctness, Graph semantics, and trace provenance separately; fixtures cannot prove live success.",
    }
