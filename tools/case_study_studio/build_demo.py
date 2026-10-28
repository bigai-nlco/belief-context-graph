"""Package an explicitly selected real run as a sanitized, hash-checked demo.

Only the listed evidence files are included. Logs, runtime profiles, credentials,
worktree files and other experiments are never recursively copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SECRET_KEY = re.compile(r"(api[_-]?key|authorization|access[_-]?token|password|secret)", re.I)
SNAPSHOT_FIELDS = ("generated_at", "problem_id", "stream_turn_index", "n_turns_ingested",
                   "stage", "finalized", "n_nodes", "n_beliefs", "n_decisions", "relations")
NODE_FIELDS = ("id", "node_type", "belief", "decision", "content", "confidence", "stance",
               "role", "entities", "event_time", "source", "evidence_ids", "tool_name",
               "tool_call_id", "query", "extraction_method")


def snapshot_view(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Keep each recorded graph, omitting repeated heavyweight audit collections.

    Node content, confidence and relations are copied from THAT snapshot. Never
    reconstruct a historical graph using the final graph. Complete final evidence
    and confidence histories are separately retained in agent/bcg/state.json.
    """
    view = {key: snapshot[key] for key in SNAPSHOT_FIELDS if key in snapshot}
    view["nodes"] = [{key: node[key] for key in NODE_FIELDS if key in node}
                     for node in snapshot["nodes"]]
    return view


def sanitize(value: Any, secrets: list[str], home: str) -> Any:
    if isinstance(value, dict):
        return {k: "[REDACTED]" if SECRET_KEY.search(k) else sanitize(v, secrets, home)
                for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, secrets, home) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value.replace(home + "/", "<local-home>/")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("demo-case.json"))
    args = parser.parse_args()
    run = args.run.resolve()
    result = json.loads((run / "result.json").read_text())
    if result["status"] != "completed" or result.get("stopped_by") == "llm_error":
        raise SystemExit("Refusing to publish an incomplete or failed task.")
    state = json.loads((run / "agent/bcg/state.json").read_text())
    graph_id = state["problem_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", graph_id):
        raise SystemExit("Invalid graph id")
    events = [json.loads(line) for line in (run / "agent/trajectories/react_agent.jsonl").read_text().splitlines() if line.strip()]
    calls = [e for e in events if e.get("t") == "llm"]
    search_calls = sum(tc.get("name", tc.get("function", {}).get("name")) == "web_search"
                       for e in calls for tc in e.get("tool_calls", []))
    # Graph usage in native state is nested under totals; derive this summary
    # from the recorded responses, including the final flush, not estimates.
    totals = state["snapshot"]["token_usage"]["llm_totals"]
    usage = result["usage"]
    usage["graph"] = {k: totals[k] for k in ("input_tokens", "output_tokens", "total_tokens")}
    usage["combined"] = {k: usage["agent"][k] + usage["graph"][k] if usage["agent"].get(k) is not None else None
                         for k in usage["graph"]}
    usage["accounting"] = "Observed LLM response usage. Input includes cache; output includes reasoning. Graph includes final flush and excludes local embedding tokens, which remain in the source snapshot."
    result["search_calls"] = search_calls

    secrets = []
    for line in (Path.home() / ".bcg/.env").read_text().splitlines():
        key, sep, value = line.removeprefix("export ").partition("=")
        if sep and SECRET_KEY.search(key) and len(value.strip().strip("'\"")) >= 12:
            secrets.append(value.strip().strip("'\""))
    home = str(Path.home())
    sources = [
        ("result.json", result),
        ("report.md", None),
        ("agent/trajectories/react_agent.jsonl", None),
        ("agent/bcg/requests.jsonl", None),
        ("agent/bcg/state.json", None),
        (f"graphs/{graph_id}/belief_graph.jsonl", None),
    ]
    files = []
    for path, derived in sources:
        raw = (run / path).read_bytes()
        if path.endswith(".jsonl"):
            rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            if path.endswith("/belief_graph.jsonl"):
                rows = [snapshot_view(row) for row in rows]
            text = "\n".join(json.dumps(sanitize(row, secrets, home), ensure_ascii=False)
                             for row in rows) + "\n"
        elif path.endswith(".json"):
            data = derived if derived is not None else json.loads(raw)
            text = json.dumps(sanitize(data, secrets, home), ensure_ascii=False, indent=2) + "\n"
        else:
            text = sanitize(raw.decode(), secrets, home)
        if any(secret in text for secret in secrets):
            raise SystemExit(f"Credential found in {path}")
        encoded = text.encode()
        files.append({"path": path, "text": text, "size": len(encoded),
                      "digest": hashlib.sha256(encoded).hexdigest(),
                      "originalDigest": hashlib.sha256(raw).hexdigest(),
                      "projection": "snapshot display fields; complete final graph in state.json" if path.endswith("/belief_graph.jsonl") else None})
    bundle = {
        "format": "bcg-real-case-demo", "version": 1, "taskId": result["task_id"],
        "provenance": {
            "status": "completed", "kind": "real_execution",
            "runDate": result["started_at"][:10], "startedAt": result["started_at"],
            "finishedAt": result["finished_at"], "graphModel": "gpt-6-luna",
            "provider": "LiteLLM", "agentEffort": "low", "graphEffort": "none",
            "recentTurns": 2, "graphChars": 6600, "searchCalls": search_calls,
            "generatedAt": datetime.now(UTC).isoformat(),
            "sourcePolicy": "Actual run outputs; credentials and local home paths redacted. Snapshot timeline keeps all recorded node contents, confidence and relations, omitting repeated audit collections; state.json preserves complete final graph and evidence. Graph LLM usage includes final flush and excludes local embedding tokens. No benchmark score or annotations added.",
        },
        "files": files,
    }
    args.output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"task": result["task_id"], "files": len(files),
                      "size": args.output.stat().st_size,
                      "usage": {k: usage[k] for k in ("agent", "graph", "combined")},
                      "search_calls": search_calls}, ensure_ascii=False))


if __name__ == "__main__":
    main()
