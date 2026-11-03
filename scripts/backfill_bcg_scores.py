"""Add explicitly labeled replay scores to a new Case bundle; preserve selections.

Uses saved per-call queries and acknowledged snapshots with the configured
embedding service. It does not rerun the Agent/PCST, alter source files, or claim
that recomputed scores are contemporaneous observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from dotenv import load_dotenv

from workflows._shared.case_study import _write_bundle, read_bundle
from workflows.stateful_react_agent.bcg.pcst import PCSTSelector
from workflows.stateful_react_agent.bcg.runtime import BCGSettings


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value).encode()).hexdigest()


def match_snapshot(event: dict[str, Any], snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Require an exact acknowledged call/graph; reject distinct duplicate states."""
    matches = {}
    for row in snapshots:
        snapshot = row.get("snapshot", {})
        identity = row.get("problem_id") or snapshot.get("problem_id")
        if (row.get("snapshot_scope") != "acknowledged_update"
                or row.get("call") != event.get("call") or not event.get("problem_id")
                or identity != event["problem_id"]):
            continue
        turns = event.get("graph_n_turns_ingested")
        if turns is not None and snapshot.get("n_turns_ingested") != turns:
            continue
        matches[digest(snapshot)] = snapshot
    if len(matches) != 1:
        raise ValueError(f"Call {event.get('call')}: expected one matching acknowledged snapshot, got {len(matches)}")
    return next(iter(matches.values()))


def replay_event(
    event: dict[str, Any], snapshot: dict[str, Any], settings: BCGSettings,
) -> dict[str, Any]:
    selection = event["selector_result"]
    strategy = selection["strategy"]
    if strategy not in {"query_pcst", "confidence_pcst"}:
        raise ValueError(f"Unsupported strategy: {strategy}")
    query = selection.get("query", {})
    if not isinstance(query.get("question"), str) or not isinstance(query.get("focus"), str):
        raise ValueError("Replay requires the recorded question and focus")
    if strategy == "query_pcst":
        recorded = selection.get("embedding", {})
        expected = {"provider": settings.embedding_provider, "url": settings.embedding_url,
                    "model": settings.embedding_model, "chunk_bytes": settings.embedding_chunk_bytes,
                    "chunking": "utf8_bytes"}
        if any(recorded.get(k) != v for k, v in expected.items()):
            raise ValueError("Configured embedding service/model/chunking differs from the recorded invocation")
    parameters = selection["parameters"]
    configured = replace(settings, retrieval=strategy, embedding_store="memory",
                         pcst_confidence_floor=parameters["confidence_floor"],
                         pcst_edge_cost=parameters["edge_cost"])
    budget = event.get("graph_chars_budget", selection.get("graph_chars_budget"))
    scores = PCSTSelector(configured).score_nodes(snapshot, query["question"], query["focus"], budget)
    nodes = scores["node_scores"]
    if not {str(i) for i in event.get("selected_node_ids", [])} <= nodes.keys():
        raise ValueError("Acknowledged snapshot is missing an injected node")
    differences = [abs(nodes[i]["prize"] - value)
                   for i, value in selection.get("selected_prizes", {}).items()]
    return {
        "kind": "retrospective_recomputation", "version": 1,
        "computed_at": datetime.now(UTC).isoformat(),
        "source_audit_sha256": digest(event), "source_snapshot_sha256": digest(snapshot),
        "node_scores": nodes, "embedding": scores["embedding"],
        "comparison": {"recorded_prizes_checked": len(differences), "tolerance": 1e-5,
                       "max_abs_difference": max(differences, default=0.0),
                       "matches_within_tolerance": all(d <= 1e-5 for d in differences) if differences else None},
        "notice": "事后补算：使用当轮快照和 query；当前 embedding 服务可能与历史不同。原始选点、边、收益、注入文本均保留。",
    }


def backfill(bundle: dict[str, Any], settings: BCGSettings) -> list[dict[str, Any]]:
    reports = []
    for case in bundle["cases"]:
        changed = False
        files = {item["path"]: item for item in case["files"]}
        for path, item in files.items():
            if PurePosixPath(path).name != "requests.jsonl":
                continue
            snapshot_file = files.get(str(PurePosixPath(path).with_name("graph_snapshots.jsonl")))
            if snapshot_file is None:
                raise ValueError(f"Missing historical graph snapshots for {path}")
            snapshots = [json.loads(line) for line in snapshot_file["text"].splitlines() if line.strip()]
            events = [json.loads(line) for line in item["text"].splitlines() if line.strip()]
            updated = False
            for event in events:
                selection = event.get("selector_result")
                if not selection or selection.get("node_scores") or selection.get("score_replay"):
                    continue
                snapshot = match_snapshot(event, snapshots)
                replay = replay_event(event, snapshot, settings)
                selection["score_replay"] = replay
                reports.append({"task_id": case["task_id"], "source": path, "call": event["call"],
                                "nodes": len(replay["node_scores"]), **replay["comparison"]})
                print(encode(reports[-1]), flush=True)
                updated = changed = True
            if updated:
                item["text"] = "\n".join(encode(event) for event in events) + "\n"
                raw = item["text"].encode()
                item["size"] = len(raw)
                item["digest"] = hashlib.sha256(raw).hexdigest()
        if changed:
            case.setdefault("warnings", []).append(
                "评分补算版：缺失的 relevance/prize 来自事后补算；原始选图、注入文本和已有评分保留。前三轮等未运行检索的调用不会补造评分。")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if len({p.resolve() for p in (args.bundle, args.out, args.report)}) != 3:
        parser.error("Input bundle, output bundle and report must be distinct")
    if args.out.exists() or args.report.exists():
        parser.error("Output already exists; choose a new path")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    settings = BCGSettings.from_agent({"bcg": {"embedding_store": "memory"}})
    bundle = read_bundle(args.bundle)
    for case in bundle["cases"]:
        for item in case["files"]:
            raw = item["text"].encode()
            if len(raw) != item["size"] or hashlib.sha256(raw).hexdigest() != item["digest"]:
                raise ValueError(f"Invalid source file integrity: {item['path']}")
    reports = backfill(bundle, settings)
    _write_bundle(args.out, bundle["cases"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"source_bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
                                      "rounds": reports}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
