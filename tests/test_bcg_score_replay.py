"""Historical score enrichment must preserve observations and graph identity."""

import copy
import hashlib
import json

import pytest

from scripts.backfill_bcg_scores import backfill, encode, match_snapshot, replay_event
from workflows.stateful_react_agent.bcg.pcst import PCSTSelector
from workflows.stateful_react_agent.bcg.runtime import BCGSettings


def records():
    settings = BCGSettings()
    event = {"call": 4, "problem_id": "graph-a", "graph_n_turns_ingested": 7,
             "selected_node_ids": [1], "selected_relation_ids": [], "graph_chars": 100,
             "selector_result": {"strategy": "query_pcst", "selected_prizes": {"1": .3},
                                 "objective": .3, "query": {"question": "q", "focus": "f"},
                                 "parameters": {"confidence_floor": .2, "edge_cost": .2},
                                 "embedding": {"provider": "openai", "url": settings.embedding_url,
                                               "model": settings.embedding_model, "chunk_bytes": 240,
                                               "chunking": "utf8_bytes"}}}
    snapshot = {"call": 4, "problem_id": "graph-a", "snapshot_scope": "acknowledged_update",
                "snapshot": {"problem_id": "graph-a", "n_turns_ingested": 7,
                             "nodes": [{"id": 1, "belief": "fact"}, {"id": 2, "belief": "other"}],
                             "relations": []}}
    return event, snapshot


def test_matching_rejects_other_graphs_turns_and_distinct_duplicate_snapshots():
    event, row = records()
    assert match_snapshot(event, [row, copy.deepcopy(row)]) == row["snapshot"]
    for field, value in (("problem_id", "other"), ("call", 5), ("snapshot_scope", "final")):
        changed = copy.deepcopy(row)
        changed[field] = value
        with pytest.raises(ValueError, match="matching acknowledged"):
            match_snapshot(event, [changed])
    changed = copy.deepcopy(row)
    changed["snapshot"]["nodes"][0]["belief"] = "different"
    with pytest.raises(ValueError, match="got 2"):
        match_snapshot(event, [row, changed])
    event["graph_n_turns_ingested"] = 8
    with pytest.raises(ValueError):
        match_snapshot(event, [row])


def test_replay_never_changes_original_observations_and_checks_recorded_prizes(monkeypatch):
    event, snapshot = records()
    before = copy.deepcopy(event)
    monkeypatch.setattr(PCSTSelector, "score_nodes", lambda *_: {
        "node_scores": {"1": {"prize": .31, "relevance": .5}, "2": {"prize": 0, "relevance": 0}},
        "embedding": {"model": "test"}})
    replay = replay_event(event, snapshot["snapshot"], BCGSettings())
    assert event == before
    assert replay["kind"] == "retrospective_recomputation"
    assert replay["comparison"]["matches_within_tolerance"] is False
    assert replay["comparison"]["max_abs_difference"] == pytest.approx(.01)
    assert replay["node_scores"]["2"]["prize"] == 0
    event["selector_result"]["embedding"]["model"] = "another-model"
    with pytest.raises(ValueError, match="differs"):
        replay_event(event, snapshot["snapshot"], BCGSettings())


def test_bundle_replay_preserves_original_fields_and_only_changes_request_digest(monkeypatch):
    event, snapshot = records()
    monkeypatch.setattr(PCSTSelector, "score_nodes", lambda *_: {
        "node_scores": {"1": {"prize": .3}, "2": {"prize": 0}}, "embedding": {}})
    def file(path, rows):
        text = "\n".join(encode(row) for row in rows) + "\n"
        return {"path": path, "text": text, "size": len(text.encode()),
                "digest": hashlib.sha256(text.encode()).hexdigest()}
    early = {"call": 1, "selected_node_ids": []}
    bundle = {"cases": [{"task_id": "35", "files": [file("agent/bcg/requests.jsonl", [early, event]),
                                                       file("agent/bcg/graph_snapshots.jsonl", [snapshot])]}]}
    original = copy.deepcopy(bundle)
    report = backfill(bundle, BCGSettings())
    assert len(report) == 1 and report[0]["nodes"] == 2
    files = bundle["cases"][0]["files"]
    assert files[1] == original["cases"][0]["files"][1]
    events = [json.loads(line) for line in files[0]["text"].splitlines()]
    assert events[0] == early
    del events[1]["selector_result"]["score_replay"]
    assert events[1] == event
    assert files[0]["digest"] == hashlib.sha256(files[0]["text"].encode()).hexdigest()
