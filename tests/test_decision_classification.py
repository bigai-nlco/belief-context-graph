"""Offline ablation contract: real prompts, parsing, projector and accounting; no API key or network."""

import json
import os
from pathlib import Path

import httpx
import pytest

from bcg.config.loader import defaults_dict
from bcg.config.schema import BCGSettings
from bcg.construct.unified import decisions, extract, llm, stream
from bcg.construct.unified.prompts import build_node_extraction_prompt
from bcg.construct.unified.task_state import classify_node, task_history


@pytest.fixture
def wire(monkeypatch):
    monkeypatch.setattr("bcg.core.env.load_project_env", lambda: {})
    monkeypatch.setenv("TEST_DECISION_KEY", "fixture-not-real")
    config = dict(
        mode="decision_model",
        model="gpt-6-luna-decision",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_DECISION_KEY",
    )
    requests = []
    tracker = llm.TokenUsageTracker()
    tok = llm.bind_usage_tracker(tracker)

    def make(select):
        def post(url, *, headers, json, **kwargs):
            body = json
            assert url == "https://example.invalid/v1/decisions"
            assert headers == {"Authorization": "Bearer fixture-not-real"}
            assert body["model"] == "gpt-6-luna-decision" and "messages" not in body
            rows = json_load(body["input"])
            answers = []
            for q in body["questions"]:
                _, n, field = q["name"].split("_", 2)
                choice = select(field, rows[int(n)])
                assert choice in [c["value"] for c in q["choices"]]
                answers.append(
                    {"type": "choice", "choice": choice, "name": q["name"], "confidence": 1.0}
                )
            requests.append(body)
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "model": "gpt-6-luna",
                    "answers": answers[::-1],
                    "usage": {"input_tokens": 14, "output_tokens": 0, "total_tokens": 14},
                },
            )

        return decisions.DecisionClassifier(config, post=post)

    json_load = json.loads
    try:
        yield make, config, requests, tracker
    finally:
        llm.unbind_usage_tracker(tok)


def test_nodes_extract_then_classify_and_verify_parent(wire, monkeypatch):
    make, _, calls, tracker = wire
    parent = {"id": 8, "task_role": "task", "belief": "Compare sales in Peru and Chile."}
    source = "Report on Peru only."

    def generate(_client, _model, prompt, **kw):
        schema = prompt.split("## Output (JSON only)\n")[-1].split("## Current turn")[0]
        assert '"nodes"' in schema and '"entities"' in schema and '"parent_quote"' in schema
        assert '"task_role"' not in schema and '"kind"' not in schema and "## Stance" not in prompt
        return json.dumps(
            {
                "nodes": [
                    {
                        "text": source,
                        "entities": ["Peru"],
                        "supporting_sentence_indices": [0],
                        "task_role": "plan",
                        "stance": "unknown",  # unsolicited classes MUST NOT win
                        "task_ids": [8],
                        "task_refinements": [
                            {
                                "parent_id": 8,
                                "parent_quote": "Chile",
                                "child_quote": "Peru only",
                                "kind": "new_deliverable",
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(extract.llm, "call_model", generate)

    def class_result(kind, quote):
        classifier = make(
            lambda f, r: dict(stance="confirmed", node_type="belief", task_role="task", kind=kind)[
                f
            ]
        )
        out = extract.extract_nodes(
            None,
            "chat-graph",
            role="assistant",
            content=source,
            sentences=[source],
            graph_nodes_str=task_history([], [parent], 12000),
            task_state_enabled=True,
            classifier=classifier,
        )
        (item,) = out["nodes"]
        obj = {
            "role": "assistant",
            "belief": item["belief"],
            "node_type": "belief",
            "supporting_excerpts": [quote],
        }
        classify_node(obj, item, [parent])
        assert (
            item["entities"] == ["Peru"]
            and item["stance"] == "confirmed"
            and item["supporting_sentence_indices"] == [0]
        )
        return obj

    assert class_result("narrower_scope", source)["task_refinements"][0]["kind"] == "narrower_scope"
    assert class_result("none", source)["task_role"] == "other"
    assert class_result("narrower_scope", "Unrelated quote.")["task_role"] == "other"
    assert len(calls) == 6 and tracker.llm_totals()["total_tokens"] == 84
    assert tracker.llm_cost({"input_per_1k": 99, "output_per_1k": 99})["unpriced_calls"] == 6
    summary = tracker.summary({"input_per_1k": 99, "output_per_1k": 99})
    assert summary["decision_totals"] == {
        "n_calls": 6,
        "input_tokens": 84,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 84,
    }
    assert "estimated_cost" not in summary


def test_conditional_status_and_final_decision(wire):
    make, _, calls, _ = wire

    def label(f, r):
        final = "final" in r["text"]
        return dict(
            stance="possible",
            node_type="decision" if final else "belief",
            task_role="progress",
            task_status="blocked",
        )[f]

    classifier = make(label)
    result = classifier.nodes(
        [
            {"text": t, "supporting_excerpts": [t]}
            for t in ("blocked on access", "final answer is A")
        ],
        role="assistant",
        sentences=[],
        content="...",
        task_state=True,
        shown_tasks={},
    )
    assert result["beliefs"][0]["task_status"] == "blocked"
    assert result["decisions"][0]["decision"] == "final answer is A"
    assert "task_status" not in result["decisions"][0]
    assert [len(c["questions"]) for c in calls] == [6, 1]
    calls.clear()
    classifier.nodes(
        [{"text": "ask"}],
        role="user",
        sentences=[],
        content="ask",
        task_state=False,
        shown_tasks={},
    )
    assert [q["name"] for q in calls[0]["questions"]] == ["item_0_stance"]


@pytest.mark.parametrize("layered", [True, False])
def test_directed_edge_classification_and_none_veto(wire, monkeypatch, layered):
    make, _, calls, tracker = wire
    answers = [
        {"from": 2, "to": 1, "type": "contradicts", "note": "source explains value"},
        {"from": 2, "to": 3},
        {"from": 2, "to": 99},
        {"from": 2, "to": 2},
        {"from": 1, "to": 3},
    ]

    def generate(_client, _model, prompt, **_):
        assert '"type":' not in prompt.split("## Output (JSON only)")[-1]
        return json.dumps({"selected_previous_layer": 1, "relations": answers})

    monkeypatch.setattr(extract.llm, "call_model", generate)
    llm.USAGE.set_label("t3.relations")
    common = dict(
        role="assistant" if layered else "tool",
        content="",
        current_nodes_str='[{"id":2,"content":"answer"}]',
        previous_nodes_str='[{"id":1,"content":"source"},{"id":3,"content":"distractor"}]',
        new_node_ids={2},
    )

    def rel(decide):
        classifier = make(lambda _f, row: decide if row["B"] == "source" else "none")
        if layered:
            return extract.extract_layered_relations(
                None, "chat", **common, candidate_layers=[], classifier=classifier
            )
        return extract.extract_relations(None, "chat", **common, classifier=classifier)

    assert rel("depends_on")["relations"] == [
        {"from": 2, "to": 1, "type": "depends_on", "note": "source explains value"}
    ]
    vetoed = rel("none")
    assert vetoed["relations"] == [] and vetoed.get("selected_previous_layer") is None
    assert len(calls[0]["questions"]) == 2 and ".decisions:relations" in tracker.records[0]["label"]


def test_studio_profile_and_baseline_no_decision_requests(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("Baseline used Decisions")

    monkeypatch.setattr(decisions, "DecisionClassifier", forbidden)
    opts = stream.StreamOptions()
    opts.apply_belief_graph_config(defaults_dict()["pipeline"])
    builder = stream.StreamingBeliefBuilder(
        client=None, model="graph", options=opts, item_id="test", out_dir=tmp_path
    )
    monkeypatch.setattr(
        llm,
        "call_model",
        lambda *a, **kw: (
            '{"beliefs":[{"belief":"Request","stance":"confirmed","entities":[],"supporting_sentence_indices":[0]}]}'
        ),
    )
    builder.ingest_turn("user", "Request")
    assert builder.graph.active()[0]["stance_model"] == "graph_model:graph"
    p = build_node_extraction_prompt("user", task_state_enabled=True, sentences_block="Request")
    assert "## Stance" in p and '"beliefs"' in p and "## Task-state annotations" in p
    cfg = defaults_dict()
    cfg["pipeline"]["classification"]["mode"] = "decision_model"
    cfg["backend"] = "hybrid"
    with pytest.raises(ValueError, match="unified"):
        BCGSettings.model_validate(cfg)


def test_initial_query_decision_arm_only_classifies_stance(tmp_path, monkeypatch, wire):
    make, _, requests, _ = wire
    builder = stream.StreamingBeliefBuilder(client=None, model="fixture", item_id="opening",
        out_dir=tmp_path, options=stream.StreamOptions(task_state_enabled=True, incremental_merge=False))
    builder.classifier = make(lambda field, _row: {"stance": "confirmed"}[field])
    def respond(_c, _m, prompt, **kwargs):
        assert "task_role" not in prompt and "task_refinements" not in prompt
        return json.dumps({"nodes": [{"text": "Compare Nepal and India.", "entities": ["Nepal", "India"],
                                     "supporting_sentence_indices": [0]}]})
    monkeypatch.setattr(extract.llm, "call_model", respond)
    builder.ingest_turn("user", "Compare Nepal and India.")
    assert [[q["name"] for q in call["questions"]] for call in requests] == [["item_0_stance"]]
    assert builder.graph.active()[0]["task_role"] == "other"
    assert builder.graph.snapshot()["task_state"]["nodes"] == []


@pytest.mark.parametrize("batch", [True, False])
def test_tool_facts_keep_own_evidence_and_rules_remain_free(wire, monkeypatch, batch):
    make, _, calls, _ = wire

    def select(field, row):
        assert field == "stance" and len(row["source_evidence"]) == 1
        assert row["text"] in row["source_evidence"][0]["title"]
        return "probable"

    classifier = make(select)
    items = [
        {
            "content": "<tool_result>"
            + json.dumps(
                {
                    "name": "web_search",
                    "tool_call_id": "id-" + title,
                    "content": "## Search Results\n[1] **"
                    + title
                    + "**\nThe building opened in 2024.\nURL: https://example.org/building\n",
                }
            )
            + "</tool_result>",
            "query": title,
        }
        for title in ("Museum", "Library")
    ]

    def respond(_client, _model, prompt, **kw):
        assert "classify stance" not in prompt and '"stance"' not in prompt
        facts = [
            [
                {
                    "belief": item["query"],
                    "stance": "unknown",
                    "entities": ["building"],
                    "source_rank": 1,
                }
            ]
            for item in items
        ]
        return json.dumps(
            {
                "items": [
                    {"item_index": 1, "beliefs": facts[1]},
                    {"item_index": 0, "beliefs": facts[0]},
                ]
            }
            if batch
            else {"beliefs": facts[0]}
        )

    monkeypatch.setattr(llm, "call_model", respond)
    if batch:
        results = extract.extract_compact_tool_result_nodes_batch(
            None, "text", items=items, mode="excerpt", classifier=classifier
        )
    else:
        results = [
            extract.extract_compact_tool_result_nodes(
                None, "text", role="tool", mode="excerpt", classifier=classifier, **items[0]
            )
        ]
    for result, item in zip(results, items, strict=batch):
        (node,) = result["nodes"]
        assert (node["tool_call_id"], node["stance"], node["entities"]) == (
            "id-" + item["query"],
            "probable",
            ["building"],
        )
    assert len(calls) == 1
    tool = '<tool_call>{"name":"web_search","arguments":{"query":"Museum"},"id":"id-Museum"}</tool_call>'
    result = extract.extract_nodes(
        None, "text", role="assistant", content=tool, mode="excerpt", classifier=classifier
    )
    assert result["nodes"][0]["extraction_method"] == "rule_tool_call" and len(calls) == 1


def test_request_batches_are_bounded_and_name_matched(wire):
    make, _, calls, _ = wire
    cls = make(lambda f, r: "confirmed" if r["text"] == "0" else "possible")
    nodes = cls.nodes(
        [{"text": str(i)} for i in range(33)],
        role="user",
        content="",
        sentences=[],
        task_state=False,
        shown_tasks={},
    )["beliefs"]
    assert [len(c["questions"]) for c in calls] == [16, 16, 1]
    assert [n["stance"] for n in nodes] == ["confirmed", *(["possible"] * 32)]


@pytest.mark.asyncio
async def test_decision_service_error_cannot_silently_run_the_agent_on_full_context(tmp_path):
    from tests.test_react_bcg import history
    from workflows.stateful_react_agent.bcg.client import DecisionGraphError, HttpGraphBackend
    from workflows.stateful_react_agent.bcg.runtime import BCGMemory, BCGSettings

    def error(_):
        return httpx.Response(
            502, json={"code": "decision_classification_failed", "error": "no label"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(error)) as client:
        backend = HttpGraphBackend("http://graph.test", 1, client=client)
        memory = BCGMemory(BCGSettings(), tmp_path, "synthetic", backend=backend)
        with pytest.raises(DecisionGraphError, match="decision_classification_failed"):
            await memory.prepare(history(3))
        assert memory.last_event["error"] == "decision_classification_failed"
        assert memory.last_event["fallback"] is None


def test_refused_missing_duplicate_and_invalid_are_not_baseline_fallback(wire):
    _, config, _, tracker = wire
    q = decisions.Question("stance", "support?", decisions.STANCES)
    good = {"type": "choice", "name": "item_0_stance", "choice": "possible"}
    for answers in ([], [good, good], [{**good, "type": "refusal"}], [{**good, "choice": "maybe"}]):

        def post(url, answers=answers, **kw):
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "answers": answers,
                    "usage": {"input_tokens": 11, "output_tokens": 0, "total_tokens": 11},
                },
            )

        with pytest.raises(decisions.DecisionError):
            decisions.DecisionClassifier(config, post=post).ask("nodes", [{"text": "X"}], [q])
    assert tracker.llm_totals()["input_tokens"] == 44


@pytest.mark.skipif(
    os.getenv("BCG_DECISION_LIVE_SMOKE") != "1", reason="opt-in paid synthetic Graph smoke test"
)
def test_live_synthetic_graph(tmp_path):
    profile = str(
        Path(__file__).resolve().parents[1] / "tools/case_study_studio/profiles/graph.yaml"
    )
    cfg = llm.load_config(profile, model_key="graph")
    options = stream.StreamOptions(incremental_merge=False)
    options.apply_belief_graph_config(llm.load_belief_graph_config(profile, model_key="graph"))
    options.classification["mode"] = "decision_model"
    usage = llm.TokenUsageTracker()
    token = llm.bind_usage_tracker(usage)
    try:
        b = stream.StreamingBeliefBuilder(
            client=llm.make_client(cfg),
            model=cfg["model"],
            item_id="synthetic",
            out_dir=tmp_path,
            options=options,
            max_tokens=1600,
        )
        b.ingest_turn(
            "user",
            "A fictional archive lists Cedar Hall's opening in 2001 and Harbor Hall's in 2005. Which opened first?",
        )
        b.ingest_turn(
            "assistant",
            "My final answer is Cedar Hall. The archive dates put its opening four years before Harbor Hall.",
        )
        stages = [(r["label"], r["total_tokens"]) for r in usage.records]
        print(
            "Synthetic graph:",
            [(n["id"], n["node_type"], n.get("task_role")) for n in b.graph.active()],
            "edges:",
            [(e["from_id"], e["to_id"], e["type"]) for e in b.graph.relations],
            "calls:",
            stages,
        )
        assert any("extract:user.decisions:nodes" in k for k, _ in stages)
        assert any("extract:assistant.decisions:nodes" in k for k, _ in stages)
        assert any(".decisions:relations" in k for k, _ in stages)
    finally:
        llm.unbind_usage_tracker(token)
