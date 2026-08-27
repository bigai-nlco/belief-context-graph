from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import pytest

from bcg import BCG, BCGMemory, BCGRunner
from bcg.construct._shared.llm import TokenUsageTracker, _coerce_usage
from bcg.construct.hybrid.confidence import (
    init_belief_confidence as init_hybrid_belief_confidence,
)
from bcg.construct.hybrid.confidence import (
    recompute_evidence_confidence_from_node as recompute_hybrid_evidence_confidence,
)
from bcg.construct.hybrid.extractor import ExtractedNode, QwenChunkExtractor
from bcg.construct.hybrid.llm import (
    call_model,
    parse_json_response,
    temperature_request_value,
    thinking_request_options,
)
from bcg.construct.hybrid.online import SessionManager as HybridSessionManager
from bcg.construct.hybrid.split import semantic_breakpoint_chunks, split_sentences
from bcg.construct.hybrid.stance import LocalZeroShotStanceClassifier, StancePrediction
from bcg.construct.unified import BeliefGraphPipeline
from bcg.construct.unified.confidence import (
    init_belief_confidence,
    initial_confidence,
    recompute_evidence_confidence_from_node,
)
from bcg.construct.unified.evidence import evidence_from_excerpt, locate_excerpt
from bcg.construct.unified.extract import (
    extract_compact_tool_result_nodes,
    extract_compact_tool_result_nodes_batch,
    extract_nodes,
    extract_rule_tool_result_nodes,
    format_extraction_nodes,
    format_graph_nodes,
    format_relation_nodes,
)
from bcg.construct.unified.graph import BeliefGraph
from bcg.construct.unified.llm import call_model as call_api_model
from bcg.construct.unified.online import SessionManager as UnifiedSessionManager
from bcg.construct.unified.prompts import (
    build_layered_relation_extraction_prompt,
    build_node_extraction_prompt,
)
from bcg.construct.unified.stream import (
    StreamingBeliefBuilder as UnifiedStreamingBeliefBuilder,
)
from bcg.construct.unified.stream import (
    StreamOptions as UnifiedStreamOptions,
)
from bcg.core.confidence import posterior_confidence
from bcg.core.graph import (
    BCGEdge,
    BCGNode,
    BeliefPayload,
    BeliefSource,
    RelationPayload,
)
from bcg.core.llm import LLMResponse
from bcg.core.runner import _bcg_from_construct, _ConstructClientAdapter

T = TypeVar("T")


def test_graph_usage_tracks_reasoning_and_excludes_embeddings_from_llm_totals() -> None:
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=30,
        total_tokens=130,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=12),
    )
    counts = _coerce_usage(usage)
    tracker = TokenUsageTracker()
    tracker.record(
        model="graph-model",
        prompt_tokens=counts["prompt_tokens"],
        completion_tokens=counts["completion_tokens"],
        reasoning_tokens=counts["reasoning_tokens"],
        total_tokens=counts["total_tokens"],
        label="extractor",
    )
    tracker.record(
        model="embedding-model",
        prompt_tokens=50,
        completion_tokens=0,
        total_tokens=50,
        label="embedding:merge",
    )

    assert tracker.summary()["llm_totals"] == {
        "n_calls": 1,
        "input_tokens": 100,
        "output_tokens": 30,
        "reasoning_tokens": 12,
        "total_tokens": 130,
    }


@pytest.mark.parametrize("manager_type", [HybridSessionManager, UnifiedSessionManager])
def test_online_manager_releases_session_memory(manager_type: type[Any]) -> None:
    manager = object.__new__(manager_type)
    manager._sessions = {"problem": object()}
    manager._sessions_lock = threading.Lock()

    assert manager.release("problem") == {
        "problem_id": "problem",
        "released": True,
    }
    assert manager.release("problem") == {
        "problem_id": "problem",
        "released": False,
    }
    assert manager._sessions == {}


def test_hybrid_parser_accepts_one_json_object_with_trailing_brace() -> None:
    response = '{"beliefs": ["A"], "decisions": []}\n}'

    assert parse_json_response(response) == {
        "beliefs": ["A"],
        "decisions": [],
    }


def test_hybrid_call_model_forwards_json_response_format() -> None:
    captured: dict[str, Any] = {}

    class Completions:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"beliefs": []}'))
                ],
                usage=None,
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    response = call_model(
        client,
        "Qwen3.5-4B",
        "Return JSON.",
        retries=1,
        response_format={"type": "json_object"},
    )

    assert response == '{"beliefs": []}'
    assert captured["response_format"] == {"type": "json_object"}


def test_hybrid_non_thinking_controls_are_provider_safe() -> None:
    assert thinking_request_options("gpt-5.6-luna", enabled=False) == (
        "none",
        None,
    )
    assert thinking_request_options("Qwen3.5-4B", enabled=False) == (
        None,
        {"chat_template_kwargs": {"enable_thinking": False}},
    )
    assert thinking_request_options("gpt-5.6-luna", enabled=True) == (
        "medium",
        None,
    )
    assert temperature_request_value("gpt-5.6-luna", 0) is None
    assert temperature_request_value("Qwen3.5-4B", 0) == 0


@pytest.mark.parametrize(
    ("model", "configured", "expected"),
    [
        ("gpt-5.6-luna", None, "none"),
        ("gpt-5.6-luna", "low", "low"),
        ("gpt-5.5", None, "medium"),
    ],
)
def test_unified_reasoning_effort_is_model_aware_and_configurable(
    model: str,
    configured: str | None,
    expected: str,
) -> None:
    captured: dict[str, Any] = {}

    class Completions:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"beliefs": []}'))
                ],
                usage=None,
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    call_api_model(
        client,
        model,
        "Return JSON.",
        retries=1,
        reasoning_effort=configured,
    )

    assert captured["reasoning_effort"] == expected


def test_unified_event_time_is_assigned_by_graph_builder(tmp_path: Path) -> None:
    prompt = build_node_extraction_prompt(
        "user",
        mode="sentences",
        sentences_block="[0] The event happened yesterday.",
        current_date="2026-08-11",
    )
    assert prompt is not None
    assert '"event_time"' not in prompt
    assert '"time_text"' not in prompt
    assert '"entities"' in prompt
    assert '"stance"' in prompt
    assert "The current turn is dated" not in prompt

    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="gpt-5.6-luna",
        item_id="code-owned-event-time",
        out_dir=tmp_path,
    )
    node = builder._make_node(
        {
            "node_type": "belief",
            "belief": "The event happened yesterday.",
            "stance": "asserted",
            "entities": [],
            # Model-provided temporal metadata must never win.
            "event_time": "1900-01-01T00:00:00Z",
            "time_text": "yesterday",
        },
        {"turn_id": 0, "item_id": "code-owned-event-time"},
        [],
        "user",
    )

    assert node["event_time"] != "1900-01-01T00:00:00Z"
    assert node["event_time"].endswith("+00:00")
    assert "time_text" not in node


def test_unified_uses_stance_and_entities_from_graph_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_nodes",
        lambda *args, **kwargs: {
            "nodes": [
                {
                    "tmp_id": "n0",
                    "node_type": "belief",
                    "belief": "Alice may prefer green tea.",
                    "stance": "speculated",
                    "entities": ["Alice", "green tea"],
                    "supporting_sentence_indices": [0],
                }
            ],
            "beliefs": [],
            "decisions": [],
            "raw_output": '{"beliefs":[{"belief":"Alice may prefer green tea.","stance":"speculated","entities":["Alice","green tea"]}]}',
            "skipped": False,
        },
    )
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_relations",
        lambda *args, **kwargs: {
            "relations": [],
            "raw_output": '{"relations":[]}',
            "skipped": False,
        },
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="graph-model",
        item_id="graph-model-metadata",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(
            incremental_merge=False,
            stance_config={"model_path": "local-stance"},
            entity_config={"method": "ml"},
        ),
        stance_classifier=object(),
        entity_recognizer=object(),
    )

    builder.ingest_turn("user", "Alice may prefer green tea.")

    node = builder.graph.active()[0]
    assert node["stance"] == "speculated"
    assert node["stance_confidence"] == 0.0
    assert node["stance_model"] == "graph_model:graph-model"
    assert node["entities"] == ["Alice", "green tea"]


def test_unified_query_tool_call_is_never_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = (
        '<tool_call>\n{"name":"web_search","arguments":'
        '{"query":"exact query with \\"quotes\\""}}\n</tool_call>'
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("pure tool calls must bypass model extraction")

    monkeypatch.setattr("bcg.construct.unified.extract.llm.call_model", fail_if_called)

    result = extract_nodes(
        object(),
        "graph-model",
        role="assistant",
        mode="excerpt",
        content=content,
    )

    assert len(result["beliefs"]) == 1
    node = result["beliefs"][0]
    assert node["tool_name"] == "web_search"
    assert node["query"] == 'exact query with "quotes"'
    assert node["belief"] == (
        'The assistant is using web_search to search for "exact query with \\"quotes\\"".'
    )
    assert node["tool_arguments"] == {"query": 'exact query with "quotes"'}
    assert node["tool_call_index"] == 0
    assert node["extraction_method"] == "rule_tool_call"
    assert node["stance"] == "asserted"
    assert node["entities"] == ["web_search"]
    assert node["supporting_excerpts"] == [content]


def test_hybrid_query_metadata_is_code_owned() -> None:
    extractor = object.__new__(QwenChunkExtractor)
    extractor.config = {"require_excerpt": False}
    content = (
        '<tool_call>\n{"name":"serper_search","arguments":'
        '{"query":"World Bank savings 2001"}}\n</tool_call>'
    )

    nodes = extractor._parse_response(
        '{"beliefs":[{"belief":"The assistant searches for World Bank savings."}]}',
        chunk_index=0,
        role="assistant",
        chunk_text=content,
    )

    assert len(nodes) == 1
    assert nodes[0].text == (
        'The assistant is using serper_search to search for "World Bank savings 2001".'
    )
    assert nodes[0].tool_name == "serper_search"
    assert nodes[0].query == "World Bank savings 2001"


def test_hybrid_pure_tool_call_bypasses_extractor_client() -> None:
    extractor = object.__new__(QwenChunkExtractor)
    extractor.model = "unused-model"
    extractor.config = {
        "context_scope": "none",
        "enable_thinking": False,
        "include_turn_content": False,
        "require_excerpt": False,
        "dynamic_node_cap": False,
        "node_cap_unit": "char",
        "node_cap_ratio": 1.0,
        "node_cap_min": 1,
        "node_cap_max": 0,
        "max_concurrency": 1,
    }
    extractor._ensure_client = lambda: (_ for _ in ()).throw(
        AssertionError("pure tool calls must not create a model client")
    )
    content = (
        '<tool_call>{"name":"web_search","arguments":'
        '{"query":"exact hybrid query","num":10}}</tool_call>'
    )
    second = (
        '<tool_call>{"name":"read_file","arguments":{"path":"notes.txt"}}</tool_call>'
    )

    groups = extractor.extract_turn(
        [
            SimpleNamespace(text=content, chunk_id=4),
            SimpleNamespace(text=second, chunk_id=5),
        ],
        "assistant",
        turn_content=f"{content}\n{second}",
    )

    assert len(groups) == 2 and all(len(group) == 1 for group in groups)
    node = groups[0][0]
    assert node.text == (
        'The assistant is using web_search to search for "exact hybrid query".'
    )
    assert node.tool_name == "web_search"
    assert node.query == "exact hybrid query"
    assert node.tool_arguments == {"query": "exact hybrid query", "num": 10}
    assert node.extraction_method == "rule_tool_call"
    assert node.stance == "asserted"
    assert node.tool_call_index == 0
    assert groups[1][0].tool_call_index == 1
    assert groups[1][0].tool_arguments == {"path": "notes.txt"}


def test_unified_rule_tool_call_node_has_complete_graph_attributes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_relations",
        lambda *args, **kwargs: {"relations": [], "raw_output": "{}", "skipped": False},
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="unused-model",
        item_id="rule-complete",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(incremental_merge=False),
    )
    content = (
        '<tool_call>{"name":"web_search","arguments":'
        '{"query":"complete attributes"}}</tool_call>'
    )

    builder.ingest_turn("assistant", content)
    node = builder.graph.active()[0]

    required = {
        "id",
        "node_type",
        "belief",
        "stance",
        "role",
        "entities",
        "event_time",
        "source",
        "evidence_ids",
        "supporting_excerpts",
        "tool_name",
        "query",
        "tool_arguments",
        "tool_call_index",
        "tool_call_id",
        "extraction_method",
        "initial_confidence",
        "evidence_confidence",
        "factor_confidence",
        "confidence",
        "confidence_history",
    }
    assert required <= node.keys()
    assert node["tool_call_id"] == "rule-complete:t0:c0"


def test_unified_initial_user_beliefs_skip_internal_relation_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relation_calls: list[str] = []

    def fake_extract_nodes(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        content = str(kwargs["content"])
        return {
            "nodes": [
                {
                    "tmp_id": "n0",
                    "node_type": "belief",
                    "belief": content,
                    "stance": "asserted",
                    "entities": [],
                    "event_time": None,
                    "time_text": None,
                    "supporting_sentence_indices": [0],
                },
                {
                    "tmp_id": "n1",
                    "node_type": "belief",
                    "belief": f"Supporting detail for {content}",
                    "stance": "asserted",
                    "entities": [],
                    "event_time": None,
                    "time_text": None,
                    "supporting_sentence_indices": [0],
                },
            ],
            "raw_output": "{}",
            "skipped": False,
        }

    def fake_extract_relations(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        relation_calls.append(str(kwargs["graph_nodes_str"]))
        return {"relations": [], "raw_output": "{}", "skipped": False}

    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_nodes", fake_extract_nodes
    )
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_relations", fake_extract_relations
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="gpt-5.6-luna",
        item_id="initial-user-root-layer",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(incremental_merge=False),
    )

    initial = builder.ingest_turn("user", "Initial question beliefs.")

    assert relation_calls == []
    assert builder.graph.relations == []
    assert initial["edge_attempts"] == []
    assert (
        initial["edge_skip_reason"]
        == "initial user belief turn does not create internal relations"
    )

    builder.ingest_turn("user", "Later user evidence.")
    assert len(relation_calls) == 1


def test_unified_relation_search_stops_after_four_non_empty_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relation_calls: list[str] = []

    def fake_extract_nodes(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        content = str(kwargs["content"])
        return {
            "nodes": [
                {
                    "tmp_id": "n0",
                    "node_type": "belief",
                    "belief": content,
                    "stance": "asserted",
                    "entities": [],
                    "event_time": None,
                    "time_text": None,
                    "supporting_sentence_indices": [0],
                }
            ],
            "raw_output": "{}",
            "skipped": False,
        }

    def fake_extract_relations(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        relation_calls.append(str(kwargs["graph_nodes_str"]))
        return {"relations": [], "raw_output": "{}", "skipped": False}

    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_nodes", fake_extract_nodes
    )
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_relations", fake_extract_relations
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="gpt-5.6-luna",
        item_id="bounded-history",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(
            incremental_merge=False,
            max_previous_windows=4,
        ),
    )

    for index in range(5):
        builder.ingest_turn("user", f"Historical statement {index}.")
    relation_calls.clear()
    event = builder.ingest_turn("user", "Current statement.")

    assert len(relation_calls) == 4
    assert len(event["edge_attempts"]) == 4
    assert event["edge_window_limit_reached"] is True


def test_unified_layered_relation_prompt_requires_one_previous_layer() -> None:
    prompt = build_layered_relation_extraction_prompt(
        role="assistant",
        content="I will use the newest relevant evidence.",
        graph_nodes='[{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]',
        graph_edges="[]",
        new_node_ids="[4]",
        candidate_layers=json.dumps(
            [
                {"layer": 1, "trajectory_index": 2, "node_ids": [3]},
                {"layer": 2, "trajectory_index": 1, "node_ids": [2]},
                {"layer": 3, "trajectory_index": 0, "node_ids": [1]},
            ]
        ),
    )

    assert '"selected_previous_layer"' in prompt
    assert "ZERO OR ONE previous layer" in prompt
    assert "Layer 1 is the nearest" in prompt
    assert "Never connect nodes from two different previous layers" in prompt
    assert "Shared entities alone do not justify a relation" in prompt
    assert "The user can be charged a late fee" not in prompt


def test_unified_relation_nodes_include_only_id_and_content() -> None:
    rendered = format_relation_nodes(
        [
            {
                "id": 7,
                "node_type": "belief",
                "belief": "A compact semantic fact.",
                "role": "assistant",
                "stance": "speculated",
                "confidence": 0.6,
                "entities": ["fact"],
                "source": {"turn_index": 3},
            }
        ],
        char_budget=None,
    )

    assert json.loads(rendered) == [{"id": 7, "content": "A compact semantic fact."}]


def test_unified_extraction_history_includes_only_content() -> None:
    rendered = format_extraction_nodes(
        [
            {
                "id": 7,
                "node_type": "belief",
                "belief": "A prior semantic fact.",
                "role": "assistant",
                "stance": "speculated",
                "confidence": 0.6,
                "entities": ["fact"],
                "event_time": "2026-08-27T00:00:00Z",
                "source": {"turn_index": 3},
            }
        ],
        char_budget=None,
    )

    assert json.loads(rendered) == [{"content": "A prior semantic fact."}]
    assert "stance" not in rendered
    assert "confidence" not in rendered
    assert "entities" not in rendered


def test_unified_assistant_relations_judge_three_layers_in_one_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layered_calls: list[dict[str, Any]] = []

    def fake_extract_nodes(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        content = str(kwargs["content"])
        return {
            "nodes": [
                {
                    "tmp_id": "n0",
                    "node_type": "belief",
                    "belief": content,
                    "stance": "asserted",
                    "entities": [],
                    "supporting_sentence_indices": [0],
                }
            ],
            "raw_output": "{}",
            "skipped": False,
        }

    def no_relations(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {"relations": [], "raw_output": "{}", "skipped": False}

    def fake_layered(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        layers = kwargs["candidate_layers"]
        layered_calls.append(kwargs)
        current_id = min(kwargs["new_node_ids"])
        nearest_id = layers[0]["node_ids"][0]
        return {
            "selected_previous_layer": 1,
            "relations": [
                {
                    "from": current_id,
                    "to": nearest_id,
                    "type": "depends_on",
                    "note": "The current reasoning uses the nearest evidence.",
                }
            ],
            "raw_output": "{}",
            "skipped": False,
        }

    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_nodes", fake_extract_nodes
    )
    monkeypatch.setattr("bcg.construct.unified.stream.extract_relations", no_relations)
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_layered_relations", fake_layered
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="gpt-5.6-luna",
        item_id="assistant-layer-bundle",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(
            incremental_merge=False,
            max_previous_windows=3,
        ),
    )
    builder.ingest_turn("user", "Oldest evidence.")
    builder.ingest_turn("user", "Middle evidence.")
    builder.ingest_turn("user", "Nearest evidence.")
    event = builder.ingest_turn(
        "assistant",
        "<thinking>Current reasoning.</thinking>\n"
        '<tool_call>{"name":"web_search","arguments":{"query":"secret query"}}</tool_call>\n'
        "Visible reasoning.",
    )

    assert len(layered_calls) == 1
    assert [layer["layer"] for layer in layered_calls[0]["candidate_layers"]] == [
        1,
        2,
        3,
    ]
    assert all(
        "trajectory_index" not in layer
        for layer in layered_calls[0]["candidate_layers"]
    )
    assert layered_calls[0]["content"] == ""
    assert '"stance"' not in layered_calls[0]["graph_nodes_str"]
    assert '"entities"' not in layered_calls[0]["graph_nodes_str"]
    assert event["edge_attempts"][0]["validation_passed"] is True
    assert event["edge_attempts"][0]["selected_previous_layer"] == 1
    assert event["edge_linked_previous_trajectory_index"] == 2
    assert event["relations_added"] == 1


def test_unified_assistant_layer_bundle_retries_then_keeps_most_used_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layered_calls: list[str | None] = []

    def fake_extract_nodes(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        content = str(kwargs["content"])
        count = 2 if content == "Middle evidence." else 1
        return {
            "nodes": [
                {
                    "tmp_id": f"n{index}",
                    "node_type": "belief",
                    "belief": f"{content} fact {index}",
                    "stance": "asserted",
                    "entities": [],
                    "supporting_sentence_indices": [0],
                }
                for index in range(count)
            ],
            "raw_output": "{}",
            "skipped": False,
        }

    def no_relations(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {"relations": [], "raw_output": "{}", "skipped": False}

    def invalid_layered(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        layered_calls.append(kwargs.get("validation_feedback"))
        layers = kwargs["candidate_layers"]
        current_id = min(kwargs["new_node_ids"])
        layer_1_id = layers[0]["node_ids"][0]
        layer_2_ids = layers[1]["node_ids"]
        return {
            "selected_previous_layer": 1,
            "relations": [
                {
                    "from": current_id,
                    "to": layer_1_id,
                    "type": "depends_on",
                    "note": "One edge to the nearest layer.",
                },
                *[
                    {
                        "from": current_id,
                        "to": node_id,
                        "type": "supplements",
                        "note": "Two edges to the second layer.",
                    }
                    for node_id in layer_2_ids
                ],
            ],
            "raw_output": "{}",
            "skipped": False,
        }

    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_nodes", fake_extract_nodes
    )
    monkeypatch.setattr("bcg.construct.unified.stream.extract_relations", no_relations)
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_layered_relations", invalid_layered
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="gpt-5.6-luna",
        item_id="assistant-layer-bundle-fallback",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(
            incremental_merge=False,
            max_previous_windows=3,
        ),
    )
    builder.ingest_turn("user", "Oldest evidence.")
    builder.ingest_turn("user", "Middle evidence.")
    builder.ingest_turn("user", "Nearest evidence.")
    event = builder.ingest_turn("assistant", "Current reasoning.")

    attempt = event["edge_attempts"][0]
    assert len(layered_calls) == 3
    assert layered_calls[0] is None
    assert all(feedback for feedback in layered_calls[1:])
    assert attempt["validation_passed"] is False
    assert attempt["fallback_pruned"] is True
    assert attempt["selected_previous_layer"] == 2
    assert attempt["selected_previous_trajectory_index"] == 1
    assert attempt["relations_added"] == 2
    retained_previous_ids = {relation["to_id"] for relation in builder.graph.relations}
    assert retained_previous_ids == set(attempt["candidate_layers"][1]["node_ids"])


def test_unified_node_extraction_prompt_omits_edges() -> None:
    nodes = [
        {"id": 1, "node_type": "belief", "belief": "The earliest node."},
        {"id": 2, "node_type": "belief", "belief": "The latest node."},
    ]
    rendered_nodes = format_graph_nodes(nodes, char_budget=None)
    prompt = build_node_extraction_prompt(
        "tool",
        mode="excerpt",
        content="A current tool result.",
        graph_nodes=rendered_nodes,
        graph_edges='[{"from": 1, "to": 2, "type": "depends_on"}]',
    )

    assert prompt is not None
    assert "The earliest node." in prompt
    assert "The latest node." in prompt
    assert "Existing relations" not in prompt
    assert '"from": 1' not in prompt
    assert '"tool_name"' not in prompt
    assert '"query"' not in prompt
    assert "query-bearing tool call" not in prompt


def test_unified_node_extraction_prompt_omits_empty_context_and_tmp_ids() -> None:
    prompt = build_node_extraction_prompt(
        "user",
        mode="sentences",
        sentences_block="[0] The user asks a question.",
        graph_nodes="[]",
    )

    assert prompt is not None
    assert "Existing belief nodes" not in prompt
    assert '"tmp_id"' not in prompt
    assert '"decisions"' not in prompt
    assert "decision" not in prompt.lower()
    assert '"stance"' in prompt
    assert '"entities"' in prompt
    assert '"supporting_sentence_indices"' in prompt
    assert "confidence is assigned downstream" not in prompt
    assert "event metadata is assigned" not in prompt
    assert "You maintain a belief graph INCREMENTALLY" not in prompt
    assert "independently searchable numbered clues" in prompt
    assert "task-defining" in prompt
    assert "qualified roles" in prompt


def test_unified_assistant_node_prompt_uses_compact_role_specific_sections() -> None:
    prompt = build_node_extraction_prompt(
        "assistant",
        mode="sentences",
        sentences_block="[0] The assistant identifies the answer.",
        graph_nodes='[{"id": 1, "belief": "Earlier evidence."}]',
    )

    assert prompt is not None
    assert "Maintain the belief graph incrementally" in prompt
    assert "## What is a belief" in prompt
    assert "## What is a decision" in prompt
    assert "## Source role: ASSISTANT" in prompt
    assert "Tool-call JSON is extracted deterministically by code" in prompt
    assert "## Existing belief nodes" in prompt
    assert "Earlier evidence." in prompt
    assert "## Sentence input" in prompt
    assert "## Hard constraints" in prompt
    assert '"decision": "<self-contained final selected answer>"' in prompt
    assert "Do not infer a decision from a question" in prompt
    assert "Task-defining criteria" in prompt
    assert "Inspect every current-turn sentence" in prompt
    assert "NON-EMPTY" in prompt
    assert "substantive source claim is represented exactly once" in prompt
    assert "## Current turn sentences" in prompt
    assert "Too fine-grained" not in prompt
    assert prompt.index("## Existing belief nodes") < prompt.index(
        "## Hard constraints"
    )
    assert prompt.index("## Output") < prompt.index("## Current turn sentences")


def test_unified_node_extraction_assigns_code_owned_tmp_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call(*args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return json.dumps(
            {
                "beliefs": [
                    {
                        "tmp_id": "model-chosen-id",
                        "belief": "First belief.",
                        "stance": "asserted",
                        "entities": ["First"],
                        "supporting_sentence_indices": [0],
                    },
                    {
                        "tmp_id": "model-chosen-id",
                        "belief": "Second belief.",
                        "stance": "judged",
                        "entities": ["Second"],
                        "supporting_sentence_indices": [1],
                    },
                ],
                "decisions": [
                    {
                        "tmp_id": "another-model-id",
                        "decision": "Final decision.",
                        "stance": "judged",
                        "entities": ["Final"],
                        "supporting_sentence_indices": [1],
                    }
                ],
            }
        )

    monkeypatch.setattr("bcg.construct.unified.extract.llm.call_model", fake_call)
    result = extract_nodes(
        object(),
        "graph-model",
        role="assistant",
        mode="sentences",
        sentences=["First belief.", "Second belief and final decision."],
    )

    assert [node["tmp_id"] for node in result["nodes"]] == ["n0", "n1", "d2"]
    assert result["nodes"][1]["stance"] == "judged"
    assert result["nodes"][1]["entities"] == ["Second"]


def test_stream_node_extraction_respects_context_chars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_contexts: list[str] = []

    def fake_extract_nodes(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        content = str(kwargs["content"])
        extraction_contexts.append(str(kwargs["graph_nodes_str"]))
        return {
            "nodes": [
                {
                    "tmp_id": "n0",
                    "node_type": "belief",
                    "belief": content,
                    "stance": "asserted",
                    "entities": [],
                    "supporting_excerpts": [content],
                }
            ],
            "raw_output": "{}",
            "skipped": False,
        }

    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_nodes", fake_extract_nodes
    )
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_relations",
        lambda *args, **kwargs: {"relations": [], "raw_output": "{}", "skipped": False},
    )
    options = UnifiedStreamOptions(
        evidence_mode="excerpt",
        incremental_merge=False,
        context_chars=450,
    )
    assert options.verify_merge is False
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="graph-model",
        item_id="bounded-node-extraction",
        out_dir=tmp_path,
        options=options,
    )
    builder.ingest_turn("user", "oldest " + "a" * 260)
    builder.ingest_turn("user", "newest " + "b" * 260)
    expected = format_extraction_nodes(builder.graph.active(), char_budget=450)
    builder.ingest_turn("user", "current")

    assert extraction_contexts[-1] == expected
    assert "oldest" not in extraction_contexts[-1]
    assert "newest" in extraction_contexts[-1]


def test_stream_node_extraction_keeps_only_latest_graph_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_contexts: list[str] = []

    def fake_extract_nodes(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        content = str(kwargs["content"])
        extraction_contexts.append(str(kwargs["graph_nodes_str"]))
        return {
            "nodes": [
                {
                    "tmp_id": "n0",
                    "node_type": "belief",
                    "belief": content,
                    "stance": "asserted",
                    "entities": [],
                    "supporting_excerpts": [content],
                }
            ],
            "raw_output": "{}",
            "skipped": False,
        }

    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_nodes", fake_extract_nodes
    )
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_relations",
        lambda *args, **kwargs: {"relations": [], "raw_output": "{}", "skipped": False},
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="graph-model",
        item_id="two-turn-node-extraction-history",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(
            evidence_mode="excerpt",
            incremental_merge=False,
            context_chars=100_000,
            extraction_history_turns=2,
        ),
    )

    builder.ingest_turn("user", "turn one")
    builder.ingest_turn("user", "turn two")
    builder.ingest_turn("user", "turn three")
    builder.ingest_turn("user", "turn four")

    assert "turn one" not in extraction_contexts[-1]
    assert "turn two" in extraction_contexts[-1]
    assert "turn three" in extraction_contexts[-1]


def test_stream_options_load_extraction_history_turns() -> None:
    options = UnifiedStreamOptions()

    options.apply_belief_graph_config({"runtime": {"extraction_history_turns": 2}})

    assert options.extraction_history_turns == 2
    assert options.to_dict()["extraction_history_turns"] == 2


def test_rule_tool_result_is_compact_and_structured() -> None:
    content = """[Tool result: web_search]
[1] First result
URL: https://example.com/one
Snippet: A useful first result with an exact fact.
Source type: organic

[2] Second result
URL: https://example.org/two
Snippet: A useful second result.
Source type: organic
"""

    result = extract_rule_tool_result_nodes(
        role="tool",
        content=content,
        mode="excerpt",
        max_results=1,
        max_snippet_chars=80,
    )

    assert result is not None
    node = result["nodes"][0]
    assert node["extraction_method"] == "rule_tool_result"
    assert node["stance"] == "recalled"
    assert node["tool_name"] == "web_search"
    assert node["tool_result_count"] == 2
    assert node["tool_result_truncated_count"] == 1
    assert node["tool_result_items"] == [
        {
            "rank": 1,
            "title": "First result",
            "url": "https://example.com/one",
            "snippet": "A useful first result with an exact fact.",
            "source_type": "organic",
        }
    ]
    assert "Second result" not in node["belief"]
    assert node["supporting_excerpts"] == [content]


def test_mixed_thinking_and_tool_calls_use_one_assistant_turn_with_exact_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def fake_call(*args: Any, **kwargs: Any) -> str:
        prompt = str(kwargs.get("prompt") or (args[2] if len(args) > 2 else ""))
        prompts.append(prompt)
        return json.dumps(
            {
                "beliefs": [
                    {
                        "belief": "The assistant is comparing the candidates.",
                        "stance": "judged",
                        "entities": [],
                        "supporting_excerpts": [
                            "<thinking>Compare the candidates.</thinking>"
                        ],
                    }
                ],
                "decisions": [],
            }
        )

    monkeypatch.setattr("bcg.construct.unified.extract.llm.call_model", fake_call)
    content = """<thinking>Compare the candidates.</thinking>
<tool_call>{"id":"call-a","name":"web_search","arguments":{"query":"alpha"}}</tool_call>
<tool_call>{"id":"call-b","name":"web_search","arguments":{"query":"beta"}}</tool_call>"""

    result = extract_nodes(
        client=object(),
        model="gpt-5.6-luna",
        role="assistant",
        mode="excerpt",
        content=content,
    )

    thinking = [
        node for node in result["nodes"] if node.get("source_component") == "thinking"
    ]
    calls = [
        node
        for node in result["nodes"]
        if node.get("extraction_method") == "rule_tool_call"
    ]
    assert len(thinking) == 1
    assert [node["tool_call_id"] for node in calls] == ["call-a", "call-b"]
    assert [node["tool_call_index"] for node in calls] == [0, 1]
    current_turn = prompts[0].split("## Current turn content", 1)[1]
    assert "<thinking>Compare the candidates.</thinking>" in current_turn
    assert "<tool_call>" not in current_turn


def test_grouped_parallel_results_pair_exact_calls_then_model_link_thinking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relation_windows: list[str] = []
    relation_contents: list[str] = []

    def fake_extract_nodes(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        role = str(kwargs["role"])
        content = str(kwargs["content"])
        if role == "user":
            nodes = [
                {
                    "tmp_id": "n0",
                    "node_type": "belief",
                    "belief": content,
                    "stance": "asserted",
                    "entities": [],
                    "supporting_excerpts": [content],
                }
            ]
        else:
            nodes = [
                {
                    "tmp_id": "n0",
                    "node_type": "belief",
                    "belief": "The assistant is comparing alpha and beta.",
                    "stance": "judged",
                    "entities": [],
                    "source_component": "thinking",
                    "supporting_excerpts": ["compare alpha and beta"],
                },
                {
                    "tmp_id": "n1",
                    "node_type": "belief",
                    "belief": "The assistant is using web_search to search for alpha.",
                    "stance": "asserted",
                    "entities": ["web_search"],
                    "tool_name": "web_search",
                    "tool_arguments": {"query": "alpha"},
                    "tool_call_index": 0,
                    "tool_call_id": "call-a",
                    "extraction_method": "rule_tool_call",
                    "supporting_excerpts": [content],
                },
                {
                    "tmp_id": "n2",
                    "node_type": "belief",
                    "belief": "The assistant is using web_search to search for beta.",
                    "stance": "asserted",
                    "entities": ["web_search"],
                    "tool_name": "web_search",
                    "tool_arguments": {"query": "beta"},
                    "tool_call_index": 1,
                    "tool_call_id": "call-b",
                    "extraction_method": "rule_tool_call",
                    "supporting_excerpts": [content],
                },
            ]
        return {"nodes": nodes, "raw_output": "{}", "skipped": False}

    thinking_id: int | None = None

    def fake_extract_relations(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        graph_nodes = str(kwargs["graph_nodes_str"])
        relation_windows.append(graph_nodes)
        relation_contents.append(str(kwargs["content"]))
        if thinking_id is not None and "Alpha result" in graph_nodes:
            current_ids = sorted(kwargs["new_node_ids"])
            return {
                "relations": [
                    {
                        "from": current_ids[0],
                        "to": thinking_id,
                        "type": "supplements",
                        "note": "The result informs the prior comparison.",
                    },
                    {
                        "from": current_ids[0],
                        "to": current_ids[1],
                        "type": "supplements",
                        "note": "This invalid cross-result edge must be rejected.",
                    },
                ],
                "raw_output": "{}",
                "skipped": False,
            }
        return {"relations": [], "raw_output": "{}", "skipped": False}

    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_nodes", fake_extract_nodes
    )
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_relations", fake_extract_relations
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="gpt-5.6-luna",
        item_id="parallel-pairing",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(incremental_merge=False, evidence_mode="excerpt"),
    )
    builder.ingest_turn("user", "Compare alpha and beta.")
    builder.ingest_turn("assistant", "thinking plus two tool calls")
    thinking_id = next(
        node["id"]
        for node in builder.graph.active()
        if node.get("source_component") == "thinking"
    )
    relation_windows.clear()
    grouped = """<tool_result>
{"tool_call_id":"call-a","name":"web_search","is_error":false,"content":"[1] Alpha result\\nURL: https://a.example"}
</tool_result>
<tool_result>
{"tool_call_id":"call-b","name":"web_search","is_error":false,"content":"[1] Beta result\\nURL: https://b.example"}
</tool_result>"""
    event = builder.ingest_turn("tool", grouped)

    results = [
        node
        for node in builder.graph.active()
        if node.get("extraction_method") == "rule_tool_result"
    ]
    calls = {
        node["tool_call_id"]: node["id"]
        for node in builder.graph.active()
        if node.get("extraction_method") == "rule_tool_call"
    }
    assert len(results) == 2
    assert {node["source"]["turn_id"] for node in results} == {2}
    paired = {
        (relation["from_id"], relation["to_id"])
        for relation in builder.graph.relations
        if relation["type"] == "depends_on"
    }
    assert (results[0]["id"], calls["call-a"]) in paired
    assert (results[1]["id"], calls["call-b"]) in paired
    assert not any(
        relation["from_id"] in {node["id"] for node in results}
        and relation["to_id"] in {node["id"] for node in results}
        for relation in builder.graph.relations
    )
    assert len(relation_windows) == 1
    assert json.loads(relation_windows[0]) == [
        {"id": node["id"], "content": node["belief"]}
        for node in [
            next(
                graph_node
                for graph_node in builder.graph.active()
                if graph_node["id"] == thinking_id
            ),
            *results,
        ]
    ]
    assert relation_contents[0] == ""
    assert "The assistant is comparing alpha and beta." in relation_windows[0]
    assert "using web_search" not in relation_windows[0]
    assert event["edge_attempts"][0]["pairing_strategy"] == "tool_call_id"


def test_grouped_results_without_thinking_only_pair_exact_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relation_roles: list[str] = []

    def fake_extract_nodes(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        role = str(kwargs["role"])
        content = str(kwargs["content"])
        if role == "assistant":
            nodes = [
                {
                    "tmp_id": "n0",
                    "node_type": "belief",
                    "belief": "The assistant is using web_search to search for alpha.",
                    "stance": "asserted",
                    "entities": ["web_search"],
                    "tool_name": "web_search",
                    "tool_arguments": {"query": "alpha"},
                    "tool_call_index": 0,
                    "tool_call_id": "call-a",
                    "extraction_method": "rule_tool_call",
                    "supporting_excerpts": [content],
                },
                {
                    "tmp_id": "n1",
                    "node_type": "belief",
                    "belief": "The assistant is using web_search to search for beta.",
                    "stance": "asserted",
                    "entities": ["web_search"],
                    "tool_name": "web_search",
                    "tool_arguments": {"query": "beta"},
                    "tool_call_index": 1,
                    "tool_call_id": "call-b",
                    "extraction_method": "rule_tool_call",
                    "supporting_excerpts": [content],
                },
            ]
        else:
            nodes = [
                {
                    "tmp_id": "n0",
                    "node_type": "belief",
                    "belief": content,
                    "stance": "asserted",
                    "entities": [],
                    "supporting_excerpts": [content],
                }
            ]
        return {"nodes": nodes, "raw_output": "{}", "skipped": False}

    def fake_extract_relations(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        relation_roles.append(str(kwargs["role"]))
        return {"relations": [], "raw_output": "{}", "skipped": False}

    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_nodes", fake_extract_nodes
    )
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_relations", fake_extract_relations
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="gpt-5.6-luna",
        item_id="parallel-pairing-no-thinking",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(incremental_merge=False, evidence_mode="excerpt"),
    )
    builder.ingest_turn("user", "Find alpha and beta.")
    builder.ingest_turn("assistant", "two tool calls")
    relation_roles.clear()
    grouped = """<tool_result>
{"tool_call_id":"call-a","name":"web_search","is_error":false,"content":"Alpha result"}
</tool_result>
<tool_result>
{"tool_call_id":"call-b","name":"web_search","is_error":false,"content":"Beta result"}
</tool_result>"""
    event = builder.ingest_turn("tool", grouped)

    results = {
        node["tool_call_id"]: node["id"]
        for node in builder.graph.active()
        if node.get("extraction_method") == "rule_tool_result"
    }
    calls = {
        node["tool_call_id"]: node["id"]
        for node in builder.graph.active()
        if node.get("extraction_method") == "rule_tool_call"
    }
    pairings = {
        (relation["from_id"], relation["to_id"])
        for relation in builder.graph.relations
        if relation["type"] == "depends_on"
    }
    assert pairings >= {
        (results["call-a"], calls["call-a"]),
        (results["call-b"], calls["call-b"]),
    }
    assert relation_roles == []
    assert len(event["edge_attempts"]) == 1


def test_semantic_tool_result_uses_small_query_only_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def fake_call(*args: Any, **kwargs: Any) -> str:
        prompts.append(str(args[2]))
        assert kwargs["max_tokens"] == 768
        return (
            '{"beliefs":[{"belief":"The exact answer is SAS 9.1.",'
            '"stance":"asserted","entities":["SAS 9.1"]}]}'
        )

    monkeypatch.setattr("bcg.construct.unified.extract.llm.call_model", fake_call)
    content = """[Tool result: web_search]
[1] Statistical analysis plan
URL: https://example.com/sap
Snippet: Statistical analyses were performed in SAS version 9.1 or higher.
Source type: organic
"""
    result = extract_compact_tool_result_nodes(
        object(),
        "graph-model",
        role="tool",
        content=content,
        mode="excerpt",
        query="earliest possible SAS version",
    )

    assert result is not None
    assert result["nodes"][0]["belief"] == "The exact answer is SAS 9.1."
    assert result["nodes"][0]["extraction_method"] == "compact_llm_tool_result"
    assert result["nodes"][0]["stance"] == "asserted"
    assert result["nodes"][0]["entities"] == ["SAS 9.1"]
    assert len(prompts) == 1
    assert "earliest possible SAS version" in prompts[0]
    assert "Existing belief nodes" not in prompts[0]
    assert '"entities"' in prompts[0]
    assert '"stance"' in prompts[0]


def test_tool_result_batch_uses_one_call_and_keeps_items_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def fake_call(*args: Any, **kwargs: Any) -> str:
        del kwargs
        prompts.append(str(args[2]))
        return json.dumps(
            {
                "items": [
                    {
                        "item_index": 0,
                        "beliefs": [
                            {
                                "belief": "Alpha was founded in 1999.",
                                "entities": ["Alpha"],
                            }
                        ],
                    },
                    {
                        "item_index": 1,
                        "beliefs": [
                            {
                                "belief": "Beta was founded in 2007.",
                                "entities": ["Beta"],
                            }
                        ],
                    },
                ]
            }
        )

    monkeypatch.setattr("bcg.construct.unified.extract.llm.call_model", fake_call)
    alpha = """[Tool result: web_search]
[1] Alpha history
URL: https://alpha.example/history
Snippet: Alpha was founded in 1999.
"""
    beta = """[Tool result: web_search]
[1] Beta history
URL: https://beta.example/history
Snippet: Beta was founded in 2007.
"""
    results = extract_compact_tool_result_nodes_batch(
        object(),
        "graph-model",
        items=[
            {"content": alpha, "query": "Alpha founding year"},
            {"content": beta, "query": "Beta founding year"},
        ],
        mode="excerpt",
    )

    assert len(prompts) == 1
    assert "Treat every item independently" in prompts[0]
    assert results[0] is not None and results[1] is not None
    assert results[0]["nodes"][0]["belief"] == "Alpha was founded in 1999."
    assert results[1]["nodes"][0]["belief"] == "Beta was founded in 2007."
    assert results[0]["nodes"][0]["tool_result_items"][0]["url"] == (
        "https://alpha.example/history"
    )
    assert results[1]["nodes"][0]["tool_result_items"][0]["url"] == (
        "https://beta.example/history"
    )


def test_builder_batches_results_and_links_each_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_call(*args: Any, **kwargs: Any) -> str:
        nonlocal calls
        del args, kwargs
        calls += 1
        return json.dumps(
            {
                "items": [
                    {
                        "item_index": 0,
                        "beliefs": [{"belief": "Alpha fact.", "entities": ["Alpha"]}],
                    },
                    {
                        "item_index": 1,
                        "beliefs": [{"belief": "Beta fact.", "entities": ["Beta"]}],
                    },
                ]
            }
        )

    monkeypatch.setattr("bcg.construct.unified.extract.llm.call_model", fake_call)
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="unused-model",
        item_id="batch-provenance",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(
            incremental_merge=False,
            evidence_mode="excerpt",
        ),
    )
    builder.ingest_turn(
        "assistant",
        """<tool_call>{"name":"web_search","arguments":{"query":"alpha query"}}</tool_call>
<tool_call>{"name":"web_search","arguments":{"query":"beta query"}}</tool_call>""",
    )
    alpha = "[Tool result: web_search]\n[1] Alpha\nURL: https://alpha.example\nSnippet: Alpha fact."
    beta = "[Tool result: web_search]\n[1] Beta\nURL: https://beta.example\nSnippet: Beta fact."

    assert builder.prepare_tool_result_batch([alpha, beta]) == 2
    builder.ingest_turn("tool", alpha)
    builder.ingest_turn("tool", beta)

    assert calls == 1
    nodes = builder.graph.active()
    alpha_query = next(node for node in nodes if node.get("query") == "alpha query")
    beta_query = next(node for node in nodes if node.get("query") == "beta query")
    alpha_fact = next(node for node in nodes if node.get("belief") == "Alpha fact.")
    beta_fact = next(node for node in nodes if node.get("belief") == "Beta fact.")
    assert (alpha_fact["source"] or {})["turn_id"] == 1
    assert (beta_fact["source"] or {})["turn_id"] == 2
    relation_pairs = {
        (relation["from_id"], relation["to_id"]) for relation in builder.graph.relations
    }
    assert (alpha_fact["id"], alpha_query["id"]) in relation_pairs
    assert (beta_fact["id"], beta_query["id"]) in relation_pairs
    assert (alpha_fact["id"], beta_query["id"]) not in relation_pairs
    assert (beta_fact["id"], alpha_query["id"]) not in relation_pairs


def test_rule_tool_result_bypasses_llm_and_links_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_relations(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("rule Tool Results must not call the relation model")

    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_relations", fail_relations
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="unused-model",
        item_id="efficient-provenance",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(
            incremental_merge=False,
            tool_result_semantic_extraction=False,
            evidence_mode="excerpt",
        ),
    )
    builder.ingest_turn(
        "assistant",
        '<tool_call>{"name":"web_search","arguments":{"query":"exact query"}}</tool_call>',
    )
    event = builder.ingest_turn(
        "tool",
        """[Tool result: web_search]
[1] Exact result
URL: https://example.com/result
Snippet: The exact answer is 9.1.
Source type: organic
""",
    )

    nodes = builder.graph.active()
    query_node = next(node for node in nodes if node.get("query") == "exact query")
    result_node = next(
        node for node in nodes if node.get("extraction_method") == "rule_tool_result"
    )
    assert result_node["tool_result_id"] == "efficient-provenance:t1:result"
    assert result_node["tool_result_items"][0]["url"] == "https://example.com/result"
    assert result_node["confidence"] == initial_confidence("tool", "recalled")
    assert result_node["factor_confidence"] == 0.0
    assert builder.graph.relations == [
        {
            "id": 0,
            "from_id": result_node["id"],
            "to_id": query_node["id"],
            "type": "depends_on",
            "note": "The tool result was produced by the preceding tool call.",
            "weight": 0.0,
            "activated_condition": {"input_conf_threshold": 1.0},
        }
    ]
    assert event["edge_attempts"][0]["strategy"] == "deterministic_provenance"


def test_thinking_and_tool_results_use_separate_extraction_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_prompts: list[str] = []

    def fake_call(*args: Any, **kwargs: Any) -> str:
        del kwargs
        prompt = str(args[2])
        extraction_prompts.append(prompt)
        if "Items:" in prompt:
            return json.dumps(
                {
                    "items": [
                        {
                            "item_index": 0,
                            "beliefs": [
                                {
                                    "belief": "Alpha fact.",
                                    "stance": "asserted",
                                    "entities": ["Alpha"],
                                }
                            ],
                        }
                    ]
                }
            )
        return json.dumps(
            {
                "beliefs": [
                    {
                        "tmp_id": "n0",
                        "belief": "The assistant hypothesizes that Alpha is relevant.",
                        "stance": "speculated",
                        "entities": ["Alpha"],
                        "supporting_sentence_indices": [0],
                    }
                ],
                "decisions": [],
            }
        )

    def no_relations(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {"relations": [], "raw_output": '{"relations":[]}'}

    monkeypatch.setattr("bcg.construct.unified.extract.llm.call_model", fake_call)
    monkeypatch.setattr("bcg.construct.unified.stream.extract_relations", no_relations)
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="unused-model",
        item_id="assistant-tool-combined",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(
            incremental_merge=False,
            evidence_mode="sentence",
        ),
    )
    assistant = """<thinking>Alpha may be relevant.</thinking>
<tool_call>{"name":"web_search","arguments":{"query":"alpha query"}}</tool_call>"""
    tool = """[Tool result: web_search]
[1] Alpha
URL: https://alpha.example
Snippet: Alpha fact."""

    assert builder.prepare_assistant_tool_result_batch(assistant, [tool]) == 2
    builder.ingest_turn("assistant", assistant)
    builder.ingest_turn("tool", tool)

    assert len(extraction_prompts) == 2
    assistant_prompt, tool_prompt = extraction_prompts
    assert "## Existing belief nodes" not in assistant_prompt
    assert '"tmp_id"' not in assistant_prompt
    assert "Alpha may be relevant." in assistant_prompt
    assert "alpha query" not in assistant_prompt
    assert "Items:" in tool_prompt
    assert "Existing belief nodes" not in tool_prompt
    nodes = builder.graph.active()
    reasoning = next(
        node
        for node in nodes
        if node.get("belief") == "The assistant hypothesizes that Alpha is relevant."
    )
    query = next(node for node in nodes if node.get("query") == "alpha query")
    fact = next(node for node in nodes if node.get("belief") == "Alpha fact.")
    assert reasoning["source"]["turn_id"] == 0
    assert query["source"]["turn_id"] == 0
    assert fact["source"]["turn_id"] == 1
    assert reasoning.get("source_component") == "thinking"
    assert reasoning["stance"] == "speculated"
    assert reasoning["entities"] == ["Alpha"]
    assert fact["stance"] == "asserted"
    assert fact["entities"] == ["Alpha"]


def test_tool_call_without_thinking_skips_assistant_model_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_prompts: list[str] = []

    def fake_call(*args: Any, **kwargs: Any) -> str:
        del kwargs
        prompt = str(args[2])
        extraction_prompts.append(prompt)
        assert "Items:" in prompt
        assert "Existing belief nodes" not in prompt
        return json.dumps(
            {
                "items": [
                    {
                        "item_index": 0,
                        "beliefs": [
                            {
                                "belief": "Alpha fact.",
                                "stance": "asserted",
                                "entities": ["Alpha"],
                            }
                        ],
                    },
                    {
                        "item_index": 1,
                        "beliefs": [
                            {
                                "belief": "Beta fact.",
                                "stance": "asserted",
                                "entities": ["Beta"],
                            }
                        ],
                    },
                ]
            }
        )

    def no_relations(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {"relations": [], "raw_output": '{"relations":[]}'}

    monkeypatch.setattr("bcg.construct.unified.extract.llm.call_model", fake_call)
    monkeypatch.setattr("bcg.construct.unified.stream.extract_relations", no_relations)
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="unused-model",
        item_id="assistant-tool-no-thinking",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(
            incremental_merge=False,
            evidence_mode="sentence",
        ),
    )
    assistant = (
        '<tool_call>{"name":"web_search","arguments":'
        '{"query":"alpha query"}}</tool_call>\n'
        '<tool_call>{"name":"web_search","arguments":'
        '{"query":"beta query"}}</tool_call>'
    )
    alpha_tool = """[Tool result: web_search]
[1] Alpha
URL: https://alpha.example
Snippet: Alpha fact."""
    beta_tool = """[Tool result: web_search]
[1] Beta
URL: https://beta.example
Snippet: Beta fact."""

    assert (
        builder.prepare_assistant_tool_result_batch(
            assistant,
            [alpha_tool, beta_tool],
        )
        == 3
    )
    builder.ingest_turn("assistant", assistant)
    builder.ingest_turn("tool", alpha_tool)
    builder.ingest_turn("tool", beta_tool)

    assert len(extraction_prompts) == 1
    nodes = builder.graph.active()
    assert any(node.get("query") == "alpha query" for node in nodes)
    assert any(node.get("query") == "beta query" for node in nodes)
    assert any(node.get("belief") == "Alpha fact." for node in nodes)
    assert any(node.get("belief") == "Beta fact." for node in nodes)


def test_stance_classifier_dynamically_batches_concurrent_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classifier = LocalZeroShotStanceClassifier(
        {
            "model_path": "unused",
            "local_files_only": False,
            "dynamic_batching": True,
            "dynamic_batch_wait_ms": 50,
            "dynamic_batch_max_texts": 16,
        }
    )
    physical_batches: list[list[str]] = []

    def fake_direct(texts: list[str]) -> list[StancePrediction]:
        physical_batches.append(list(texts))
        return [
            StancePrediction(
                stance="asserted",
                confidence=1.0,
                scores={
                    "asserted": 1.0,
                    "recalled": 0.0,
                    "judged": 0.0,
                    "speculated": 0.0,
                },
                model_path="fake",
            )
            for _ in texts
        ]

    monkeypatch.setattr(classifier, "_classify_texts_direct", fake_direct)
    barrier = threading.Barrier(3)
    results: dict[str, list[StancePrediction]] = {}

    def classify(key: str, texts: list[str]) -> None:
        barrier.wait()
        results[key] = classifier.classify_texts(texts)

    left = threading.Thread(target=classify, args=("left", ["a", "b"]))
    right = threading.Thread(target=classify, args=("right", ["c"]))
    left.start()
    right.start()
    barrier.wait()
    left.join(timeout=2)
    right.join(timeout=2)

    assert not left.is_alive()
    assert not right.is_alive()
    assert len(physical_batches) == 1
    assert sorted(physical_batches[0]) == ["a", "b", "c"]
    assert len(results["left"]) == 2
    assert len(results["right"]) == 1


def test_semantic_tool_result_call_cap_falls_back_to_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compact_calls: list[str] = []

    def fake_compact(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        compact_calls.append(str(kwargs["content"]))
        rule = extract_rule_tool_result_nodes(
            role=kwargs["role"],
            content=kwargs["content"],
            mode=kwargs["mode"],
            sentences=kwargs["sentences"],
        )
        assert rule is not None
        rule["extraction_method"] = "compact_llm_tool_result"
        for node in rule["nodes"]:
            node["extraction_method"] = "compact_llm_tool_result"
        return rule

    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_compact_tool_result_nodes",
        fake_compact,
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="unused-model",
        item_id="semantic-cap",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(
            incremental_merge=False,
            tool_result_max_semantic_calls=1,
            evidence_mode="excerpt",
        ),
    )
    first = "[Tool result: web_search]\n[1] First\nURL: https://one.example"
    second = "[Tool result: web_search]\n[1] Second\nURL: https://two.example"

    first_event = builder.ingest_turn("tool", first)
    second_event = builder.ingest_turn("tool", second)

    assert compact_calls == [first]
    methods = [node.get("extraction_method") for node in builder.graph.active()]
    assert methods == ["compact_llm_tool_result", "rule_tool_result"]
    assert first_event["semantic_tool_result_calls"] == 1
    assert second_event["semantic_tool_result_calls"] == 1


def test_tool_result_path_uses_bounded_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_extract_nodes(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        calls.append(str(kwargs["content"]))
        return {
            "nodes": [
                {
                    "tmp_id": "n0",
                    "node_type": "belief",
                    "belief": "Model-extracted tool result.",
                    "stance": "asserted",
                    "entities": [],
                    "supporting_excerpts": [kwargs["content"]],
                }
            ],
            "raw_output": "{}",
            "skipped": False,
        }

    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_nodes", fake_extract_nodes
    )
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_relations",
        lambda *args, **kwargs: {"relations": [], "raw_output": "{}", "skipped": False},
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="unused-model",
        item_id="canonical-path",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(
            incremental_merge=False,
            evidence_mode="excerpt",
        ),
    )
    content = "[Tool result: web_search]\nNo web results were returned."
    builder.ingest_turn("tool", content)

    assert calls == []
    node = builder.graph.active()[0]
    assert node["belief"] == "The web_search tool returned no results."
    assert node["extraction_method"] == "rule_tool_result"


def test_parallel_tool_result_facts_are_extracted_once_and_paired_by_call_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def fake_call(*args: Any, **kwargs: Any) -> str:
        del kwargs
        prompts.append(str(args[2]))
        return json.dumps(
            {
                "items": [
                    {
                        "item_index": 0,
                        "beliefs": [
                            {
                                "belief": "Alpha was founded in 1999.",
                                "entities": ["Alpha"],
                            },
                            {
                                "belief": "Alpha is based in Paris.",
                                "entities": ["Alpha", "Paris"],
                            },
                        ],
                    },
                    {
                        "item_index": 1,
                        "beliefs": [
                            {
                                "belief": "Beta was founded in 2007.",
                                "entities": ["Beta"],
                            },
                            {
                                "belief": "Beta is based in Rome.",
                                "entities": ["Beta", "Rome"],
                            },
                        ],
                    },
                ]
            }
        )

    monkeypatch.setattr("bcg.construct.unified.extract.llm.call_model", fake_call)
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_relations",
        lambda *args, **kwargs: {"relations": [], "raw_output": "{}", "skipped": False},
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="graph-model",
        item_id="llm-parallel-facts",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(
            incremental_merge=False,
            evidence_mode="excerpt",
            tool_result_max_facts=3,
        ),
    )
    builder.ingest_turn(
        "assistant",
        """<tool_call>{"id":"call-a","name":"web_search","arguments":{"query":"alpha query"}}</tool_call>
<tool_call>{"id":"call-b","name":"web_search","arguments":{"query":"beta query"}}</tool_call>""",
    )
    grouped = """<tool_result>
{"tool_call_id":"call-a","name":"web_search","is_error":false,"content":"[1] Alpha profile\\nURL: https://alpha.example\\nSnippet: Alpha was founded in 1999 and is based in Paris."}
</tool_result>
<tool_result>
{"tool_call_id":"call-b","name":"web_search","is_error":false,"content":"[1] Beta profile\\nURL: https://beta.example\\nSnippet: Beta was founded in 2007 and is based in Rome."}
</tool_result>"""
    event = builder.ingest_turn("tool", grouped)

    assert len(prompts) == 1
    assert "Treat every item independently" in prompts[0]
    assert "alpha query" in prompts[0]
    assert "beta query" in prompts[0]
    facts = [
        node
        for node in builder.graph.active()
        if node.get("extraction_method") == "compact_llm_tool_result"
    ]
    assert len(facts) == 4
    assert [node["tool_call_id"] for node in facts] == [
        "call-a",
        "call-a",
        "call-b",
        "call-b",
    ]
    assert len({node["tool_result_id"] for node in facts}) == 4
    calls = {
        node["tool_call_id"]: node["id"]
        for node in builder.graph.active()
        if node.get("extraction_method") == "rule_tool_call"
    }
    paired = {
        (relation["from_id"], relation["to_id"])
        for relation in builder.graph.relations
        if relation["type"] == "depends_on"
    }
    assert all((node["id"], calls[node["tool_call_id"]]) in paired for node in facts)
    assert event["semantic_tool_result_calls"] == 2


def test_grouped_results_fall_back_to_rules_after_semantic_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def fake_call(*args: Any, **kwargs: Any) -> str:
        del kwargs
        prompts.append(str(args[2]))
        return json.dumps(
            {
                "items": [
                    {
                        "item_index": 0,
                        "beliefs": [{"belief": "Alpha fact.", "entities": ["Alpha"]}],
                    }
                ]
            }
        )

    monkeypatch.setattr("bcg.construct.unified.extract.llm.call_model", fake_call)
    monkeypatch.setattr(
        "bcg.construct.unified.stream.extract_relations",
        lambda *args, **kwargs: {"relations": [], "raw_output": "{}", "skipped": False},
    )
    builder = UnifiedStreamingBeliefBuilder(
        client=object(),
        model="graph-model",
        item_id="llm-semantic-cap",
        out_dir=tmp_path,
        options=UnifiedStreamOptions(
            incremental_merge=False,
            evidence_mode="excerpt",
            tool_result_max_semantic_calls=1,
        ),
    )
    builder.ingest_turn(
        "assistant",
        """<tool_call>{"id":"call-a","name":"web_search","arguments":{"query":"alpha"}}</tool_call>
<tool_call>{"id":"call-b","name":"web_search","arguments":{"query":"beta"}}</tool_call>""",
    )
    event = builder.ingest_turn(
        "tool",
        """<tool_result>
{"tool_call_id":"call-a","name":"web_search","content":"[1] Alpha\\nURL: https://a.example\\nSnippet: Alpha fact."}
</tool_result>
<tool_result>
{"tool_call_id":"call-b","name":"web_search","content":"[1] Beta\\nURL: https://b.example\\nSnippet: Beta fact."}
</tool_result>""",
    )

    assert len(prompts) == 1
    result_nodes = [
        node
        for node in builder.graph.active()
        if node.get("extraction_method")
        in {"compact_llm_tool_result", "rule_tool_result"}
    ]
    assert [
        (node["tool_call_id"], node["extraction_method"]) for node in result_nodes
    ] == [
        ("call-a", "compact_llm_tool_result"),
        ("call-b", "rule_tool_result"),
    ]
    assert event["semantic_tool_result_calls"] == 1


def test_hybrid_call_model_reduces_output_budget_after_context_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    max_token_calls: list[int] = []

    class Completions:
        def create(self, **kwargs: Any) -> Any:
            max_token_calls.append(kwargs["max_tokens"])
            if len(max_token_calls) == 1:
                raise RuntimeError(
                    "This model's maximum context length is 10000 tokens. "
                    "However, you requested 2048 output tokens and your prompt "
                    "contains at least 7953 input tokens, for a total of at "
                    "least 10001 tokens."
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"beliefs": []}'))
                ],
                usage=None,
            )

    monkeypatch.setattr("bcg.construct.hybrid.llm.time.sleep", lambda _: None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    response = call_model(
        client,
        "Qwen3.5-4B",
        "Long prompt.",
        max_tokens=2048,
        retries=2,
        response_format={"type": "json_object"},
    )

    assert response == '{"beliefs": []}'
    assert max_token_calls == [2048, 1024]


def test_hybrid_extractor_retries_overflow_without_historical_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def fake_call_model(
        client: Any,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        del client, model, kwargs
        prompts.append(prompt)
        if "historical fact that fills the context window" in prompt:
            raise RuntimeError(
                "All retries failed: This model's maximum context length is "
                "10000 tokens. However, you requested 512 output tokens and "
                "your prompt contains at least 9489 input tokens."
            )
        return '{"beliefs": [{"belief": "The tool reports a current fact."}]}'

    monkeypatch.setattr("bcg.construct.hybrid.extractor.call_model", fake_call_model)
    extractor = QwenChunkExtractor(
        {
            "enabled": True,
            "provider": "openai",
            "base_url": "http://127.0.0.1:8001/v1",
            "api_key": "EMPTY",
            "api_key_env": "BELIEF_GRAPH_LOCAL_API_KEY",
            "model": "Qwen3.5-4B",
            "temperature": 0,
            "max_tokens": 2048,
            "max_concurrency": 1,
            "request_timeout": 60,
            "retries": 3,
            "context_scope": "graph",
            "enable_thinking": False,
            "include_turn_content": False,
            "require_excerpt": False,
            "dynamic_node_cap": False,
            "node_cap_unit": "char",
            "node_cap_ratio": 0.004,
            "node_cap_min": 1,
            "node_cap_max": 0,
        }
    )
    extractor._client = object()
    chunk = SimpleNamespace(text="The tool reports a current fact.", chunk_id=2)

    extracted = extractor.extract_turn(
        [chunk],
        "tool",
        graph_nodes=[
            {
                "id": 1,
                "node_type": "belief",
                "belief": "historical fact that fills the context window",
            }
        ],
    )

    assert len(prompts) == 2
    assert "historical fact that fills the context window" in prompts[0]
    assert "historical fact that fills the context window" not in prompts[1]
    assert extracted[0][0].text == "The tool reports a current fact."


class DummyLLM:
    """The engine call is monkeypatched in integration tests below."""


class AsyncGenerateLLM:
    def __init__(self) -> None:
        self.models: list[str | None] = []

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del temperature, max_tokens
        self.models.append(model)
        return LLMResponse(
            content='{"beliefs": [], "decisions": []}',
            messages=messages,
            usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )


class FakeEmbedder:
    model = "fake-embedding"

    def embed(
        self, texts: list[str], *, purpose: str = "embedding"
    ) -> list[list[float]]:
        del purpose
        return [[1.0, 0.0] if "tea" in text else [0.0, 1.0] for text in texts]


def run(coro: Awaitable[T]) -> T:
    return asyncio.run(coro)


def sample_trajectory() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Alice likes tea."},
        {"role": "assistant", "content": "Final answer: Alice likes green tea."},
    ]


@pytest.fixture
def fake_construct_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    prompts: list[str] = []

    def fake_call_model(
        client: Any,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        del client, model, kwargs
        prompts.append(prompt)
        if '"relations": [' in prompt and '"beliefs": [' not in prompt:
            return json.dumps({"relations": []})
        if "Final answer:" in prompt:
            return json.dumps(
                {
                    "beliefs": [],
                    "decisions": [
                        {
                            "tmp_id": "d0",
                            "decision": "The assistant concludes Alice likes green tea.",
                            "entities": ["Alice"],
                            "stance": "asserted",
                            "supporting_sentence_indices": [0],
                            "event_time": None,
                            "time_text": None,
                        }
                    ],
                }
            )
        return json.dumps(
            {
                "beliefs": [
                    {
                        "tmp_id": "n0",
                        "belief": "The user states Alice likes tea.",
                        "entities": ["Alice"],
                        "stance": "asserted",
                        "supporting_sentence_indices": [0],
                        "event_time": None,
                        "time_text": None,
                    }
                ],
                "decisions": [],
            }
        )

    monkeypatch.setattr("bcg.construct.unified.llm.call_model", fake_call_model)
    return prompts


def test_construct_confidence_is_the_only_confidence_policy() -> None:
    node = {"role": "user", "stance": "asserted"}
    init_belief_confidence(node)

    assert node["confidence"] == initial_confidence("user", "asserted")
    assert node["initial_confidence"] == node["confidence"]
    assert node["confidence_history"] == [
        {
            "step": "initial",
            "value": node["confidence"],
            "evidence_confidence": 0.0,
            "factor_confidence": 0.0,
        }
    ]


def test_creation_turn_excerpts_do_not_self_corroborate_confidence() -> None:
    source = {"item_id": "item-1", "turn_id": 3}
    node = {
        "role": "tool",
        "stance": "recalled",
        "source": source,
        "evidence_ids": [10, 11],
    }
    evidence = {
        10: {"role": "tool", "stance": "recalled", "source": dict(source)},
        11: {"role": "tool", "stance": "recalled", "source": dict(source)},
    }
    init_belief_confidence(node)
    prior = node["confidence"]

    recompute_evidence_confidence_from_node(node, evidence, record_history=True)

    assert node["evidence_confidence"] == 0
    assert node["confidence"] == prior
    assert len(node["confidence_history"]) == 1


def test_only_later_turn_evidence_corrobates_merged_node() -> None:
    source = {"item_id": "item-1", "turn_id": 3}
    node = {
        "role": "tool",
        "stance": "recalled",
        "source": source,
        "evidence_ids": [10, 11, 12],
    }
    evidence = {
        10: {"role": "tool", "stance": "recalled", "source": dict(source)},
        11: {"role": "tool", "stance": "recalled", "source": dict(source)},
        12: {
            "role": "tool",
            "stance": "recalled",
            "source": {"item_id": "item-1", "turn_id": 9},
        },
    }
    init_belief_confidence(node)
    prior = node["confidence"]

    recompute_evidence_confidence_from_node(node, evidence, record_history=True)

    assert node["evidence_confidence"] > 0
    assert node["confidence"] > prior
    assert node["confidence_history"][-1]["scored_evidence_ids"] == [12]


def test_same_later_turn_counts_as_one_independent_observation() -> None:
    source = {"item_id": "item-1", "turn_id": 3}
    node = {
        "role": "tool",
        "stance": "recalled",
        "source": source,
        "evidence_ids": [10, 11, 12, 13],
    }
    evidence = {
        10: {"role": "tool", "stance": "recalled", "source": dict(source)},
        11: {
            "role": "tool",
            "stance": "recalled",
            "source": {"item_id": "item-1", "turn_id": 9},
        },
        12: {
            "role": "tool",
            "stance": "recalled",
            "source": {"item_id": "item-1", "turn_id": 9},
        },
        13: {
            "role": "tool",
            "stance": "recalled",
            "source": {"item_id": "item-1", "turn_id": 10},
        },
    }
    init_belief_confidence(node)

    recompute_evidence_confidence_from_node(node, evidence, record_history=True)

    assert node["evidence_confidence"] == 1.04
    assert node["confidence"] == 0.909
    assert node["confidence_history"][-1]["scored_evidence_ids"] == [11, 13]


def test_hybrid_confidence_also_deduplicates_one_source_turn() -> None:
    node = {
        "role": "tool",
        "stance": "recalled",
        "initial_evidence_count": 1,
        "evidence_ids": [10, 11, 12],
    }
    evidence = {
        10: {"role": "tool", "stance": "recalled"},
        11: {
            "role": "tool",
            "stance": "recalled",
            "source": {"item_id": "item-1", "turn_id": 9},
        },
        12: {
            "role": "tool",
            "stance": "recalled",
            "source": {"item_id": "item-1", "turn_id": 9},
        },
    }
    init_hybrid_belief_confidence(node)

    recompute_hybrid_evidence_confidence(node, evidence, record_history=True)

    assert node["evidence_confidence"] == 0.576
    assert node["confidence_history"][-1]["scored_evidence_ids"] == [11]


def test_posterior_confidence_never_rounds_to_exact_certainty() -> None:
    assert posterior_confidence(0.78, evidence_score=100.0) == 0.999
    assert posterior_confidence(0.78, evidence_score=-100.0) == 0.001


def test_construct_evidence_offsets_are_exact() -> None:
    content = "Before Alice likes tea. After"
    start, end, match = locate_excerpt("Alice likes tea.", content)
    evidence = evidence_from_excerpt(
        "Alice likes tea.",
        content,
        {"turn_id": 0, "item_id": "test"},
        role="user",
    )

    assert match == "exact"
    assert content[start:end] == evidence["text"]


def test_construct_graph_supports_current_relation_types() -> None:
    graph = BeliefGraph()
    for text in ("input", "output"):
        graph.add_belief(
            {
                "id": graph.allocate_id(),
                "node_type": "belief",
                "belief": text,
                "evidence_ids": [],
                "factor_ids": [],
            }
        )

    assert (
        graph.add_relations(
            [{"from_id": 0, "to_id": 1, "type": "depends_on", "note": "dependency"}]
        )
        == 1
    )
    assert graph.relations[0]["type"] == "depends_on"


def test_graph_keeps_basic_constructor_compatibility() -> None:
    async def exercise() -> BCG:
        graph = BCG()
        node = BCGNode(name="manual")
        await graph.add_node(node)
        await graph.add_edge(BCGEdge(source=node.uuid, target=node.uuid))
        return graph

    graph = run(exercise())
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 1


def test_public_graph_accepts_construct_relations_and_decisions() -> None:
    source = BeliefSource(
        type="assistant",
        role="assistant",
        trajectory_index=0,
        segment_index=0,
        segment_type="assistant",
    )
    graph = BCG(evidence=[{"id": 0, "text": "answer"}], factors=[])
    first = graph.add_belief(
        BeliefPayload(
            id=0,
            belief="A premise.",
            layer="reasoning",
            source=source,
        )
    )
    second = graph.add_belief(
        BeliefPayload(
            id=1,
            node_type="decision",
            belief="The final answer.",
            decision="The final answer.",
            layer="reasoning",
            source=source,
        )
    )
    graph.add_relation(
        RelationPayload(from_id=0, to_id=1, type="depends_on"),
        source_uuid=first.uuid,
        target_uuid=second.uuid,
    )

    memory = graph.to_memory_dict()
    assert len(memory["beliefs"]) == 1
    assert len(memory["decisions"]) == 1
    assert memory["relations"][0]["type"] == "depends_on"


def test_construct_snapshot_conversion_preserves_evidence_factors_and_decisions() -> (
    None
):
    graph = _bcg_from_construct(
        {
            "nodes": [
                {
                    "id": 0,
                    "node_type": "decision",
                    "decision": "Choose tea.",
                    "role": "assistant",
                    "stance": "asserted",
                    "source": {"turn_id": 1, "item_id": "case"},
                    "evidence_ids": [0],
                    "factor_ids": [0],
                    "confidence": 0.8,
                }
            ],
            "evidence": [
                {
                    "id": 0,
                    "text": "Choose tea.",
                    "start": 0,
                    "end": 11,
                    "match": "exact",
                    "via": "split_sentence",
                    "role": "assistant",
                    "source": {"turn_id": 1, "item_id": "case"},
                }
            ],
            "factors": [{"id": 0, "factor_type": "depends_on"}],
            "relations": [],
            "merges": [],
        },
        sessions=[],
        metadata={},
    )

    decision = graph.beliefs()[0]
    assert decision.node_type == "decision"
    assert decision.evidence[0].text == "Choose tea."
    assert decision.factor_ids == [0]
    assert graph.factors[0]["id"] == 0


def test_pipeline_writes_sdk_and_native_outputs(
    tmp_path: Path,
    fake_construct_calls: list[str],
) -> None:
    pipeline = BeliefGraphPipeline(
        DummyLLM(),
        output_root=tmp_path / ".bcg" / "runs",
        run_id="unit-run",
    )
    result = run(pipeline.run(sample_trajectory(), metadata={"case": "unit"}))

    assert fake_construct_calls
    assert result.counts["beliefs"] == 1
    assert result.counts["decisions"] == 1
    assert result.output_paths.graph.exists()
    assert result.output_paths.memory.exists()
    assert result.output_paths.result.exists()
    assert result.output_paths.final_graph.exists()
    assert result.output_paths.graph_stream.exists()
    assert result.output_paths.segments.parent.name == "artifacts"
    assert result.memory["engine"] == "bcg.construct.unified"


def test_memory_manual_observe_uses_construct_confidence() -> None:
    memory = BCGMemory(graph=BCG())
    observed = memory.observe(source_type="user", content="Alice likes tea.")

    assert observed.belief.confidence == initial_confidence("user", "asserted")
    assert memory.believe("Alice") == [observed.belief]


def test_runner_incremental_session_methods(
    tmp_path: Path,
    fake_construct_calls: list[str],
) -> None:
    memory = BCGMemory(graph=BCG())
    runner = BCGRunner(
        memory=memory,
        llm=DummyLLM(),
        output_root=tmp_path / ".bcg" / "runs",
    )
    runner.begin_belief_run(run_id="incremental-run")
    runner.start_session("chat-1", "2024-01-01")
    run(runner.observe_turn("user", "Alice likes tea."))
    run(runner.end_session())
    result = run(runner.finalize())

    assert fake_construct_calls
    assert result.counts["sessions"] == 1
    assert result.memory["trajectory"][0]["session_id"] == "chat-1"
    assert result.graph.metadata["engine"] == "bcg.construct.unified"


def test_async_public_llm_adapter_forwards_model_and_usage() -> None:
    llm = AsyncGenerateLLM()
    response = _ConstructClientAdapter(llm).chat.completions.create(
        model="custom-model",
        messages=[{"role": "user", "content": "{}"}],
        temperature=0,
        max_tokens=32,
    )

    assert response.choices[0].message.content.startswith("{")
    assert response.usage.total_tokens == 5
    assert llm.models == ["custom-model"]


def hybrid_belief_graph_config() -> dict[str, Any]:
    """Minimal-but-complete belief_graph section for the hybrid backend.

    Chunking / incremental_merge / edge_generation are all turned off so the
    test doesn't need a real embedder or a real Qwen edge-generator endpoint;
    the extractor/stance/NER components are monkeypatched below instead of
    loading real weights.
    """

    return {
        "extractor": {
            "enabled": True,
            "provider": "openai",
            "base_url": "http://unused.invalid/v1",
            "api_key": "unused",
            "model": "unused",
            "temperature": 0,
            "max_tokens": 4096,
            "max_concurrency": 4,
            "request_timeout": 60,
            "retries": 1,
            "context_scope": "none",
            "enable_thinking": False,
            "include_turn_content": False,
            "require_excerpt": False,
            "dynamic_node_cap": False,
            "node_cap_unit": "char",
            "node_cap_ratio": 0.004,
            "node_cap_min": 1,
            "node_cap_max": 0,
        },
        "stance": {
            "enabled": True,
            "model_path": "unused",
            "device": "cpu",
            "dtype": "auto",
            "batch_size": 16,
            "max_length": 512,
            "local_files_only": True,
            "hypothesis_template": "{description}",
            "labels": {
                "asserted": {"description": "asserted"},
                "recalled": {"description": "recalled"},
                "judged": {"description": "judged"},
                "speculated": {"description": "speculated"},
            },
        },
        "edge_generation": {
            "enabled": False,
            "provider": "openai",
            "base_url": "http://unused.invalid/v1",
            "api_key": "unused",
            "model": "unused",
            "temperature": 0,
            "max_tokens": 4096,
            "retries": 1,
            "enable_thinking": False,
            "fail_on_error": True,
            "search_previous_turns": True,
        },
        "runtime": {
            "evidence_mode": "chunk",
            "context_chars": 20000,
            "min_content_len": 0,
        },
        "incremental_merge": {
            "enabled": False,
            "threshold": 0.8,
            "keep_newest_text": False,
        },
        "entities": {
            "method": "ml",
            "spacy_model": "unused",
            "huggingface_model": "unused",
            "device": "cpu",
            "confidence_threshold": 0.5,
            "merge_overlapping": True,
            "include_standard_types": True,
            "fallback_methods": [],
            "patterns": [],
        },
        "confidence": {
            "initial_method": "weighted_average",
            "evidence_method": "product",
            "source_weight": 0.5,
            "stance_weight": 0.5,
            "default_source_reliability": 0.55,
            "default_stance_quality": 0.65,
            "source_reliability": {"user": 0.85, "assistant": 0.65},
            "stance_quality": {"asserted": 0.9},
        },
        "chunking": {
            "enabled": False,
            "breakpoint_percentile_threshold": 95.0,
            "buffer_size": 1,
            "min_chunk_sentences": 1,
            "isolate_tool_calls": True,
        },
    }


class FakeHybridExtractor:
    """Stand-in for QwenChunkExtractor: no HTTP calls, scripted node output."""

    def extract_turn(
        self,
        chunks: list[Any],
        role: str,
        *,
        turn_content: str,
        graph_nodes: list[Any],
        context_chars: int,
        turn_index: int,
    ) -> list[list[ExtractedNode]]:
        del graph_nodes, context_chars, turn_index
        if "Final answer:" in turn_content:
            node = ExtractedNode(
                chunk_index=0,
                node_type="decision",
                text="The assistant concludes Alice likes green tea.",
            )
        elif role == "user":
            node = ExtractedNode(
                chunk_index=0,
                node_type="belief",
                text="The user states Alice likes tea.",
            )
        else:
            return [[] for _ in chunks]
        # Chunking is disabled in hybrid_belief_graph_config(), so there is
        # always exactly one chunk per turn; attach the node to it.
        return [[node]] + [[] for _ in chunks[1:]]


class FakeHybridStanceClassifier:
    """Stand-in for LocalZeroShotStanceClassifier: always 'asserted'."""

    def classify_texts(self, texts: list[str]) -> list[StancePrediction]:
        return [
            StancePrediction(
                stance="asserted",
                confidence=0.99,
                scores={
                    "asserted": 0.99,
                    "recalled": 0.0,
                    "judged": 0.0,
                    "speculated": 0.01,
                },
                model_path="fake",
            )
            for _ in texts
        ]


class FakeHybridEntityRecognizer:
    """Stand-in for NamedEntityRecognizer: no spaCy/HF model loading."""

    load_errors: list[Any] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def extract_entity_texts(self, text: str, **options: Any) -> list[str]:
        del text, options
        return []


@pytest.fixture
def fake_hybrid_construct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the hybrid backend's heavy components (HTTP extractor, local
    stance model, local NER model) with lightweight fakes. Patched on
    bcg.construct.hybrid.stream because that module imports these names
    directly (``from .extractor import get_extractor`` etc.), so patching the
    defining modules would not affect stream.py's own bound references."""

    monkeypatch.setattr(
        "bcg.construct.hybrid.stream.get_extractor",
        lambda config: FakeHybridExtractor(),
    )
    monkeypatch.setattr(
        "bcg.construct.hybrid.stream.get_stance_classifier",
        lambda config: FakeHybridStanceClassifier(),
    )
    monkeypatch.setattr(
        "bcg.construct.hybrid.stream.NamedEntityRecognizer",
        FakeHybridEntityRecognizer,
    )


def test_runner_incremental_session_methods_hybrid_backend(
    tmp_path: Path,
    fake_hybrid_construct: None,
) -> None:
    memory = BCGMemory(graph=BCG())
    runner = BCGRunner(
        memory=memory,
        llm=DummyLLM(),
        output_root=tmp_path / ".bcg" / "runs",
        backend="hybrid",
    )
    runner.begin_belief_run(
        run_id="incremental-run-hybrid",
        belief_graph_config=hybrid_belief_graph_config(),
    )
    runner.start_session("chat-1", "2024-01-01")
    run(runner.observe_turn("user", "Alice likes tea."))
    run(runner.end_session())
    result = run(runner.finalize())

    assert result.counts["sessions"] == 1
    assert result.counts["beliefs"] == 1
    assert result.memory["trajectory"][0]["session_id"] == "chat-1"
    assert result.graph.metadata["engine"] == "bcg.construct.hybrid"
    belief = result.memory["beliefs"][0]
    assert belief["belief"] == "The user states Alice likes tea."


def _normalized_backend_contract(result: Any) -> dict[str, Any]:
    memory = result.memory
    stable_node_fields = (
        "id",
        "node_type",
        "belief",
        "decision",
        "role",
        "stance",
        "layer",
        "confidence",
        "initial_confidence",
        "evidence_confidence",
        "factor_confidence",
        "entities",
        "evidence_ids",
        "factor_ids",
    )

    def normalized_nodes(key: str) -> list[dict[str, Any]]:
        return [
            {field: node[field] for field in stable_node_fields if field in node}
            for node in memory.get(key, [])
        ]

    return {
        "engine": memory.get("engine"),
        "counts": result.counts,
        "beliefs": normalized_nodes("beliefs"),
        "decisions": normalized_nodes("decisions"),
        "relations": memory.get("relations"),
        "sessions": memory.get("sessions"),
        "trajectory": memory.get("trajectory"),
        "artifact_names": sorted(
            path.name for path in result.output_paths.artifacts_dir.iterdir()
        ),
    }


@pytest.mark.parametrize("backend", ["unified", "hybrid"])
def test_backend_normalized_artifact_contract(
    backend: str,
    tmp_path: Path,
    fake_construct_calls: list[str],
    fake_hybrid_construct: None,
) -> None:
    del fake_construct_calls, fake_hybrid_construct
    runner = BCGRunner(
        memory=BCGMemory(graph=BCG()),
        llm=DummyLLM(),
        output_root=tmp_path / backend,
        backend=backend,
    )
    begin_options: dict[str, Any] = {"run_id": f"contract-{backend}"}
    if backend == "hybrid":
        begin_options["belief_graph_config"] = hybrid_belief_graph_config()
    runner.begin_belief_run(**begin_options)
    runner.start_session("chat-1", "2024-01-01")
    run(runner.observe_turn("user", "Alice likes tea."))
    run(
        runner.observe_turn(
            "assistant",
            "Final answer: Alice likes green tea.",
        )
    )
    run(runner.end_session())
    result = run(runner.finalize())

    expected_path = (
        Path(__file__).parent / "fixtures" / "refactor" / f"construct_{backend}.json"
    )
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    assert _normalized_backend_contract(result) == expected
    for output_path in result.output_paths.to_dict().values():
        assert Path(output_path).exists()


def test_semantic_split_clusters_with_fake_embeddings() -> None:
    content = "Alice likes tea. Alice drinks tea. Bob codes."
    sentences = split_sentences(content)
    chunks, info = semantic_breakpoint_chunks(sentences, content, FakeEmbedder())

    assert info["n_sentences"] == 3
    assert len(chunks) < len(sentences)
