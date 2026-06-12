from __future__ import annotations

import asyncio
import importlib.util
import json
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, TypeVar

from bcg import BCG, BCGMemory, BCGRunner
from bcg.belief_graph.confidence import apply_relations, initial_confidence
from bcg.belief_graph.evidence import evidence_from_excerpt, locate_excerpt
from bcg.belief_graph.extraction import clean_belief
from bcg.belief_graph.linking import validate_relations
from bcg.belief_graph.merge import run_merge_pass
from bcg.belief_graph.pipeline import BeliefGraphPipeline
from bcg.belief_graph.segment import segment_trajectory
from bcg.belief_graph.split import cluster_sentences, split_sentences
from bcg.graph import BCGEdge, BCGNode, BeliefPayload, BeliefSource, RelationPayload
from bcg.llm import parse_json_response

T = TypeVar("T")


class FakeLLM:
    def __init__(self) -> None:
        self.labels: list[str] = []

    async def generate_text(
        self,
        prompt: str,
        *,
        label: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        del prompt, temperature, max_tokens
        self.labels.append(label or "")
        payloads = {
            "s0.t0.extract:user_input": {
                "beliefs": [
                    {
                        "belief": "User states Alice likes tea.",
                        "stance": "asserted",
                        "supporting_excerpts": ["Alice likes tea"],
                    }
                ]
            },
            "s0.t1.extract:tool_call": {
                "beliefs": [
                    {
                        "belief": "Assistant queries for Alice tea evidence.",
                        "stance": "asserted",
                        "supporting_excerpts": ["Alice tea evidence"],
                    }
                ]
            },
            "s0.t1.extract:assistant_other": {
                "beliefs": [
                    {
                        "belief": "Assistant says it will check Alice tea evidence.",
                        "stance": "asserted",
                        "supporting_excerpts": ["Checking Alice tea evidence"],
                    }
                ]
            },
            "s0.t3.extract:assistant_other": {
                "beliefs": [
                    {
                        "belief": "Assistant concludes Alice likes green tea.",
                        "stance": "asserted",
                        "supporting_excerpts": ["Alice likes green tea"],
                    }
                ]
            },
            "s0.t1.extract:think": {
                "beliefs": [
                    {
                        "belief": "Assistant speculates Alice may prefer tea.",
                        "stance": "speculated",
                        "supporting_excerpts": ["Alice may prefer tea"],
                    }
                ]
            },
            "s0.t2.extract:tool_response": {
                "beliefs": [
                    {
                        "belief": "Tool result says Alice likes green tea.",
                        "stance": "asserted",
                        "supporting_excerpts": ["Alice likes green tea"],
                    }
                ]
            },
            "s0.t1.forward": {
                "forward_relations": [
                    {
                        "from_id": 0,
                        "to_id": 1,
                        "type": "informs",
                        "note": "The user statement informs the reasoning.",
                    }
                ]
            },
            "s0.t2.forward": {"forward_relations": []},
            "s0.t3.forward": {"forward_relations": []},
            "s0.backward": {
                "relations": [
                    {
                        "from_id": 4,
                        "to_id": 1,
                        "type": "confirms",
                        "note": "The tool result confirms the speculation.",
                    }
                ]
            },
        }
        return json.dumps(payloads[label or ""])


class FakeMergeLLM:
    async def generate_text(
        self,
        prompt: str,
        *,
        label: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        del prompt, label, temperature, max_tokens
        return json.dumps(
            {
                "merge_groups": [
                    {
                        "ids": [0, 1],
                        "canonical_belief": "User likes green tea.",
                        "reason": "Both beliefs state the same preference.",
                    }
                ]
            }
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
        {"role": "user", "content": "Alice likes tea."},
        {
            "role": "assistant",
            "content": (
                "<think>Alice may prefer tea.</think>"
                '<tool_call>{"query":"Alice tea evidence"}</tool_call>'
                "Checking Alice tea evidence before answering."
            ),
        },
        {
            "role": "user",
            "content": (
                "<tool_response>Alice likes green tea according to the profile "
                "record, and the record is specific enough to confirm the "
                "preference.</tool_response>"
            ),
        },
        {
            "role": "assistant",
            "content": "Final answer: Alice likes green tea.",
        },
    ]


def test_segment_trajectory_splits_tagged_messages() -> None:
    segments = segment_trajectory(sample_trajectory())

    assert [segment.type for segment in segments] == [
        "user_input",
        "think",
        "tool_call",
        "assistant_other",
        "tool_response",
        "assistant_other",
    ]
    assert segments[-1].is_last_assistant
    for segment in segments:
        original = sample_trajectory()[segment.traj_idx]["content"]
        assert segment.content == original[segment.start : segment.end]


def test_evidence_offsets_are_exact_when_excerpt_matches() -> None:
    message = "Before <think>Alice likes tea.</think> after"
    segment = [
        item
        for item in segment_trajectory([{"role": "assistant", "content": message}])
        if item.type == "think"
    ][0]
    assert locate_excerpt("Alice likes tea.", segment.content)[2] == "exact"
    evidence = evidence_from_excerpt(
        "Alice likes tea.",
        segment,
        message,
        {
            "type": "llm_reasoning",
            "role": "assistant",
            "trajectory_index": 0,
            "segment_index": 0,
            "segment_type": "think",
        },
    )
    assert message[evidence["start"] : evidence["end"]] == evidence["text"]


def test_clean_belief_requires_text_and_excerpt() -> None:
    assert clean_belief({"belief": "No excerpt"}) is None
    assert clean_belief(
        {
            "belief": " Alice likes tea. ",
            "stance": "unknown",
            "supporting_excerpts": [" Alice likes tea "],
        }
    ) == {
        "belief": "Alice likes tea.",
        "stance": "asserted",
        "supporting_excerpts": ["Alice likes tea"],
    }


def test_parse_json_response_handles_fences_and_prefix_text() -> None:
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('model says {"b": 2} done') == {"b": 2}


def test_confidence_rules_apply_backward_relations() -> None:
    beliefs = [
        {
            "id": 0,
            "belief": "Alice may like tea.",
            "stance": "speculated",
            "confidence": initial_confidence("llm_reasoning", "speculated"),
            "source": {"type": "llm_reasoning"},
        },
        {
            "id": 1,
            "belief": "Alice likes tea.",
            "stance": "asserted",
            "confidence": initial_confidence("tool_result", "asserted"),
            "source": {"type": "tool_result"},
        },
    ]
    updated = apply_relations(
        beliefs,
        [{"from_id": 1, "to_id": 0, "type": "confirms", "note": "confirmed"}],
    )

    assert updated[0]["confidence"] > beliefs[0]["confidence"]
    assert updated[0]["confidence_history"][-1]["step"] == "confirms"
    dimensions = updated[0]["confidence_history"][-1]["dimensions"]
    assert updated[0]["confidence"] == round(sum(dimensions.values()) / 4, 3)


def test_relation_validation_filters_invalid_edges() -> None:
    beliefs = [{"id": 0}, {"id": 1}, {"id": 2}]
    relations = validate_relations(
        [
            {"from_id": 0, "to_id": 1, "type": "informs"},
            {"from_id": 2, "to_id": 1, "type": "informs"},
            {"from_id": 0, "to_id": 9, "type": "informs"},
            {"from_id": 0, "to_id": 1, "type": "informs"},
        ],
        beliefs,
        valid_types={"informs"},
        direction="forward",
    )

    assert relations == [{"from_id": 0, "to_id": 1, "type": "informs", "note": ""}]


def test_graph_keeps_constructor_compatibility_and_mutation() -> None:
    async def exercise() -> BCG:
        graph = BCG()
        node = BCGNode(name="manual")
        await graph.add_node(node)
        await graph.add_edge(BCGEdge(source=node.uuid, target=node.uuid))
        return graph

    graph = run(exercise())
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 1


def test_graph_exports_typed_memory_shape() -> None:
    source = BeliefSource(
        type="user_input",
        role="user",
        trajectory_index=0,
        segment_index=0,
        segment_type="user_input",
    )
    graph = BCG()
    node = graph.add_belief(
        BeliefPayload(
            id=0,
            belief="User states Alice likes tea.",
            stance="asserted",
            layer="io",
            source=source,
            supporting_excerpts=["Alice likes tea"],
            confidence=0.95,
        )
    )
    graph.add_relation(
        RelationPayload(from_id=0, to_id=0, type="extends"),
        source_uuid=node.uuid,
        target_uuid=node.uuid,
    )

    memory = graph.to_memory_dict()
    assert memory["counts"]["beliefs"] == 1
    assert memory["backward_relations"][0]["type"] == "extends"


def test_pipeline_writes_requested_output_layout(tmp_path: Path) -> None:
    pipeline = BeliefGraphPipeline(
        FakeLLM(),
        output_root=tmp_path / ".bcg" / "runs",
        run_id="unit-run",
    )
    result = run(pipeline.run(sample_trajectory(), metadata={"case": "unit"}))

    assert result.output_paths.graph.exists()
    assert result.output_paths.memory.exists()
    assert result.output_paths.token_usage.exists()
    assert result.output_paths.events.exists()
    assert result.output_paths.segments.exists()
    assert result.output_paths.merges.exists()
    assert result.output_paths.segments.name == "segments.json"
    assert result.output_paths.merges.name == "merges.json"
    assert result.output_paths.segments.parent.name == "artifacts"
    assert not result.output_paths.segments.name.startswith("01_")
    assert result.counts["beliefs"] == 6
    assert len(result.graph.beliefs()) == 6
    assert [belief.id for belief in result.graph.beliefs()] == list(range(6))


def test_memory_observe_and_trajectory_ingestion(tmp_path: Path) -> None:
    memory = BCGMemory(graph=BCG())
    observed = memory.observe(source_type="user_input", content="Alice likes tea.")
    assert observed.belief.belief == "Alice likes tea."
    assert observed.belief.confidence == round(
        sum(observed.belief.confidence_dimensions.values()) / 4,
        3,
    )

    result = run(
        BCGRunner(
            memory=memory,
            llm=FakeLLM(),
            output_root=tmp_path / ".bcg" / "runs",
        ).observe_trajectory(
            sample_trajectory(),
            run_id="memory-run",
        )
    )
    assert result.counts["beliefs"] == 6
    assert len(memory.believe("Alice")) == 6


def test_memory_incremental_session_methods(tmp_path: Path) -> None:
    memory = BCGMemory(graph=BCG())
    runner = BCGRunner(
        memory=memory,
        llm=FakeLLM(),
        output_root=tmp_path / ".bcg" / "runs",
    )
    runner.begin_belief_run(
        run_id="incremental-run",
    )
    runner.start_session("chat-1", "2024-01-01")
    run(runner.observe_turn("user", "Alice likes tea."))
    run(runner.end_session())
    result = run(runner.finalize())

    assert result.counts["sessions"] == 1
    assert result.memory["trajectory"][0]["session_id"] == "chat-1"


def test_semantic_split_clusters_with_fake_embeddings() -> None:
    sentences = split_sentences("Alice likes tea. Alice drinks tea. Bob codes.")
    clusters, info = cluster_sentences(
        sentences,
        FakeEmbedder(),
        similarity_threshold=0.8,
    )

    assert info["n_sentences"] == 3
    assert len(clusters) < len(sentences)


def test_merge_pass_rewrites_graph_and_records_audit() -> None:
    graph = BCG()
    source = BeliefSource(
        type="user_input",
        role="user",
        trajectory_index=0,
        segment_index=0,
        segment_type="user_input",
    )
    first = graph.add_belief(
        BeliefPayload(
            id=0,
            belief="User likes tea.",
            stance="asserted",
            layer="io",
            source=source,
            supporting_excerpts=["User likes tea"],
            confidence=0.8,
            confidence_history=[{"step": "initial", "value": 0.8}],
        )
    )
    second = graph.add_belief(
        BeliefPayload(
            id=1,
            belief="User likes tea.",
            stance="asserted",
            layer="io",
            source=source,
            supporting_excerpts=["User likes tea"],
            confidence=0.9,
            confidence_history=[{"step": "initial", "value": 0.9}],
        )
    )
    graph.add_relation(
        RelationPayload(from_id=0, to_id=1, type="informs"),
        source_uuid=first.uuid,
        target_uuid=second.uuid,
    )

    result = run(
        run_merge_pass(
            graph=graph,
            strategy="embedding",
            generator=FakeMergeLLM(),
            embedder=FakeEmbedder(),
            threshold=0.8,
        )
    )

    assert not result.skipped
    assert len(graph.beliefs()) == 1
    assert graph.beliefs()[0].merged_from == [1]
    assert len(graph.merges) == 1


def test_visualizer_smoke_renders_memory_output(tmp_path: Path) -> None:
    pipeline = BeliefGraphPipeline(
        FakeLLM(),
        output_root=tmp_path / ".bcg" / "runs",
        run_id="viz-run",
    )
    result = run(pipeline.run(sample_trajectory()))

    script = Path.cwd() / "scripts" / "visualize_belief_graph.py"
    spec = importlib.util.spec_from_file_location("visualize_belief_graph", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rendered = module.render_html(result.memory, result.output_paths.memory)
    assert "BCG Belief Graph" in rendered
    assert "Alice likes green tea" in rendered
