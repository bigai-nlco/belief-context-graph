from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, TypeVar

import pytest

from bcg import BCG, BCGMemory, BCGRunner
from bcg.construct import BeliefGraphPipeline
from bcg.construct.confidence import init_belief_confidence, initial_confidence
from bcg.construct.evidence import evidence_from_excerpt, locate_excerpt
from bcg.construct.graph import BeliefGraph
from bcg.construct.split import cluster_sentences, split_sentences
from bcg.graph import BCGEdge, BCGNode, BeliefPayload, BeliefSource, RelationPayload
from bcg.llm import LLMResponse
from bcg.runner import _bcg_from_construct, _ConstructClientAdapter

T = TypeVar("T")


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

    monkeypatch.setattr("bcg.construct.llm.call_model", fake_call_model)
    return prompts


def test_construct_confidence_is_the_only_confidence_policy() -> None:
    node = {"role": "user", "stance": "asserted"}
    init_belief_confidence(node)

    assert node["confidence"] == initial_confidence("user", "asserted")
    assert node["initial_confidence"] == node["confidence"]
    assert node["confidence_history"] == [
        {"step": "initial", "value": node["confidence"]}
    ]


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
    assert result.memory["engine"] == "bcg.construct"


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
    assert result.graph.metadata["engine"] == "bcg.construct"


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


def test_semantic_split_clusters_with_fake_embeddings() -> None:
    sentences = split_sentences("Alice likes tea. Alice drinks tea. Bob codes.")
    clusters, info = cluster_sentences(
        sentences,
        FakeEmbedder(),
        similarity_threshold=0.8,
    )

    assert info["n_sentences"] == 3
    assert len(clusters) < len(sentences)
