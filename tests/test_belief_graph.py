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
from bcg.construct.api_based import BeliefGraphPipeline
from bcg.construct.api_based.confidence import (
    init_belief_confidence,
    initial_confidence,
)
from bcg.construct.api_based.evidence import evidence_from_excerpt, locate_excerpt
from bcg.construct.api_based.graph import BeliefGraph
from bcg.construct.api_based.online import SessionManager as ApiSessionManager
from bcg.construct.light.extractor import ExtractedNode, QwenChunkExtractor
from bcg.construct.light.llm import call_model, parse_json_response
from bcg.construct.light.online import SessionManager as LightSessionManager
from bcg.construct.light.split import semantic_breakpoint_chunks, split_sentences
from bcg.construct.light.stance import StancePrediction
from bcg.graph import BCGEdge, BCGNode, BeliefPayload, BeliefSource, RelationPayload
from bcg.llm import LLMResponse
from bcg.runner import _bcg_from_construct, _ConstructClientAdapter

T = TypeVar("T")


@pytest.mark.parametrize("manager_type", [LightSessionManager, ApiSessionManager])
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


def test_light_parser_accepts_one_json_object_with_trailing_brace() -> None:
    response = '{"beliefs": ["A"], "decisions": []}\n}'

    assert parse_json_response(response) == {
        "beliefs": ["A"],
        "decisions": [],
    }


def test_light_call_model_forwards_json_response_format() -> None:
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


def test_light_call_model_reduces_output_budget_after_context_overflow(
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

    monkeypatch.setattr("bcg.construct.light.llm.time.sleep", lambda _: None)
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


def test_light_extractor_retries_overflow_without_historical_nodes(
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

    monkeypatch.setattr("bcg.construct.light.extractor.call_model", fake_call_model)
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

    monkeypatch.setattr("bcg.construct.api_based.llm.call_model", fake_call_model)
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
    assert result.memory["engine"] == "bcg.construct.api_based"


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
    assert result.graph.metadata["engine"] == "bcg.construct.api_based"


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


def light_belief_graph_config() -> dict[str, Any]:
    """Minimal-but-complete belief_graph section for the light backend.

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


class FakeLightExtractor:
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
        # Chunking is disabled in light_belief_graph_config(), so there is
        # always exactly one chunk per turn; attach the node to it.
        return [[node]] + [[] for _ in chunks[1:]]


class FakeLightStanceClassifier:
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


class FakeLightEntityRecognizer:
    """Stand-in for NamedEntityRecognizer: no spaCy/HF model loading."""

    load_errors: list[Any] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def extract_entity_texts(self, text: str, **options: Any) -> list[str]:
        del text, options
        return []


@pytest.fixture
def fake_light_construct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the light backend's heavy components (HTTP extractor, local
    stance model, local NER model) with lightweight fakes. Patched on
    bcg.construct.light.stream because that module imports these names
    directly (``from .extractor import get_extractor`` etc.), so patching the
    defining modules would not affect stream.py's own bound references."""

    monkeypatch.setattr(
        "bcg.construct.light.stream.get_extractor",
        lambda config: FakeLightExtractor(),
    )
    monkeypatch.setattr(
        "bcg.construct.light.stream.get_stance_classifier",
        lambda config: FakeLightStanceClassifier(),
    )
    monkeypatch.setattr(
        "bcg.construct.light.stream.NamedEntityRecognizer",
        FakeLightEntityRecognizer,
    )


def test_runner_incremental_session_methods_light_backend(
    tmp_path: Path,
    fake_light_construct: None,
) -> None:
    memory = BCGMemory(graph=BCG())
    runner = BCGRunner(
        memory=memory,
        llm=DummyLLM(),
        output_root=tmp_path / ".bcg" / "runs",
        backend="light",
    )
    runner.begin_belief_run(
        run_id="incremental-run-light",
        belief_graph_config=light_belief_graph_config(),
    )
    runner.start_session("chat-1", "2024-01-01")
    run(runner.observe_turn("user", "Alice likes tea."))
    run(runner.end_session())
    result = run(runner.finalize())

    assert result.counts["sessions"] == 1
    assert result.counts["beliefs"] == 1
    assert result.memory["trajectory"][0]["session_id"] == "chat-1"
    assert result.graph.metadata["engine"] == "bcg.construct.light"
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


@pytest.mark.parametrize("backend", ["api_based", "light"])
def test_backend_normalized_artifact_contract(
    backend: str,
    tmp_path: Path,
    fake_construct_calls: list[str],
    fake_light_construct: None,
) -> None:
    del fake_construct_calls, fake_light_construct
    runner = BCGRunner(
        memory=BCGMemory(graph=BCG()),
        llm=DummyLLM(),
        output_root=tmp_path / backend,
        backend=backend,
    )
    begin_options: dict[str, Any] = {"run_id": f"contract-{backend}"}
    if backend == "light":
        begin_options["belief_graph_config"] = light_belief_graph_config()
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
        Path(__file__).parent
        / "fixtures"
        / "refactor"
        / f"construct_{backend}.json"
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
