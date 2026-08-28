from __future__ import annotations

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from bcg.construct._shared.context_selection import select_connected_context
from bcg.construct.hybrid.llm import LocalEmbeddingClient as HybridLocalEmbedder
from bcg.construct.unified.llm import LocalEmbeddingClient as UnifiedLocalEmbedder


class KeywordEmbedder:
    def embed(self, texts: list[str], purpose: str = "") -> list[list[float]]:
        assert purpose == "compact_context_selection"
        return [
            [1.0, 0.0] if "target" in text.casefold() else [0.0, 1.0] for text in texts
        ]


def node(node_id: int, text: str, turn: int, **extra: Any) -> dict[str, Any]:
    return {
        "id": node_id,
        "node_type": "belief",
        "belief": text,
        "confidence": 0.8,
        "source": {"turn_id": turn},
        **extra,
    }


def test_connected_selection_preserves_path_and_filters_empty_search() -> None:
    snapshot = {
        "beliefs": [
            node(1, "support premise " + "a" * 180, 2),
            node(2, "intermediate clue " + "b" * 180, 3),
            node(3, "target answer evidence " + "c" * 180, 4),
            node(4, "unrelated distractor " + "d" * 500, 4),
            node(5, "The web_search tool returned no results.", 4),
        ],
        "relations": [
            {"id": 11, "from_id": 3, "to_id": 2, "type": "depends_on"},
            {"id": 12, "from_id": 2, "to_id": 1, "type": "depends_on"},
        ],
    }

    result = select_connected_context(
        snapshot,
        "find target answer",
        embedder=KeywordEmbedder(),
        node_char_budget=700,
        max_depth=4,
    )

    assert {1, 2, 3}.issubset(result["node_ids"])
    assert 5 not in result["node_ids"]
    assert result["relation_ids"] == [11, 12]
    assert result["retrieval"] == "embedding"


def test_connected_selection_keeps_every_eligible_node_when_all_fit() -> None:
    snapshot = {
        "beliefs": [node(1, "one", 2), node(2, "two", 3)],
        "relations": [{"id": 7, "from_id": 2, "to_id": 1, "type": "supplements"}],
    }

    result = select_connected_context(
        snapshot,
        "anything",
        embedder=None,
        node_char_budget=1_000,
    )

    assert result["node_ids"] == [1, 2]
    assert result["relation_ids"] == [7]
    assert result["retrieval"] == "all_fit"


@pytest.mark.parametrize("client_class", [UnifiedLocalEmbedder, HybridLocalEmbedder])
def test_local_embedding_encode_is_serialized_across_workers(client_class: Any) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def encode(self, batch: list[str], **_kwargs: Any) -> list[list[float]]:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.01)
            with self.lock:
                self.active -= 1
            return [[1.0, 0.0] for _ in batch]

    client = client_class(
        {"model": "fake", "device": "cpu", "batch_size": 1, "max_length": 512}
    )
    fake = FakeModel()
    client._ensure_model = lambda: fake

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                lambda index: client.embed([f"uncached-{index}"], purpose="test"),
                range(8),
            )
        )

    assert fake.max_active == 1


@pytest.mark.parametrize("client_class", [UnifiedLocalEmbedder, HybridLocalEmbedder])
@pytest.mark.parametrize(("configured_max", "expected_max"), [(8_192, 256), (128, 128)])
def test_local_embedding_max_length_never_exceeds_native_limit(
    client_class: Any,
    configured_max: int,
    expected_max: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        max_seq_length = 256

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = lambda *_args, **_kwargs: FakeModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    client = client_class(
        {"model": "fake", "device": "cpu", "max_length": configured_max}
    )

    assert client._ensure_model().max_seq_length == expected_max
