from __future__ import annotations

from typing import Any

import pytest

from bcg.construct.hybrid import confidence as hybrid_confidence
from bcg.construct.unified import confidence as unified_confidence
from bcg.core.confidence import select_independent_evidence

BACKENDS = [unified_confidence, hybrid_confidence]


def _node(confidence: float, **overrides: Any) -> dict[str, Any]:
    node = {
        "initial_confidence": confidence,
        "evidence_confidence": 0.0,
        "factor_confidence": 0.0,
        "confidence": confidence,
    }
    node.update(overrides)
    return node


def _config(*, threshold: float = 0.8, max_iterations: int = 3) -> dict[str, Any]:
    return {
        "relation_propagation": {
            "default_relation_weight": 0.5,
            "input_confidence_threshold": threshold,
            "min_confidence_delta": 0.0,
            "max_iterations": max_iterations,
        }
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_relation_activation_is_continuous_above_threshold(backend: Any) -> None:
    nodes = {1: _node(0.5), 2: _node(0.81)}
    relation = {
        "id": 1,
        "from_id": 1,
        "to_id": 2,
        "type": "depends_on",
        "weight": 0.5,
    }

    contribution = backend.relation_factor_contribution(
        relation,
        nodes,
        config=_config(),
    )

    assert contribution is not None
    assert contribution["activation_strength"] == pytest.approx(0.05)
    assert contribution["contribution"] == pytest.approx(0.025)


@pytest.mark.parametrize("backend", BACKENDS)
def test_relation_weight_is_capped_at_one(backend: Any) -> None:
    nodes = {1: _node(0.5), 2: _node(0.9)}
    relation = {
        "id": 1,
        "from_id": 1,
        "to_id": 2,
        "type": "depends_on",
        "weight": 8.0,
    }

    contribution = backend.relation_factor_contribution(
        relation,
        nodes,
        config=_config(threshold=0.0),
    )

    assert contribution is not None
    assert contribution["weight"] == 1.0
    assert contribution["contribution"] == pytest.approx(0.9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_factor_uses_only_strongest_relation_per_direction(backend: Any) -> None:
    nodes = {
        1: _node(0.5),
        2: _node(0.9),
        3: _node(0.85),
        4: _node(0.95),
    }
    relations = [
        {"id": 1, "from_id": 1, "to_id": 2, "type": "depends_on", "weight": 1},
        {"id": 2, "from_id": 1, "to_id": 3, "type": "depends_on", "weight": 1},
        {"id": 3, "from_id": 4, "to_id": 1, "type": "contradicts", "weight": 1},
    ]

    score, contributions = backend.compute_factor_confidence(
        1,
        nodes,
        relations,
        config=_config(threshold=0.0),
    )

    assert score == pytest.approx(-0.05)
    assert {item["relation_id"] for item in contributions} == {1, 3}


@pytest.mark.parametrize("backend", BACKENDS)
def test_depends_on_does_not_inflate_procedural_tool_call(backend: Any) -> None:
    nodes = {
        1: _node(
            0.45,
            role="assistant",
            extraction_method="rule_tool_call",
            query="find the answer",
        ),
        2: _node(0.9, role="user"),
    }
    relation = {
        "id": 1,
        "from_id": 1,
        "to_id": 2,
        "type": "depends_on",
        "weight": 1.0,
    }

    assert (
        backend.relation_factor_contribution(
            relation,
            nodes,
            config=_config(threshold=0.0),
        )
        is None
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_relation_iteration_reads_one_synchronous_snapshot(backend: Any) -> None:
    nodes = {0: _node(0.5), 1: _node(0.5), 2: _node(0.5)}
    relations = [
        {"id": 1, "from_id": 1, "to_id": 0, "type": "depends_on", "weight": 1},
        {"id": 2, "from_id": 2, "to_id": 1, "type": "depends_on", "weight": 1},
    ]

    backend.propagate_relation_confidences(
        nodes,
        relations,
        config=_config(threshold=0.0, max_iterations=1),
        record_history=False,
    )

    assert nodes[1]["confidence"] == 0.622
    assert nodes[2]["confidence"] == 0.622


def test_repeated_url_across_turns_is_one_independent_observation() -> None:
    records = [
        (
            1,
            {
                "text": "Result from https://example.test/report: value is 42.",
                "source": {"item_id": "task", "turn_id": 1},
            },
        ),
        (
            2,
            {
                "text": "Revisited https://example.test/report: value is 42.",
                "source": {"item_id": "task", "turn_id": 3},
            },
        ),
    ]

    evidence_ids, _ = select_independent_evidence(
        records,
        contribution=lambda _: 1.0,
    )

    assert evidence_ids == [1]


def test_repeated_normalized_content_across_turns_is_one_observation() -> None:
    records = [
        (
            1,
            {
                "text": "tool_call_id=call-a The reported value is 42.",
                "tool_call_id": "call-a",
                "source": {"item_id": "task", "turn_id": 1},
            },
        ),
        (
            2,
            {
                "text": "tool_call_id=call-b The reported value is 42.",
                "tool_call_id": "call-b",
                "source": {"item_id": "task", "turn_id": 3},
            },
        ),
    ]

    evidence_ids, _ = select_independent_evidence(
        records,
        contribution=lambda _: 1.0,
    )

    assert evidence_ids == [1]


def test_hybrid_source_type_takes_precedence_over_flat_role() -> None:
    config = {
        "source_reliability": {
            "assistant": 0.1,
            "llm_reasoning": 0.9,
        },
        "stance_quality": {"asserted": 1.0},
    }
    node = {
        "role": "assistant",
        "source_type": "llm_reasoning",
        "stance": "asserted",
    }

    hybrid_confidence.init_belief_confidence(node, config=config)

    assert node["confidence_history"][0]["confidence_config"]["source"] == (
        "llm_reasoning"
    )
    assert node["confidence"] == pytest.approx(0.95)
