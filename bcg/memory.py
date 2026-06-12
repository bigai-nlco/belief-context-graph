"""Belief memory interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bcg.belief_graph.confidence import ConfidenceConfig, init_belief_confidence
from bcg.graph import (
    BCG,
    BeliefPayload,
    BeliefSource,
)


@dataclass(frozen=True, slots=True)
class MemoryObservation:
    """Result returned by synchronous manual observations."""

    belief: BeliefPayload
    graph: BCG


@dataclass(slots=True)
class BCGMemory:
    """User-facing belief-native memory protocol."""

    namespace: str = "default"
    options: dict[str, Any] = field(default_factory=dict)
    graph: BCG | None = None
    confidence_config: ConfidenceConfig | None = None

    def observe(
        self,
        *,
        source_type: str,
        content: str,
        actor: dict[str, Any] | None = None,
        observed_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryObservation:
        """Observe one already-formed evidence episode as an asserted belief."""

        graph = self._graph()
        source = BeliefSource(
            type=source_type,
            role=str((actor or {}).get("role") or source_type),
            trajectory_index=-1,
            segment_index=0,
            segment_type=source_type,
        )
        belief_dict: dict[str, Any] = {
            "id": graph.allocate_belief_id(),
            "belief": content,
            "stance": "asserted",
            "layer": "io",
            "source": source.model_dump(mode="json"),
            "supporting_excerpts": [content],
            "evidence": [
                {
                    "text": content,
                    "start": 0,
                    "end": len(content),
                    "match": "exact",
                    "via": "manual",
                    "source": source.model_dump(mode="json"),
                }
            ],
            "metadata": {
                **(metadata or {}),
                "namespace": self.namespace,
                "actor": actor,
                "observed_at": observed_at,
            },
        }
        init_belief_confidence(belief_dict, config=self.confidence_config)
        belief = BeliefPayload.model_validate(belief_dict)
        graph.add_belief(belief)
        return MemoryObservation(belief=belief, graph=graph)

    def believe(self, target: str, **kwargs: Any) -> Any:
        """Read belief nodes matching a target substring."""

        max_results = int(kwargs.get("max_results", 10))
        target_lower = target.lower()
        matches = [
            belief
            for belief in self._graph().beliefs()
            if target_lower in belief.belief.lower()
        ]
        return matches[:max_results]

    def context(
        self,
        *,
        task: str,
        focal_entities: list[str] | None = None,
        max_variables: int = 100,
        include_conflicts: bool = True,
        include_missing_evidence: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Assemble belief-aware task context from the current graph."""

        del include_conflicts, include_missing_evidence, kwargs
        beliefs = self._graph().beliefs()
        if focal_entities:
            needles = [entity.lower() for entity in focal_entities]
            beliefs = [
                belief
                for belief in beliefs
                if any(needle in belief.belief.lower() for needle in needles)
            ]
        beliefs = sorted(beliefs, key=lambda belief: belief.confidence, reverse=True)
        return {
            "task": task,
            "namespace": self.namespace,
            "beliefs": [
                belief.model_dump(mode="json") for belief in beliefs[:max_variables]
            ],
        }

    async def search(
        self,
        *,
        query: str,
        max_results: int = 10,
        include_conflicts: bool = True,
        include_missing_evidence: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Search belief text in the current graph."""

        del include_conflicts, include_missing_evidence, kwargs
        query_lower = query.lower()
        matches = [
            belief
            for belief in self._graph().beliefs()
            if query_lower in belief.belief.lower()
        ]
        matches = sorted(matches, key=lambda belief: belief.confidence, reverse=True)
        return [belief.model_dump(mode="json") for belief in matches[:max_results]]

    def _graph(self) -> BCG:
        if self.graph is None:
            self.graph = BCG(metadata={"namespace": self.namespace})
        return self.graph
