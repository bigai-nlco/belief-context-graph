"""Belief Context Graph schema and in-memory graph container."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bcg.utils import get_random_uuid, utc_now

BeliefLayer = Literal["io", "reasoning"]
BeliefStance = Literal["asserted", "recalled", "speculated", "judged"]
RelationType = Literal[
    "depends_on",
    "supplements",
    "contradicts",
    # Legacy relation names remain readable during the compatibility window.
    "informs",
    "confirms",
    "extends",
]
NodeKind = Literal["belief", "decision", "evidence", "factor", "episode", "entity"]
CONFIDENCE_DIMENSION_KEYS = (
    "source_reliability",
    "evidence_directness",
    "claim_specificity",
    "linguistic_certainty",
)


class TrajectorySegment(BaseModel):
    """Segment-level provenance for a belief or evidence excerpt."""

    trajectory_index: int = Field(..., description="Index of the source message.")
    segment_index: int = Field(..., description="Ordinal within the source message.")
    role: str = Field(..., description="Role of the source message.")
    type: str = Field(..., description="Semantic segment type.")
    start: int = Field(..., ge=0, description="Segment start offset.")
    end: int = Field(..., ge=0, description="Segment exclusive end offset.")
    is_last_assistant: bool = False


class BeliefSource(BaseModel):
    """Source metadata for an extracted belief."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="Source type used by confidence rules.")
    role: str = Field(..., description="Role of the host message.")
    trajectory_index: int = Field(..., description="Index of the host message.")
    segment_index: int = Field(..., description="Index of the host segment.")
    segment_type: str = Field(..., description="Semantic host segment type.")
    scenario: str | None = None
    item_id: str | None = None
    session_id: str | None = None
    session_index: int | None = None
    session_date: str | None = None
    turn_index: int | None = None
    segment_start: int | None = Field(default=None, ge=0)
    segment_end: int | None = Field(default=None, ge=0)
    has_answer: bool | None = None


class EvidenceExcerpt(BaseModel):
    """Textual support for a belief."""

    model_config = ConfigDict(extra="allow")

    text: str = Field(..., min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    match: Literal["exact", "normalized", "fuzzy", "not_found"] | None = None
    via: Literal["llm_excerpt", "split_sentence", "manual"] | None = None
    source: BeliefSource | None = None


class ConfidenceHistoryEntry(BaseModel):
    """Audit record for confidence assignment and updates."""

    step: str
    value: float = Field(..., ge=0.0, le=1.0)
    reason: str = ""
    delta: float | None = None
    from_belief_id: int | None = None
    method: str = "dimension_average"
    dimensions: dict[str, float] = Field(default_factory=dict)
    formula: str | None = None
    fallback_used: bool = False
    model: str | None = None

    @model_validator(mode="after")
    def _derive_value_from_dimensions(self) -> ConfidenceHistoryEntry:
        if self.dimensions:
            self.dimensions = _normalized_confidence_dimensions(self.dimensions)
            self.value = _confidence_from_dimensions(self.dimensions)
            self.formula = (
                self.formula
                or "average(source_reliability, evidence_directness, claim_specificity, linguistic_certainty)"
            )
        return self


class BeliefPayload(BaseModel):
    """Typed payload for belief nodes."""

    model_config = ConfigDict(extra="allow")

    id: int = Field(..., ge=0)
    node_type: Literal["belief", "decision"] = "belief"
    belief: str = Field(..., min_length=1)
    decision: str | None = None
    stance: BeliefStance = "asserted"
    layer: BeliefLayer
    role: str | None = None
    entities: list[str] = Field(default_factory=list)
    source: BeliefSource
    evidence_ids: list[int] = Field(default_factory=list)
    factor_ids: list[int] = Field(default_factory=list)
    supporting_excerpts: list[str] = Field(default_factory=list)
    evidence: list[EvidenceExcerpt] = Field(default_factory=list)
    event_time: str | None = None
    time_text: str | None = None
    merged_from: list[int] = Field(default_factory=list)
    belief_original: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_dimensions: dict[str, float] = Field(default_factory=dict)
    initial_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_confidence: float = 0.0
    factor_confidence: float = 0.0
    confidence_history: list[ConfidenceHistoryEntry] = Field(default_factory=list)
    decision_history: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_evidence_and_excerpts(self) -> BeliefPayload:
        if self.confidence_dimensions:
            self.confidence_dimensions = _normalized_confidence_dimensions(
                self.confidence_dimensions
            )
            self.confidence = _confidence_from_dimensions(self.confidence_dimensions)
        if not self.evidence and self.supporting_excerpts:
            self.evidence = [
                EvidenceExcerpt(text=excerpt, source=self.source)
                for excerpt in self.supporting_excerpts
                if excerpt
            ]
        if not self.supporting_excerpts and self.evidence:
            self.supporting_excerpts = [
                evidence.text for evidence in self.evidence if evidence.text
            ]
        return self


def _normalized_confidence_dimensions(
    dimensions: dict[str, float],
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key in CONFIDENCE_DIMENSION_KEYS:
        raw = dimensions.get(key, 0.0)
        normalized[key] = min(1.0, max(0.0, round(float(raw), 3)))
    return normalized


def _confidence_from_dimensions(dimensions: dict[str, float]) -> float:
    normalized = _normalized_confidence_dimensions(dimensions)
    return round(sum(normalized.values()) / len(CONFIDENCE_DIMENSION_KEYS), 3)


class RelationPayload(BaseModel):
    """Typed payload for belief relation edges."""

    model_config = ConfigDict(extra="allow")

    from_id: int = Field(..., ge=0)
    to_id: int = Field(..., ge=0)
    type: RelationType
    note: str = ""


class BCGNode(BaseModel):
    """Node in a Belief Context Graph."""

    uuid: str = Field(
        default_factory=get_random_uuid, description="Unique identifier for the node."
    )
    name: str = Field(..., description="Human-readable node name.")
    kind: NodeKind = Field(default="belief", description="Node semantic kind.")
    probability: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Node confidence/probability."
    )
    belief: BeliefPayload | None = Field(
        default=None, description="Typed belief payload for belief nodes."
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Additional payload data."
    )
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_belief(cls, belief: BeliefPayload) -> BCGNode:
        return cls(
            name=belief.belief,
            kind=belief.node_type,
            probability=belief.confidence,
            belief=belief,
            payload=belief.model_dump(mode="json"),
            metadata={
                "belief_id": belief.id,
                "source_type": belief.source.type,
                "layer": belief.layer,
            },
        )


class BCGEdge(BaseModel):
    """Edge in a Belief Context Graph."""

    uuid: str = Field(
        default_factory=get_random_uuid, description="Unique identifier for the edge."
    )
    source: str = Field(..., description="UUID of the source node.")
    target: str = Field(..., description="UUID of the target node.")
    weight: float = Field(default=1.0, description="Edge weight.")
    relation: RelationPayload | None = Field(
        default=None, description="Typed relation payload for belief edges."
    )
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_relation(
        cls,
        relation: RelationPayload,
        *,
        source_uuid: str,
        target_uuid: str,
    ) -> BCGEdge:
        return cls(
            source=source_uuid,
            target=target_uuid,
            relation=relation,
            payload=relation.model_dump(mode="json"),
            metadata={"relation_type": relation.type},
        )


class BCG(BaseModel):
    """In-memory Belief Context Graph container."""

    nodes: list[BCGNode] = Field(default_factory=list)
    edges: list[BCGEdge] = Field(default_factory=list)
    merges: list[dict[str, Any]] = Field(default_factory=list)
    sessions: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    factors: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    async def add_node(self, node: BCGNode) -> None:
        """Add a node to the graph."""

        self.nodes.append(node)

    async def add_edge(self, edge: BCGEdge) -> None:
        """Add an edge to the graph."""

        self.edges.append(edge)

    def add_belief(self, belief: BeliefPayload) -> BCGNode:
        """Create and add a belief node."""

        self.metadata["next_belief_id"] = max(self.next_belief_id, belief.id + 1)
        node = BCGNode.from_belief(belief)
        self.nodes.append(node)
        return node

    @property
    def next_belief_id(self) -> int:
        """Return the next monotonic belief id."""

        stored = self.metadata.get("next_belief_id")
        if isinstance(stored, int):
            return stored
        return max((belief.id for belief in self.beliefs()), default=-1) + 1

    def allocate_belief_id(self) -> int:
        """Allocate a stable monotonic belief id."""

        belief_id = self.next_belief_id
        self.metadata["next_belief_id"] = belief_id + 1
        return belief_id

    def add_relation(
        self,
        relation: RelationPayload,
        *,
        source_uuid: str,
        target_uuid: str,
    ) -> BCGEdge:
        """Create and add a relation edge."""

        edge = BCGEdge.from_relation(
            relation,
            source_uuid=source_uuid,
            target_uuid=target_uuid,
        )
        self.edges.append(edge)
        return edge

    def add_relation_by_ids(self, relation: RelationPayload) -> BCGEdge | None:
        """Create a relation edge between belief ids if both endpoints exist."""

        source_node = self.node_for_belief_id(relation.from_id)
        target_node = self.node_for_belief_id(relation.to_id)
        if source_node is None or target_node is None:
            return None
        key = (relation.from_id, relation.to_id, relation.type)
        for existing in self.relations():
            if (existing.from_id, existing.to_id, existing.type) == key:
                return None
        return self.add_relation(
            relation,
            source_uuid=source_node.uuid,
            target_uuid=target_node.uuid,
        )

    def add_relations(self, relations: list[dict[str, Any]]) -> int:
        """Validate and add relation dictionaries."""

        added = 0
        for relation_dict in relations:
            relation = RelationPayload.model_validate(relation_dict)
            if self.add_relation_by_ids(relation) is not None:
                added += 1
        return added

    def belief_nodes(self) -> list[BCGNode]:
        """Return belief nodes ordered by belief id."""

        return sorted(
            [node for node in self.nodes if node.belief is not None],
            key=lambda node: node.belief.id if node.belief is not None else -1,
        )

    def beliefs(self) -> list[BeliefPayload]:
        """Return typed belief payloads ordered by belief id."""

        return [node.belief for node in self.belief_nodes() if node.belief is not None]

    def belief_dicts(self) -> list[dict[str, Any]]:
        """Return active belief payloads as JSON-compatible dictionaries."""

        return [belief.model_dump(mode="json") for belief in self.beliefs()]

    def relations(
        self, relation_type: RelationType | None = None
    ) -> list[RelationPayload]:
        """Return typed relation payloads."""

        relations = [edge.relation for edge in self.edges if edge.relation is not None]
        if relation_type is None:
            return relations
        return [relation for relation in relations if relation.type == relation_type]

    def node_for_belief_id(self, belief_id: int) -> BCGNode | None:
        """Find a node by typed belief id."""

        for node in self.nodes:
            if node.belief is not None and node.belief.id == belief_id:
                return node
        return None

    def update_belief(self, belief: BeliefPayload | dict[str, Any]) -> None:
        """Replace a belief payload while preserving its node uuid."""

        payload = (
            belief
            if isinstance(belief, BeliefPayload)
            else BeliefPayload.model_validate(belief)
        )
        node = self.node_for_belief_id(payload.id)
        if node is None:
            self.add_belief(payload)
            return
        node.name = payload.belief
        node.probability = payload.confidence
        node.belief = payload
        node.payload = payload.model_dump(mode="json")
        node.metadata.update(
            {
                "belief_id": payload.id,
                "source_type": payload.source.type,
                "layer": payload.layer,
            }
        )
        node.updated_at = utc_now()

    def remove_belief(
        self, belief_id: int, *, drop_edges: bool = True
    ) -> BCGNode | None:
        """Remove one active belief node and any incident relation edges."""

        node = self.node_for_belief_id(belief_id)
        if node is None:
            return None
        self.nodes = [item for item in self.nodes if item.uuid != node.uuid]
        if drop_edges:
            self.edges = [
                edge
                for edge in self.edges
                if edge.relation is None
                or (
                    edge.relation.from_id != belief_id
                    and edge.relation.to_id != belief_id
                )
            ]
        return node

    def relation_dicts(
        self, relation_type: RelationType | None = None
    ) -> list[dict[str, Any]]:
        """Return typed relation payloads as JSON-compatible dictionaries."""

        return [
            relation.model_dump(mode="json")
            for relation in self.relations(relation_type)
        ]

    def remap_relations(self, mapping: dict[int, int]) -> dict[str, Any]:
        """Rewrite relation endpoints after merges and drop invalid edges."""

        report: dict[str, Any] = {
            "rewritten": 0,
            "dropped_self": 0,
            "dropped_direction": [],
            "dropped_duplicate": 0,
        }
        rebuilt: list[BCGEdge] = []
        seen: set[tuple[int, int, str]] = set()
        for edge in self.edges:
            relation = edge.relation
            if relation is None:
                rebuilt.append(edge)
                continue
            from_id = mapping.get(relation.from_id, relation.from_id)
            to_id = mapping.get(relation.to_id, relation.to_id)
            if from_id == to_id:
                report["dropped_self"] += 1
                continue
            if relation.type == "informs" and not from_id < to_id:
                report["dropped_direction"].append(
                    {
                        "type": relation.type,
                        "original": relation.model_dump(mode="json"),
                        "remapped": {"from_id": from_id, "to_id": to_id},
                    }
                )
                continue
            if relation.type in {"confirms", "contradicts", "extends"} and not (
                from_id > to_id
            ):
                report["dropped_direction"].append(
                    {
                        "type": relation.type,
                        "original": relation.model_dump(mode="json"),
                        "remapped": {"from_id": from_id, "to_id": to_id},
                    }
                )
                continue
            key = (from_id, to_id, relation.type)
            if key in seen:
                report["dropped_duplicate"] += 1
                continue
            seen.add(key)
            if from_id != relation.from_id or to_id != relation.to_id:
                relation = relation.model_copy(
                    update={"from_id": from_id, "to_id": to_id}
                )
                source_node = self.node_for_belief_id(from_id)
                target_node = self.node_for_belief_id(to_id)
                if source_node is None or target_node is None:
                    continue
                edge.source = source_node.uuid
                edge.target = target_node.uuid
                edge.relation = relation
                edge.payload = relation.model_dump(mode="json")
                edge.updated_at = utc_now()
                report["rewritten"] += 1
            rebuilt.append(edge)
        self.edges = rebuilt
        return report

    def to_memory_dict(self) -> dict[str, Any]:
        """Export the graph in a compact memory-protocol shape."""

        nodes = [belief.model_dump(mode="json") for belief in self.beliefs()]
        beliefs = [node for node in nodes if node.get("node_type") == "belief"]
        decisions = [node for node in nodes if node.get("node_type") == "decision"]
        relations = [relation.model_dump(mode="json") for relation in self.relations()]
        forward_relations = [
            relation
            for relation in relations
            if relation["type"] in {"depends_on", "supplements", "informs"}
        ]
        backward_relations = [
            relation
            for relation in relations
            if relation["type"] in {"contradicts", "confirms", "extends"}
        ]
        return {
            "beliefs": beliefs,
            "nodes": nodes,
            "decisions": decisions,
            "relations": relations,
            "forward_relations": forward_relations,
            "backward_relations": backward_relations,
            "merges": list(self.merges),
            "sessions": list(self.sessions),
            "evidence": list(self.evidence),
            "factors": list(self.factors),
            "graph": self.model_dump(mode="json"),
            "counts": {
                "beliefs": len(beliefs),
                "forward_relations": len(forward_relations),
                "backward_relations": len(backward_relations),
                "merges": len(self.merges),
                "sessions": len(self.sessions),
                "nodes": len(self.nodes),
                "edges": len(self.edges),
            },
            "metadata": dict(self.metadata),
        }


class EpisodeBCG(BCG):
    """Belief Context Graph for a single episode."""

    uuid: str = Field(default_factory=get_random_uuid)
