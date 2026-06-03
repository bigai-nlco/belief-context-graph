"""Top-level Belief Context Graph facade."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from bcg.utils import get_random_uuid, utc_now


class BCGNode(BaseModel):
    """Belief Context Graph Node"""

    uuid: str = Field(
        default_factory=get_random_uuid, description="Unique identifier for the node"
    )
    name: str = Field(..., description="Name of the node")
    probability: float = Field(
        default=1.0, description="Probability associated with the belief"
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Payload data associated with the node"
    )
    created_at: datetime = Field(
        default_factory=utc_now, description="Timestamp of when the node was created"
    )
    updated_at: datetime = Field(
        ..., description="Timestamp of when the node was last updated"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the node"
    )


class BCGEdge(BaseModel):
    """Belief Context Graph Edge"""

    uuid: str = Field(
        default_factory=get_random_uuid, description="Unique identifier for the edge"
    )
    source: str = Field(..., description="UUID of the source node")
    target: str = Field(..., description="UUID of the target node")
    weight: float = Field(default=1.0, description="Weight associated with the edge")
    created_at: datetime = Field(
        default_factory=utc_now, description="Timestamp of when the edge was created"
    )
    updated_at: datetime = Field(
        ..., description="Timestamp of when the edge was last updated"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the edge"
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Payload data associated with the edge"
    )


class BCG(BaseModel):
    """Belief Context Graph"""

    nodes: list[BCGNode] = Field(
        default_factory=list, description="List of nodes in the graph"
    )
    edges: list[BCGEdge] = Field(
        default_factory=list, description="List of edges in the graph"
    )

    async def add_node(self, node: BCGNode) -> None:
        """Add a node to the graph."""
        raise NotImplementedError("BCG.add_node is not implemented yet.")

    async def add_edge(self, edge: BCGEdge) -> None:
        """Add an edge to the graph."""
        raise NotImplementedError("BCG.add_edge is not implemented yet.")


class EpisodeBCG(BCG):
    """Belief Context Graph for an episode"""

    uuid: str = Field(
        default_factory=get_random_uuid, description="Unique identifier for the episode"
    )
