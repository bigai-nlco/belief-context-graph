"""Belief memory interface skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bcg.graph import BCG


@dataclass(slots=True)
class BCGMemory:
    """Interface for a belief-native memory runtime."""

    namespace: str = "default"
    options: dict[str, Any] = field(default_factory=dict)
    graph: BCG | None = None

    def observe(
        self,
        *,
        source_type: str,
        content: str,
        actor: dict[str, Any] | None = None,
        observed_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Observe an evidence episode."""

        raise NotImplementedError("BCGMemory.observe is not implemented yet.")

    def believe(self, target: str, **kwargs: Any) -> Any:
        """Read the current state of a belief variable."""

        raise NotImplementedError("BCGMemory.believe is not implemented yet.")

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
        """Assemble belief-aware task context."""

        raise NotImplementedError("BCGMemory.context is not implemented yet.")

    async def search(
        self,
        *,
        query: str,
        max_results: int = 10,
        include_conflicts: bool = True,
        include_missing_evidence: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Search for relevant beliefs and evidence."""

        raise NotImplementedError("BCGMemory.search is not implemented yet.")
