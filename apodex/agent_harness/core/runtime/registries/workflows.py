"""WorkflowContext — registration interface for external workflow plugins.

External workflows receive a WorkflowContext in their ``register()``
function.  It exposes only the registration APIs needed to wire up a
pipeline — no direct access to internal kernel services.

Usage (inside a workflow package)::

    from agent_harness.core.runtime.registries.workflows import WorkflowContext

    def register(ctx: WorkflowContext) -> None:
        ctx.register_pipeline(MY_SPEC)
        ctx.register_agent(MY_AGENT_DEF)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_harness.core.runtime.registries.agents import AgentRegistry
    from agent_harness.models.agent_definition import AgentDefinition
    from agent_harness.models.pipeline_spec import PipelineSpec
    from agent_harness.scheduling.pipeline_registry import PipelineRegistry

logger = logging.getLogger(__name__)


class WorkflowContext:
    """Safe registration facade passed to workflow ``register()`` hooks."""

    def __init__(
        self,
        pipeline_registry: PipelineRegistry,
        agent_registry: AgentRegistry,
    ) -> None:
        self._pipelines = pipeline_registry
        self._agents = agent_registry

    # -- Pipeline registration ------------------------------------------------

    def register_pipeline(self, spec: PipelineSpec) -> None:
        """Register a PipelineSpec so it can be selected at runtime."""
        if self._pipelines.has(spec.pipeline_id):
            raise ValueError(
                f"Pipeline '{spec.pipeline_id}' is already registered; "
                "plugin registration cannot override existing pipelines"
            )
        self._pipelines.register(spec)
        logger.info(
            "Workflow plugin registered pipeline: %s", spec.pipeline_id,
        )

    # -- Agent registration ---------------------------------------------------

    def register_agent(self, definition: AgentDefinition) -> None:
        """Register a custom AgentDefinition (role, prompt, tools)."""
        self._agents.register(definition)

    def register_agents(self, definitions: list[AgentDefinition]) -> None:
        """Convenience: register multiple AgentDefinitions at once."""
        for defn in definitions:
            self.register_agent(defn)

    # -- Introspection (read-only) --------------------------------------------

    def has_pipeline(self, pipeline_id: str) -> bool:
        return self._pipelines.has(pipeline_id)

    def has_agent(self, role_id: str) -> bool:
        return self._agents.has(role_id)
