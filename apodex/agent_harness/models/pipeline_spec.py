"""Declarative pipeline specification models.

A PipelineSpec describes the complete topology of a pipeline DAG:
which phases exist, what agent roles they use, and how they connect.
Each spec can also bundle the agent definitions it needs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_harness.models.agent_definition import AgentDefinition


# ---------------------------------------------------------------------------
# Node-level declarative models (pluggable node architecture)
# ---------------------------------------------------------------------------


class ContextPolicy(BaseModel):
    """Declares what subset of pipeline state a node receives."""

    include_fields: list[str] | None = Field(
        default=None,
        description="Whitelist of state fields. None = full state.",
    )
    filter_fn: str | None = Field(
        default=None,
        description=(
            "Dotted import path to (state: dict) -> dict. "
            "Takes precedence over include_fields."
        ),
    )
    inject_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Fields always injected by framework regardless of "
            "include_fields."
        ),
    )


class CompressionConfig(BaseModel):
    """Per-node compression strategy for SummarizationMiddleware."""

    enabled: bool = Field(
        default=True,
        description="Whether compression is active for this node.",
    )
    threshold: int = Field(
        default=80000,
        description="Token count that triggers auto-compact.",
    )
    keep_recent: int = Field(
        default=6,
        description="Recent messages to preserve during compression.",
    )
    max_field_tokens: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Truncate state fields before prompt. "
            "E.g. {'evidence_cards': 8000}."
        ),
    )


class NodeExecutionPolicy(BaseModel):
    """Declares tool gating owned by a node/workflow, not by the kernel."""

    allow_tools: list[str] | None = Field(
        default=None,
        description=(
            "Allowlist of tool names for this node. None = all role-permitted "
            "tools remain available."
        ),
    )
    deny_tools: list[str] = Field(
        default_factory=list,
        description="Explicit tool-name denials for this node.",
    )
    deny_tool_prefixes: list[str] = Field(
        default_factory=list,
        description="Prefix-based tool denials for this node.",
    )


class NodeDefinition(BaseModel):
    """Declares a node's identity, behavior, and context requirements."""

    node_id: str = Field(
        ...,
        description="Unique within pipeline, e.g. 'clarify'",
    )
    role_id: str = Field(
        ...,
        description="Agent role -> AgentDefinition (prompt, tools, model)",
    )
    context_policy: ContextPolicy = Field(default_factory=ContextPolicy)
    execution_policy: NodeExecutionPolicy | None = Field(
        default=None,
        description=(
            "Node-owned runtime policy such as tool gating. "
            "None = use workflow/framework defaults."
        ),
    )
    compression: CompressionConfig | None = Field(
        default=None,
        description=(
            "Per-node compression config. None = framework defaults."
        ),
    )
    node_function: str | None = Field(
        default=None,
        description="Dotted import path to async def(state: dict) -> dict.",
    )
    prompt_template: str = Field(
        default="",
        description=(
            "Jinja2 template. Used by generic executor when "
            "node_function is None."
        ),
    )
    output_fields: list[str] = Field(default_factory=list)
    display_label: str = Field(
        default="",
        description=(
            "Human-readable label emitted on the PHASE_TRANSITION event "
            "(e.g. 'Searching and collecting evidence'). Empty falls back "
            "to f'Entering {node_id} phase'. Replaces the hardcoded "
            "_PHASE_LABELS dict in EventEmitterMiddleware."
        ),
    )
    metadata: dict = Field(default_factory=dict)


class PhaseSpec(BaseModel):
    """Legacy phase-based node spec — prefer ``NodeDefinition``.

    ``PipelineSpec.resolved_nodes`` auto-converts both forms.
    """

    phase_id: str = Field(..., description="Unique ID within the pipeline, e.g. 'clarify'")
    role_id: str = Field(..., description="Agent role that executes this phase")
    phase_type: str = Field(
        default="llm_reasoning",
        description="Type: llm_reasoning | tool_execution | parallel_search | custom",
    )
    prompt_template: str = Field(default="", description="Jinja2 template for LLM prompt")
    node_function: str | None = Field(
        default=None,
        description="Dotted import path to a custom node function. If set, overrides prompt_template.",
    )
    execution_policy: NodeExecutionPolicy | None = Field(
        default=None,
        description="Node-owned tool / runtime policy; forwarded on conversion.",
    )
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    display_label: str = Field(
        default="",
        description=(
            "Human-readable label for PHASE_TRANSITION events. "
            "Forwarded to NodeDefinition on conversion."
        ),
    )
    metadata: dict = Field(default_factory=dict)

    def to_node_definition(self) -> NodeDefinition:
        """Convert PhaseSpec to NodeDefinition (migration bridge)."""
        include = self.input_fields if self.input_fields else None
        return NodeDefinition(
            node_id=self.phase_id,
            role_id=self.role_id,
            node_function=self.node_function,
            prompt_template=self.prompt_template,
            context_policy=ContextPolicy(include_fields=include),
            execution_policy=self.execution_policy,
            output_fields=self.output_fields,
            display_label=self.display_label,
            metadata=self.metadata,
        )


class TransitionSpec(BaseModel):
    """Describes an edge in the pipeline DAG."""

    from_phase: str
    to_phase: str  # "__END__" for terminal edges
    condition: str | None = Field(
        default=None,
        description="Dotted import path to a condition function, or None for unconditional",
    )


class PipelineSpec(BaseModel):
    """Complete declarative description of a pipeline topology."""

    pipeline_id: str = Field(..., description="Unique ID, e.g. 'deep_research'")
    name: str = Field(default="")
    description: str = Field(default="")
    required_roles: list[str] = Field(default_factory=list)
    agent_definitions: list[AgentDefinition] = Field(
        default_factory=list,
        description="Agent roles bundled with this pipeline. Auto-registered at execution time.",
    )
    phases: list[PhaseSpec] = Field(default_factory=list)
    nodes: list[NodeDefinition] = Field(default_factory=list)
    transitions: list[TransitionSpec] = Field(default_factory=list)
    entry_point: str = Field(..., description="phase_id of the first phase")
    terminal_phases: list[str] = Field(default_factory=list)
    terminal_nodes: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict, description="Pipeline-level metadata")
    hidden: bool = Field(
        default=False,
        description=(
            "If True, this pipeline is omitted from the default /pipelines "
            "list (so it does not clutter the UI dropdown). It remains "
            "registered and selectable via explicit pipeline_id in API "
            "requests. Use for benchmark-only, backwards-compat, or "
            "developer-template pipelines."
        ),
    )
    state_type: str | None = Field(
        default=None,
        description="Dotted import path to the state TypedDict class. If None, uses BaseTaskState.",
    )

    @property
    def resolved_nodes(self) -> list[NodeDefinition]:
        """Return NodeDefinitions — auto-converts phases if nodes is empty."""
        if self.nodes:
            return self.nodes
        return [p.to_node_definition() for p in self.phases]

    @property
    def resolved_terminal_nodes(self) -> list[str]:
        """Return terminal node IDs — falls back to terminal_phases."""
        if self.terminal_nodes:
            return self.terminal_nodes
        return self.terminal_phases
