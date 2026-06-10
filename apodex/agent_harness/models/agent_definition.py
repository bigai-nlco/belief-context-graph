"""AgentDefinition — declarative, runtime-configurable agent role."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentDefinition(BaseModel):
    """Describes an agent role that can be dynamically registered.

    This replaces the hardcoded AgentRole enum — new roles can be added
    at runtime via AgentRegistry.register() without modifying any kernel code.
    """

    role_id: str = Field(..., description="Unique identifier, e.g. 'researcher', 'critic', or any custom role")
    display_name: str = Field(..., description="Human-readable name for UI display")
    system_prompt: str = Field(default="", description="System prompt sent to LLM when this agent acts")
    allowed_tools: list[str] = Field(default_factory=list, description="Tool names this agent can use")

    color: str = Field(default="#6b7280", description="Hex color for frontend visualization")
    icon: str = Field(default="agent", description="Icon identifier for frontend")
    description: str = Field(default="", description="Brief description of this agent's purpose")
    metadata: dict = Field(default_factory=dict, description="Extensible metadata")
