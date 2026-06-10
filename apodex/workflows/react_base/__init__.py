"""react_base — single-agent ReAct workflow with keep-last-N tool result hygiene.

One ReAct agent calls ``web_search`` / ``web_fetch`` / ``run_python_code``
directly (no sub-agent fan-out). Old tool results are replaced by short
placeholders once they fall outside the last N via
:class:`KeepLastNToolResultsCompactor`, applied every turn through
:class:`AlwaysCompactPolicy`.
"""

from __future__ import annotations

from agent_harness.core.runtime.registries.workflows import WorkflowContext
from agent_harness.models.agent_definition import AgentDefinition
from workflows.react_base.spec import REACT_BASE_SPEC

_PLACEHOLDER = "Computed per-task."


MAIN_AGENT_DEF = AgentDefinition(
    role_id="react_solver",
    display_name="ReAct Solver",
    system_prompt=_PLACEHOLDER,
    allowed_tools=[
        "web_search",
        "web_fetch",
        "run_python_code",
    ],
    color="#10b981",
    icon="cpu",
    description=(
        "Single ReAct agent: directly searches / fetches / runs Python "
        "with keep-last-N tool result hygiene."
    ),
)


def register(ctx: WorkflowContext) -> None:
    ctx.register_agent(MAIN_AGENT_DEF)
    ctx.register_pipeline(REACT_BASE_SPEC)
