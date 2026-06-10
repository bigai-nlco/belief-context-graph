"""Pipeline spec: ``react_base``.

Single-agent ReAct with keep-last-N tool result hygiene. One agent does
the research and then writes a terminal Markdown report from the
accumulated reasoning and tool results.
"""

from __future__ import annotations

from agent_harness.models.pipeline_spec import (
    CompressionConfig,
    ContextPolicy,
    NodeDefinition,
    PipelineSpec,
    TransitionSpec,
)

REACT_BASE_SPEC = PipelineSpec(
    pipeline_id="react_base",
    name="AgentHarness ReAct Base",
    description=(
        "Single ReAct agent that directly uses web_search / web_fetch / "
        "run_python_code, with old tool results replaced by placeholders "
        "after the last N (keep-last-N tool result hygiene)."
    ),
    entry_point="main_agent",
    terminal_nodes=["main_agent"],
    nodes=[
        NodeDefinition(
            node_id="main_agent",
            role_id="react_solver",
            node_function=(
                "workflows.react_base.nodes.main_agent.main_agent_node"
            ),
            context_policy=ContextPolicy(
                include_fields=[
                    "original_question", "language", "task_id", "metadata",
                ],
            ),
            compression=CompressionConfig(enabled=False),
            output_fields=[
                "final_answer", "report", "final_content",
                "answer_confidence", "react_steps",
            ],
        ),
    ],
    transitions=[
        TransitionSpec(from_phase="main_agent", to_phase="__END__"),
    ],
)
