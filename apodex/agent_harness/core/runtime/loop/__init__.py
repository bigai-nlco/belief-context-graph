"""Generic ReAct agent loop — LLM call + tool execution + runtime primitives.

Groups the agent_loop engine with its immediate dependencies:
tool-call parsing/repair, model profiles, message trimming, execution
context, context-budget estimation, and guardrails.
"""

from agent_harness.core.runtime.loop.agent_loop import run_agent_loop
from agent_harness.core.runtime.loop.compact import (
    DefaultCompactionPolicy,
    DefaultMessageCompactor,
)
from agent_harness.core.runtime.loop.context_budget import estimate_tokens
from agent_harness.core.execution_context import (
    ExecutionScope,
    build_execution_scope,
    ensure_trace_metadata,
    get_current_execution_scope,
    normalize_execution_context,
    reset_current_execution_scope,
    set_current_execution_scope,
)
from agent_harness.core.runtime.loop.model_profile import (
    DefaultMessageNormalizer,
    DefaultThinkingParser,
    HistoryPolicy,
    MessageNormalizer,
    ModelProfile,
    ThinkingParser,
    ThinkingResult,
)
from agent_harness.core.runtime.loop.tool_call_parser import (
    LEAKED_REASONING_KEY,
    DefaultToolCallParser,
    MultiFormatToolCallParser,
    ToolCallParser,
    extract_leaked_reasoning,
)
from agent_harness.core.runtime.loop.tool_exec import (
    DefaultToolResultPostProcessor,
    ToolResultPostProcessor,
)

__all__ = [
    "run_agent_loop",
    "DefaultMessageCompactor",
    "DefaultCompactionPolicy",
    "ExecutionScope",
    "build_execution_scope",
    "ensure_trace_metadata",
    "get_current_execution_scope",
    "normalize_execution_context",
    "reset_current_execution_scope",
    "set_current_execution_scope",
    "NullTrimmer",
    "KeepRecentTrimmer",
    "TaskBoundaryTrimmer",
    "ModelProfile",
    "HistoryPolicy",
    "ThinkingResult",
    "ThinkingParser",
    "MessageNormalizer",
    "DefaultThinkingParser",
    "DefaultMessageNormalizer",
    "ToolCallParser",
    "DefaultToolCallParser",
    "MultiFormatToolCallParser",
    "LEAKED_REASONING_KEY",
    "extract_leaked_reasoning",
    "ToolResultPostProcessor",
    "DefaultToolResultPostProcessor",
    "estimate_tokens",
]
