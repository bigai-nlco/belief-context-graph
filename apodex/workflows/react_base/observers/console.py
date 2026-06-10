"""Rich console observer for the react_base main agent.

Prints turn-by-turn agent activity to the terminal in real time.
"""

from __future__ import annotations

import json

from rich.console import Console
from rich.markup import escape as _escape

from agent_harness.core.loop_types import (
    AgentLoopResult,
    BaseObserver,
    LoopConfig,
    ToolResult,
    TurnContext,
)

# Shared module-level console — thread-safe via Rich's internal lock.
# width=200 avoids line-wrapping; force_terminal=True keeps ANSI in tee/logs.
_console = Console(width=200, force_terminal=True)

_CONTENT_PREVIEW = 2000
_ARGS_PREVIEW = 400
_RESULT_PREVIEW = 400

_STOP_ICON: dict[str, str] = {
    "no_tool":          "✅",
    "BudgetObserver":   "💰",
    "max_turns":        "⏱️",
    "no_tool_budget":   "⚠️",
    "llm_error":        "💥",
    "paused":           "⏸️",
}


class RichConsoleObserver(BaseObserver):
    """Prints react_base execution to the terminal in real time."""

    critical: bool = False

    async def on_loop_start(self, config: LoopConfig) -> None:
        _console.print(
            f"\n[bold magenta]{'━' * 60}[/bold magenta]"
            f"\n[bold magenta]🚀 [PROFILE] task started[/bold magenta]"
            f"  task_id=[cyan]{_escape(config.task_id)}[/cyan]"
            f"  max_turns={config.max_turns}"
        )

    async def on_llm_response(self, ctx: TurnContext) -> None:
        tokens = ""
        if ctx.usage:
            inp = ctx.usage.get("input_tokens") or ctx.usage.get("prompt_tokens") or 0
            out = ctx.usage.get("output_tokens") or ctx.usage.get("completion_tokens") or 0
            tokens = f"  [dim]tokens: {inp}→{out}[/dim]"

        _console.print(
            f"\n[bold cyan]═══ [PROFILE] Turn {ctx.turn}/{ctx.max_turns}[/bold cyan]"
            f"  [dim]{_escape(ctx.role_id)}[/dim]{tokens}"
        )

        text = ctx.ai_text.strip()
        if text:
            _console.print(_escape(text[:_CONTENT_PREVIEW]))
            if len(text) > _CONTENT_PREVIEW:
                _console.print(f"[dim]... ({len(text)} chars total)[/dim]")

        for tc in ctx.tool_calls or []:
            raw = json.dumps(tc.get("args", {}), ensure_ascii=False)
            preview = raw[:_ARGS_PREVIEW] + ("…" if len(raw) > _ARGS_PREVIEW else "")
            _console.print(
                f"  [yellow]🔧 {_escape(tc.get('name', '?'))}[/yellow]"
                f"  [dim]{_escape(preview)}[/dim]"
            )

    async def on_tool_result(self, ctx: TurnContext, result: ToolResult) -> None:
        preview = (
            result.result[:_RESULT_PREVIEW]
            + ("…" if len(result.result) > _RESULT_PREVIEW else "")
        )
        style = "red" if result.is_error else "green"
        prefix = "ERROR: " if result.is_error else ""
        _console.print(f"  [{style}]↳ {_escape(prefix + preview)}[/{style}]")

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        icon = _STOP_ICON.get(result.stopped_by, "⏹")
        _console.print(
            f"\n[bold magenta]{icon} [PROFILE] done[/bold magenta]"
            f"  turns={result.turns_used}"
            f"  tool_calls={result.tool_calls_count}"
            f"  stopped_by={_escape(result.stopped_by)}"
        )
