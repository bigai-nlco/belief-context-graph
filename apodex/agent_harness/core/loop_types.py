"""Loop type contracts for the agent-loop engine.

Framework-level types and dispatch helpers shared by the kernel loop,
observers, sdk and workflow code. Lives at ``agent_harness.core`` to avoid
a reverse dependency from ``kernel/loop`` into the observer layer.

Exports:

* Types — ``LoopObserver`` / ``LoopConfig`` / ``LoopPolicy`` /
  ``TurnContext`` / ``ToolResult`` / ``Intervention`` / ``AgentLoopResult``
* Dispatch — ``notify_observers`` / ``merge_interventions``

The convenience base class ``BaseObserver`` lives in
``agent_harness.observers.base`` (observer-side utility, not a contract).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopPolicy:
    """Policy knobs for the generic agent loop.

    The loop engine owns the mechanics of turn execution. Workflow-specific
    behavior such as phase identity, terminal-tool hints, and how to recover
    from a no-tool turn is injected through this policy object.
    """

    phase_id: str = ""
    no_tool_behavior: Literal["stop", "nudge"] = "nudge"
    no_tool_nudge_message: str = ""
    terminal_tool_names: tuple[str, ...] = ()


@dataclass
class LoopConfig:
    max_turns: int = 50
    max_tool_calls_per_turn: int = 5
    tool_timeout: int = 120
    llm_timeout: int = 180
    context_token_limit: int = 120_000
    compact_after_turns: int = 12
    keep_recent: int = 16
    # 2 retries — more tends to produce refusal / dithering loops under
    # large contexts. No-tool recovery text is owned by LoopPolicy.
    no_tool_max_retries: int = 2
    # Workflows hitting noisy endpoints may bump to 10-15 in their own config.
    max_llm_retries: int = 5
    # Fixed wait (seconds) between LLM retry attempts; ``None`` uses the
    # default exponential 2/4/8/16/32 schedule. A flat 60-90s helps when
    # server-side recovery takes longer than the early exponential waits.
    retry_wait_fixed: int | None = None
    task_id: str = ""
    role_id: str = ""
    loop_policy: LoopPolicy = field(default_factory=LoopPolicy)
    # Cap each tool message content at this many chars before it enters
    # history. ``None`` (default) keeps full output.
    tool_result_max_chars: int | None = None
    # Pluggable message-history compactor. ``None`` → ``DefaultMessageCompactor``
    # (system-prefix + middle squash + recent window). Typed as ``Any`` to
    # keep this module a pure type layer with no kernel imports.
    compactor: Any = None
    # Pluggable policy deciding WHEN to compact. ``None`` →
    # ``DefaultCompactionPolicy(compact_after_turns, context_token_limit)``.
    # Pair with ``compactor`` — policy is *when*, compactor is *how*.
    compaction_policy: Any = None
    # Per-tool post-processor that shapes a ``ToolResult`` into the
    # ``tool message`` content string. ``None`` →
    # ``DefaultToolResultPostProcessor(tool_result_max_chars)``.
    tool_result_post_processor: Any = None

    # ── Proactive context-overflow guard.
    #
    # When ``context_overflow_guard`` is True, the loop estimates after
    # every tool-execution step whether the next LLM call (including the
    # safety-net summarize prompt) would exceed ``max_context_length``.
    # On overflow it pops the trailing tool message(s) + last assistant message with
    # tool_calls and exits with ``stopped_by="context_limit_reached"``, so
    # the workflow's summarize path runs on a fitting history.
    #
    # ``max_completion_tokens`` is the completion budget for the next call;
    # ``summary_prompt`` is the summarize text the estimator should also
    # count (×1.5 buffer).
    context_overflow_guard: bool = False
    max_context_length: int = 262_144
    max_completion_tokens: int = 32_768
    summary_prompt: str = ""


@dataclass
class TurnContext:
    turn: int
    max_turns: int
    task_id: str
    role_id: str
    ai_text: str
    thinking: str
    tool_calls: list[dict]
    messages: list
    usage: dict | None
    metadata: dict
    # Reasoning recovered from leaked ``<think_never_used>`` / ``<model:thinking>``
    # / ``<think>`` tags that the model emitted as content. Empty string when
    # no leak was detected. Populated by the tool-call parser and surfaced
    # separately from ``thinking`` (which comes from the model's native
    # reasoning format) so consumers can distinguish provenance.
    leaked_reasoning: str = ""


@dataclass
class LLMDeltaContext:
    turn: int
    max_turns: int
    task_id: str
    role_id: str
    delta: str
    accumulated_text: str
    delta_index: int
    metadata: dict
    # Per-chunk thinking / reasoning content from the provider's native
    # reasoning channel (Anthropic ``thinking`` content blocks, OpenAI
    # ``reasoning_content``, DeepSeek ``reasoning_content``). Empty for
    # providers/models that don't emit reasoning, or when the chunk is a
    # pure visible-content delta. Surfaced separately from ``delta`` so
    # observers can route it to a distinct stream (Agent Protocol v1's
    # ``reasoning_steps`` field) without polluting visible content.
    thinking_delta: str = ""


@dataclass
class ToolResult:
    name: str
    args: dict
    result: str
    duration_ms: int
    tool_call_id: str
    is_error: bool


@dataclass
class Intervention:
    inject_messages: list[str] | None = None
    stop_reason: str | None = None
    skip_tool_execution: bool = False
    # Pop the last message off the history before the next turn. Used by
    # workflows that want to retry an LLM turn after a malformed response
    # (truncation, refusal) without leaving the bad assistant message in history.
    # Applied AFTER ``inject_messages``, BEFORE ``continue_to_next_turn``.
    pop_last_message: bool = False
    # Short-circuit: skip the rest of the turn (tool execution, on_turn_end,
    # compaction) and jump straight to the next iteration. Used by rollback
    # observers that detect a malformed assistant message and want to retry without
    # spending a full turn budget on it.
    continue_to_next_turn: bool = False


@dataclass
class ToolCallIntervention:
    """Per-tool-call hook return value.

    Used by ``notify_tool_call`` to let observers rewrite args, short-
    circuit with a cached result, or stash metadata before dispatch.
    """

    rewrite_args: dict | None = None
    skip_with_result: str | None = None
    metadata_updates: dict | None = None


@dataclass
class AgentLoopResult:
    messages: list
    final_content: str = ""
    turns_used: int = 0
    tool_calls_count: int = 0
    stopped_by: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LoopObserver Protocol + BaseObserver default impl
# ---------------------------------------------------------------------------


@runtime_checkable
class LoopObserver(Protocol):
    """Protocol for agent-loop observers.

    Critical observers (critical=True) are awaited and their return values
    collected as Intervention objects.  Passive observers (critical=False)
    are fired-and-forgotten via asyncio.create_task().

    Two of the hooks (``on_tool_call`` and ``on_tool_result``) drive
    per-tool-call fan-out and have their own merge semantics in the
    loop engine — see ``run_agent_loop``. They are awaited regardless
    of ``critical``.
    """

    critical: bool

    async def on_loop_start(self, config: LoopConfig) -> None: ...

    async def on_llm_delta(self, ctx: LLMDeltaContext) -> Intervention | None: ...

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None: ...

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ) -> ToolCallIntervention | None: ...

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None: ...

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None: ...

    async def on_loop_end(self, result: AgentLoopResult) -> None: ...


class BaseObserver:
    """Convenience no-op base class for ``LoopObserver`` implementations.

    Subclass and override only the hooks you need. Lives next to
    ``LoopObserver`` so observer authors only need a single import:
    ``from agent_harness.core.loop_types import BaseObserver``.
    """

    critical: bool = False

    async def on_loop_start(self, config: LoopConfig) -> None:
        pass

    async def on_llm_delta(self, ctx: LLMDeltaContext) -> Intervention | None:
        return None

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        return None

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ) -> ToolCallIntervention | None:
        return None

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        return None

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        return None

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        pass


# ---------------------------------------------------------------------------
# Notification helpers
# ---------------------------------------------------------------------------


# Prevent GC of fire-and-forget observer tasks.
_background_tasks: set[asyncio.Task] = set()


# Restart-scoped ``(observer_class, method)`` dedupe set. First failure
# emits WARNING + traceback; later occurrences degrade to DEBUG so a
# chatty broken observer cannot flood production logs.
_warned_observer_errors: set[tuple[str, str]] = set()


def _handle_observer_error(
    observer: Any, method: str, exc: BaseException,
) -> None:
    """Log an observer crash. Never raises.

    First failure per ``(observer_class, method)`` emits WARNING with
    traceback; later occurrences degrade to DEBUG. ``loop_types`` is
    deliberately kept free of imports from ``core.runtime`` (layering
    rule 7), so the WARNING is the only telemetry channel — that is
    enough to surface a broken observer in production logs.
    """
    obs_class = type(observer).__name__
    key = (obs_class, method)
    if key in _warned_observer_errors:
        logger.debug(
            "Observer %s.%s raised (suppressed)", obs_class, method,
            exc_info=True,
        )
        return
    _warned_observer_errors.add(key)
    logger.warning(
        "Observer %s.%s raised: %s — subsequent failures DEBUG only",
        obs_class, method, exc, exc_info=True,
    )


async def notify_observers(
    observers: list[Any],
    method: str,
    *args: Any,
    **kwargs: Any,
) -> list[Intervention]:
    """Call *method* on every observer, collecting Intervention returns.

    - critical=True  → awaited; Intervention return values are collected.
    - critical=False → fire-and-forget via asyncio.create_task(); returns ignored.
    - ALL errors are swallowed (logged at DEBUG) — observers must never crash the loop.

    Returns a (possibly empty) list of Intervention objects from critical observers.
    """
    interventions: list[Intervention] = []

    for obs in observers:
        fn = getattr(obs, method, None)
        if fn is None:
            continue

        if getattr(obs, "critical", False):
            try:
                rv = await fn(*args, **kwargs)
                if isinstance(rv, Intervention):
                    interventions.append(rv)
            except Exception as exc:
                _handle_observer_error(obs, method, exc)
        else:
            # Fire-and-forget: errors are swallowed inside the wrapper.
            async def _run(observer=obs, m=method, f=fn, a=args, kw=kwargs):
                try:
                    await f(*a, **kw)
                except Exception as exc:
                    _handle_observer_error(observer, m, exc)

            task = asyncio.create_task(_run())
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

    return interventions


def merge_interventions(interventions: list[Intervention]) -> Intervention:
    """Merge a list of Interventions into one.

    Rules:
    - inject_messages: concatenation of all non-None lists.
    - stop_reason: first non-None wins.
    - skip_tool_execution: True if any is True.
    - pop_last_message: True if any is True.
    - continue_to_next_turn: True if any is True.
    """
    all_messages: list[str] = []
    stop_reason: str | None = None
    skip: bool = False
    pop_last: bool = False
    continue_turn: bool = False

    for iv in interventions:
        if iv.inject_messages:
            all_messages.extend(iv.inject_messages)
        if stop_reason is None and iv.stop_reason is not None:
            stop_reason = iv.stop_reason
        if iv.skip_tool_execution:
            skip = True
        if iv.pop_last_message:
            pop_last = True
        if iv.continue_to_next_turn:
            continue_turn = True

    return Intervention(
        inject_messages=all_messages if all_messages else None,
        stop_reason=stop_reason,
        skip_tool_execution=skip,
        pop_last_message=pop_last,
        continue_to_next_turn=continue_turn,
    )


async def notify_tool_call(
    observers: list[Any], ctx: "TurnContext", tool_call: dict,
) -> "ToolCallIntervention":
    """Fan-out per-tool-call hook with first-skip-wins semantics.

    Calls ``on_tool_call`` on every observer that has it. Returns a
    merged ``ToolCallIntervention`` where:

    * ``rewrite_args`` is the last observer's value to set it (later
      observers can override earlier rewrites).
    * ``skip_with_result`` is the first observer's value to set it
      (once skipped, no further mutation makes sense).
    * ``metadata_updates`` are merged dict-wise.

    Errors are swallowed and logged at DEBUG so a buggy observer can
    never crash a tool dispatch.
    """
    rewrite: dict | None = None
    skip: str | None = None
    meta_updates: dict = {}

    for obs in observers:
        fn = getattr(obs, "on_tool_call", None)
        if fn is None:
            continue
        try:
            rv = await fn(ctx, tool_call)
        except Exception as exc:
            _handle_observer_error(obs, "on_tool_call", exc)
            continue
        if rv is None:
            continue
        if rv.rewrite_args is not None:
            rewrite = rv.rewrite_args
        if skip is None and rv.skip_with_result is not None:
            skip = rv.skip_with_result
        if rv.metadata_updates:
            meta_updates.update(rv.metadata_updates)

    return ToolCallIntervention(
        rewrite_args=rewrite,
        skip_with_result=skip,
        metadata_updates=meta_updates or None,
    )


async def notify_tool_result(
    observers: list[Any], ctx: "TurnContext", result: "ToolResult",
) -> "ToolResult":
    """Fan-out per-tool-result hook with last-mutation-wins semantics.

    Each observer that returns a non-None ``ToolResult`` replaces the
    one passed to subsequent observers, so the final result reflects
    the chained mutation. Observers that return ``None`` see the
    current result without mutating it. Errors are swallowed.
    """
    current = result
    for obs in observers:
        fn = getattr(obs, "on_tool_result", None)
        if fn is None:
            continue
        try:
            rv = await fn(ctx, current)
        except Exception as exc:
            _handle_observer_error(obs, "on_tool_result", exc)
            continue
        if rv is not None:
            current = rv
    return current


__all__ = [
    "AgentLoopResult",
    "BaseObserver",
    "Intervention",
    "LoopConfig",
    "LoopObserver",
    "LoopPolicy",
    "ToolCallIntervention",
    "ToolResult",
    "TurnContext",
    "merge_interventions",
    "notify_observers",
]
