"""Core Agent Loop Engine — config-driven ReAct loop with zero domain logic.

All agent-specific behaviour comes from observers. The loop only does:
LLM call -> parse tools -> execute -> observe -> compress.

Thin-kernel guardrail: do not add workflow-specific phase names, terminal-
tool semantics, or no-tool recovery policy here. Those belong in injected
loop policy owned by pipeline/workflow assembly.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Awaitable, Callable

from agent_harness.core.llm import LLMClient
from agent_harness.core.messages import (
    Message,
    is_assistant_msg,
    is_tool_msg,
    system_msg,
    tool_msg,
    user_msg,
)
from agent_harness.core.runtime.loop.compact import (
    DefaultCompactionPolicy,
    DefaultMessageCompactor,
    estimate_tokens,
)
from agent_harness.core.runtime.loop.llm_client import (
    call_llm,
    estimate_message_tokens,
    estimate_text_tokens,
    extract_final_content,
    extract_leaked_reasoning,
    extract_usage,
)
from agent_harness.core.runtime.loop.model_profile import (
    DefaultMessageNormalizer,
    DefaultThinkingParser,
    HistoryPolicy,
    ModelProfile,
)
from agent_harness.core.runtime.loop.tool_call_parser import (
    MultiFormatToolCallParser,
    ToolCallParser,
)
from agent_harness.core.runtime.loop.tool_exec import (
    DefaultToolResultPostProcessor,
    execute_tools,
)
from agent_harness.core.tool import Tool
from agent_harness.core.loop_types import (
    AgentLoopResult,
    LLMDeltaContext,
    LoopConfig,
    LoopPolicy,
    ToolResult,
    TurnContext,
    merge_interventions,
    notify_observers,
    notify_tool_call,
    notify_tool_result,
)

logger = logging.getLogger(__name__)


# Rollback iterations do not consume ``max_turns`` but total iterations are
# capped at ``max_turns + EXTRA_ATTEMPTS_BUFFER`` to prevent runaway loops.
EXTRA_ATTEMPTS_BUFFER = 200


TurnCompleteHook = Callable[
    [int, list[Message], dict[str, Any]], Awaitable[None],
]
PauseCheckHook = Callable[[], Awaitable[bool]]


async def run_agent_loop(
    *,
    system_prompt: str,
    user_message: str,
    llm: LLMClient,
    tools: list[Tool],
    config: LoopConfig | None = None,
    observers: list[Any] | None = None,
    parser: ToolCallParser | None = None,
    model_profile: ModelProfile | None = None,
    history_policy: HistoryPolicy | None = None,
    initial_messages: list[Message] | None = None,
    on_turn_complete: TurnCompleteHook | None = None,
    pause_check: PauseCheckHook | None = None,
    scope_metadata: dict[str, Any] | None = None,
) -> AgentLoopResult:
    """Run a generic ReAct agent loop.

    Optional hooks (exceptions are caught — they never abort a run):

    * ``on_turn_complete(turn, messages, metadata)`` — per-turn persistence;
      fires after ``on_turn_end`` and message compaction.
    * ``pause_check() -> bool`` — graceful-pause probe after the checkpoint
      is written. ``True`` stops with ``stopped_by="paused"``.
    * ``scope_metadata`` — merged into ``ExecutionScope.metadata`` so
      workflow-specific tools can read per-task config.
    """
    cfg = config or LoopConfig()
    obs = observers or []
    tc_parser = parser or MultiFormatToolCallParser()
    profile = model_profile or ModelProfile(model_id="default", provider="openai")
    policy = history_policy or HistoryPolicy()
    thinking_parser = DefaultThinkingParser()
    normalizer = DefaultMessageNormalizer()

    tool_map: dict[str, Tool] = {t.name: t for t in tools}
    tool_names: set[str] = set(tool_map.keys())
    tool_schemas: list[dict[str, Any]] = [t.to_openai_schema() for t in tools]

    # Resume path: pass ``initial_messages`` with empty ``user_message`` to
    # re-enter without modifying history; pass a non-empty user_message to
    # start a new turn.
    if initial_messages is not None:
        messages: list[Message] = list(initial_messages)
        if user_message:
            messages.append(user_msg(user_message))
    else:
        messages = [system_msg(system_prompt), user_msg(user_message)]

    metadata: dict[str, Any] = {"role_id": cfg.role_id}

    from agent_harness.core.execution_context import (
        ExecutionScope,
        reset_current_execution_scope,
        set_current_execution_scope,
    )
    scope_meta: dict[str, Any] = {"agent_id": cfg.role_id}
    if scope_metadata:
        scope_meta.update(scope_metadata)
    scope = ExecutionScope(
        task_id=cfg.task_id,
        role_id=cfg.role_id,
        phase_id=_resolve_phase_id(cfg),
        metadata=scope_meta,
    )
    scope_token = set_current_execution_scope(scope)

    try:
        return await _run_loop_inner(
            cfg, obs, tc_parser, profile, policy, thinking_parser,
            normalizer, tool_map, tool_names, tool_schemas, llm,
            messages, metadata, on_turn_complete, pause_check,
        )
    finally:
        reset_current_execution_scope(scope_token)


async def _run_loop_inner(
    cfg: LoopConfig,
    obs: list,
    tc_parser: Any,
    profile: Any,
    policy: Any,
    thinking_parser: Any,
    normalizer: Any,
    tool_map: dict[str, Tool],
    tool_names: set[str],
    tool_schemas: list[dict[str, Any]],
    llm: LLMClient,
    messages: list[Message],
    metadata: dict[str, Any],
    on_turn_complete: TurnCompleteHook | None = None,
    pause_check: PauseCheckHook | None = None,
) -> AgentLoopResult:
    await notify_observers(obs, "on_loop_start", cfg)

    stop_reason = ""
    total_tool_calls = 0
    no_tool_retries = 0
    last_input_tokens = 0
    last_output_tokens = 0
    stream_llm_tokens = any(
        bool(getattr(observer, "wants_llm_delta", False))
        for observer in obs
    )

    # ``turn`` counts successful turns; ``attempts`` includes rollbacks
    # and is capped at ``max_attempts`` to prevent infinite rollback.
    max_attempts = cfg.max_turns + EXTRA_ATTEMPTS_BUFFER
    turn = 0
    attempts = 0
    while turn < cfg.max_turns and attempts < max_attempts:
        turn += 1
        attempts += 1
        # Per-turn temperature override (consume-on-read).
        temp_override = metadata.pop("_llm_temp_override", None)
        # Per-turn tool-stripping: ``LastTurnForcer`` plants this on
        # ``max_turns - 1`` so the final turn runs tool-free.
        strip_tools = metadata.pop("_llm_strip_tools", False)
        call_tools = None if strip_tools else (tool_schemas or None)

        async def _on_delta(
            delta: str, accumulated: str, delta_index: int,
            thinking_delta: str = "",
        ) -> None:
            ctx = LLMDeltaContext(
                turn=turn, max_turns=cfg.max_turns,
                task_id=cfg.task_id, role_id=cfg.role_id,
                delta=delta, accumulated_text=accumulated,
                delta_index=delta_index, metadata=metadata,
                thinking_delta=thinking_delta,
            )
            await notify_observers(obs, "on_llm_delta", ctx)

        response = await call_llm(
            llm, messages,
            timeout=cfg.llm_timeout,
            max_retries=cfg.max_llm_retries,
            turn=turn,
            tools=call_tools,
            temperature=temp_override,
            on_delta=_on_delta if stream_llm_tokens else None,
            retry_wait_fixed=cfg.retry_wait_fixed,
        )
        if response is None:
            stop_reason = "llm_error"
            break

        tr = thinking_parser.extract(response, profile)

        history_msg = normalizer.to_history(response, tr, policy)
        messages.append(history_msg)

        # When native thinking uses ``<think>`` tags (which the leak-stripper
        # also matches), sync visible_content into the response before parsing
        # to avoid double-capture as leaked reasoning.
        if tr.thinking and profile.thinking_format == "tag":
            response.content = tr.visible_content

        # Side effect: parser strips leaked private-tag content from
        # ``response.content`` and stashes recovered text on
        # ``response.response_metadata[LEAKED_REASONING_KEY]``.
        parsed_calls = tc_parser.parse(response, tool_names)

        # Some Qwen variants emit Hermes-style tool calls inside ``<think>``.
        # Re-run the text fallback on the recovered thinking.
        if not parsed_calls and tr.thinking and hasattr(tc_parser, "parse_text"):
            parsed_calls = tc_parser.parse_text(tr.thinking, tool_names)
            if parsed_calls:
                logger.warning(
                    "turn=%d recovered %d tool_call(s) leaked into <think>",
                    turn, len(parsed_calls),
                )

        if len(parsed_calls) > cfg.max_tool_calls_per_turn:
            parsed_calls = parsed_calls[:cfg.max_tool_calls_per_turn]

        usage = extract_usage(response)
        if usage:
            last_input_tokens = int(usage.get("prompt_tokens", 0) or 0)
            last_output_tokens = int(usage.get("completion_tokens", 0) or 0)
        leaked_reasoning = extract_leaked_reasoning(response)
        # Surface ``LLMFallbackChain`` markers on metadata so observers can
        # flag failover runs. Reset both first so a successful primary turn
        # after a previous failover doesn't carry stale state.
        rmd = response.response_metadata or {}
        metadata.pop("llm_fallback_used", None)
        metadata.pop("llm_model_actually_used", None)
        if "fallback_used" in rmd:
            metadata["llm_fallback_used"] = rmd["fallback_used"]
        if "model_actually_used" in rmd:
            metadata["llm_model_actually_used"] = rmd["model_actually_used"]
        post_content = response.content
        ai_text = post_content if isinstance(post_content, str) else tr.visible_content
        ctx = TurnContext(
            turn=turn, max_turns=cfg.max_turns,
            task_id=cfg.task_id, role_id=cfg.role_id,
            ai_text=ai_text, thinking=tr.thinking,
            tool_calls=parsed_calls, messages=messages,
            usage=usage, metadata=metadata,
            leaked_reasoning=leaked_reasoning,
        )

        llm_interventions = await notify_observers(obs, "on_llm_response", ctx)
        merged_llm = merge_interventions(llm_interventions)
        if merged_llm.stop_reason:
            stop_reason = merged_llm.stop_reason
            break

        # pop_last_message + inject + continue_to_next_turn — used by
        # rollback observers to retry a turn without consuming turn budget.
        # ``attempts`` is NOT rolled back, capping total rollback iterations.
        if merged_llm.pop_last_message and messages:
            messages.pop()
        if merged_llm.continue_to_next_turn:
            if merged_llm.inject_messages:
                for msg_text in merged_llm.inject_messages:
                    messages.append(user_msg(msg_text))
            turn -= 1
            continue

        if not parsed_calls:
            no_tool_retries += 1
            if (
                cfg.loop_policy.no_tool_behavior != "nudge"
                or no_tool_retries >= cfg.no_tool_max_retries
            ):
                stop_reason = "no_tool"
                break
            messages.append(user_msg(_build_no_tool_nudge(cfg.loop_policy)))
            continue

        no_tool_retries = 0

        if not merged_llm.skip_tool_execution:
            # Fire on_tool_call per call so observers can rewrite args or
            # short-circuit with a synthetic result. Order is preserved so
            # each tool message stays paired with its assistant tool_call.
            executable: list[tuple[int, dict]] = []
            synthetic: list[tuple[int, ToolResult]] = []
            for idx, tc in enumerate(parsed_calls):
                tcv = await notify_tool_call(obs, ctx, tc)
                if tcv.metadata_updates:
                    metadata.update(tcv.metadata_updates)
                if tcv.rewrite_args is not None:
                    tc = {**tc, "args": tcv.rewrite_args}
                if tcv.skip_with_result is not None:
                    synthetic.append((idx, ToolResult(
                        name=tc.get("name", ""),
                        args=tc.get("args", {}) or {},
                        result=tcv.skip_with_result,
                        duration_ms=0,
                        tool_call_id=tc.get("id") or f"call_{turn}_{idx}",
                        is_error=False,
                    )))
                else:
                    executable.append((idx, tc))

            executed_results: list[ToolResult] = []
            if executable:
                executed_results = await execute_tools(
                    [tc for _, tc in executable],
                    tool_map, cfg.tool_timeout, turn, total_tool_calls,
                )

            ordered: list[ToolResult] = [None] * len(parsed_calls)  # type: ignore[list-item]
            for (idx, _), result in zip(executable, executed_results):
                ordered[idx] = result
            for idx, result in synthetic:
                ordered[idx] = result

            total_tool_calls += len(parsed_calls)

            processor = cfg.tool_result_post_processor or (
                DefaultToolResultPostProcessor(cfg.tool_result_max_chars)
            )
            for tr_result in ordered:
                tr_result = await notify_tool_result(obs, ctx, tr_result)
                messages.append(
                    tool_msg(processor.process(tr_result), tr_result.tool_call_id),
                )

        # Context-overflow guard. If the next turn would exceed the server's
        # context limit, drop the trailing tool messages + the assistant
        # message that requested them and exit so the workflow's safety-net
        # summarize call runs on a fitting history.
        if cfg.context_overflow_guard and messages:
            trailing_tool_idx = len(messages)
            while trailing_tool_idx > 0 and is_tool_msg(
                messages[trailing_tool_idx - 1],
            ):
                trailing_tool_idx -= 1
            buffer_factor = 1.5
            trailing_tool_tokens = 0
            for m in messages[trailing_tool_idx:]:
                trailing_tool_tokens += int(
                    estimate_message_tokens(m) * buffer_factor
                )
            summary_tokens = int(
                estimate_text_tokens(cfg.summary_prompt) * buffer_factor
            )
            estimated_total = (
                last_input_tokens
                + last_output_tokens
                + trailing_tool_tokens
                + summary_tokens
                + cfg.max_completion_tokens
                + 1000
            )
            if estimated_total >= cfg.max_context_length:
                logger.warning(
                    "Context overflow guard tripped at turn=%d "
                    "(estimated=%d / limit=%d, last_input=%d, last_output=%d, "
                    "trailing_tool=%d, summary=%d). Dropping trailing tool "
                    "messages + assistant request and exiting with "
                    "stopped_by='context_limit_reached'.",
                    turn, estimated_total, cfg.max_context_length,
                    last_input_tokens, last_output_tokens,
                    trailing_tool_tokens, summary_tokens,
                )
                while messages and is_tool_msg(messages[-1]):
                    messages.pop()
                if messages and is_assistant_msg(messages[-1]):
                    messages.pop()
                stop_reason = "context_limit_reached"
                break

        if merged_llm.inject_messages:
            for msg_text in merged_llm.inject_messages:
                messages.append(user_msg(msg_text))

        turn_interventions = await notify_observers(obs, "on_turn_end", ctx)
        merged_turn = merge_interventions(turn_interventions)

        if merged_turn.inject_messages:
            for msg_text in merged_turn.inject_messages:
                messages.append(user_msg(msg_text))

        if merged_turn.stop_reason:
            stop_reason = merged_turn.stop_reason
            break

        est_tokens = estimate_tokens(messages)
        compaction_policy = cfg.compaction_policy or DefaultCompactionPolicy(
            cfg.compact_after_turns, cfg.context_token_limit,
        )
        if compaction_policy.should_compact(turn, messages, est_tokens):
            compactor = cfg.compactor or DefaultMessageCompactor()
            # ``compact`` may be sync or async; await transparently.
            result = compactor.compact(messages, cfg.keep_recent)
            messages = await result if inspect.isawaitable(result) else result

        # Per-turn persistence hook. Fires only for turns that completed
        # normally; early-exit branches bypass this so a checkpoint always
        # represents a "ready to resume" state.
        if on_turn_complete is not None:
            try:
                await on_turn_complete(turn, messages, metadata)
            except Exception as exc:
                logger.warning(
                    "on_turn_complete hook failed at turn %d: %s",
                    turn, exc,
                )

        # Graceful pause probe — always runs AFTER the checkpoint so an
        # acknowledged pause is resumable from the turn that just finished.
        if pause_check is not None:
            try:
                if await pause_check():
                    stop_reason = "paused"
                    break
            except Exception as exc:
                logger.warning(
                    "pause_check failed at turn %d: %s", turn, exc,
                )

    else:
        # Loop condition became False without a break: either we ran out of
        # ``max_turns`` budget, or we hit ``max_attempts`` while still under
        # ``max_turns`` (rollback budget exhausted).
        if attempts >= max_attempts and turn < cfg.max_turns:
            stop_reason = "max_attempts"
            logger.warning(
                "loop exhausted rollback budget at turn=%d (attempts=%d/%d)",
                turn, attempts, max_attempts,
            )
        else:
            stop_reason = "max_turns"

    # When a terminal tool latched, its ``content`` arg is the canonical
    # answer. Prefer it over the last assistant text (typically a one-line
    # "submitting…" prelude).
    latched_answer = metadata.get("final_answer") if isinstance(metadata, dict) else None
    if isinstance(latched_answer, str) and latched_answer.strip():
        final_content = latched_answer
    else:
        final_content = extract_final_content(messages)

    result = AgentLoopResult(
        messages=messages,
        final_content=final_content,
        turns_used=turn,
        tool_calls_count=total_tool_calls,
        stopped_by=stop_reason,
        metadata=metadata,
    )

    await notify_observers(obs, "on_loop_end", result)
    return result


def _resolve_phase_id(cfg: LoopConfig) -> str:
    """Phase ID priority: explicit LoopPolicy → role_id → empty."""
    return cfg.loop_policy.phase_id or cfg.role_id or ""


def _build_no_tool_nudge(policy: LoopPolicy) -> str:
    """Render the no-tool recovery message from injected loop policy."""
    if policy.no_tool_nudge_message.strip():
        return policy.no_tool_nudge_message.strip()

    if policy.terminal_tool_names:
        terminals = ", ".join(f"`{name}`" for name in policy.terminal_tool_names)
        if len(policy.terminal_tool_names) == 1:
            finish_hint = f"If you are done, call {terminals}. "
        else:
            finish_hint = (
                "If you are done, call one of the terminal tools: "
                f"{terminals}. "
            )
    else:
        finish_hint = "If your workflow defines a terminal action, use it now. "

    return (
        "This loop requires a structured tool call to continue. "
        + finish_hint
        + "Otherwise, call an appropriate tool instead of replying in plain text."
    )
