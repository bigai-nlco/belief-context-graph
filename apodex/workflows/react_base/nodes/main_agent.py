"""Main-agent node — react_base.

A plain ReAct loop with no sub-agent fan-out. The agent calls
``web_search`` / ``web_fetch`` / ``run_python_code`` directly to gather
evidence and produce a final answer. Old tool results are replaced by
short placeholders after the last N via
:class:`KeepLastNToolResultsCompactor`.

Termination is natural ReAct: the agent stops when it emits content
without calling any tool. The node then writes a full Markdown report
from the accumulated reasoning and tool results. When the loop exits
without a final answer (``max_turns`` / ``llm_error`` / budget / etc.),
:func:`_force_final_answer` first runs one tool-free LLM call to extract
a plain-text answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import date
from pathlib import Path
from typing import Any

from agent_harness.core.messages import text_of, user_msg
from agent_harness.core.runtime import registry
from agent_harness.state.event_store.sqlite import EventStore
from agent_harness.core.runtime.loop.agent_loop import run_agent_loop
from agent_harness.core.runtime.loop.llm_client import extract_model_name
from agent_harness.core.runtime.loop.model_profile import HistoryPolicy
from agent_harness.core.runtime.pause_check import pause_check_from_state
from agent_harness.core.runtime.resources.manager import ResourceManager
from agent_harness.models.node_context import NodeContext
from agent_harness.core.loop_types import (
    AgentLoopResult,
    LoopConfig,
    LoopPolicy,
)
from agent_harness.components.observers.budget_observer import BudgetObserver
from agent_harness.components.observers.context_size_guard import ContextSizeGuard
from agent_harness.components.observers.leaked_tool_call_retry import (
    LeakedToolCallRetryObserver,
)
from agent_harness.components.observers.react_step_tracker import ReactStepTracker
from agent_harness.components.observers.sse_observer import SSEObserver

from workflows.react_base.compactors import (
    AlwaysCompactPolicy,
    KeepLastNToolResultsCompactor,
)
from workflows.react_base.observers.console import RichConsoleObserver
from workflows.react_base.observers.duplicate_query_rollback import (
    DuplicateQueryRollbackObserver,
)
from workflows.react_base.observers.empty_search_rollback import (
    EmptySearchRollbackObserver,
)
from workflows.react_base.observers.refusal_rollback import (
    RefusalRollbackObserver,
)
from workflows.react_base.observers.tool_call_normalizer import (
    ToolCallArgsNormalizer,
)
from workflows.react_base.observers.trajectory import (
    TrajectoryFileObserver,
)
from workflows.react_base.prompts import (
    get_report_prompt,
    get_summarize_prompt,
    get_system_prompt,
)

logger = logging.getLogger(__name__)

# Defaults — overridable by profile YAML's ``agent`` section.
# max_turns=600, keep_tool_result=5, TOOL_RESULT_MAX_LENGTH=100_000.
MAIN_MAX_TURNS = 600
MAIN_KEEP_TOOL_RESULT = 5
MAIN_TOOL_TIMEOUT_S = 900
MAIN_LLM_TIMEOUT_S = 600
MAIN_BUDGET_TOKENS: int | None = None
MAIN_TOOL_RESULT_MAX_CHARS = 100_000

# Stop reasons that go into the **summarize** path of ``_force_final_answer``.
#
# - ``max_turns``: the loop sets ``reached_max_turns=True``
# - ``context_limit_reached``: the loop sets
#   ``turn_count = max_turns`` when ``ensure_summary_context`` pops the
#   tail to fit, so this also ends up in the same summarize branch.
# - ``no_tool``: only when the visible answer is empty (the ``</think>``
#   tail got truncated by the endpoint, leaving a tool_call-XML residue
#   or a bare ``\n\n``). the loop could return
#   ``"No final answer generated."`` in
#   this corner case — we instead run summarize for a chance to recover
#   the answer from the thinking history. Falls through to the
#   summarize block below; the non-empty-visible path returns early.
#
# Other stop reasons (``llm_error`` / ``max_attempts`` / ``budget_exhausted``
# / observer-driven stops) get a ``raise`` — the LLM client retry
# raises on exhaustion too, and fail-fast makes infra failures easier to
# spot in the trial logs than a hallucinated summary built from an empty
# message history.
_FORCED_FINAL_STOP_REASONS: frozenset[str] = frozenset(
    {
        "max_turns",
        "context_limit_reached",
        "no_tool",
    }
)

# One LLM client per profile name, shared across tasks.
_llm_cache: dict[str, tuple[dict[str, Any], Any, Any]] = {}


def _strip_thinking(text: str) -> str:
    """Strip thinking content from an LLM response.

    Returns everything after the **last** ``</think>`` tag if present,
    else the text unchanged. Handles common SGLang / Qwen quirks where the
    opening ``<think>`` may be missing but the closing tag is always
    emitted.
    """
    text = text or ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return text.strip()


def _strip_thinking_from_report_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove assistant thinking blocks before the report-writing call."""
    clean_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            clean_messages.append(message)
            continue
        if message.get("role") != "assistant":
            clean_messages.append(dict(message))
            continue
        next_message = dict(message)
        content = next_message.get("content")
        if isinstance(content, str):
            next_message["content"] = _strip_thinking(content)
        clean_messages.append(next_message)
    return clean_messages


def _normalize_report_reference_numbers(text: str) -> str:
    """Make report citations compact: References [1], [2], ... [N]."""
    heading_match = re.search(
        r"(?im)^#{1,6}\s+References\s*:?\s*$",
        text,
    )
    if not heading_match:
        return text

    body = text[: heading_match.start()]
    heading = text[heading_match.start() : heading_match.end()]
    refs = text[heading_match.end() :].lstrip("\n")
    old_to_new: dict[str, str] = {}
    next_index = 1
    ref_lines: list[str] = []

    for line in refs.splitlines():
        ref_match = re.match(r"^(\s*)(?:[-*]\s*)?\[(\d+)\](\s+.*)$", line)
        if not ref_match:
            ref_lines.append(line)
            continue
        old_label = ref_match.group(2)
        if old_label not in old_to_new:
            old_to_new[old_label] = str(next_index)
            next_index += 1
        ref_lines.append(
            f"{ref_match.group(1)}[{old_to_new[old_label]}]{ref_match.group(3)}",
        )

    if not old_to_new:
        return text

    def replace_citation(match: re.Match[str]) -> str:
        label = match.group(1)
        return f"[{old_to_new.get(label, label)}]"

    normalized_body = re.sub(r"\[(\d+)\]", replace_citation, body)
    return f"{normalized_body}{heading}\n{chr(10).join(ref_lines)}".rstrip()


def _resolve_llm_and_profile(
    profile_name: str | None,
) -> tuple[Any, dict[str, Any] | None, Any]:
    """Return ``(llm, profile_dict, model_profile)`` for this run.

    With a named profile → load YAML, build LLM + ModelProfile, cache.
    Without → use ResourceManager's ``react_solver`` LLM at temperature=1.0.
    """
    if profile_name:
        if profile_name not in _llm_cache:
            from workflows.react_base.profile import (
                build_model_profile,
                create_llm,
                load_profile,
            )

            profile = load_profile(profile_name)
            _llm_cache[profile_name] = (
                profile,
                create_llm(profile),
                build_model_profile(profile),
            )
        profile, llm, model_profile = _llm_cache[profile_name]
        return llm, profile, model_profile

    # Temperature is forwarded per-call in the agent loop now, so we don't
    # need provider-level binding here.
    llm = registry.get(ResourceManager).get_llm("react_solver")
    return llm, None, None


def _resolve_trajectory_dir(state: dict[str, Any], task_id: str) -> Path:
    """``<trial_dir>/agent/trajectories/`` when a trial dir is set in metadata,
    else ``logs/react_base/<task_id>/trajectories/``."""
    trial_dir = (state.get("metadata") or {}).get("_trial_dir")
    if trial_dir:
        return Path(trial_dir) / "agent" / "trajectories"
    return Path("logs") / "react_base" / task_id / "trajectories"


def _build_observers(
    *,
    traj_dir: Path,
    task_id: str,
    budget_tokens: int | None,
    max_input_tokens: int | None,
    event_store: Any,
    tool_names: list[str],
    tools: list[Any] | None = None,
    system_prompt: str | None = None,
    user_message: str | None = None,
    model_name: str | None = None,
) -> list[Any]:
    observers: list[Any] = [
        # Run before LeakedToolCallRetry / ReactStepTracker so downstream
        # observers see the normalized tool_calls in ctx.messages[-1].
        ToolCallArgsNormalizer(),
        LeakedToolCallRetryObserver(tool_names=tool_names),
        # Duplicate web_search query rollback. Sits AFTER normalizer so
        # it sees the canonical ``q`` argument shape, and BEFORE
        # ReactStepTracker / Trajectory observers so a rolled-back turn
        # is not counted as a real step.
        DuplicateQueryRollbackObserver(),
        # Refusal-keyword rollback. Fires on the same
        # ``on_llm_response`` phase as DuplicateQueryRollback, so order
        # within this group doesn't matter — ``merge_interventions``
        # ORs the pop / continue flags. Cheap substring check.
        RefusalRollbackObserver(),
        # Empty-search-result rollback. Fires on ``on_turn_end``, after
        # tools have run, so its phase doesn't overlap with the
        # on_llm_response observers above.
        EmptySearchRollbackObserver(),
        RichConsoleObserver(),
        TrajectoryFileObserver(
            traj_dir,
            tools=tools or [],
            model_name=model_name,
            system_prompt=system_prompt,
            user_message=user_message,
        ),
        ReactStepTracker(),
    ]
    if budget_tokens:
        observers.append(BudgetObserver(max_tokens=budget_tokens))
    if max_input_tokens:
        observers.append(ContextSizeGuard(max_input_tokens=max_input_tokens))
    if event_store is not None:
        observers.append(SSEObserver(event_store=event_store, task_id=task_id))
    return observers


def _loop_policy() -> LoopPolicy:
    """Stop on no-tool turns — natural ReAct termination."""
    return LoopPolicy(no_tool_behavior="stop")


def _append_traj_event(traj_dir: Path | None, event: dict[str, Any]) -> None:
    """Append a single JSONL event to the trial's trajectory file.

    Used by ``_force_final_answer`` to record summarize calls that happen
    outside ``run_agent_loop``'s observer chain — without this, the
    trajectory's last event is the agent loop's ``end`` and there's no
    record of the summarize prompt, response, retry attempts, or token
    usage that produced ``result.json.predicted_answer``.

    Best-effort: silently skips when the dir/file is missing (e.g.,
    standalone debug run with no trajectory observer) or when write
    fails — the trial result still has ``predicted_answer``, this just
    augments the trajectory.
    """
    if traj_dir is None or not traj_dir.is_dir():
        return
    jsonls = list(traj_dir.glob("*.jsonl"))
    if not jsonls:
        return
    try:
        with jsonls[0].open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 — best-effort logging
        logger.debug("Trajectory append failed (non-fatal): %s", exc)


async def _force_final_answer(
    result: AgentLoopResult,
    llm: Any,
    timeout: float,
    task_description: str,
    task_id: str = "",
    traj_dir: Path | None = None,
) -> AgentLoopResult:
    """Ensure ``result.metadata['final_answer']`` carries a plain-text answer.

    - **Natural finish** (``stopped_by == "no_tool"``,
      ``reached_max_turns=False``): use the last assistant response —
      ``_strip_thinking(last_response)``.
    - **max_turns / context_limit_reached** (``reached_max_turns=True``):
      append the summarize prompt and run the LLM with a 10× retry × 30s
      fixed sleep. If all 10 attempts fail → ``raise`` so the harbor task
      records ``pipeline_error`` rather than hiding as a wrong verdict.
    - **Anything else** (``llm_error`` / ``max_attempts`` /
      ``budget_exhausted`` / observer-driven stops): ``raise``. These all
      mean the message history is unreliable (the loop never produced
      any turns / rollback wiped state / budget cut mid-call).

    Context-length errors (``Error code: 400 ... longer than the model``)
    inside summarize retries are deterministic, not transient — skip
    retry on these.
    """
    if result.metadata.get("final_answer"):
        return result

    if result.stopped_by == "no_tool":
        final = _strip_thinking(result.final_content)
        if final:
            result.metadata["final_answer"] = final
            result.final_content = final
            return result
        # Empty visible after no_tool stop — usually endpoint truncated
        # the ``</think>`` tail. Fall through to summarize.
        logger.info(
            "force_final: no_tool stop with empty visible (likely </think> "
            "truncation) — falling through to summarize (task=%s, turns=%d)",
            task_id,
            result.turns_used,
        )

    if result.stopped_by not in _FORCED_FINAL_STOP_REASONS:
        # Fail-fast: ``llm_error`` / ``max_attempts`` / ``budget_exhausted``
        # / observer-driven stops all leave the message history in an
        # unreliable state. Don't paper over with a hallucinated summary.
        raise RuntimeError(
            f"agent_loop stopped with stopped_by={result.stopped_by!r}; "
            f"force_final cannot recover (message history unreliable). "
            f"turns_used={result.turns_used}, "
            f"messages={len(result.messages)}",
        )

    summarize_prompt = get_summarize_prompt(task_description)

    # Strip a trailing user nudge (observer-injected) before appending the
    # summary prompt. Tool messages stay — the summary needs them.
    if (
        result.messages
        and isinstance(result.messages[-1], dict)
        and result.messages[-1].get("role") == "user"
    ):
        result.messages.pop()
    result.messages.append(user_msg(summarize_prompt))

    del task_id  # noqa: F841 — kept on signature for compat

    # Record summarize setup in the trajectory so prompt→response is
    # visible alongside ``result.json.predicted_answer``.
    _append_traj_event(
        traj_dir,
        {
            "t": "summarize_start",
            "ts": time.time(),
            "stopped_by_reason": result.stopped_by,
            "messages_count": len(result.messages),
            "summarize_prompt_chars": len(summarize_prompt),
            "summarize_prompt_preview": summarize_prompt[:500],
        },
    )

    max_retries = 10
    base_wait_time = 30
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = await asyncio.wait_for(
                llm.chat(result.messages, timeout=timeout),
                timeout=timeout,
            )
            content = text_of(resp.content)
            text = _normalize_report_reference_numbers(_strip_thinking(content))
            if text:
                result.messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
                result.metadata["final_answer"] = text
                result.final_content = text
                logger.info(
                    "force_final_answer success after %s (attempt %d/%d, %d chars)",
                    result.stopped_by,
                    attempt + 1,
                    max_retries,
                    len(text),
                )
                _append_traj_event(
                    traj_dir,
                    {
                        "t": "summarize_response",
                        "ts": time.time(),
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "content": text,
                        "usage": dict(resp.usage or {}),
                    },
                )
                return result
            # Empty content slot — count as a failure so we retry. The
            # SGLang server occasionally returns ``""`` under load.
            last_exc = RuntimeError("LLM returned empty content")
            logger.warning(
                "force_final_answer attempt %d/%d returned empty content",
                attempt + 1,
                max_retries,
            )
            _append_traj_event(
                traj_dir,
                {
                    "t": "summarize_retry",
                    "ts": time.time(),
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "error": "empty content",
                },
            )
        except Exception as exc:
            msg = str(exc)
            # Deterministic context-length failure — skip retry.
            if "Error code: 400" in msg and "longer than the model" in msg:
                logger.error(
                    "force_final_answer context-length error (non-retryable): %s",
                    exc,
                )
                _append_traj_event(
                    traj_dir,
                    {
                        "t": "summarize_failed",
                        "ts": time.time(),
                        "reason": "context_length_non_retryable",
                        "error": str(exc),
                    },
                )
                raise
            last_exc = exc
            logger.warning(
                "force_final_answer attempt %d/%d failed: %s",
                attempt + 1,
                max_retries,
                exc,
            )
            _append_traj_event(
                traj_dir,
                {
                    "t": "summarize_retry",
                    "ts": time.time(),
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "error": str(exc)[:500],
                },
            )

        if attempt < max_retries - 1:
            await asyncio.sleep(base_wait_time)

    # All retries exhausted. Re-raise so agent.py records pipeline_error
    # and the trial result becomes ``<ANSWER_NOT_FOUND>`` (judge fail).
    # No fallback to leftover ``result.final_content`` — silently shipping
    # the popped mid-search assistant message as the "answer" makes failures
    # harder to debug.
    logger.error(
        "force_final_answer exhausted %d retries: %s",
        max_retries,
        last_exc,
    )
    _append_traj_event(
        traj_dir,
        {
            "t": "summarize_failed",
            "ts": time.time(),
            "reason": "retries_exhausted",
            "max_retries": max_retries,
            "last_error": str(last_exc)[:500] if last_exc else "",
        },
    )
    raise RuntimeError(
        f"force_final_answer failed after {max_retries} attempts: {last_exc}",
    ) from last_exc


async def _force_final_report(
    result: AgentLoopResult,
    llm: Any,
    timeout: float,
    task_description: str,
    task_id: str = "",
    traj_dir: Path | None = None,
) -> AgentLoopResult:
    """Generate the canonical Markdown report from accumulated evidence."""
    existing = result.metadata.get("report")
    if isinstance(existing, str) and existing.strip():
        return result

    report_prompt = get_report_prompt(task_description)

    if result.stopped_by not in _FORCED_FINAL_STOP_REASONS:
        # Fail-fast: ``llm_error`` / ``max_attempts`` / ``budget_exhausted``
        # / observer-driven stops all leave the message history in an
        # unreliable state. Don't paper over with a hallucinated summary.
        raise RuntimeError(
            f"agent_loop stopped with stopped_by={result.stopped_by!r}; "
            f"force_final cannot recover (message history unreliable). "
            f"turns_used={result.turns_used}, "
            f"messages={len(result.messages)}",
        )

    # Strip a trailing user nudge (observer-injected) before appending the
    # report prompt. Tool messages stay — the report needs them.
    report_messages = _strip_thinking_from_report_messages(result.messages)
    if (
        report_messages
        and isinstance(report_messages[-1], dict)
        and report_messages[-1].get("role") == "user"
    ):
        report_messages.pop()
    report_messages.append(user_msg(report_prompt))

    del task_id  # noqa: F841 — kept on signature for compat

    # Record summarize setup in the trajectory so prompt→response is
    # visible alongside ``result.json.predicted_answer``.
    _append_traj_event(
        traj_dir,
        {
            "t": "report_start",
            "ts": time.time(),
            "stopped_by_reason": result.stopped_by,
            "messages_count": len(report_messages),
            "report_prompt_chars": len(report_prompt),
            "report_prompt_preview": report_prompt[:500],
        },
    )

    max_retries = 10
    base_wait_time = 30
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = await asyncio.wait_for(
                llm.chat(report_messages, timeout=timeout),
                timeout=timeout,
            )
            content = text_of(resp.content)
            text = _normalize_report_reference_numbers(_strip_thinking(content))
            if text:
                result.messages = report_messages + [{
                    "role": "assistant",
                    "content": content,
                }]
                result.metadata["report"] = text
                result.metadata["final_answer"] = text
                result.final_content = text
                logger.info(
                    "force_final_report success after %s (attempt %d/%d, %d chars)",
                    result.stopped_by,
                    attempt + 1,
                    max_retries,
                    len(text),
                )
                _append_traj_event(
                    traj_dir,
                    {
                        "t": "report_response",
                        "ts": time.time(),
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "content": text,
                        "usage": dict(resp.usage or {}),
                    },
                )
                return result
            # Empty content slot — count as a failure so we retry. The
            # SGLang server occasionally returns ``""`` under load.
            last_exc = RuntimeError("LLM returned empty content")
            logger.warning(
                "force_final_report attempt %d/%d returned empty content",
                attempt + 1,
                max_retries,
            )
            _append_traj_event(
                traj_dir,
                {
                    "t": "report_retry",
                    "ts": time.time(),
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "error": "empty content",
                },
            )
        except Exception as exc:
            msg = str(exc)
            # Deterministic context-length failure — skip retry.
            if "Error code: 400" in msg and "longer than the model" in msg:
                logger.error(
                    "force_final_report context-length error (non-retryable): %s",
                    exc,
                )
                _append_traj_event(
                    traj_dir,
                    {
                        "t": "report_failed",
                        "ts": time.time(),
                        "reason": "context_length_non_retryable",
                        "error": str(exc),
                    },
                )
                raise
            last_exc = exc
            logger.warning(
                "force_final_report attempt %d/%d failed: %s",
                attempt + 1,
                max_retries,
                exc,
            )
            _append_traj_event(
                traj_dir,
                {
                    "t": "report_retry",
                    "ts": time.time(),
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "error": str(exc)[:500],
                },
            )

        if attempt < max_retries - 1:
            await asyncio.sleep(base_wait_time)

    # All retries exhausted. Re-raise so agent.py records pipeline_error
    # and the trial result becomes ``<ANSWER_NOT_FOUND>`` (judge fail).
    # No fallback to leftover ``result.final_content`` — silently shipping
    # the popped mid-search assistant message as the "answer" makes failures
    # harder to debug.
    logger.error(
        "force_final_report exhausted %d retries: %s",
        max_retries,
        last_exc,
    )
    _append_traj_event(
        traj_dir,
        {
            "t": "report_failed",
            "ts": time.time(),
            "reason": "retries_exhausted",
            "max_retries": max_retries,
            "last_error": str(last_exc)[:500] if last_exc else "",
        },
    )
    raise RuntimeError(
        f"force_final_report failed after {max_retries} attempts: {last_exc}",
    ) from last_exc


async def main_agent_node(
    state: dict[str, Any],
    ctx: NodeContext,
) -> dict[str, Any]:
    """Run the react_base agent as a single ReAct loop."""
    question = state.get("original_question", "")
    if not question:
        raise ValueError("main_agent_node requires 'original_question' in state")

    # Strip the "# Task\n\n" header that the harbor task generator wraps
    # questions in; feed the bare question text to the model.
    if question.startswith("# Task\n\n"):
        question = question[len("# Task\n\n") :]
    question = question.rstrip()

    profile_name: str | None = (state.get("metadata") or {}).get(
        "profile",
    )
    llm, profile, model_profile = _resolve_llm_and_profile(profile_name)

    agent_cfg = (profile or {}).get("agent", {})

    max_turns = int(agent_cfg.get("main_max_turns", MAIN_MAX_TURNS))
    keep_tool_result = int(
        agent_cfg.get("keep_tool_result", MAIN_KEEP_TOOL_RESULT),
    )
    budget_raw = agent_cfg.get("budget_tokens", MAIN_BUDGET_TOKENS)
    budget_tokens = int(budget_raw) if budget_raw else None
    max_input_raw = agent_cfg.get("max_input_tokens")
    max_input_tokens = int(max_input_raw) if max_input_raw else None
    tool_timeout = float(agent_cfg.get("tool_timeout_s", MAIN_TOOL_TIMEOUT_S))
    llm_timeout = float(agent_cfg.get("llm_timeout_s", MAIN_LLM_TIMEOUT_S))
    tool_result_max_chars = int(
        agent_cfg.get(
            "tool_result_max_chars",
            MAIN_TOOL_RESULT_MAX_CHARS,
        )
    )
    thinking_in_history = bool(agent_cfg.get("thinking_in_history", False))

    resource_mgr = registry.get(ResourceManager)
    tools = resource_mgr.get_tools_for_role("react_solver")

    if profile is not None:
        from workflows.react_base.profile import apply_fastmcp_schemas

        apply_fastmcp_schemas(tools)

    tool_names = [getattr(t, "name", "") for t in tools if getattr(t, "name", "")]
    system_prompt = get_system_prompt(date.today().isoformat())
    # Build terminal prompts up front so the kernel's context-overflow
    # guard can budget for the final extraction and report-writing calls.
    # final_summary_prompt = get_summarize_prompt(question)
    report_prompt = get_report_prompt(question)
    # summary_prompt = f"{final_summary_prompt}\n\n{report_prompt}"
    # Completion budget used by the context guard to reserve tokens
    # for the next response. Profile YAML is authoritative; fall back
    # to 32K if not exposed by the LLM client.
    profile_llm_cfg = (profile or {}).get("llm") or {}
    max_completion_tokens = int(
        profile_llm_cfg.get("max_tokens") or getattr(llm, "max_tokens", None) or 32_768,
    )
    # Server-side context cap (YAML ``agent.max_input_tokens`` overrides).
    max_context_length = int(
        agent_cfg.get("max_input_tokens", 262_144) or 262_144,
    )
    model_name = extract_model_name(llm, profile)

    event_store = registry.get_optional(EventStore)
    traj_dir = _resolve_trajectory_dir(state, ctx.task_id)
    observers = _build_observers(
        traj_dir=traj_dir,
        task_id=ctx.task_id,
        budget_tokens=budget_tokens,
        max_input_tokens=max_input_tokens,
        event_store=event_store,
        tool_names=tool_names,
        tools=tools,
        system_prompt=system_prompt,
        user_message=question,
        model_name=model_name,
    )
    # Callers may inject extra observers via ``state.metadata["sdk_extra_observers"]``.
    metadata = state.get("metadata") or {}
    extra_observers = metadata.get("sdk_extra_observers") or []
    if extra_observers:
        observers = list(observers) + list(extra_observers)

    logger.info(
        "react_base starting (task_id=%s, tools=%s, max_turns=%d, "
        "keep_tool_result=%d, tool_result_max_chars=%d)",
        ctx.task_id,
        tool_names,
        max_turns,
        keep_tool_result,
        tool_result_max_chars,
    )

    # Forward the pause closure so SIGTERM breaks the loop at the next
    # turn boundary.
    result = await run_agent_loop(
        system_prompt=system_prompt,
        user_message=question,
        llm=llm,
        tools=tools,
        config=LoopConfig(
            max_turns=max_turns,
            task_id=ctx.task_id,
            role_id="react_solver",
            no_tool_max_retries=0,
            tool_timeout=int(tool_timeout),
            llm_timeout=int(llm_timeout),
            tool_result_max_chars=tool_result_max_chars,
            # 15 × 90s fixed-interval retries — accommodates slow
            # backend recovery between attempts.
            max_llm_retries=15,
            retry_wait_fixed=90,
            loop_policy=_loop_policy(),
            compactor=KeepLastNToolResultsCompactor(keep_tool_result),
            compaction_policy=AlwaysCompactPolicy(),
            # Proactive context-overflow guard. Pops the trailing
            # tool message(s) + last assistant message before the next request
            # would exceed ``max_context_length`` and exits with
            # ``stopped_by="context_limit_reached"`` so
            # ``_force_final_answer`` runs summarize on a fitting
            # history instead of catching a 400 with no recoverable
            # answer.
            context_overflow_guard=True,
            max_context_length=max_context_length,
            max_completion_tokens=max_completion_tokens,
            summary_prompt=report_prompt,
        ),
        model_profile=model_profile,
        history_policy=HistoryPolicy(thinking_in_history=thinking_in_history),
        observers=observers,
        pause_check=pause_check_from_state(state),
    )
    # result = await _force_final_answer(
    #     result, llm, llm_timeout,
    #     task_description=question, task_id=ctx.task_id,
    #     traj_dir=traj_dir,
    # )
    result = await _force_final_report(
        result,
        llm,
        llm_timeout,
        task_description=question,
        task_id=ctx.task_id,
        traj_dir=traj_dir,
    )

    logger.info(
        "react_base done (task=%s) stopped_by=%s turns=%d",
        ctx.task_id,
        result.stopped_by,
        result.turns_used,
    )

    final_text = (
        result.metadata.get("report")
        or result.metadata.get("final_answer")
        or result.final_content
    )
    return {
        "final_answer": final_text,
        "report": final_text,
        # Mirror to ``final_content`` so the scheduler's terminal-output
        # check passes when react_base has no separate ``report`` node.
        "final_content": final_text,
        "answer_confidence": "",
        "react_steps": result.metadata.get("react_steps", []),
    }


__all__ = ["main_agent_node"]
