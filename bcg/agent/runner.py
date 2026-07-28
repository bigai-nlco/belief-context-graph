"""Runner for agent workflow rollouts.

Drives BeliefTracerAgent + BeliefTracerEnvironment + BeliefTracerWorkflow through
AgentWorkflowEngine using an in-process vLLM engine by default. Set
``backend="openai"`` to fall back to talking to an external
OpenAI-compatible server.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

import rllm.engine.agent_workflow_engine as _awe_mod
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine

from bcg.agent.benchmark_loader import AgenticTask, load_benchmark
from bcg.agent.config import AgentRolloutConfig, model_output_dir_name
from bcg.agent.context_memory import BELIEF_GRAPH_MODE, uses_belief_graph_service
from bcg.agent.reward import (
    BrowseCompJudgeConfig,
    build_dispatching_reward_fn,
    build_reward_fn,
)
from bcg.agent.rllm_compat import BeliefTracerAgent, BeliefTracerEnvironment
from bcg.agent.tokenizer_compat import (
    build_sampling_params_compat,
    is_qwen35_or_qwen36,
    load_tokenizer_compat,
    qwen_thinking_token_report,
    qwen_thinking_chat_template_kwargs,
)
from bcg.agent.workflow import BeliefTracerWorkflow
from bcg.agent.belief_graph_client import BeliefGraphClient

console = Console()
logger = logging.getLogger(__name__)
RUN_STATE_HEARTBEAT_SECONDS = 5.0

# Suppress the per-rollout "Rollout completed. Rewards: ..." log from rllm.
_orig_colorful_print = _awe_mod.colorful_print


def _filtered_colorful_print(msg, **kwargs):
    if "Rollout completed." not in str(msg):
        _orig_colorful_print(msg, **kwargs)


_awe_mod.colorful_print = _filtered_colorful_print

# Fixed namespaces so problem_id / trajectory_id are stable across runs.
_PROBLEM_NS = uuid.UUID("6f3b8b1e-2a2a-4a1e-9b4a-1f5f1e3d7c01")
_TRAJECTORY_NS = uuid.UUID("6f3b8b1e-2a2a-4a1e-9b4a-1f5f1e3d7c02")


def _iso(ts: float | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _problem_uuid(data_source: str, task_id: str, question: str) -> str:
    return str(uuid.uuid5(_PROBLEM_NS, f"{data_source}\x1f{task_id}\x1f{question}"))


def _trajectory_uuid(problem_id: str, episode) -> str:
    """Derive a uuid5 from the full trajectory content.

    Identical trajectories (same model turns + tool outputs) map to the same
    id; any divergence in steps, tool results, or termination state produces a
    different id.
    """
    parts: list[str] = [problem_id]
    for traj in getattr(episode, "trajectories", []) or []:
        for step in getattr(traj, "steps", []) or []:
            parts.append(str(getattr(step, "model_response", "") or ""))
            parts.append(str(getattr(step, "action", "") or ""))
            parts.append(str(getattr(step, "observation", "") or ""))
    term = getattr(episode, "termination_reason", None)
    parts.append(getattr(term, "value", "") if term is not None else "")
    return str(uuid.uuid5(_TRAJECTORY_NS, "\x1e".join(parts)))

def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples < k:
        return 1.0 if num_correct == num_samples else 0.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def estimate_pass_hat_k(num_samples: int, num_correct: int, k: int) -> float:
    """Pass^k: probability that *all* k independently drawn samples are correct.

    C(c, k) / C(n, k). Returns 0.0 when k > num_correct (impossible).
    """
    if num_samples < k:
        return 1.0 if num_correct == num_samples else 0.0
    if num_correct < k:
        return 0.0
    return math.comb(num_correct, k) / math.comb(num_samples, k)


def _build_sampling_params(cfg: AgentRolloutConfig) -> dict[str, Any]:
    return build_sampling_params_compat(
        cfg.model,
        enable_thinking=cfg.enable_thinking,
        max_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        min_p=cfg.min_p,
        presence_penalty=cfg.presence_penalty,
        repetition_penalty=cfg.repetition_penalty,
    )


def _build_openai_sampling_params(cfg: AgentRolloutConfig) -> dict[str, Any]:
    sampling_params = _build_sampling_params(cfg)
    extra_body = dict(sampling_params.pop("extra_body", {}) or {})
    chat_template_kwargs = qwen_thinking_chat_template_kwargs(
        cfg.model, disable_thinking=not cfg.enable_thinking
    )
    if chat_template_kwargs:
        template_kwargs = dict(extra_body.get("chat_template_kwargs", {}) or {})
        template_kwargs.update(chat_template_kwargs)
        extra_body["chat_template_kwargs"] = template_kwargs

    for key in ("top_k", "min_p", "repetition_penalty"):
        if key in sampling_params:
            extra_body[key] = sampling_params.pop(key)

    if extra_body:
        sampling_params["extra_body"] = extra_body
    return sampling_params


def _present_config_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _compact_config(value: Any) -> Any:
    if isinstance(value, dict):
        compacted = {
            key: _compact_config(item)
            for key, item in value.items()
            if _present_config_value(item)
        }
        return {
            key: item
            for key, item in compacted.items()
            if _present_config_value(item)
        }
    if isinstance(value, list):
        return [
            item
            for item in (_compact_config(item) for item in value)
            if _present_config_value(item)
        ]
    return value


def _sampling_param_sources(cfg: AgentRolloutConfig) -> dict[str, str]:
    explicit = {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "min_p": cfg.min_p,
        "presence_penalty": cfg.presence_penalty,
        "repetition_penalty": cfg.repetition_penalty,
    }
    qwen35 = is_qwen35_or_qwen36(cfg.model)
    if qwen35:
        default_source = (
            "qwen3.5/3.6 thinking default"
            if cfg.enable_thinking
            else "qwen3.5/3.6 no-thinking default"
        )
    elif cfg.backend in {"sglang", "sglang_dp"}:
        default_source = "BeliefTracer SGLang default"
    else:
        default_source = "BeliefTracer default"

    sources = {
        key: ("cli override" if value is not None else default_source)
        for key, value in explicit.items()
    }
    sources["max_tokens"] = "max_new_tokens"
    return sources


def _effective_run_config(cfg: AgentRolloutConfig) -> dict[str, Any]:
    sampling_params = _build_sampling_params(cfg)
    engine: dict[str, Any] = {
        "backend": cfg.backend,
        "max_model_len": cfg.vllm_max_model_len
        or (cfg.max_prompt_length + cfg.max_response_length),
        "max_prompt_length": cfg.max_prompt_length,
        "max_response_length": cfg.max_response_length,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "trust_remote_code": cfg.vllm_trust_remote_code,
    }
    if cfg.backend in {"vllm", "ray_vllm"}:
        engine.update(
            {
                "dtype": cfg.vllm_dtype,
                "enforce_eager": cfg.vllm_enforce_eager,
            }
        )
    if cfg.backend in {"sglang", "sglang_dp"}:
        engine.update(
            {
                "sglang_context_length": engine["max_model_len"],
                "sglang_max_total_tokens": engine["max_model_len"],
                "sglang_disable_cuda_graph": True,
                "sglang_disable_piecewise_cuda_graph": True,
            }
        )
    if cfg.backend == "sglang_dp":
        engine.update(
            {
                "data_parallel_size": cfg.data_parallel_size,
                "data_parallel_devices": cfg.data_parallel_devices,
            }
        )
    if cfg.backend == "ray_vllm":
        engine.update(
            {
                "ray_num_replicas": cfg.ray_num_replicas,
                "ray_address": cfg.ray_address,
            }
        )
    if cfg.backend == "openai":
        engine.update({"base_url": cfg.resolved_base_url()})

    effective = {
        "model": {
            "path": cfg.model,
            "enable_thinking": cfg.enable_thinking,
        },
        "benchmarks": {
            "tasks": cfg.tasks,
            "max_problems": cfg.max_problems,
            "shuffle": cfg.shuffle,
            "shuffle_seed": cfg.shuffle_seed,
            "num_samples": cfg.num_samples,
            "passk": cfg.passk,
        },
        "agent": {
            "parser_name": cfg.parser_name,
            "tools": cfg.tools,
            "max_steps": cfg.max_steps,
            "system_prompt_enabled": bool(cfg.system_prompt),
            "system_prompt": cfg.system_prompt,
            "context_memory_mode": cfg.context_memory_mode,
            "context_memory_recent_observations": cfg.context_memory_recent_observations,
            "context_memory_tail_turns": cfg.context_memory_tail_turns,
            "context_memory_max_chars": cfg.context_memory_max_chars,
            "context_memory_tool_summary_chars": cfg.context_memory_tool_summary_chars,
            "context_memory_interval": cfg.context_memory_interval,
            "context_memory_summarizer": cfg.context_memory_summarizer,
            "context_memory_summarizer_max_tokens": cfg.context_memory_summarizer_max_tokens,
            "context_memory_summarizer_timeout": cfg.context_memory_summarizer_timeout,
            "context_memory_summarizer_failure_limit": cfg.context_memory_summarizer_failure_limit,
            "context_memory_log_preview_chars": cfg.context_memory_log_preview_chars,
            "context_memory_summarizer_model": cfg.model,
            "context_memory_summarizer_base_url": cfg.resolved_base_url(),
        },
        "engine": engine,
        "sampling": sampling_params,
        "sampling_sources": {
            key: source
            for key, source in _sampling_param_sources(cfg).items()
            if key in sampling_params
        },
        "retrieval": {
            "server_url": cfg.retrieval_server_url,
            "max_results": cfg.retrieval_max_results,
            "timeout": cfg.retrieval_timeout,
        },
        "scheduler": {
            "n_parallel_tasks": cfg.n_parallel_tasks,
            "retry_limit": cfg.retry_limit,
            "mixed_rollouts": cfg.mixed_rollouts,
        },
        "artifacts": {
            "output_dir": str(_model_output_dir(cfg)),
            "overwrite": cfg.overwrite,
            "auto_ui": cfg.auto_ui,
        },
    }
    return _compact_config(effective)


def _extract_from_tool_calls(step) -> str:
    """Extract answer from the finish tool call in chat_completions or action."""
    # Search chat_completions for the last assistant message with a finish tool call
    for msg in reversed(getattr(step, "chat_completions", None) or []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            func = tc if isinstance(tc, dict) and "name" in tc else tc.get("function", {})
            if func.get("name") == "finish":
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        return args
                answer = str(args.get("answer", "")) if isinstance(args, dict) else str(args)
                # JSON double-parse turns \\boxed into \x08oxed (backspace).
                # Restore so downstream extractors can find \boxed{}.
                answer = answer.replace("\x08oxed", "\\boxed")
                return answer
    return ""


def _extract_final_answer(
    episode, ground_truth=None, data_source=None, question=None
) -> tuple[str, bool, dict]:
    """Pull the extracted answer, correctness flag, and reward metadata.

    The last step of the agent trajectory carries an ``info`` dict populated
    by ``ToolEnvironment.step`` which includes ``is_correct`` and
    ``metadata`` from the reward function (with ``extracted_answer``).

    We additionally re-apply the boxed-first extractor at save time: the
    reward path can return an empty or raw-response ``extracted_answer`` if
    the env's reward_fn was mis-wired or if the model emitted the boxed
    span without a ``finish`` tool call. Re-extracting here guarantees the
    JSON record carries the short predicted string (``\\boxed{...}``
    contents) rather than the full trajectory text.

    When ``ground_truth`` is provided, we re-score the extracted answer to
    fix stale correctness flags from rollout-time scoring of raw responses.
    """
    from bcg.agent.reward import (
        MCQ_DATA_SOURCES,
        _BoxedFirstSearchFn,
        build_reward_fn,
    )  # local import to avoid cycles
    from rllm.rewards.reward_types import RewardConfig, RewardInput

    extracted = ""
    is_correct = False
    metadata: dict[str, Any] = {}
    if episode.trajectories and episode.trajectories[0].steps:
        last = episode.trajectories[0].steps[-1]
        is_correct = bool(last.info.get("is_correct", False))
        metadata = last.info.get("metadata") or {}
        # Structured tool_calls is the most reliable source: it comes
        # directly from the API's parsed finish-tool arguments.
        extracted = _extract_from_tool_calls(last)

        # Try re-extracting from model_response (with corruption fix)
        # before trusting metadata, which may have been corrupted by
        # the \boxed → \x08oxed double-parse bug in the reward function.
        if not extracted and last.model_response:
            fixed = last.model_response.replace("\x08oxed", "\\boxed")
            if "\\boxed" in fixed or "boxed{" in fixed:
                try:
                    _fn = _BoxedFirstSearchFn.__new__(_BoxedFirstSearchFn)
                    reextracted = _fn.extract_answer_from_response(fixed)
                    if reextracted:
                        extracted = reextracted
                except Exception:
                    pass

        if not extracted:
            meta_answer = str(metadata.get("extracted_answer", ""))
            extracted = meta_answer.replace("\x08oxed", "\\boxed")

        if not extracted and last.model_response:
            extracted = last.model_response

        # Re-unbox if the stored value is clearly a raw response (contains a
        # boxed anchor or is long enough to be a trajectory).
        if extracted and (
            "\\boxed" in extracted or "boxed{" in extracted or len(extracted) > 200
        ):
            try:
                _fn = _BoxedFirstSearchFn.__new__(_BoxedFirstSearchFn)
                reextracted = _fn.extract_answer_from_response(extracted)
                if reextracted and reextracted != extracted:
                    extracted = reextracted
            except Exception:
                pass

        if data_source in {"browsecomp", "browse_comp"}:
            # The rollout-time LLM judge is authoritative for BrowseComp. Do
            # not overwrite its verdict with the generic EM/F1 rescoring path
            # below (and do not pay for a duplicate judge call at save time).
            judge_extracted = str(metadata.get("extracted_answer") or "").strip()
            if judge_extracted and judge_extracted.lower() != "none":
                extracted = judge_extracted
            return extracted, is_correct, metadata

        if data_source in MCQ_DATA_SOURCES and ground_truth is not None:
            # MCQ benchmarks must be rescored from the raw final response so
            # values/prose/LaTeX wrappers can be mapped back to A/B/C/D.
            score_action = last.model_response or extracted
            task_info = {
                "ground_truth": ground_truth,
                "data_source": data_source or "",
            }
            if question is not None:
                task_info["question"] = question
            result = build_reward_fn(data_source)(task_info, score_action)
            is_correct = bool(result.is_correct)
            metadata.update(dict(result.metadata or {}))
            extracted = str(metadata.get("extracted_answer", extracted))
            return extracted, is_correct, metadata

        # Re-score if ground_truth provided and extraction changed
        if ground_truth is not None and extracted:
            cfg = RewardConfig(
                **{
                    k: v
                    for k, v in {
                        "toolcall_bonus": 0.0,
                        "apply_repetition_penalty": False,
                        "correct_reward": 1.0,
                        "incorrect_reward": 0.0,
                        "enable_step_bonus": False,
                    }.items()
                    if k in inspect.signature(RewardConfig).parameters
                }
            )
            fn = _BoxedFirstSearchFn(cfg)
            task_info = {"ground_truth": ground_truth, "data_source": data_source or ""}
            if question is not None:
                # MCQ datasets whose GT is option prose (e.g. ``medqa``) need
                # the question text so ``evaluate_answer`` can map a bare
                # letter like ``"D"`` back to the D-option prose. Without it
                # the reward sees ``"D"`` vs prose GT and scores 0.
                task_info["question"] = question
            result = fn(RewardInput(task_info=task_info, action=extracted))
            is_correct = result.is_correct
            metadata.update(result.metadata)

    return extracted, is_correct, metadata


def _json_safe(value: Any) -> Any:
    """Recursively coerce a value to JSON-serializable primitives."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@lru_cache(maxsize=8)
def _load_metadata_tokenizer(model: str, trust_remote_code: bool):
    return load_tokenizer_compat(model, trust_remote_code=trust_remote_code)


def _count_model_tokens(
    text: str | None, *, model: str, trust_remote_code: bool
) -> int:
    if not text:
        return 0
    if not Path(model).exists():
        return 0
    try:
        tokenizer = _load_metadata_tokenizer(model, trust_remote_code)
    except Exception:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _extract_trajectory_messages(episode) -> list[dict[str, Any]]:
    """Return the full chat transcript (system/user/assistant/tool turns).

    ``ToolAgent`` stores the cumulative message list on every ``Step`` via
    ``chat_completions``; the last step holds the complete conversation
    including the agent's final assistant turn. We return that list
    directly (JSON-sanitized) so consumers can replay every role/content
    pair — plus tool-call metadata (``tool_calls`` on assistant turns,
    ``tool_call_id`` on tool turns) — exactly as the model saw it.
    """
    if not episode.trajectories:
        return []
    steps = episode.trajectories[0].steps
    if not steps:
        return []
    messages = getattr(steps[-1], "chat_completions", None) or []
    return _json_safe(messages)


def _extract_model_io(episode) -> list[dict[str, Any]]:
    """Return per-turn model input/output recorded during rollout."""
    if not episode.trajectories:
        return []
    steps = episode.trajectories[0].steps
    if not steps:
        return []
    for s in reversed(steps):
        s_info = s.info or {}
        if "model_io" in s_info:
            return _json_safe(s_info["model_io"])
    return []


def _extract_last_step_field(episode, key: str, default=None):
    if not episode.trajectories:
        return default
    steps = episode.trajectories[0].steps
    if not steps:
        return default
    for s in reversed(steps):
        val = (s.info or {}).get(key)
        if val is not None:
            return val
    return default


def _extract_model_step_metadata(
    episode, *, model: str, trust_remote_code: bool
) -> list[dict[str, Any]]:
    """Return compact token/reasoning diagnostics for each model call."""

    if not episode.trajectories:
        return []

    records: list[dict[str, Any]] = []
    for idx, step in enumerate(getattr(episode.trajectories[0], "steps", []) or []):
        model_output = getattr(step, "model_output", None)
        prompt_ids = (
            getattr(model_output, "prompt_ids", None)
            if model_output is not None
            else getattr(step, "prompt_ids", None)
        )
        completion_ids = (
            getattr(model_output, "completion_ids", None)
            if model_output is not None
            else getattr(step, "response_ids", None)
        )
        prompt_tokens = (
            len(prompt_ids)
            if prompt_ids is not None
            else int(getattr(model_output, "prompt_length", 0) or 0)
            if model_output is not None
            else 0
        )
        completion_tokens = (
            len(completion_ids)
            if completion_ids is not None
            else int(getattr(model_output, "completion_length", 0) or 0)
            if model_output is not None
            else 0
        )
        reasoning = (
            getattr(model_output, "reasoning", "") or ""
            if model_output is not None
            else getattr(step, "thought", "") or ""
        )
        content = (
            getattr(model_output, "content", "") or ""
            if model_output is not None
            else ""
        )
        records.append(
            {
                "step_index": idx,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": getattr(model_output, "finish_reason", None)
                if model_output is not None
                else None,
                "reasoning_tokens": _count_model_tokens(
                    reasoning, model=model, trust_remote_code=trust_remote_code
                ),
                "content_tokens": _count_model_tokens(
                    content, model=model, trust_remote_code=trust_remote_code
                ),
                "text_preview": (getattr(model_output, "text", "") or "")[:500]
                if model_output is not None
                else "",
                "reasoning_preview": reasoning[:500],
                "qwen_thinking_tokens": qwen_thinking_token_report(
                    prompt_ids,
                    completion_ids,
                ),
            }
        )
    return records


def _extract_token_usage_summary(
    episode, model_steps: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    usage = _extract_last_step_field(episode, "token_usage")
    if isinstance(usage, dict):
        return _json_safe(usage)

    steps = model_steps or []
    turns: list[dict[str, int]] = []
    total_prompt = 0
    total_completion = 0
    for idx, step in enumerate(steps):
        prompt_tokens = int(step.get("prompt_tokens") or 0)
        completion_tokens = int(step.get("completion_tokens") or 0)
        total_prompt += prompt_tokens
        total_completion += completion_tokens
        turns.append({
            "turn": int(step.get("step_index", idx) or 0),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cumulative_prompt_tokens": total_prompt,
            "cumulative_completion_tokens": total_completion,
        })

    return {
        "turns": turns,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_output_tokens": total_completion,
        "num_model_calls": len(turns),
    }


def _aggregate_problem_token_usage(samples: list[dict[str, Any]]) -> dict[str, Any]:
    sample_records: list[dict[str, Any]] = []
    total_prompt = 0
    total_completion = 0
    total_calls = 0
    for idx, sample in enumerate(samples):
        usage = sample.get("token_usage") or {}
        prompt_tokens = int(usage.get("total_prompt_tokens") or 0)
        completion_tokens = int(
            usage.get("total_completion_tokens")
            or usage.get("total_output_tokens")
            or 0
        )
        calls = int(usage.get("num_model_calls") or len(usage.get("turns") or []))
        total_prompt += prompt_tokens
        total_completion += completion_tokens
        total_calls += calls
        sample_records.append({
            "sample_index": idx,
            "trajectory_id": sample.get("trajectory_id"),
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_output_tokens": completion_tokens,
            "num_model_calls": calls,
            "turns": usage.get("turns") or [],
        })

    return {
        "samples": sample_records,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_output_tokens": total_completion,
        "num_model_calls": total_calls,
    }


def _build_sample_record(
    *,
    problem_id: str,
    task: AgenticTask,
    episode,
    model: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    gt = task.ground_truth[0] if len(task.ground_truth) == 1 else task.ground_truth
    extracted, is_correct, meta = _extract_final_answer(
        episode,
        ground_truth=gt,
        data_source=task.data_source,
        question=task.question,
    )
    model_steps = _extract_model_step_metadata(
        episode, model=model, trust_remote_code=trust_remote_code
    )
    token_usage = _extract_token_usage_summary(episode, model_steps)
    return {
        "trajectory_id": _trajectory_uuid(problem_id, episode),
        "extracted_answer": extracted,
        "is_correct": is_correct,
        "reward": float(getattr(episode.trajectories[0], "reward", 0.0))
        if episode.trajectories
        else 0.0,
        "termination_reason": episode.termination_reason.value
        if episode.termination_reason
        else None,
        "f1_score": meta.get("f1_score"),
        "exact_match": meta.get("exact_match"),
        "evaluation_metadata": _json_safe(meta),
        "num_steps": len(episode.trajectories[0].steps)
        if episode.trajectories
        else 0,
        "graph_problem_id": _extract_last_step_field(episode, "graph_problem_id"),
        "trajectory": _extract_trajectory_messages(episode),
        "model_io": _extract_model_io(episode),
        "model_steps": model_steps,
        "token_usage": token_usage,
        "output_tokens_by_turn": [
            turn.get("completion_tokens", 0)
            for turn in token_usage.get("turns", [])
        ],
        "total_output_tokens": token_usage.get("total_output_tokens", 0),
    }


def _build_stream_entry(
    *,
    cfg: AgentRolloutConfig,
    run_id: str,
    started_at: float,
    bench: str,
    task: AgenticTask,
    rollout_idx: int,
    episode,
) -> dict[str, Any]:
    problem_id = _problem_uuid(task.data_source, task.task_id, task.question)
    sample = _build_sample_record(
        problem_id=problem_id,
        task=task,
        episode=episode,
        model=cfg.model,
        trust_remote_code=cfg.vllm_trust_remote_code,
    )
    completed_at = time.time()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "model": cfg.model,
        "benchmark": bench,
        "data_source": task.data_source,
        "problem_id": problem_id,
        "task_id": task.task_id,
        "sample_index": rollout_idx,
        "completed_at": completed_at,
        "completed_iso": _iso(completed_at),
        "elapsed_seconds": completed_at - started_at,
        "question": task.question,
        "ground_truth": task.ground_truth,
        "sample": sample,
    }


class _TrajectoryJsonlWriter:
    """Append completed trajectory samples so the UI can read them mid-run."""

    def __init__(self, path: Path, append: bool = False):
        self.path = path
        self.count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._fh = self.path.open(mode, encoding="utf-8")

    def append(self, entry: dict[str, Any]) -> None:
        self._fh.write(json.dumps(_json_safe(entry), ensure_ascii=False) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        self._fh.close()


async def _execute_tasks_streaming(
    engine: AgentWorkflowEngine,
    tasks: list[dict],
    task_ids: list[str],
    *,
    on_complete: Callable[[str, int, Any], None] | None = None,
    **kwargs,
) -> list[Any]:
    """Run tasks and invoke ``on_complete`` as each episode finishes.

    This mirrors rLLM's ``AgentWorkflowEngine.execute_tasks`` behavior but
    exposes the already-available completion point before the full batch ends.
    """
    if engine.workflow_queue is None:
        await engine.initialize_pool()

    task_states = defaultdict(
        lambda: {
            "idx": None,
            "task": None,
            "episodes": [],
            "completed": 0,
            "total_rollouts": 0,
        }
    )

    futures = []
    idx_counter = 0
    for task, task_id in zip(tasks, task_ids, strict=True):
        state = task_states[task_id]
        if state["idx"] is None:
            state["idx"] = idx_counter
            state["task"] = task
            idx_counter += 1
        rollout_idx = state["total_rollouts"]
        futures.append(engine.process_task_with_retry(task, task_id, rollout_idx, **kwargs))
        state["total_rollouts"] += 1

    with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
        for future in asyncio.as_completed(futures):
            task_id, rollout_idx, episode = await future
            state = task_states[task_id]
            state["episodes"].append(episode)
            state["completed"] += 1
            if on_complete is not None:
                on_complete(task_id, rollout_idx, episode)
            pbar.update(1)

    results = []
    sorted_tasks = sorted(
        task_states.keys(), key=lambda task_id: task_states[task_id]["idx"]
    )
    for task_id in sorted_tasks:
        results.extend(task_states[task_id]["episodes"])

    if engine.episode_logger is not None:
        try:
            engine.episode_logger.log_episodes_batch(
                results, engine.current_step, engine.current_mode, engine.current_epoch
            )
        except Exception as exc:
            console.log(f"[yellow]Failed to log episodes: {exc}")

    return results


def _model_output_dir(cfg: AgentRolloutConfig) -> Path:
    return Path(cfg.output_dir) / model_output_dir_name(
        cfg.model,
        enable_thinking=cfg.enable_thinking,
        save_alias=cfg.save_alias,
    )


def _benchmark_output_dir(cfg: AgentRolloutConfig, bench: str) -> Path:
    return _model_output_dir(cfg) / bench


def _artifact_stem_path(cfg: AgentRolloutConfig, bench: str, stem: str, suffix: str) -> Path:
    out_dir = _benchmark_output_dir(cfg, bench)
    # Keep a stable path so non-overwrite runs can inspect trajectories.jsonl,
    # skip completed task IDs, and append only the missing trajectories.  The
    # previous timestamped path made the resume logic below inspect a new empty
    # file, so every interrupted run silently started over.
    return out_dir / f"{stem}{suffix}"


def _planned_result_path(cfg: AgentRolloutConfig, bench: str) -> Path:
    return _artifact_stem_path(cfg, bench, "results", ".json")


def _planned_stream_path(cfg: AgentRolloutConfig, bench: str) -> Path:
    return _artifact_stem_path(cfg, bench, "trajectories", ".jsonl")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=4)
    tmp.replace(path)


def _redact_value(key: str, value: Any) -> Any:
    lowered = key.lower()
    if any(token in lowered for token in ("password", "passwd", "secret", "token", "api_key", "apikey")):
        return "[redacted]" if value else value
    return value


def _redact_argv(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for item in argv:
        key = item.lstrip("-").replace("-", "_")
        if redact_next:
            redacted.append("[redacted]")
            redact_next = False
            continue
        if any(token in key.lower() for token in ("password", "passwd", "secret", "token", "api_key", "apikey")):
            if "=" in item:
                name, _value = item.split("=", 1)
                redacted.append(f"{name}=[redacted]")
            else:
                redacted.append(item)
                redact_next = True
            continue
        redacted.append(item)
    return redacted


def _selected_environment() -> dict[str, Any]:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CONDA_DEFAULT_ENV",
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "RAY_ADDRESS",
        "OPENAI_BASE_URL",
        "SGLANG_APPLY_CONFIG_BACKUP",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    return {key: _redact_value(key, os.environ.get(key, "")) for key in keys if key in os.environ}


def _memory_total_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    try:
        return int(pages) * int(page_size)
    except (TypeError, ValueError):
        return None


def _gpu_info() -> dict[str, Any]:
    query = "index,name,uuid,memory.total,driver_version"
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "nvidia-smi timed out"}
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
        return {"available": False, "error": stderr or f"nvidia-smi rc={proc.returncode}"}
    gpus: list[dict[str, Any]] = []
    for line in (proc.stdout or b"").decode("utf-8", "replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        memory_mb: int | None
        try:
            memory_mb = int(parts[3])
        except ValueError:
            memory_mb = None
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "uuid": parts[2],
                "memory_total_mb": memory_mb,
                "driver_version": parts[4],
            }
        )
    return {"available": True, "count": len(gpus), "gpus": gpus}


def _machine_info() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "platform": {
            "system": uname.system,
            "node": uname.node,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
            "platform": platform.platform(),
        },
        "cpu": {
            "count": os.cpu_count(),
        },
        "memory": {
            "total_bytes": _memory_total_bytes(),
        },
        "gpu": _gpu_info(),
    }


def _run_config_payload(
    cfg: AgentRolloutConfig,
    *,
    run_id: str,
    started_at: float,
    status: str,
    phase: str = "",
    completed_at: float | None = None,
    live_paths: list[str] | None = None,
    result_paths: list[str] | None = None,
    summaries: list[dict[str, Any]] | None = None,
    error: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "phase": phase,
        "created_at": started_at,
        "created_iso": _iso(started_at),
        "updated_at": time.time(),
        "updated_iso": _iso(time.time()),
        "completed_at": completed_at,
        "completed_iso": _iso(completed_at),
        "elapsed_seconds": (completed_at or time.time()) - started_at,
        "effective_config": _effective_run_config(cfg),
        "config": asdict(cfg),
        "sampling_params": _build_sampling_params(cfg),
        "process": {
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "argv": _redact_argv(sys.argv),
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "implementation": platform.python_implementation(),
            },
            "environment": _selected_environment(),
        },
        "machine": _machine_info(),
        "artifacts": {
            "output_dir": str(_model_output_dir(cfg)),
            "latest_config_path": str(_model_output_dir(cfg) / "run_config.json"),
            "history_config_path": str(_model_output_dir(cfg) / "run_configs" / f"{run_id}.json"),
            "live_paths": live_paths or [],
            "result_paths": result_paths or [],
        },
        "summaries": summaries or [],
        "error": error,
    }


def _write_run_config(
    cfg: AgentRolloutConfig,
    *,
    run_id: str,
    started_at: float,
    status: str,
    phase: str = "",
    completed_at: float | None = None,
    live_paths: list[str] | None = None,
    result_paths: list[str] | None = None,
    summaries: list[dict[str, Any]] | None = None,
    error: str = "",
) -> None:
    payload = _run_config_payload(
        cfg,
        run_id=run_id,
        started_at=started_at,
        status=status,
        phase=phase,
        completed_at=completed_at,
        live_paths=live_paths,
        result_paths=result_paths,
        summaries=summaries,
        error=error,
    )
    out_dir = _model_output_dir(cfg)
    latest_path = out_dir / "run_config.json"
    history_path = out_dir / "run_configs" / f"{run_id}.json"
    try:
        _write_json_atomic(latest_path, payload)
        _write_json_atomic(history_path, payload)
    except Exception as exc:
        console.log(f"[yellow]Could not write run_config.json: {exc}")


def _write_run_state(
    cfg: AgentRolloutConfig,
    *,
    run_id: str,
    started_at: float,
    status: str,
    phase: str,
    completed_benchmarks: int = 0,
    total_benchmarks: int | None = None,
    completed_samples: int = 0,
    total_samples: int | None = None,
    current_benchmark: str | None = None,
    current_expected_samples: int | None = None,
    live_paths: list[str] | None = None,
    result_paths: list[str] | None = None,
    summaries: list[dict[str, Any]] | None = None,
    error: str = "",
) -> None:
    """Persist a lightweight run heartbeat consumed by ``bcg agent ui``.

    Completed trajectories are appended to ``trajectories*.jsonl`` during
    generation, while final aggregates are saved in ``results.json`` at
    benchmark boundaries. This file lets the web UI report run progress and
    discover the live stream paths.
    """
    payload = {
        "run_id": run_id,
        "pid": os.getpid(),
        "status": status,
        "phase": phase,
        "started_at": started_at,
        "updated_at": time.time(),
        "elapsed_seconds": time.time() - started_at,
        "completed_benchmarks": completed_benchmarks,
        "total_benchmarks": total_benchmarks,
        "completed_samples": completed_samples,
        "total_samples": total_samples,
        "current_benchmark": current_benchmark,
        "current_expected_samples": current_expected_samples,
        "live_paths": live_paths or [],
        "result_paths": result_paths or [],
        "summaries": summaries or [],
        "error": error,
        "config": asdict(cfg),
    }
    try:
        _write_json_atomic(_model_output_dir(cfg) / "run_state.json", payload)
    except Exception as exc:
        console.log(f"[yellow]Could not write run_state.json: {exc}")


class _RunStateHeartbeat:
    def __init__(self, cfg: AgentRolloutConfig, **state: Any):
        self._cfg = cfg
        self._state = dict(state)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name="BeliefTracerRunStateHeartbeat",
            daemon=True,
        )

    def start(self) -> "_RunStateHeartbeat":
        self.flush()
        self._thread.start()
        return self

    def update(self, **state: Any) -> None:
        with self._lock:
            self._state.update(state)
        self.flush()

    def flush(self) -> None:
        with self._lock:
            state = dict(self._state)
        _write_run_state(self._cfg, **state)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.flush()

    def _loop(self) -> None:
        while not self._stop.wait(RUN_STATE_HEARTBEAT_SECONDS):
            self.flush()


def _build_rollout_engine(cfg: AgentRolloutConfig):
    """Instantiate the rollout engine once for the whole rollout run.

    Loading a multi-billion parameter model per benchmark is wasteful, so
    the engine is constructed once and reused for every benchmark-specific
    ``AgentWorkflowEngine`` below.
    """
    backend = (cfg.backend or "vllm").lower()
    if backend == "vllm":
        try:
            from bcg.agent.vllm_engine import VLLMEngine
        except ModuleNotFoundError as exc:
            if exc.name != "vllm":
                raise
            console.log("[yellow]vLLM is not installed; falling back to SGLang backend.")
            backend = "sglang"
        else:
            console.log(
                f"[cyan]Initializing in-process vLLM engine for {cfg.model} "
                f"(tp={cfg.tensor_parallel_size}, mem={cfg.gpu_memory_utilization})"
            )
            return VLLMEngine(
                model=cfg.model,
                max_prompt_length=cfg.max_prompt_length,
                max_response_length=cfg.max_response_length,
                max_model_length=cfg.vllm_max_model_len,
                sampling_params=_build_sampling_params(cfg),
                tensor_parallel_size=cfg.tensor_parallel_size,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                dtype=cfg.vllm_dtype,
                enforce_eager=cfg.vllm_enforce_eager,
                trust_remote_code=cfg.vllm_trust_remote_code,
                disable_thinking=not cfg.enable_thinking,
            )

    if backend == "sglang":
        from bcg.agent.sglang_engine import SGLangEngine

        console.log(
            f"[cyan]Initializing in-process SGLang engine for {cfg.model} "
            f"(tp={cfg.tensor_parallel_size}, mem={cfg.gpu_memory_utilization})"
        )
        return SGLangEngine(
            model=cfg.model,
            max_prompt_length=cfg.max_prompt_length,
            max_response_length=cfg.max_response_length,
            max_model_length=cfg.vllm_max_model_len,
            sampling_params=_build_sampling_params(cfg),
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            trust_remote_code=cfg.vllm_trust_remote_code,
            disable_thinking=not cfg.enable_thinking,
        )
    if backend == "sglang_dp":
        from bcg.agent.sglang_dp_engine import SGLangDataParallelEngine

        console.log(
            f"[cyan]Initializing data-parallel SGLang pool for {cfg.model} "
            f"(dp={cfg.data_parallel_size}, tp_per_worker={cfg.tensor_parallel_size}, "
            f"mem={cfg.gpu_memory_utilization})"
        )
        return SGLangDataParallelEngine(
            model=cfg.model,
            data_parallel_size=cfg.data_parallel_size,
            data_parallel_devices=cfg.data_parallel_devices,
            max_prompt_length=cfg.max_prompt_length,
            max_response_length=cfg.max_response_length,
            max_model_length=cfg.vllm_max_model_len,
            sampling_params=_build_sampling_params(cfg),
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            trust_remote_code=cfg.vllm_trust_remote_code,
            disable_thinking=not cfg.enable_thinking,
        )
    if backend == "ray_vllm":
        from bcg.agent.ray_vllm_engine import RayVLLMEngine

        console.log(
            f"[cyan]Initializing Ray-distributed vLLM pool for {cfg.model} "
            f"(replicas={cfg.ray_num_replicas}, tp={cfg.tensor_parallel_size}, "
            f"mem={cfg.gpu_memory_utilization})"
        )
        return RayVLLMEngine(
            model=cfg.model,
            num_replicas=cfg.ray_num_replicas,
            tensor_parallel_size=cfg.tensor_parallel_size,
            ray_address=cfg.ray_address or None,
            max_prompt_length=cfg.max_prompt_length,
            max_response_length=cfg.max_response_length,
            max_model_length=cfg.vllm_max_model_len,
            sampling_params=_build_sampling_params(cfg),
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            dtype=cfg.vllm_dtype,
            enforce_eager=cfg.vllm_enforce_eager,
            trust_remote_code=cfg.vllm_trust_remote_code,
            disable_thinking=not cfg.enable_thinking,
        )
    if backend == "openai":
        from rllm.engine.rollout.openai_engine import OpenAIEngine

        sampling_params = _build_openai_sampling_params(cfg)

        console.log(f"[cyan]Using OpenAI-compatible endpoint {cfg.resolved_base_url()}")
        return OpenAIEngine(
            model=cfg.model,
            tokenizer=None,
            base_url=cfg.resolved_base_url(),
            api_key=cfg.resolved_api_key(),
            max_prompt_length=cfg.max_prompt_length,
            max_response_length=cfg.max_response_length,
            sampling_params=sampling_params,
        )
    if backend == "api":
        from bcg.agent.api_engine import APIEngine

        # Clean sampling params: only keep standard OpenAI fields
        sampling_params: dict[str, Any] = {}
        if cfg.temperature is not None:
            sampling_params["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            sampling_params["top_p"] = cfg.top_p
        if cfg.presence_penalty is not None:
            sampling_params["presence_penalty"] = cfg.presence_penalty

        console.log(
            f"[cyan]Using API engine: {cfg.resolved_base_url()} (model: {cfg.model})"
        )
        return APIEngine(
            model=cfg.model,
            base_url=cfg.resolved_base_url(),
            api_key=cfg.resolved_api_key(),
            max_prompt_length=cfg.max_prompt_length,
            max_response_length=cfg.max_response_length,
            sampling_params=sampling_params,
        )
    raise ValueError(
        f"Unknown backend: {cfg.backend!r} "
        "(expected 'vllm', 'sglang', 'sglang_dp', 'ray_vllm', 'openai', or 'api')"
    )


def _resolve_tools(cfg: AgentRolloutConfig) -> dict[str, Any]:
    """Translate ``cfg.tools`` into kwargs for BeliefTracerAgent / BeliefTracerEnvironment.

    Tools registered in ``rllm.tools.tool_registry`` are passed through as the
    legacy ``tools=[...]`` list. ``local_search`` is not in the default registry,
    so when it's requested we switch to a ``tool_map={name: cls}`` construction
    that points at ``examples.search.local_retrieval_tool.LocalRetrievalTool``.

    Mixing registered tools with ``local_search`` in the same run isn't supported
    by rllm's ``MultiTool`` (which rejects both arguments together), so this
    raises if the user asks for that combination.
    """
    names = list(cfg.tools)
    wants_local = "local_search" in names
    wants_averitec = "averitec_search" in names
    wants_bcp = "bcp_search" in names
    wants_serper_search = "serper_search" in names
    wants_serper_scrape = "serper_scrape" in names
    wants_serper = wants_serper_search or wants_serper_scrape
    search_tool_name: str | None = None
    search_tool_class: Any | None = None
    serper_scrape_tool_class: Any | None = None

    # read_file is a sandboxed companion tool; it shares the averitec tool_map
    # (no tools/tool_map conflict) and may be requested explicitly or implied by
    # archive/file-read flags.
    wants_read_file = (
        "read_file" in names or cfg.enable_file_read or cfg.enable_archive
    )

    if wants_averitec or wants_bcp or wants_serper:
        selected_searches = [
            name
            for name, selected in (
                ("averitec_search", wants_averitec),
                ("bcp_search", wants_bcp),
                ("serper_search", wants_serper_search),
            )
            if selected
        ]
        if len(selected_searches) > 1:
            raise ValueError(
                "Only one search backend can be selected in a run because they share "
                f"the rllm tool_map wiring. Got {selected_searches}."
            )
        if wants_bcp:
            extra = [n for n in names if n not in ("bcp_search", "read_file")]
            if extra:
                raise ValueError(
                    "'bcp_search' cannot be combined with other tools in the same run "
                    f"(except 'read_file'). Got tools={names}."
                )
            from bcg.agent.tools.bcp_search import BCPSearchTool

            class ConfiguredBCPSearchTool(BCPSearchTool):
                def __init__(self, **kwargs):
                    max_results = kwargs.pop("max_results", cfg.retrieval_max_results)
                    max_output_chars = kwargs.pop("max_output_chars", cfg.bcp_max_output_chars)
                    super().__init__(
                        index_dir=cfg.bcp_index_dir or None,
                        embedding_model=cfg.hero_embedding_model,
                        embedding_url=cfg.hero_embedding_url,
                        max_results=max_results,
                        max_output_chars=max_output_chars,
                        **kwargs,
                    )

            search_tool_name = "bcp_search"
            search_tool_class = ConfiguredBCPSearchTool
            console.log("[cyan]Using BCPSearchTool over local BrowseComp-Plus corpus")
            console.log(f"[cyan]  - Index dir: {cfg.bcp_index_dir or os.environ.get('BCP_INDEX_DIR') or 'default'}")
            console.log(f"[cyan]  - Embedding model: {cfg.hero_embedding_model}")
            console.log(f"[cyan]  - Embedding URL: {cfg.hero_embedding_url}")
        elif wants_serper:
            extra = [
                n
                for n in names
                if n not in ("serper_search", "serper_scrape", "read_file")
            ]
            if extra:
                raise ValueError(
                    "Serper tools cannot be combined with other search tools in the "
                    f"same run (except 'read_file'). Got tools={names}."
                )
            if wants_serper_search:
                from bcg.agent.tools.serper_search import SerperSearchTool

                class ConfiguredSerperSearchTool(SerperSearchTool):
                    def __init__(self, **kwargs):
                        max_results = kwargs.pop("max_results", cfg.retrieval_max_results)
                        max_output_chars = kwargs.pop(
                            "max_output_chars", cfg.serper_max_output_chars
                        )
                        super().__init__(
                            endpoint=cfg.serper_endpoint,
                            country=cfg.serper_country,
                            language=cfg.serper_language,
                            max_results=max_results,
                            max_output_chars=max_output_chars,
                            timeout=cfg.serper_timeout,
                            **kwargs,
                        )

                search_tool_name = "serper_search"
                search_tool_class = ConfiguredSerperSearchTool
                console.log("[cyan]Using SerperSearchTool for live Google web search")
                console.log(f"[cyan]  - Search endpoint: {cfg.serper_endpoint}")
                console.log(
                    f"[cyan]  - Locale: gl={cfg.serper_country or '(default)'}, "
                    f"hl={cfg.serper_language or '(default)'}"
                )
            if wants_serper_scrape:
                from bcg.agent.tools.serper_scrape import SerperScrapeTool

                class ConfiguredSerperScrapeTool(SerperScrapeTool):
                    def __init__(self, **kwargs):
                        max_output_chars = kwargs.pop(
                            "max_output_chars", cfg.serper_scrape_max_output_chars
                        )
                        super().__init__(
                            endpoint=cfg.serper_scrape_endpoint,
                            max_output_chars=max_output_chars,
                            timeout=cfg.serper_scrape_timeout,
                            **kwargs,
                        )

                serper_scrape_tool_class = ConfiguredSerperScrapeTool
                console.log("[cyan]Using SerperScrapeTool for full web-page content")
                console.log(f"[cyan]  - Scrape endpoint: {cfg.serper_scrape_endpoint}")
                console.log(
                    "[cyan]  - Max page output: "
                    f"{cfg.serper_scrape_max_output_chars} chars"
                )
            console.log("[cyan]  - API key: project-root .env / SERPER_API_KEY")
        else:
            search_tool_name = "averitec_search"
            search_tool_class = None

    if wants_averitec:
        extra = [n for n in names if n not in ("averitec_search", "read_file")]
        if extra:
            raise ValueError(
                "'averitec_search' cannot be combined with other tools in the same run "
                "(except 'read_file') because rllm's MultiTool accepts either 'tools' or "
                f"'tool_map', not both. Got tools={names}."
            )

        # Select retrieval method based on config
        if cfg.retrieval_method in ("hero", "hero4"):
            from bcg.agent.tools.hero_search_tool import HerOSearchTool

            _four_stage = cfg.retrieval_method == "hero4"

            # Create HerO tool with config parameters
            class ConfiguredHerOSearchTool(HerOSearchTool):
                def __init__(self, **kwargs):
                    super().__init__(
                        bm25_top_k=cfg.hero_bm25_top_k,
                        embedding_model=cfg.hero_embedding_model,
                        embedding_device=cfg.hero_embedding_device,
                        batch_size=cfg.hero_batch_size,
                        embedding_url=cfg.hero_embedding_url,
                        max_results=cfg.retrieval_max_results,
                        four_stage=_four_stage,
                        stage1_bm25_k=cfg.stage1_bm25_k,
                        stage2_embed_k=cfg.stage2_embed_k,
                        stage3_rerank_k=cfg.stage3_rerank_k,
                        rerank_url=cfg.rerank_url,
                        rerank_model=cfg.rerank_model,
                        enable_judge=cfg.enable_judge,
                        judge_model=cfg.judge_model or cfg.model,
                        judge_base_url=cfg.judge_base_url,
                        judge_api_key=cfg.judge_api_key,
                        judge_max_workers=cfg.judge_max_workers,
                        judge_max_items=cfg.judge_max_items,
                        **kwargs
                    )

            search_tool_class = ConfiguredHerOSearchTool
            emb_backend = f"remote API ({cfg.hero_embedding_url})" if cfg.hero_embedding_url else "local SentenceTransformer"
            if _four_stage:
                console.log("[cyan]Using HerOSearchTool 4-stage (BM25 -> Embedding -> Reranker -> LLM judge)")
                console.log(f"[cyan]  - Stages: BM25 min({cfg.stage1_bm25_k},N) -> embed {cfg.stage2_embed_k} -> rerank {cfg.stage3_rerank_k} -> judge={cfg.enable_judge}")
                console.log(f"[cyan]  - Reranker: {cfg.rerank_model} @ {cfg.rerank_url}")
                console.log(f"[cyan]  - Judge model: {cfg.judge_model or cfg.model}")
            else:
                console.log("[cyan]Using HerOSearchTool (BM25 + Embedding Reranking) over the per-claim knowledge store")
                console.log(f"[cyan]  - BM25 top-k: {cfg.hero_bm25_top_k}")
            console.log(f"[cyan]  - Embedding model: {cfg.hero_embedding_model}")
            console.log(f"[cyan]  - Embedding backend: {emb_backend}")
            console.log(f"[cyan]  - Batch size: {cfg.hero_batch_size}")
        else:
            from bcg.agent.tools.averitec_search import AVeriTeCSearchTool

            search_tool_class = AVeriTeCSearchTool
            console.log("[cyan]Using AVeriTeCSearchTool (BM25 only) over the per-claim knowledge store")
        search_tool_name = "averitec_search"

    if wants_averitec or wants_bcp or wants_serper:
        # Build the tool_map first; usage text is then assembled dynamically by
        # asking each tool for its own usage_prompt() — no hardcoded blob, and
        # only the tools actually present contribute instructions.
        tool_map: dict[str, Any] = {}
        if search_tool_name and search_tool_class:
            tool_map[search_tool_name] = search_tool_class
        if wants_serper_scrape and serper_scrape_tool_class:
            tool_map["serper_scrape"] = serper_scrape_tool_class
        if wants_read_file:
            file_root = cfg.file_tool_root or os.environ.get(
                "BELIEF_TRACER_FILE_ROOT", "ai_workspace"
            )

            from bcg.agent.tools.file_read_tool import FileReadTool

            class _ConfiguredFileReadTool(FileReadTool):
                def __init__(self, **kwargs):
                    super().__init__(root=file_root, **kwargs)

            tool_map["read_file"] = _ConfiguredFileReadTool
            console.log(
                f"[cyan]Exposing read_file tool over sandbox root {file_root!r}"
            )

        # Assemble usage text from each tool instance's usage_prompt(). We build
        # two variants: a detailed block (for the user message in layered mode /
        # the system prompt in legacy mode) and a brief block (the system-prompt
        # tool listing in layered mode — JSON shape + arg names, no examples).
        usage_blocks: list[str] = []
        usage_blocks_brief: list[str] = []
        for tool_cls in tool_map.values():
            try:
                # Build the usage instance with the same retrieval_max_results the
                # runtime tool uses, so the "default top_k" printed in the prompt
                # matches what the model actually gets (not the bare fallback).
                try:
                    inst = tool_cls(max_results=cfg.retrieval_max_results)
                except TypeError:
                    inst = tool_cls()
                fn = getattr(inst, "usage_prompt", None)
                if callable(fn):
                    try:
                        usage_blocks.append(fn(hyde=cfg.hyde, detail=True))
                        usage_blocks_brief.append(fn(hyde=cfg.hyde, detail=False))
                    except TypeError:
                        # Tool without a `detail` parameter (e.g. local_search):
                        # reuse the single variant for both slots.
                        block = fn(hyde=cfg.hyde)
                        usage_blocks.append(block)
                        usage_blocks_brief.append(block)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to build usage prompt for %s: %s", tool_cls, exc)
        if wants_averitec:
            usage_blocks.append(
                "- finish(answer: string): finish the rollout. Put the final AVeriTeC "
                "label in \\boxed{} inside the answer.\n"
                "  Example:\n"
                "  <tool_call>\n"
                '  {"name": "finish", "arguments": {"answer": "short justification. '
                'Final label: \\boxed{Supported}"}}\n'
                "  </tool_call>"
            )
            usage_blocks_brief.append(
                "- finish: finish the rollout. Call as:\n"
                "  <tool_call>\n"
                '  {"name": "finish", "arguments": {"answer": "...\\boxed{Label}"}}\n'
                "  </tool_call>"
            )
        else:
            usage_blocks.append(
                "- finish(answer: string): finish the rollout. Put the succinct exact "
                "answer in \\boxed{} inside the answer.\n"
                "  Example:\n"
                "  <tool_call>\n"
                '  {"name": "finish", "arguments": {"answer": "Evidence-based '
                'conclusion. Final answer: \\boxed{exact answer}"}}\n'
                "  </tool_call>"
            )
            usage_blocks_brief.append(
                "- finish: finish the rollout with the exact answer in \\boxed{}. Call as:\n"
                "  <tool_call>\n"
                '  {"name": "finish", "arguments": {"answer": "...\\boxed{answer}"}}\n'
                "  </tool_call>"
            )

        # When parallel tool execution is enabled (--max-tool-workers > 1), tell
        # the model it may issue multiple <tool_call> blocks in one response for
        # independent queries. Otherwise keep the historical "one per turn"
        # instruction so single-worker runs behave exactly as before.
        if cfg.max_tool_workers > 1:
            tool_prompt_header = (
                "Available tools for this benchmark (Qwen tool-call JSON, no text "
                "after the tool call). You may issue multiple <tool_call> blocks in "
                "a single response when the queries are independent — they will run "
                "in parallel:\n\n"
            )
            tool_prompt_brief_header = (
                "Available tools (Qwen tool-call JSON, no text after the tool call). "
                "You may issue multiple <tool_call> blocks in one response for "
                "independent queries — they run in parallel:\n\n"
            )
        else:
            tool_prompt_header = (
                "Available tools for this benchmark (call one per turn, Qwen tool-call "
                "JSON, no text after the tool call):\n\n"
            )
            tool_prompt_brief_header = (
                "Available tools (call one per turn, Qwen tool-call JSON, no text after "
                "the tool call):\n\n"
            )

        tool_prompt = tool_prompt_header + "\n".join(usage_blocks)
        tool_prompt_brief = tool_prompt_brief_header + "\n".join(usage_blocks_brief)

        return {
            "tool_map": tool_map,
            "tool_prompt": tool_prompt,
            "tool_prompt_brief": tool_prompt_brief,
        }

    if not wants_local:
        return {"tools": names}

    if len(names) > 1:
        raise ValueError(
            "'local_search' cannot be combined with other tools in the same run "
            "because rllm's MultiTool accepts either 'tools' or 'tool_map', not both. "
            f"Got tools={names}."
        )

    from examples.search.local_retrieval_tool import LocalRetrievalTool

    server_url = (
        cfg.retrieval_server_url
        or os.environ.get("RETRIEVAL_SERVER_URL")
        or "http://127.0.0.1:8000"
    )
    os.environ.setdefault("RETRIEVAL_SERVER_URL", server_url)
    max_results = cfg.retrieval_max_results
    timeout = cfg.retrieval_timeout

    class _LocalSearch(LocalRetrievalTool):
        def __init__(self, name: str = "local_search", description: str | None = None):
            super().__init__(
                name=name,
                description=description or LocalRetrievalTool.DESCRIPTION,
                server_url=server_url,
                timeout=timeout,
                max_results=max_results,
            )

    console.log(
        f"[cyan]Using LocalRetrievalTool against {server_url} "
        f"(top_k={max_results}, summarize={os.environ.get('RLLM_RETRIEVAL_SUMMARIZE', '0') == '1'})"
    )
    return {"tool_map": {"local_search": _LocalSearch}}


def _browsecomp_judge_config(cfg: AgentRolloutConfig) -> BrowseCompJudgeConfig:
    return BrowseCompJudgeConfig(
        model=cfg.resolved_browsecomp_grader_model(),
        base_url=cfg.resolved_browsecomp_grader_base_url(),
        api_key=cfg.resolved_browsecomp_grader_api_key(),
        timeout=cfg.browsecomp_grader_timeout,
        max_tokens=cfg.browsecomp_grader_max_tokens,
        max_retries=cfg.browsecomp_grader_max_retries,
    )


def _build_workflow_engine(
    cfg: AgentRolloutConfig,
    data_source: str,
    rollout_engine,
    reward_fn=None,
    belief_client: BeliefGraphClient | None = None,
) -> AgentWorkflowEngine:
    """Wrap ``rollout_engine`` in an ``AgentWorkflowEngine`` for one benchmark.

    Each benchmark gets its own workflow pool because ``reward_fn`` bakes in
    the benchmark-specific ``data_source``, but they all share the single
    loaded rollout engine passed in.

    In mixed-rollouts mode callers pass a dispatching ``reward_fn`` instead so a
    single workflow pool can serve rollouts from multiple data sources.
    """
    if reward_fn is None:
        reward_fn = build_reward_fn(
            data_source,
            browsecomp_judge_config=_browsecomp_judge_config(cfg),
        )
    tool_kwargs = _resolve_tools(cfg)

    effective_belief_client = (
        belief_client
        if uses_belief_graph_service(cfg.context_memory_mode, cfg.belief_graph_mode)
        else None
    )

    workflow_args = {
        "agent_cls": BeliefTracerAgent,
        "env_cls": BeliefTracerEnvironment,
        "agent_args": {
            "system_prompt": cfg.system_prompt,
            "parser_name": cfg.parser_name,
            "model": cfg.model,
            "enable_thinking": cfg.enable_thinking,
            "belief_graph_mode": cfg.belief_graph_mode,
            "graph_format": cfg.graph_format,
            "deepseek_v4_payload_format": cfg.deepseek_v4_payload_format,
            "graph_include_relations": cfg.graph_include_relations,
            "belief_graph_placement": cfg.belief_graph_placement,
            "archive_enabled": cfg.enable_archive,
            "layered_context": cfg.layered_context,
            "recent_turns": cfg.recent_turns,
            "user_rules_prompt": cfg.user_rules_prompt,
            "context_memory_mode": cfg.context_memory_mode,
            **tool_kwargs,
        },
        "env_args": {
            **tool_kwargs,
            "reward_fn": reward_fn,
            "max_steps": cfg.max_steps,
            "max_tool_workers": cfg.max_tool_workers,
        },
        "max_steps": cfg.max_steps,
        "belief_client": effective_belief_client,
        "graph_output_dir": str(_benchmark_output_dir(cfg, data_source) / "belief_graphs"),
        "archive_enabled": cfg.enable_archive,
        "file_tool_root": cfg.file_tool_root,
        "belief_graph_interval": cfg.belief_graph_interval,
        "recent_turns": cfg.recent_turns,
        "context_memory_config": {
            "mode": cfg.context_memory_mode,
            "recent_observations": cfg.context_memory_recent_observations,
            "tail_turns": cfg.context_memory_tail_turns,
            "max_chars": cfg.context_memory_max_chars,
            "tool_summary_chars": cfg.context_memory_tool_summary_chars,
            "interval": cfg.context_memory_interval,
            "summarizer": cfg.context_memory_summarizer,
            "summarizer_model": cfg.model,
            "summarizer_base_url": cfg.resolved_base_url(),
            "summarizer_api_key": cfg.resolved_api_key(),
            "summarizer_max_tokens": cfg.context_memory_summarizer_max_tokens,
            "summarizer_timeout": cfg.context_memory_summarizer_timeout,
            "summarizer_failure_limit": cfg.context_memory_summarizer_failure_limit,
            "log_preview_chars": cfg.context_memory_log_preview_chars,
        },
        "tonggraph_sync_config": {
            "enabled": cfg.tonggraph_sync,
            "base_url": cfg.tonggraph_base_url,
            "token": cfg.tonggraph_token,
            "graph": cfg.tonggraph_graph,
            "logical_graph_id": cfg.tonggraph_logical_graph_id,
            "timeout": cfg.tonggraph_timeout,
            "text_index": cfg.tonggraph_text_index or None,
            "embedding_url": cfg.tonggraph_embedding_url,
            "embedding_model": cfg.tonggraph_embedding_model,
            "embedding_index": cfg.tonggraph_embedding_index or None,
            "embedding_batch_size": cfg.tonggraph_embedding_batch_size,
        },
    }

    return AgentWorkflowEngine(
        workflow_cls=BeliefTracerWorkflow,
        workflow_args=workflow_args,
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=cfg.n_parallel_tasks,
        retry_limit=cfg.retry_limit,
        raise_on_error=False,
    )


async def _run_one_benchmark(
    cfg: AgentRolloutConfig,
    bench: str,
    tasks: list[AgenticTask],
    rollout_engine,
    *,
    run_id: str,
    started_at: float,
    stream_path: Path | None = None,
    on_stream_entry: Callable[[dict[str, Any]], None] | None = None,
    belief_client: BeliefGraphClient | None = None,
) -> dict[str, Any]:
    console.rule(
        f"[bold cyan]{bench.strip()} (n={len(tasks)}, samples={cfg.num_samples})"
    )
    data_source = tasks[0].data_source if tasks else bench

    engine = _build_workflow_engine(
        cfg, data_source=data_source, rollout_engine=rollout_engine,
        belief_client=belief_client,
    )

    # Skip tasks that already have completed trajectories on disk.
    all_tasks = list(tasks)
    completed_task_ids: set[str] = set()
    prev_records: dict[str, dict] = {}
    if not cfg.overwrite and stream_path and stream_path.exists():
        try:
            with stream_path.open("r", encoding="utf-8") as _sf:
                for line in _sf:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    tid = entry.get("task_id")
                    if tid is not None:
                        completed_task_ids.add(str(tid))
        except Exception as exc:
            console.log(f"[yellow]Failed to read existing trajectories: {exc}[/yellow]")
    # Load previous results for merging
    if completed_task_ids:
        result_path = _planned_result_path(cfg, bench)
        if result_path.exists():
            try:
                with result_path.open("r", encoding="utf-8") as _rf:
                    prev_payload = json.load(_rf)
                for rec in prev_payload.get("records", []):
                    tid = str(rec.get("task_id", ""))
                    if tid in completed_task_ids:
                        prev_records[tid] = rec
            except Exception as exc:
                console.log(f"[yellow]Failed to load previous results: {exc}[/yellow]")
        original_count = len(tasks)
        tasks = [t for t in tasks if t.task_id not in completed_task_ids]
        skipped = original_count - len(tasks)
        console.log(f"[green]Skipping {skipped} already-completed tasks ({len(tasks)} remaining)")

    # Replicate each task num_samples times; carry identical task_ids to group.
    expanded_tasks: list[dict] = []
    expanded_ids: list[str] = []
    task_by_id: dict[str, AgenticTask] = {}
    for t in tasks:
        env_task = t.to_env_task()
        task_by_id[t.task_id] = t
        for _ in range(cfg.num_samples):
            expanded_tasks.append(env_task)
            expanded_ids.append(t.task_id)

    writer = _TrajectoryJsonlWriter(stream_path, append=bool(completed_task_ids)) if stream_path else None

    incremental_episodes: dict[str, list] = {}
    incremental_result_path = _planned_result_path(cfg, bench)

    def _on_complete(task_id: str, rollout_idx: int, episode) -> None:
        incremental_episodes.setdefault(task_id, []).append(episode)

        try:
            completed_tasks = [t for t in tasks if t.task_id in incremental_episodes]
            elapsed_so_far = time.time() - t0
            payload = _score_benchmark_results(
                cfg, bench, completed_tasks, incremental_episodes, elapsed_so_far
            )
            # Merge with previous records for incremental save
            if prev_records:
                payload["records"] = list(prev_records.values()) + payload["records"]
            _save_results(cfg, bench, payload, fpath=incremental_result_path)
            console.print(
                f"[dim]Incremental save: {len(completed_tasks)}/{len(tasks)} tasks "
                f"(+{len(prev_records)} previous)[/dim]"
            )
        except Exception as exc:
            console.print(f"[yellow]Incremental save failed: {exc}[/yellow]")

        if writer is None:
            return
        task = task_by_id[task_id]
        entry = _build_stream_entry(
            cfg=cfg,
            run_id=run_id,
            started_at=started_at,
            bench=bench,
            task=task,
            rollout_idx=rollout_idx,
            episode=episode,
        )
        writer.append(entry)
        if on_stream_entry is not None:
            on_stream_entry(entry)

    t0 = time.time()
    try:
        episodes = await _execute_tasks_streaming(
            engine,
            expanded_tasks,
            task_ids=expanded_ids,
            on_complete=_on_complete,
        )
    finally:
        if writer is not None:
            writer.close()
    elapsed = time.time() - t0

    # Group episodes back by task_id in input order.
    per_task: dict[str, list] = {}
    for tid, ep in zip(expanded_ids, episodes):
        per_task.setdefault(tid, []).append(ep)

    payload = _score_benchmark_results(cfg, bench, tasks, per_task, elapsed)
    # Merge previously completed records into the final payload
    if prev_records:
        payload["records"] = list(prev_records.values()) + payload["records"]
        # Update summary counts
        summary = payload["summary"]
        total_tasks_count = len(tasks) + len(prev_records)
        total_correct = sum(
            r.get("num_correct", 0) for r in payload["records"]
        )
        total_samples = sum(
            r.get("num_samples", 0) for r in payload["records"]
        )
        summary["num_tasks"] = total_tasks_count
        summary["accuracy_mean"] = (total_correct / total_samples) if total_samples else 0.0
    return payload


def _score_benchmark_results(
    cfg: AgentRolloutConfig,
    bench: str,
    tasks: list[AgenticTask],
    per_task: dict[str, list],
    elapsed: float,
) -> dict[str, Any]:
    """Aggregate per-task episodes into the per-benchmark payload.

    Shared by the sequential and mixed-rollouts paths — the only difference is
    whether ``per_task`` came from one ``execute_tasks`` call covering this
    benchmark only, or from the slice of a pooled mixed-rollouts run.
    """
    data_source = tasks[0].data_source if tasks else bench

    records: list[dict[str, Any]] = []
    total_correct = 0
    total_samples = 0
    passk_hits = 0
    per_task_accuracy: list[float] = []
    per_task_pass1: list[float] = []
    per_task_passn: list[float] = []
    passk_ks = sorted(
        {k for k in [1, 2, 4, 8, cfg.num_samples] if 1 < k <= cfg.num_samples}
    )
    per_task_passk_vals: dict[int, list[float]] = {k: [] for k in passk_ks}
    passhatk_ks = sorted(set(passk_ks) | {cfg.passk})
    per_task_passhatk_vals: dict[int, list[float]] = {k: [] for k in passhatk_ks}
    # is_correct[task_idx][sample_idx]. A "run" = one sample index across
    # every task; std across runs is the std of per-run aggregate metrics
    # (matches parallel_reasoner rollout inter-run std semantics).
    per_task_sample_correct: list[list[int]] = []

    for t in tasks:
        eps = per_task.get(t.task_id, [])
        problem_id = _problem_uuid(t.data_source, t.task_id, t.question)
        samples: list[dict[str, Any]] = []
        num_correct = 0
        sample_correct: list[int] = []
        for ep in eps:
            sample = _build_sample_record(
                problem_id=problem_id,
                task=t,
                episode=ep,
                model=cfg.model,
                trust_remote_code=cfg.vllm_trust_remote_code,
            )
            is_correct = bool(sample.get("is_correct"))
            num_correct += int(is_correct)
            sample_correct.append(int(bool(is_correct)))
            samples.append(sample)
        total_correct += num_correct
        total_samples += len(eps)
        per_task_sample_correct.append(sample_correct)
        pass1 = estimate_pass_at_k(len(eps), num_correct, 1) if eps else 0.0
        passk_val = estimate_pass_at_k(len(eps), num_correct, cfg.passk) if eps else 0.0
        passk_hits += int(passk_val > 0.5)
        per_task_accuracy.append(num_correct / len(eps) if eps else 0.0)
        per_task_pass1.append(pass1)
        passn_val = estimate_pass_at_k(len(eps), num_correct, len(eps)) if eps else 0.0
        per_task_passn.append(passn_val)
        for k in passk_ks:
            per_task_passk_vals[k].append(
                estimate_pass_at_k(len(eps), num_correct, k) if eps else 0.0
            )
        passhatk_val = (
            estimate_pass_hat_k(len(eps), num_correct, cfg.passk) if eps else 0.0
        )
        for k in passhatk_ks:
            per_task_passhatk_vals[k].append(
                estimate_pass_hat_k(len(eps), num_correct, k) if eps else 0.0
            )

        # Log per-problem metrics
        if cfg.passk == 1:
            console.log(
                f"[dim]{t.task_id}[/dim] Pass@1={pass1:.2f} ({num_correct}/{len(eps)})"
            )
        else:
            console.log(
                f"[dim]{t.task_id}[/dim] Pass@1={pass1:.2f} "
                f"Pass@{cfg.passk}={passk_val:.2f} "
                f"Pass^{cfg.passk}={passhatk_val:.2f} "
                f"({num_correct}/{len(eps)})"
            )

        problem_token_usage = _aggregate_problem_token_usage(samples)
        records.append(
            {
                "problem_id": problem_id,
                "task_id": t.task_id,
                "data_source": t.data_source,
                "question": t.question,
                "ground_truth": t.ground_truth,
                "num_samples": len(eps),
                "num_correct": num_correct,
                f"pass@{cfg.passk}": passk_val,
                f"pass^{cfg.passk}": passhatk_val,
                "token_usage": problem_token_usage,
                "total_output_tokens": problem_token_usage["total_output_tokens"],
                "samples": samples,
            }
        )

    accuracy = (total_correct / total_samples) if total_samples else 0.0
    passk_acc = (passk_hits / len(tasks)) if tasks else 0.0
    n = len(per_task_accuracy)

    def _mean_std(vals: list[float]) -> tuple[float, float]:
        mean = sum(vals) / len(vals) if vals else 0.0
        std = (
            math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
            if len(vals) > 1
            else 0.0
        )
        return mean, std

    def _per_run_accuracy(run_idx: int) -> float:
        """Mean over tasks of ``is_correct`` at sample index ``run_idx``."""
        hits = 0
        denom = 0
        for sc in per_task_sample_correct:
            if run_idx < len(sc):
                hits += sc[run_idx]
                denom += 1
        return (hits / denom) if denom else 0.0

    def _per_run_passk(k: int) -> list[float]:
        """Partition each task's samples into ``num_samples // k`` disjoint
        k-slices; a run's pass@k = mean over tasks of "any hit in that slice"."""
        if k < 1 or cfg.num_samples < k:
            return []
        n_runs = cfg.num_samples // k
        runs: list[float] = []
        for r in range(n_runs):
            hit_sum = 0
            denom = 0
            for sc in per_task_sample_correct:
                if len(sc) < (r + 1) * k:
                    continue
                window = sc[r * k : (r + 1) * k]
                hit_sum += 1 if any(window) else 0
                denom += 1
            runs.append((hit_sum / denom) if denom else 0.0)
        return runs

    # Std across runs (per-sample-index aggregates), not across tasks.
    per_run_acc = [_per_run_accuracy(i) for i in range(cfg.num_samples)]
    accuracy_std = _mean_std(per_run_acc)[1]
    pass1_std = _mean_std(_per_run_passk(cfg.passk))[1]
    summary = {
        "benchmark": bench,
        "data_source": data_source,
        "num_tasks": len(tasks),
        "num_samples_per_task": cfg.num_samples,
        "accuracy_mean": accuracy,
        "accuracy_std": accuracy_std,
        f"pass@{cfg.passk}": passk_acc,
        f"pass@{cfg.passk}_std": pass1_std,
        "elapsed_seconds": elapsed,
        "model": cfg.model,
        "tools": list(cfg.tools),
        "max_steps": cfg.max_steps,
    }
    for k, vals in per_task_passk_vals.items():
        summary[f"pass@{k}"] = sum(vals) / len(vals) if vals else 0.0
        summary[f"pass@{k}_std"] = _mean_std(_per_run_passk(k))[1]
    for k, vals in per_task_passhatk_vals.items():
        mean, std = _mean_std(vals)
        summary[f"pass^{k}"] = mean
        summary[f"pass^{k}_std"] = std

    return {"summary": summary, "records": records}


async def _run_benchmarks_mixed(
    cfg: AgentRolloutConfig,
    bench_tasks: list[tuple[str, list[AgenticTask]]],
    rollout_engine,
    *,
    run_id: str,
    started_at: float,
    stream_paths: dict[str, Path] | None = None,
    on_stream_entry: Callable[[dict[str, Any]], None] | None = None,
    belief_client: BeliefGraphClient | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Run every (bench, tasks) pair through one shared workflow engine.

    Every benchmark's rollouts share a single ``execute_tasks`` call so a
    long tail on one benchmark overlaps with fast-finishing rollouts on
    another, keeping the rollout pool saturated instead of draining
    between benchmarks. After the merged run, episodes are split back per
    benchmark and scored exactly as in the sequential path.
    """
    total_tasks = sum(len(ts) for _, ts in bench_tasks)
    bench_names = ", ".join(b for b, _ in bench_tasks)
    console.rule(
        f"[bold cyan]MIXED  [{bench_names}]  (n={total_tasks}, samples={cfg.num_samples})"
    )

    # Single workflow pool with a dispatching reward fn that resolves the
    # right grader from each rollout's ``data_source``.
    engine = _build_workflow_engine(
        cfg,
        data_source="__mixed__",
        rollout_engine=rollout_engine,
        reward_fn=build_dispatching_reward_fn(
            browsecomp_judge_config=_browsecomp_judge_config(cfg)
        ),
        belief_client=belief_client,
    )

    # Bench-prefixed engine task_ids so identical task_ids across
    # benchmarks (e.g. ``hotpotqa-0`` and ``2wiki-0``) don't collide.
    sep = "\x1f"
    expanded_tasks: list[dict] = []
    expanded_ids: list[str] = []
    task_by_mixed_id: dict[str, tuple[str, AgenticTask]] = {}
    for bench, tasks in bench_tasks:
        for t in tasks:
            env_task = t.to_env_task()
            mixed_id = f"{bench}{sep}{t.task_id}"
            task_by_mixed_id[mixed_id] = (bench, t)
            for _ in range(cfg.num_samples):
                expanded_tasks.append(env_task)
                expanded_ids.append(mixed_id)

    writers: dict[str, _TrajectoryJsonlWriter] = {}
    for bench, path in (stream_paths or {}).items():
        writers[bench] = _TrajectoryJsonlWriter(path)

    def _on_complete(mixed_id: str, rollout_idx: int, episode) -> None:
        bench, task = task_by_mixed_id[mixed_id]
        writer = writers.get(bench)
        if writer is None:
            return
        entry = _build_stream_entry(
            cfg=cfg,
            run_id=run_id,
            started_at=started_at,
            bench=bench,
            task=task,
            rollout_idx=rollout_idx,
            episode=episode,
        )
        writer.append(entry)
        if on_stream_entry is not None:
            on_stream_entry(entry)

    t0 = time.time()
    try:
        episodes = await _execute_tasks_streaming(
            engine,
            expanded_tasks,
            task_ids=expanded_ids,
            on_complete=_on_complete if writers else None,
        )
    finally:
        for writer in writers.values():
            writer.close()
    elapsed = time.time() - t0

    per_bench: dict[str, dict[str, list]] = {b: {} for b, _ in bench_tasks}
    for mid, ep in zip(expanded_ids, episodes):
        bench, _, task_id = mid.partition(sep)
        per_bench.setdefault(bench, {}).setdefault(task_id, []).append(ep)

    out: list[tuple[str, dict[str, Any]]] = []
    for bench, tasks in bench_tasks:
        payload = _score_benchmark_results(
            cfg, bench, tasks, per_bench.get(bench, {}), elapsed
        )
        out.append((bench, payload))
    return out


def _save_results(
    cfg: AgentRolloutConfig,
    bench: str,
    payload: dict[str, Any],
    *,
    fpath: Path | None = None,
) -> Path:
    fpath = fpath or _planned_result_path(cfg, bench)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with fpath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
    return fpath


def _fmt(mean: float, std: float) -> str:
    return f"{mean * 100:.1f} (±{std * 100:.1f})"


def _print_summary_table(summaries: list[dict[str, Any]], passk: int) -> None:
    if not summaries:
        return
    num_samples = summaries[0]["num_samples_per_task"]
    extra_ks = sorted(
        {k for k in [2, 4, 8, num_samples] if k > passk and k <= num_samples}
    )
    tbl = Table(title="Agent rollout summary", show_lines=False)
    tbl.add_column("Benchmark", style="cyan")
    tbl.add_column("N tasks", justify="right")
    tbl.add_column("Samples/task", justify="right")
    tbl.add_column("Accuracy", justify="right")
    tbl.add_column(f"Pass@{passk}", justify="right")
    for k in extra_ks:
        tbl.add_column(f"Pass@{k}", justify="right")
    tbl.add_column(f"Pass^{passk}", justify="right")
    for k in extra_ks:
        tbl.add_column(f"Pass^{k}", justify="right")
    tbl.add_column("Elapsed (s)", justify="right")
    for s in summaries:
        row = [
            s["benchmark"],
            str(s["num_tasks"]),
            str(s["num_samples_per_task"]),
            _fmt(s["accuracy_mean"], s.get("accuracy_std", 0.0)),
            _fmt(s[f"pass@{passk}"], s.get(f"pass@{passk}_std", 0.0)),
        ]
        for k in extra_ks:
            row.append(_fmt(s.get(f"pass@{k}", 0.0), s.get(f"pass@{k}_std", 0.0)))
        row.append(_fmt(s.get(f"pass^{passk}", 0.0), s.get(f"pass^{passk}_std", 0.0)))
        for k in extra_ks:
            row.append(_fmt(s.get(f"pass^{k}", 0.0), s.get(f"pass^{k}_std", 0.0)))
        row.append(f"{s['elapsed_seconds']:.1f}")
        tbl.add_row(*row)
    console.print(tbl)


async def run_agent_rollouts_async(cfg: AgentRolloutConfig) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    result_paths: list[str] = []
    live_paths: list[str] = []
    run_id = str(uuid.uuid4())
    started_at = time.time()
    completed_samples = 0
    _write_run_config(
        cfg,
        run_id=run_id,
        started_at=started_at,
        status="starting",
        phase="initializing_backend",
    )
    heartbeat = _RunStateHeartbeat(
        cfg,
        run_id=run_id,
        started_at=started_at,
        status="starting",
        phase="initializing_backend",
        summaries=summaries,
    ).start()
    rollout_engine = None
    belief_client: BeliefGraphClient | None = None
    try:
        rollout_engine = _build_rollout_engine(cfg)

        # Initialize belief graph client only for the original graph path.
        if uses_belief_graph_service(cfg.context_memory_mode, cfg.belief_graph_mode) and cfg.belief_graph_url:
            belief_client = BeliefGraphClient(
                base_url=cfg.belief_graph_url,
                timeout=cfg.belief_graph_timeout,
            )
            if not await belief_client.health_check():
                console.log(
                    f"[yellow]Belief graph service unreachable at {cfg.belief_graph_url}, disabling."
                )
                await belief_client.close()
                belief_client = None
            else:
                console.log(f"[green]Belief graph service connected at {cfg.belief_graph_url}")
        elif cfg.belief_graph_url and cfg.context_memory_mode != BELIEF_GRAPH_MODE:
            console.log(
                f"[yellow]Ignoring Belief Graph URL because context_memory_mode="
                f"{cfg.context_memory_mode}; graph service will not be called."
            )

        heartbeat.update(
            status="running",
            phase="loading_benchmarks",
            summaries=summaries,
        )
        loaded: list[tuple[str, list[AgenticTask]]] = []
        for bench in cfg.tasks:
            try:
                tasks = load_benchmark(
                    bench,
                    artifacts_dir=cfg.artifacts_dir or None,
                    max_problems=None if (cfg.task_ids or cfg.exclude_ids) else cfg.max_problems,
                    shuffle=cfg.shuffle,
                    shuffle_seed=cfg.shuffle_seed,
                )
            except (KeyError, FileNotFoundError) as e:
                console.log(f"[yellow]Skipping {bench}: {e}")
                continue
            if cfg.task_ids:
                task_id_set = set(cfg.task_ids)
                tasks = [t for t in tasks if t.task_id in task_id_set]
                console.log(f"[cyan]Filtered to {len(tasks)} tasks by --task-ids: {cfg.task_ids}")
            if cfg.exclude_ids:
                exclude_set = set(cfg.exclude_ids)
                tasks = [t for t in tasks if t.task_id not in exclude_set]
                console.log(f"[cyan]Excluded {len(exclude_set)} task IDs, {len(tasks)} remaining")
            if (cfg.task_ids or cfg.exclude_ids) and cfg.max_problems:
                tasks = tasks[:cfg.max_problems]
            if not tasks:
                console.log(f"[yellow]Benchmark {bench} yielded 0 tasks, skipping.")
                continue
            loaded.append((bench, tasks))

        total_samples = sum(len(tasks) * cfg.num_samples for _, tasks in loaded)
        planned_result_paths = {
            bench: _planned_result_path(cfg, bench) for bench, _ in loaded
        }
        planned_stream_paths = {
            bench: _planned_stream_path(cfg, bench) for bench, _ in loaded
        }
        live_paths = [str(path) for path in planned_stream_paths.values()]

        def _on_stream_entry(entry: dict[str, Any]) -> None:
            nonlocal completed_samples
            completed_samples += 1
            heartbeat.update(
                status="running",
                phase="streaming_trajectories",
                completed_benchmarks=len(summaries),
                total_benchmarks=len(loaded),
                completed_samples=completed_samples,
                total_samples=total_samples,
                current_benchmark=entry.get("benchmark"),
                live_paths=live_paths,
                result_paths=result_paths,
                summaries=summaries,
            )

        heartbeat.update(
            status="running",
            phase="benchmarks_loaded",
            completed_benchmarks=0,
            total_benchmarks=len(loaded),
            completed_samples=0,
            total_samples=total_samples,
            live_paths=live_paths,
            summaries=summaries,
        )

        if cfg.mixed_rollouts and len(loaded) > 1:
            console.log(
                f"[bold green]Mixed rollouts enabled: pooling rollouts across "
                f"{len(loaded)} benchmarks into a single workflow engine."
            )
            heartbeat.update(
                status="running",
                phase="generating_mixed_rollouts",
                completed_benchmarks=0,
                total_benchmarks=len(loaded),
                completed_samples=0,
                total_samples=total_samples,
                current_benchmark="__mixed__",
                current_expected_samples=total_samples,
                live_paths=live_paths,
                result_paths=result_paths,
                summaries=summaries,
            )
            results = await _run_benchmarks_mixed(
                cfg,
                loaded,
                rollout_engine,
                run_id=run_id,
                started_at=started_at,
                stream_paths=planned_stream_paths,
                on_stream_entry=_on_stream_entry,
                belief_client=belief_client,
            )
            for idx, (bench, payload) in enumerate(results, start=1):
                out_path = _save_results(
                    cfg, bench, payload, fpath=planned_result_paths[bench]
                )
                result_paths.append(str(out_path))
                console.log(f"[green]Saved {bench} -> {out_path}")
                summaries.append(payload["summary"])
                heartbeat.update(
                    status="running",
                    phase="saving_results",
                    completed_benchmarks=idx,
                    total_benchmarks=len(loaded),
                    completed_samples=completed_samples,
                    total_samples=total_samples,
                    live_paths=live_paths,
                    result_paths=result_paths,
                    summaries=summaries,
                )
        else:
            if cfg.mixed_rollouts and len(loaded) <= 1:
                console.log(
                    "[yellow]--mixed-rollouts set but only one benchmark loaded; "
                    "falling back to the sequential path."
                )
            for idx, (bench, tasks) in enumerate(loaded, start=1):
                expected = len(tasks) * cfg.num_samples
                heartbeat.update(
                    status="running",
                    phase="generating_rollouts",
                    completed_benchmarks=idx - 1,
                    total_benchmarks=len(loaded),
                    completed_samples=completed_samples,
                    total_samples=total_samples,
                    current_benchmark=bench,
                    current_expected_samples=expected,
                    live_paths=live_paths,
                    result_paths=result_paths,
                    summaries=summaries,
                )
                payload = await _run_one_benchmark(
                    cfg,
                    bench,
                    tasks,
                    rollout_engine,
                    run_id=run_id,
                    started_at=started_at,
                    stream_path=planned_stream_paths[bench],
                    on_stream_entry=_on_stream_entry,
                    belief_client=belief_client,
                )
                out_path = _save_results(
                    cfg, bench, payload, fpath=planned_result_paths[bench]
                )
                result_paths.append(str(out_path))
                console.log(f"[green]Saved {bench} -> {out_path}")
                summaries.append(payload["summary"])
                heartbeat.update(
                    status="running",
                    phase="benchmark_finished",
                    completed_benchmarks=idx,
                    total_benchmarks=len(loaded),
                    completed_samples=completed_samples,
                    total_samples=total_samples,
                    live_paths=live_paths,
                    result_paths=result_paths,
                    summaries=summaries,
                )
    except BaseException as exc:
        _write_run_config(
            cfg,
            run_id=run_id,
            started_at=started_at,
            status="failed",
            phase="failed",
            completed_at=time.time(),
            live_paths=live_paths,
            result_paths=result_paths,
            summaries=summaries,
            error=f"{type(exc).__name__}: {exc}",
        )
        heartbeat.update(
            status="failed",
            phase="failed",
            completed_benchmarks=len(summaries),
            total_benchmarks=None,
            completed_samples=completed_samples,
            live_paths=live_paths,
            result_paths=result_paths,
            summaries=summaries,
            error=f"{type(exc).__name__}: {exc}",
        )
        heartbeat.close()
        raise
    finally:
        shutdown = getattr(rollout_engine, "shutdown", None) if rollout_engine else None
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        if belief_client:
            try:
                await belief_client.close()
            except Exception:
                pass

    _print_summary_table(summaries, passk=cfg.passk)

    if summaries:
        overall_path = _model_output_dir(cfg) / "overall_summary.json"
        overall_path.parent.mkdir(parents=True, exist_ok=True)
        with overall_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"config": asdict(cfg), "summaries": summaries},
                f,
                ensure_ascii=False,
                indent=4,
            )
        console.log(f"[green]Overall summary -> {overall_path}")
        completed_at = time.time()
        _write_run_config(
            cfg,
            run_id=run_id,
            started_at=started_at,
            status="completed",
            phase="completed",
            completed_at=completed_at,
            live_paths=live_paths,
            result_paths=result_paths,
            summaries=summaries,
        )
        heartbeat.update(
            status="completed",
            phase="completed",
            completed_benchmarks=len(summaries),
            total_benchmarks=len(summaries),
            completed_samples=sum(
                int(s.get("num_tasks", 0) * s.get("num_samples_per_task", 0))
                for s in summaries
            ),
            total_samples=sum(
                int(s.get("num_tasks", 0) * s.get("num_samples_per_task", 0))
                for s in summaries
            ),
            live_paths=live_paths,
            result_paths=result_paths,
            summaries=summaries,
        )
    else:
        completed_at = time.time()
        _write_run_config(
            cfg,
            run_id=run_id,
            started_at=started_at,
            status="completed",
            phase="no_results",
            completed_at=completed_at,
            live_paths=live_paths,
            result_paths=result_paths,
            summaries=summaries,
        )
        heartbeat.update(
            status="completed",
            phase="no_results",
            completed_benchmarks=0,
            total_benchmarks=0,
            completed_samples=0,
            total_samples=0,
            live_paths=live_paths,
            result_paths=result_paths,
            summaries=summaries,
        )
    heartbeat.close()
    return summaries


def run_agent_rollouts(cfg: AgentRolloutConfig) -> list[dict[str, Any]]:
    """Synchronous entry point: loads every benchmark in ``cfg.tasks`` and
    runs it, writing per-benchmark and overall summaries under
    ``cfg.output_dir``.
    """
    return asyncio.run(run_agent_rollouts_async(cfg))


run_agentic_eval_async = run_agent_rollouts_async
run_agentic_eval = run_agent_rollouts
AgenticEvalConfig = AgentRolloutConfig


__all__ = [
    "AgentRolloutConfig",
    "estimate_pass_at_k",
    "estimate_pass_hat_k",
    "run_agent_rollouts",
    "run_agent_rollouts_async",
    "AgenticEvalConfig",
    "run_agentic_eval",
    "run_agentic_eval_async",
]
