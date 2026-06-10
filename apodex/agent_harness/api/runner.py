"""Runtime bridge used by the FastAPI server.

The benchmark adapter already has a small bootstrapper, but the API should
live inside the package instead of importing benchmark-only modules.  This
runner wires the same kernel services and executes one ReAct inquiry at a
time against the selected pipeline.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from agent_harness.api.schemas import InquiryResponse, TraceStep
from agent_harness.core.loop_types import (
    BaseObserver,
    LLMDeltaContext,
    LoopConfig,
    ToolCallIntervention,
    ToolResult,
    TurnContext,
)

logger = logging.getLogger(__name__)


@dataclass
class ActiveRun:
    """Runtime handle for a currently streaming API session."""

    session_id: str
    queue: asyncio.Queue[dict[str, Any]]
    worker_task: asyncio.Task[None] | None = None
    harness_task_id: str | None = None
    stop_requested: bool = False


class StreamingTraceObserver(BaseObserver):
    """Observer that bridges agent-loop events into an async API queue."""

    critical = True
    wants_llm_delta = True

    def __init__(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._queue = queue
        self._next_step_index = 1
        self._call_indices: dict[str, int] = {}

    async def on_loop_start(self, config: LoopConfig) -> None:
        await self._emit(
            "loop_start",
            {
                "task_id": config.task_id,
                "role_id": config.role_id,
                "max_turns": config.max_turns,
                "max_context_length": config.max_context_length,
                "max_tokens": config.max_completion_tokens,
            },
        )

    async def on_llm_delta(self, ctx: LLMDeltaContext) -> None:
        if not ctx.delta and not ctx.thinking_delta:
            return
        await self._emit(
            "llm_delta",
            {
                "turn": ctx.turn,
                "delta": ctx.delta,
                "reasoning_delta": ctx.thinking_delta,
                "delta_index": ctx.delta_index,
            },
        )

    async def on_llm_response(self, ctx: TurnContext) -> None:
        await self._emit(
            "llm_response",
            {
                "turn": ctx.turn,
                "text": ctx.ai_text or "",
                "reasoning": ctx.thinking or ctx.leaked_reasoning or "",
                "tool_calls_count": len(ctx.tool_calls or []),
                "usage": ctx.usage or {},
            },
        )

    async def on_tool_call(
        self,
        ctx: TurnContext,
        tool_call: dict,
    ) -> ToolCallIntervention | None:
        tool_name = str(tool_call.get("name") or "")
        args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
        call_id = str(tool_call.get("id") or f"turn-{ctx.turn}-call-{self._next_step_index}")
        index = self._next_step_index
        self._next_step_index += 1
        self._call_indices[call_id] = index
        await self._emit(
            "tool_call",
            {
                "turn": ctx.turn,
                "call_id": call_id,
                "step": _live_tool_step(
                    index=index,
                    turn=ctx.turn,
                    tool_name=tool_name,
                    args=args,
                    status="running",
                ),
            },
        )
        return None

    async def on_tool_result(
        self,
        ctx: TurnContext,
        result: ToolResult,
    ) -> None:
        index = self._call_indices.get(result.tool_call_id)
        if index is None:
            index = self._next_step_index
            self._next_step_index += 1
        await self._emit(
            "tool_result",
            {
                "turn": ctx.turn,
                "call_id": result.tool_call_id,
                "step": _trace_step_from_tool_result(index, ctx, result).model_dump(),
            },
        )

    async def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        await self._queue.put({"type": event_type, "payload": payload})


class HarnessAPIRunner:
    """Scoped AgentHarness runtime for web requests."""

    def __init__(self) -> None:
        self._scheduler: Any | None = None
        self._pm: Any | None = None
        self._registry_snapshot: dict[type, Any] | None = None
        self._lock = asyncio.Lock()
        self._active_runs: dict[str, ActiveRun] = {}
        self._active_lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._scheduler is not None and self._pm is not None

    async def __aenter__(self) -> "HarnessAPIRunner":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()

    async def start(self) -> None:
        """Bootstrap kernel services once for the API process."""
        if self.ready:
            return

        from agent_harness.core.runtime import registry

        self._registry_snapshot = registry.snapshot()
        try:
            await self._bootstrap()
        except Exception:
            registry.restore(self._registry_snapshot)
            self._registry_snapshot = None
            self._scheduler = None
            self._pm = None
            raise

    async def stop(self) -> None:
        """Restore the registry snapshot captured during startup."""
        if self._registry_snapshot is None:
            return

        from agent_harness.core.runtime import registry

        registry.restore(self._registry_snapshot)
        self._registry_snapshot = None
        self._scheduler = None
        self._pm = None

    async def run_inquiry(
        self,
        *,
        query: str,
        session_id: str | None,
        pipeline_id: str,
        profile: str | None,
        model: str | None,
        wall_time_s: int | None,
    ) -> InquiryResponse:
        """Execute a question through AgentHarness and normalize the result."""
        if not self.ready:
            await self.start()

        assert self._scheduler is not None
        assert self._pm is not None

        # The harness registry is process-global. Serializing requests avoids
        # cross-run observer and checkpoint state bleed while keeping the API
        # implementation small and predictable.
        async with self._lock:
            started = time.monotonic()
            task = await self._pm.create_task(query)
            task_id = str(task.id)
            meta: dict[str, Any] = {
                "source": "api",
                "session_id": session_id or "",
                "model": model or "",
            }
            if profile:
                meta["profile"] = profile

            input_data: dict[str, Any] = {
                "task_id": task_id,
                "original_question": query,
                "file_path": "",
                "metadata": meta,
                "clarified_questions": [],
                "evidence_cards": [],
                "assertions": [],
                "react_steps": [],
                "report": None,
                "current_phase": "",
                "errors": [],
                "messages": [],
                "language": "auto",
            }

            state: dict[str, Any] = {}
            status = "completed"
            error: str | None = None
            try:
                async for _mode, _chunk in self._scheduler.execute(
                    task.id,
                    input_data,
                    pipeline_id=pipeline_id,
                    wall_time_s=wall_time_s,
                ):
                    pass
                state = await self._scheduler.get_state(task.id)
            except Exception as exc:  # noqa: BLE001 - API returns structured failure
                logger.exception("AgentHarness inquiry failed")
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                try:
                    state = await self._scheduler.get_state(task.id)
                except Exception:
                    state = {}

            final_answer = _extract_final_answer(state)
            trace = normalize_trace_steps(state.get("react_steps") or [])
            duration = time.monotonic() - started
            return InquiryResponse(
                id=task_id,
                session_id=session_id,
                query=query,
                status=status,  # type: ignore[arg-type]
                final_answer=final_answer,
                trace=trace,
                duration_seconds=round(duration, 2),
                pipeline_id=pipeline_id,
                profile=profile,
                model=model,
                error=error,
            )

    async def stop_inquiry(self, session_id: str) -> dict[str, Any]:
        """Cancel the currently running stream for a frontend session."""
        async with self._active_lock:
            active = self._active_runs.get(session_id)

        if active is None:
            return {
                "stopped": False,
                "session_id": session_id,
                "status": "not_found",
            }

        active.stop_requested = True
        if active.harness_task_id and self._pm is not None:
            with suppress(Exception):
                await self._pm.abort_task(active.harness_task_id, "Stopped by user")

        await active.queue.put(
            {
                "type": "stopped",
                "payload": {
                    "session_id": session_id,
                    "task_id": active.harness_task_id,
                    "message": "Stopped by user",
                },
            },
        )

        if active.worker_task is not None and not active.worker_task.done():
            active.worker_task.cancel()

        return {
            "stopped": True,
            "session_id": session_id,
            "task_id": active.harness_task_id,
            "status": "stopping",
        }

    async def run_inquiry_stream(
        self,
        *,
        query: str,
        session_id: str | None,
        pipeline_id: str,
        profile: str | None,
        model: str | None,
        wall_time_s: int | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute a question and yield real-time trace events."""
        if not self.ready:
            await self.start()

        assert self._scheduler is not None
        assert self._pm is not None

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        done = object()
        active = ActiveRun(session_id=session_id or "", queue=queue)

        async def worker() -> None:
            try:
                await queue.put(
                    {
                        "type": "queued",
                        "payload": {
                            "session_id": session_id,
                            "query": query,
                            "pipeline_id": pipeline_id,
                            "profile": profile,
                            "model": model,
                        },
                    },
                )
                async with self._lock:
                    started = time.monotonic()
                    task = await self._pm.create_task(query)
                    task_id = str(task.id)
                    active.harness_task_id = task_id
                    observer = StreamingTraceObserver(queue)
                    await queue.put(
                        {
                            "type": "run_start",
                            "payload": {
                                "id": task_id,
                                "session_id": session_id,
                                "query": query,
                                "pipeline_id": pipeline_id,
                                "profile": profile,
                                "model": model,
                            },
                        },
                    )
                    meta: dict[str, Any] = {
                        "source": "api",
                        "session_id": session_id or "",
                        "model": model or "",
                        "sdk_extra_observers": [observer],
                    }
                    if profile:
                        meta["profile"] = profile

                    input_data: dict[str, Any] = {
                        "task_id": task_id,
                        "original_question": query,
                        "file_path": "",
                        "metadata": meta,
                        "clarified_questions": [],
                        "evidence_cards": [],
                        "assertions": [],
                        "react_steps": [],
                        "report": None,
                        "current_phase": "",
                        "errors": [],
                        "messages": [],
                        "language": "auto",
                    }

                    state: dict[str, Any] = {}
                    status = "completed"
                    error: str | None = None
                    try:
                        async for _mode, chunk in self._scheduler.execute(
                            task.id,
                            input_data,
                            pipeline_id=pipeline_id,
                            wall_time_s=wall_time_s,
                        ):
                            await queue.put(
                                {
                                    "type": "phase_update",
                                    "payload": {"chunk": _json_safe(chunk)},
                                },
                            )
                        state = await self._scheduler.get_state(task.id)
                    except asyncio.CancelledError:
                        if active.stop_requested:
                            await queue.put(
                                {
                                    "type": "stopped",
                                    "payload": {
                                        "session_id": session_id,
                                        "task_id": task_id,
                                        "message": "Stopped by user",
                                    },
                                },
                            )
                            return
                        raise
                    except Exception as exc:  # noqa: BLE001 - stream structured failure
                        logger.exception("AgentHarness streamed inquiry failed")
                        status = "failed"
                        error = f"{type(exc).__name__}: {exc}"
                        await queue.put(
                            {
                                "type": "error",
                                "payload": {"message": error},
                            },
                        )
                        with suppress(Exception):
                            state = await self._scheduler.get_state(task.id)

                    final_answer = _extract_final_answer(state)
                    trace = normalize_trace_steps(state.get("react_steps") or [])
                    duration = time.monotonic() - started
                    response = InquiryResponse(
                        id=task_id,
                        session_id=session_id,
                        query=query,
                        status=status,  # type: ignore[arg-type]
                        final_answer=final_answer,
                        trace=trace,
                        duration_seconds=round(duration, 2),
                        pipeline_id=pipeline_id,
                        profile=profile,
                        model=model,
                        error=error,
                    )
                    await queue.put(
                        {
                            "type": "final",
                            "payload": response.model_dump(),
                        },
                    )
                await queue.put({"type": "done", "payload": {}})
            except asyncio.CancelledError:
                if active.stop_requested:
                    await queue.put(
                        {
                            "type": "stopped",
                            "payload": {
                                "session_id": session_id,
                                "task_id": active.harness_task_id,
                                "message": "Stopped by user",
                            },
                        },
                    )
                    await queue.put({"type": "done", "payload": {}})
                    return
                raise
            except Exception as exc:  # noqa: BLE001 - keep SSE clients from hanging
                logger.exception("AgentHarness stream worker failed")
                await queue.put(
                    {
                        "type": "error",
                        "payload": {"message": f"{type(exc).__name__}: {exc}"},
                    },
                )
            finally:
                if session_id:
                    async with self._active_lock:
                        current = self._active_runs.get(session_id)
                        if current is active:
                            self._active_runs.pop(session_id, None)
                await queue.put({"type": "_done", "payload": done})

        if session_id:
            async with self._active_lock:
                self._active_runs[session_id] = active
        worker_task = asyncio.create_task(worker())
        active.worker_task = worker_task
        try:
            while True:
                item = await queue.get()
                if item.get("payload") is done:
                    break
                yield item
        finally:
            if not worker_task.done():
                worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await worker_task

    async def _bootstrap(self) -> None:
        """Wire the runtime services used by the ReAct pipeline."""
        from dotenv import load_dotenv

        from agent_harness.components.middleware.llm import (
            LLMMiddlewareChain,
            SummarizationMiddleware,
        )
        from agent_harness.core.runtime import registry
        from agent_harness.core.runtime.dag.graph_builder import DynamicGraphBuilder
        from agent_harness.core.runtime.events.bus import EventBus
        from agent_harness.core.runtime.registries.agents import AgentRegistry
        from agent_harness.core.runtime.resources.manager import ResourceManager
        from agent_harness.infra.config import get_config
        from agent_harness.infra.llm_adapter import create_llm
        from agent_harness.scheduling.pipeline_registry import PipelineRegistry
        from agent_harness.scheduling.process_manager import ProcessManager
        from agent_harness.scheduling.scheduler import Scheduler
        from agent_harness.scheduling.workflow_loader import load_workflow_plugins
        from agent_harness.state.event_store.sqlite import EventStore
        from plugins.tools import get_builtin_tools

        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env", override=True)
        os.makedirs("data", exist_ok=True)

        event_store = EventStore()
        registry.register(EventStore, event_store)
        registry.register(EventBus, EventBus())

        pm = ProcessManager(event_store, session_factory=None)
        registry.register(ProcessManager, pm)

        agent_reg = AgentRegistry()
        registry.register(AgentRegistry, agent_reg)

        llm = create_llm(get_config())
        tools_map = dict(get_builtin_tools())
        registry.register(ResourceManager, ResourceManager(llm=llm, tools=tools_map))

        llm_mw_chain = LLMMiddlewareChain()
        llm_mw_chain.add(SummarizationMiddleware(threshold=80_000, keep_recent=10))
        registry.register(LLMMiddlewareChain, llm_mw_chain)

        pipeline_reg = PipelineRegistry()
        load_workflow_plugins(pipeline_reg, agent_reg)
        _discover_pipeline_specs(pipeline_reg)
        registry.register(PipelineRegistry, pipeline_reg)

        graph_builder = DynamicGraphBuilder()
        registry.register(DynamicGraphBuilder, graph_builder)

        scheduler = Scheduler(
            None,
            pm,
            event_store,
            graph_builder=graph_builder,
            pipeline_registry=pipeline_reg,
            checkpointer=None,
        )
        registry.register(Scheduler, scheduler)

        self._scheduler = scheduler
        self._pm = pm


def _discover_pipeline_specs(pipeline_reg: Any) -> None:
    """Register module-level PipelineSpec objects under ``workflows/*/spec.py``."""
    from agent_harness.models.pipeline_spec import PipelineSpec

    project_root = Path(__file__).resolve().parents[2]
    workflows_dir = project_root / "workflows"
    if not workflows_dir.is_dir():
        return

    import sys

    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    for spec_file in sorted(workflows_dir.glob("*/spec.py")):
        if spec_file.name.startswith("_"):
            continue
        mod_name = ".".join(spec_file.relative_to(project_root).with_suffix("").parts)
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            logger.warning("Skipping pipeline spec %s", mod_name, exc_info=True)
            continue
        for attr in dir(mod):
            obj = getattr(mod, attr, None)
            if isinstance(obj, PipelineSpec):
                pipeline_reg.register(obj)


def _extract_final_answer(state: dict[str, Any]) -> str:
    report = state.get("report")
    if isinstance(report, str) and report.strip():
        return report.strip()
    if isinstance(report, dict):
        for key in ("summary", "answer", "final_answer"):
            value = report.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("final_answer", "final_content"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_trace_steps(raw_steps: list[Any]) -> list[TraceStep]:
    """Convert harness react_steps into reader-safe frontend steps."""
    steps: list[TraceStep] = []
    for idx, raw in enumerate(raw_steps, start=1):
        if not isinstance(raw, dict):
            continue
        tool_name = str(raw.get("tool_name") or "")
        args = _parse_tool_args(raw.get("tool_args"))
        observation = _truncate(_squash(raw.get("tool_result_preview") or ""), 2_000)
        is_error = bool(raw.get("is_error"))
        steps.append(
            TraceStep(
                index=idx,
                turn=_int_or_none(raw.get("turn")),
                title=_title_for_tool(tool_name),
                summary=_summary_for_tool(tool_name, args, is_error),
                tool_name=tool_name or None,
                tool_args=args,
                observation=observation,
                duration_ms=_int_or_none(raw.get("duration_ms")),
                status="error" if is_error else "completed",
            )
        )
    return steps


def _live_tool_step(
    *,
    index: int,
    turn: int,
    tool_name: str,
    args: dict[str, Any] | str | None,
    status: str,
) -> dict[str, Any]:
    return {
        "index": index,
        "turn": turn,
        "title": _title_for_tool(tool_name),
        "summary": _summary_for_tool(tool_name, args, is_error=False),
        "tool_name": tool_name or None,
        "tool_args": _json_safe(args),
        "observation": "",
        "duration_ms": None,
        "status": status,
    }


def _trace_step_from_tool_result(
    index: int,
    ctx: TurnContext,
    result: ToolResult,
) -> TraceStep:
    observation = _truncate(_squash(result.result or ""), 2_000)
    return TraceStep(
        index=index,
        turn=ctx.turn,
        title=_title_for_tool(result.name),
        summary=_summary_for_tool(result.name, result.args, result.is_error),
        tool_name=result.name or None,
        tool_args=_json_safe(result.args),
        observation=observation,
        duration_ms=result.duration_ms,
        status="error" if result.is_error else "completed",
    )


def _parse_tool_args(value: Any) -> dict[str, Any] | str | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return str(value)
    import json

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return parsed if isinstance(parsed, dict) else value


def _title_for_tool(tool_name: str) -> str:
    labels = {
        "web_search": "Search",
        "web_fetch": "Inspect Source",
        "run_python_code": "Compute",
    }
    return labels.get(tool_name, "Reasoning Step")


def _summary_for_tool(
    tool_name: str,
    args: dict[str, Any] | str | None,
    is_error: bool,
) -> str:
    if is_error:
        return "The agent attempted this step, but the tool returned an error."
    if tool_name == "web_search":
        query = _arg(args, "q") or _arg(args, "query")
        return f"Searched for evidence about {_quote(query)}." if query else "Searched for relevant evidence."
    if tool_name == "web_fetch":
        url = _arg(args, "url")
        return f"Opened and inspected {_quote(url)}." if url else "Opened a source for closer inspection."
    if tool_name == "run_python_code":
        return "Ran a Python check to compute or verify an intermediate result."
    return "Used an available tool to advance the answer."


def _arg(args: dict[str, Any] | str | None, key: str) -> str:
    if isinstance(args, dict):
        value = args.get(key)
        return str(value).strip() if value is not None else ""
    return ""


def _quote(value: str) -> str:
    clean = _truncate(_squash(value), 120)
    return f'"{clean}"' if clean else "the query"


def _squash(value: Any) -> str:
    return " ".join(str(value).split())


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    import json

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(value)
