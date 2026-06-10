"""DAG Scheduler — wraps MiniDAG execution with OS semantics.

Translates task-level operations (run, suspend, resume) into
compiled graph operations on the current DAG runner.

Supports multiple pipelines via PipelineRegistry + DynamicGraphBuilder.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncIterator

from typing import Any as _GraphType  # MiniDAGRunner via duck typing

from agent_harness.core.types import PipelineEventType, TaskId, TaskStatus
from agent_harness.state.event_store.sqlite import EventStore
from agent_harness.scheduling.process_manager import ProcessManager

logger = logging.getLogger(__name__)


_WALL_TIME_ENV = "AGENT_HARNESS_TASK_WALL_TIME_S"


class TaskWallTimeExceeded(RuntimeError):
    """Task exceeded its configured wall-time budget.

    Raised by :meth:`Scheduler.execute` when ``astream`` does not finish
    within ``wall_time_s`` (caller-supplied or read from
    ``AGENT_HARNESS_TASK_WALL_TIME_S``). The scheduler treats this as a
    budget abort rather than an implementation failure.
    """


def _resolve_wall_time_s(
    explicit: int | None,
    pipeline_id: str | None = None,  # noqa: ARG001 — kept for API stability
) -> int | None:
    """Resolve the effective wall-time budget.

    Priority: caller-supplied ``explicit`` > ``AGENT_HARNESS_TASK_WALL_TIME_S``
    env var > ``None`` (no timeout).

    Returns ``None`` (no deadline) or a positive int.
    """
    if explicit is not None:
        return explicit if explicit > 0 else None
    raw = os.environ.get(_WALL_TIME_ENV, "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; ignoring", _WALL_TIME_ENV, raw)
        else:
            return parsed if parsed > 0 else None
    return None


class Scheduler:
    """Executes research task DAGs via the configured graph runner."""

    def __init__(
        self,
        graph: _GraphType,
        process_manager: ProcessManager,
        event_store: EventStore,
        *,
        graph_builder: Any | None = None,
        pipeline_registry: Any | None = None,
        checkpointer: Any | None = None,  # noqa: ARG002  (kept for backward compat)
    ) -> None:
        self._default_graph = graph
        self._pm = process_manager
        self._event_store = event_store
        self._graph_builder = graph_builder
        self._pipeline_registry = pipeline_registry
        self._compiled_cache: dict[str, _GraphType] = {}

    def _get_graph(self, pipeline_id: str | None = None) -> _GraphType:
        """Get compiled graph for a pipeline, or the default.

        All pipelines (including "deep_research") are now built via
        DynamicGraphBuilder for uniform middleware wrapping.
        The _default_graph is the middleware-wrapped deep_research graph
        built at app startup.
        """
        if not pipeline_id:
            return self._default_graph

        # Check if we have a dynamically compiled version
        if pipeline_id in self._compiled_cache:
            return self._compiled_cache[pipeline_id]

        # Pre-built defaults (already middleware-wrapped at startup). Benchmark
        # sessions may construct Scheduler with no default graph and rely on
        # dynamic PipelineSpec compilation instead.
        if pipeline_id in ("deep_research", "react_research") and self._default_graph is not None:
            return self._default_graph

        if not self._pipeline_registry or not self._graph_builder:
            return self._default_graph

        spec = self._pipeline_registry.get(pipeline_id)
        self._register_pipeline_agents(spec)
        # Use ResearchState for specs that reference existing pipeline nodes
        # (they expect Annotated[list, operator.add] fan-in reducers).
        # Specs with only generic executors work fine with dict.
        state_type = self._resolve_state_type(spec)
        compiled = self._graph_builder.compile(
            spec, state_type=state_type, checkpointer=None,
        )
        self._compiled_cache[pipeline_id] = compiled

        return self._compiled_cache[pipeline_id]

    def _resolve_state_type(self, spec: Any) -> type | None:
        """Determine the correct state type for a PipelineSpec.

        Specs that reference existing pipeline nodes (via node_function)
        need ResearchState for proper fan-in reducers (Annotated[list, operator.add]).
        Specs with only generic executors (prompt_template) work with dict.
        """
        if hasattr(spec, "state_type") and spec.state_type:
            try:
                from agent_harness.core.runtime.dag.graph_builder import _import_dotted
                return _import_dotted(spec.state_type)
            except Exception as exc:  # noqa: BLE001 — import failure falls back to dict
                logger.debug("state_type %r unresolved: %s", spec.state_type, exc)

        return None  # Default: dict

    def _register_pipeline_agents(self, spec: Any) -> None:
        """Auto-register agent definitions bundled with a PipelineSpec."""
        from agent_harness.core.runtime import registry as kernel_registry
        from agent_harness.core.runtime.registries.agents import AgentRegistry

        if not hasattr(spec, "agent_definitions") or not spec.agent_definitions:
            return
        try:
            agent_reg = kernel_registry.get(AgentRegistry)
            for defn in spec.agent_definitions:
                if not agent_reg.has(defn.role_id):
                    agent_reg.register(defn)
                    logger.info("Auto-registered agent '%s' from pipeline '%s'", defn.role_id, spec.pipeline_id)
        except Exception as e:
            logger.warning("Failed to auto-register agents: %s", e)

    async def execute(
        self,
        task_id: TaskId,
        input_data: dict[str, Any] | None,
        *,
        pipeline_id: str | None = None,
        config_override: dict[str, Any] | None = None,
        execution_token: str | None = None,
        wall_time_s: int | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Execute a research task, yielding stream events.

        Wraps the graph runner's `astream()` with OS-level lifecycle management.

        ``wall_time_s`` is a hard wall-time ceiling for graph execution.
        Resolution priority: caller-supplied value >
        ``AGENT_HARNESS_TASK_WALL_TIME_S`` env var > workflow default
        (``workflows/<name>/defaults.py:TASK_WALL_TIME_S``, looked up
        by ``pipeline_id``) > no deadline. ``0`` means unlimited at
        every layer (matches the existing HLE budget convention).
        On expiry the scheduler marks the task ABORTED and raises
        :class:`TaskWallTimeExceeded`.
        """
        task = await self._pm.get_task(task_id)
        if task.status != TaskStatus.RUNNING:
            await self._pm.update_status(task_id, TaskStatus.RUNNING)

        graph = self._get_graph(pipeline_id)

        config = config_override or {
            "configurable": {
                "thread_id": task.thread_id,
            },
        }

        wall_time = _resolve_wall_time_s(wall_time_s, pipeline_id=pipeline_id)

        try:
            async for chunk in self._astream_with_wall_time(
                graph, input_data, config,
                task_id=task_id, wall_time=wall_time,
            ):
                latest_task = await self._pm.get_task(task_id)
                if self._is_stale_execution(latest_task, execution_token):
                    logger.info(
                        "Task %s execution token %s is stale before delivering chunk; stopping runner",
                        task_id,
                        execution_token,
                    )
                    return

                # chunk is {"node_name": {state_updates}}
                # Note: middleware hooks are applied inside node wrappers
                # (see DynamicGraphBuilder.wrap_with_middleware), NOT here.
                # By the time we receive a chunk, the node has already executed
                # with full middleware coverage (before/after/error).
                for node_name in chunk:
                    phase = node_name
                    logger.info("Task %s: phase → %s", task_id, phase)

                yield "updates", chunk

                latest_task = await self._pm.get_task(task_id)
                if self._is_stale_execution(latest_task, execution_token):
                    logger.info(
                        "Task %s execution token %s was superseded after chunk; stopping runner",
                        task_id,
                        execution_token,
                    )
                    return
                if latest_task.status == TaskStatus.SUSPENDED:
                    logger.info("Task %s suspended after phase boundary; stopping execution", task_id)
                    return
                if latest_task.status == TaskStatus.ABORTED:
                    logger.info("Task %s aborted after phase boundary; stopping execution", task_id)
                    return

            latest_task = await self._pm.get_task(task_id)
            if self._is_stale_execution(latest_task, execution_token):
                logger.info("Task %s execution token %s is stale at finalization; stopping runner", task_id, execution_token)
                return
            if latest_task.status == TaskStatus.SUSPENDED:
                logger.info("Task %s suspended; skipping completion transition", task_id)
                return
            if latest_task.status == TaskStatus.ABORTED:
                logger.info("Task %s finished after abort; skipping completion transition", task_id)
                return
            if latest_task.status == TaskStatus.COMPLETED:
                logger.info("Task %s already completed by another runner; skipping duplicate completion", task_id)
                return

            final_state = await self._get_state_for_config(graph, config)
            if not self._has_terminal_output(final_state):
                raise RuntimeError(
                    self._format_no_terminal_error(task_id, final_state)
                )

            latest_task = await self._pm.get_task(task_id)
            if self._is_stale_execution(latest_task, execution_token):
                logger.info(
                    "Task %s execution token %s was superseded after final state read; skipping completion",
                    task_id,
                    execution_token,
                )
                return
            if latest_task.status == TaskStatus.COMPLETED:
                logger.info("Task %s already completed after final state read; skipping duplicate completion", task_id)
                return

            await self._pm.update_status(task_id, TaskStatus.COMPLETED)
            await self._event_store.append(
                task_id=task_id,
                event_type=PipelineEventType.REPORT_GENERATED,
                payload={"summary": "Research completed successfully"},
            )

        except Exception as e:
            try:
                latest_task = await self._pm.get_task(task_id)
            except Exception:
                latest_task = None

            if latest_task and self._is_stale_execution(latest_task, execution_token):
                logger.info("Task %s stale execution token %s raised after being superseded; suppressing failure", task_id, execution_token)
                return
            if latest_task and latest_task.status == TaskStatus.COMPLETED:
                logger.info("Task %s raised after another runner already completed it; suppressing failure", task_id)
                return
            if latest_task and latest_task.status == TaskStatus.SUSPENDED:
                logger.info("Task %s suspended during execution; suppressing failure transition", task_id)
                return
            if latest_task and latest_task.status == TaskStatus.ABORTED:
                logger.info("Task %s aborted during execution; suppressing failure transition", task_id)
                return
            if isinstance(e, TaskWallTimeExceeded):
                logger.warning("Task %s wall-time exceeded; aborting", task_id)
                await self._pm.abort_task(task_id, str(e))
                raise

            logger.exception("Task %s failed: %s", task_id, e)
            await self._pm.set_error(task_id, str(e))
            raise

    async def _astream_with_wall_time(
        self,
        graph: _GraphType,
        input_data: dict[str, Any] | None,
        config: dict[str, Any],
        *,
        task_id: TaskId,
        wall_time: int | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield ``graph.astream`` chunks under a wall-time deadline.

        ``wall_time=None`` is the no-deadline fast path. On expiry we
        raise :class:`TaskWallTimeExceeded`; the outer ``execute`` arm
        does the abort + status transition.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wall_time if wall_time is not None else None
        stream = graph.astream(
            input_data, config=config, stream_mode="updates",
        )
        try:
            while True:
                try:
                    if deadline is None:
                        chunk = await stream.__anext__()
                    else:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            raise TaskWallTimeExceeded(
                                f"task {task_id} wall-time exceeded "
                                f"after {wall_time}s"
                            )
                        chunk = await asyncio.wait_for(
                            stream.__anext__(), timeout=remaining,
                        )
                except StopAsyncIteration:
                    return
                except asyncio.TimeoutError as e:
                    raise TaskWallTimeExceeded(
                        f"task {task_id} wall-time exceeded "
                        f"after {wall_time}s"
                    ) from e
                yield chunk
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()

    async def get_state(self, task_id: TaskId) -> dict[str, Any]:
        """Get the current state snapshot for a task.

        Tries the task's actual pipeline graph first, falls back to default,
        then falls back to raw checkpoint read.
        """
        task = await self._pm.get_task(task_id)
        pid = getattr(task, "pipeline_id", None)
        if pid == "auto" or pid is None:
            pid = "deep_research"
        config = {"configurable": {"thread_id": task.thread_id}}

        # Try with the specific pipeline's graph
        for try_pid in [pid, "deep_research"] if pid != "deep_research" else ["deep_research"]:
            try:
                graph = self._get_graph(try_pid)
                snapshot = await graph.aget_state(config)
                if snapshot and snapshot.values:
                    return snapshot.values
            except Exception:
                continue

        # Try all compiled pipeline graphs as a catch-all (MiniDAGRunner.aget_state
        # is a pure in-memory dict lookup so this is safe and cheap).
        for cached_graph in self._compiled_cache.values():
            try:
                snapshot = await cached_graph.aget_state(config)
                if snapshot and snapshot.values:
                    return snapshot.values
            except Exception:
                continue

        return {}

    async def _get_state_for_config(
        self,
        graph: _GraphType | None,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Best-effort state read for a specific checkpoint config."""
        if graph is not None:
            try:
                snapshot = await graph.aget_state(config)
                if snapshot and snapshot.values:
                    return snapshot.values
            except Exception:
                pass
        return {}

    @staticmethod
    def _has_terminal_output(state: dict[str, Any]) -> bool:
        """A task is terminal only when it has a reader-facing output.

        ``report`` is the canonical field for multi-phase research
        pipelines. Single-solver topologies (solo, debate) sometimes
        populate ``final_content`` instead — both are valid terminal
        outputs for the user-facing purpose.
        """
        for key in ("report", "final_content"):
            value = state.get(key)
            if isinstance(value, str) and value.strip():
                return True
            if isinstance(value, dict) and value:
                return True
        return False

    @staticmethod
    def _format_no_terminal_error(
        task_id: Any, state: dict[str, Any],
    ) -> str:
        """Build a human-actionable error for empty pipeline output.

        Surfaces the phase the task was in, any accumulated errors,
        and truncates long values so the message stays readable when
        it lands in a DB column and ultimately a UI banner.
        """
        phase = state.get("current_phase") or "?"
        bits = [
            f"Task {task_id} produced no final answer at phase '{phase}'."
        ]
        errors = state.get("errors")
        if isinstance(errors, list) and errors:
            # Cap at 3 error strings, each ≤200 chars. Anything longer
            # is probably a stack trace already logged by the node.
            shortened = [str(e)[:200] for e in errors[:3]]
            bits.append("Errors: " + " | ".join(shortened))
        iteration = state.get("iteration_count")
        if isinstance(iteration, int) and iteration > 0:
            bits.append(f"iterations={iteration}")
        return " ".join(bits)

    @staticmethod
    def _is_stale_execution(task: Any, execution_token: str | None) -> bool:
        """Detect whether a newer runner has taken over this task."""
        if not execution_token:
            return False
        context = getattr(task, "context", {}) or {}
        return context.get("active_execution_token") != execution_token
