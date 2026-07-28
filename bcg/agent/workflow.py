"""Project-owned workflow entrypoint for BeliefTracer rollouts."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

from bcg.agent.context_memory import ContextMemoryConfig, build_context_memory
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.agents.agent import Episode
from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
from rllm.workflows.workflow import TerminationEvent, TerminationReason

if TYPE_CHECKING:
    from bcg.agent.belief_graph_client import BeliefGraphClient

logger = logging.getLogger(__name__)


def _evict_graph_turn_payloads(
    window: list[list[dict[str, Any]]],
    completed_turn: list[dict[str, Any]],
    raw_turn_limit: int | None,
) -> list[dict[str, Any]]:
    """Return graph payloads for turns that just left the raw context.

    ``raw_turn_limit=None`` means raw history is unbounded, so no completed
    turn is graph-eligible.  A limit of zero is used by graph-only mode.
    """
    if raw_turn_limit is None:
        return []
    window.append(completed_turn)
    evicted: list[dict[str, Any]] = []
    while len(window) > raw_turn_limit:
        evicted.extend(window.pop(0))
    return evicted


def _is_context_length_error(exc: BaseException) -> bool:
    """Return True for server-side context-window overflow errors."""
    seen: set[int] = set()
    current: BaseException | None = exc
    parts: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        parts.extend(str(arg) for arg in getattr(current, "args", ()) if arg is not current)
        current = current.__cause__ or current.__context__

    text = "\n".join(parts).lower()
    signals = (
        "context length",
        "context window",
        "context_length_exceeded",
        "context_window_exceeded",
        "maximum context",
        "maximum model length",
        "max model len",
        "prompt is too long",
        "prompt too long",
        "input is too long",
        "input too long",
        "too many tokens",
        "exceeds the context",
        "exceed context",
        "maximum number of tokens",
    )
    return any(signal in text for signal in signals)


def _model_output_token_counts(output: ModelOutput) -> dict[str, int]:
    prompt_ids = getattr(output, "prompt_ids", None)
    completion_ids = getattr(output, "completion_ids", None)
    prompt_tokens = (
        len(prompt_ids)
        if prompt_ids is not None
        else int(getattr(output, "prompt_length", 0) or 0)
    )
    completion_tokens = (
        len(completion_ids)
        if completion_ids is not None
        else int(getattr(output, "completion_length", 0) or 0)
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


class BeliefTracerWorkflow(MultiTurnWorkflow):
    """Named workflow wrapper used by BeliefTracer runtime entrypoints."""

    def __init__(self, *args, belief_client: "BeliefGraphClient | None" = None,
                 graph_output_dir: str = "", archive_enabled: bool = False,
                 file_tool_root: str = "", belief_graph_interval: int = 1,
                 recent_turns: int = 0,
                 tonggraph_sync_config: dict[str, Any] | None = None,
                 context_memory_config: dict[str, Any] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.belief_client = belief_client
        self._graph_output_dir = graph_output_dir
        self._archive_enabled = bool(archive_enabled)
        self._file_tool_root = file_tool_root
        # Legacy interval path used only when archive mode is disabled. Archive
        # mode aligns graph updates directly with raw-context eviction.
        self._belief_graph_interval = max(1, int(belief_graph_interval or 1))
        parsed_recent_turns = int(recent_turns)
        self._recent_turns = -1 if parsed_recent_turns < 0 else parsed_recent_turns
        # In archive mode, graph ingestion follows the same sliding-window
        # boundary as prompt assembly: a completed turn enters the graph only
        # when it is evicted from the raw recent-turn context.  This prevents a
        # turn from appearing both verbatim and as graph beliefs.  The initial
        # system + question graph remains a deliberate shared task anchor.
        self._eviction_aligned_graph = self._archive_enabled
        self._tonggraph_sync_config = dict(tonggraph_sync_config or {})
        self._context_memory_config = dict(context_memory_config or {})

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        """Execute a multi-step workflow while preserving raw model outputs."""

        self._graph_snapshots: list[dict[str, Any]] = []
        self._model_io: list[dict[str, Any]] = []
        self._token_usage_turns: list[dict[str, int]] = []
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._problem_id: str = f"{uid}:{uuid.uuid4().hex[:8]}"
        self._context_memory = build_context_memory(
            ContextMemoryConfig(**self._context_memory_config)
            if self._context_memory_config
            else None
        )
        # Turns buffered since the last graph push, flushed together when the
        # interval triggers (or at trajectory end). See __init__ docstring.
        self._pending_graph_turns: list[dict[str, Any]] = []
        # Completed assistant/tool turns still present verbatim in the model
        # context.  Each entry is one logical turn and its graph payloads.
        self._raw_graph_turn_window: list[list[dict[str, Any]]] = []
        self._graph_raw_turn_limit: int | None = (
            0
            if getattr(self.agent, "belief_graph_mode", "augment") == "only"
            else (None if self._recent_turns == -1 else self._recent_turns)
        )
        logger.info("[Workflow] Starting problem_id=%s  uid=%s", self._problem_id, uid)

        # Two-layer archive of tool results (layered context mode).
        self._archive = None
        if self._archive_enabled:
            from bcg.agent.archive import ArchiveWriter

            self._archive = ArchiveWriter(self._problem_id, root=self._file_tool_root or None)
            set_url = getattr(self.agent, "set_archive_tool_urls", None)
            if callable(set_url):
                set_url(self._archive.tool_index_urls)

        observation, info = await self.timed_env_call(self.reset, task=task, uid=uid)
        self.agent.update_from_env(observation, 0, False, info)

        question = ""
        if isinstance(observation, dict):
            question = str(observation.get("question") or observation.get("prompt") or "")
        else:
            question = str(observation or "")
        system_content = (
            self.agent.chat_completions[0].get("content", "")
            if self.agent.chat_completions
            else getattr(self.agent, "system_prompt", "")
        )
        if self._context_memory:
            self._context_memory.observe_initial(system=system_content, question=question)
            set_memory = getattr(self.agent, "set_context_memory_message", None)
            if callable(set_memory):
                set_memory(self._context_memory.render_message())

        # Push initial system + user turns individually
        if self.belief_client:
            await self.belief_client.push_turn(
                problem_id=self._problem_id,
                role="system",
                content=system_content,
                is_message_end=True,
                is_trajectory_end=False,
            )
            snapshot = await self.belief_client.push_turn(
                problem_id=self._problem_id,
                role="user",
                content=question,
                is_message_end=True,
                is_trajectory_end=False,
            )
            if snapshot:
                await self._record_graph_snapshot(snapshot, phase="initial")
                self.agent.inject_belief_graph(snapshot, self.belief_client)

        import time as _time

        for _ in range(1, self.max_steps + 1):
            _t_llm_start = _time.time()
            try:
                output: ModelOutput = await self.timed_llm_call(
                    self.agent.chat_completions,
                    application_id=uid,
                    **kwargs,
                )
            except Exception as exc:
                if _is_context_length_error(exc):
                    raise TerminationEvent(
                        TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED
                    ) from exc
                raise
            _t_llm = _time.time() - _t_llm_start
            response = output.text or ""
            turn_index = len(self._model_io)
            token_counts = _model_output_token_counts(output)
            self._total_prompt_tokens += token_counts["prompt_tokens"]
            self._total_completion_tokens += token_counts["completion_tokens"]
            turn_token_usage = {
                "turn": turn_index,
                **token_counts,
                "cumulative_prompt_tokens": self._total_prompt_tokens,
                "cumulative_completion_tokens": self._total_completion_tokens,
            }
            self._token_usage_turns.append(turn_token_usage)
            logger.info(
                "[Workflow] Turn %d tokens: output=%d total_output=%d prompt=%d",
                turn_index,
                token_counts["completion_tokens"],
                self._total_completion_tokens,
                token_counts["prompt_tokens"],
            )

            self._model_io.append({
                "turn": turn_index,
                "input": [dict(m) for m in self.agent.chat_completions],
                "output": {
                    "content": getattr(output, "content", "") or "",
                    "reasoning": getattr(output, "reasoning", "") or "",
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in (output.tool_calls or [])
                    ] if output.tool_calls else [],
                    "finish_reason": getattr(output, "finish_reason", None),
                },
                "tokens": turn_token_usage,
                "timings": {},
            })

            action = self.agent.update_from_model(response, model_output=output)
            if self._context_memory:
                self._context_memory.observe_assistant(
                    content=response,
                    model_io=self._model_io[-1],
                )

            # Record both the model's unmodified output and the canonicalized
            # (call_id-tagged) text that was actually stored in the agent's
            # message history -- kept as two separate fields so a divergence
            # between them (protocol rewriting gone wrong) is visible in the
            # saved trajectory, not silently overwritten.
            self._model_io[-1]["output"]["raw_assistant_output"] = getattr(
                self.agent, "_pending_raw_assistant_output", ""
            )
            self._model_io[-1]["output"]["canonical_assistant_output"] = getattr(
                self.agent, "_pending_canonical_assistant_output", ""
            )
            # Re-tag tool_calls with the call_id BeliefTracerAgent assigned
            # (LabeledToolCall.id), so the saved trajectory can be cross-
            # referenced against tool_metadata's "tool_call_id" without
            # re-parsing raw_assistant_output.
            action_calls = action.action if isinstance(action.action, list) else []
            if action_calls and len(action_calls) == len(
                self._model_io[-1]["output"]["tool_calls"]
            ):
                for tc_record, call in zip(
                    self._model_io[-1]["output"]["tool_calls"], action_calls
                ):
                    tc_record["id"] = getattr(call, "id", "") or ""

            # Push the assistant turn to the belief graph NOW — don't wait for the tool.
            # It only needs `response` (already produced), so the graph build overlaps
            # with tool execution below. Awaited later (before the tool push) to keep
            # assistant-before-tool ingestion order on the server. Only for the
            # interval==1 (default) path — buffered intervals batch this turn into
            # _pending_graph_turns instead (see below), so kicking off a real POST
            # here would double-push it.
            _assistant_push_task = None
            if (
                self.belief_client
                and not self._eviction_aligned_graph
                and self._belief_graph_interval == 1
            ):
                _assistant_push_task = asyncio.ensure_future(self.belief_client.push_turn(
                    problem_id=self._problem_id,
                    role="assistant",
                    content=response,
                    is_message_end=True,
                    is_trajectory_end=False,
                    meta={"timings": {"llm": round(_t_llm, 3)}},
                ))

            _t_tool_start = _time.time()
            next_obs, reward, done, info = await self.timed_env_call(
                self.env.step, action
            )
            _t_tool = _time.time() - _t_tool_start
            tool_meta = (info or {}).get("tool_metadata") or []

            # Archive each retrieved evidence separately (two-layer), then refresh
            # the manifest URL the agent advertises in its system prompt.
            if self._archive is not None:
                wrote = False
                for tm in tool_meta:
                    if tm.get("name") == "finish":
                        continue
                    # Skip recall-only tools (e.g. read_file): their results were
                    # already archived when first retrieved; re-archiving is
                    # redundant. Routed by capability flag, not tool name.
                    if not tm.get("feeds_memory", True):
                        continue
                    md = tm.get("metadata") or {}
                    evidences = md.get("evidences") or []
                    query = str(md.get("query") or (tm.get("arguments") or {}).get("query") or "")
                    # Local call_id (e.g. "call_1") from this turn's protocol-
                    # layer numbering, plus a globally-unique id for archive
                    # cross-referencing (e.g. "claim469:abcd1234_round3_call_1").
                    # Only the local id is ever sent back to the model (in the
                    # <tool_result> block rendered by rllm_compat.step()); the
                    # global id is archive/log-only.
                    call_id = str(tm.get("tool_call_id") or "")
                    global_call_id = (
                        f"{self._problem_id}_round{len(self._model_io)}_{call_id}"
                        if call_id else ""
                    )
                    archive_entries: list[dict[str, Any]] = []
                    if evidences:
                        archive_entries = self._archive.add_evidences(
                            turn=len(self._model_io),
                            tool_name=tm.get("name", "averitec_search"),
                            query=query,
                            evidences=evidences,
                            call_id=call_id,
                            global_call_id=global_call_id,
                        )
                        wrote = True
                    else:
                        # Fallback: no structured evidences — archive the tool's
                        # own output blob (use its tagged output, not the joined
                        # next_obs, so we don't mix in other tools' text).
                        obs_for_archive = str(tm.get("output") or "")
                        if obs_for_archive:
                            archive_entry = self._archive.add(
                                turn=len(self._model_io),
                                tool_name=tm.get("name", "tool"),
                                tool_arguments=tm.get("arguments") or {},
                                tool_result=obs_for_archive,
                                call_id=call_id,
                                global_call_id=global_call_id,
                            )
                            archive_entries = [archive_entry]
                            wrote = True
                    if archive_entries:
                        tm["archive_entries"] = archive_entries
                if wrote:
                    set_url = getattr(self.agent, "set_archive_tool_urls", None)
                    if callable(set_url):
                        set_url(self._archive.tool_index_urls)

            graph_tool_parts = [
                str(tm.get("output") or "")
                for tm in tool_meta
                if tm.get("feeds_memory", True)
                and tm.get("name") != "finish"
                and tm.get("output")
            ]
            obs_str = "\n".join(graph_tool_parts)
            context_memory_message = None
            if self._context_memory:
                self._context_memory.observe_tool(content=obs_str, tool_metadata=tool_meta)
                await self._context_memory.maybe_compact()
                context_memory_message = self._context_memory.render_message()

            # In archive mode, feed only turns that just fell out of the raw
            # recent-turn window.  Legacy/non-trimming modes retain the older
            # interval-based graph ingestion behavior.
            _bg_snapshot = None
            _t_graph = 0.0
            if self.belief_client:
                _t_graph_start = _time.time()
                # Tool content fed to the graph: only from memory-feeding tools
                # (exclude read_file etc.). The assistant turn — including its
                # <think> reasoning — is ALWAYS pushed, even on a read_file-only
                # turn, because the belief evolution it expresses is new signal.
                logger.info(
                    "[Workflow] Assistant response (%d chars): %s",
                    len(response), response[:150] + ("..." if len(response) > 150 else ""),
                )
                if obs_str:
                    logger.info(
                        "[Workflow] Tool result fed to graph (%d chars): %s",
                        len(obs_str), obs_str[:150] + ("..." if len(obs_str) > 150 else ""),
                    )
                if self._eviction_aligned_graph:
                    completed_turn = [{
                        "problem_id": self._problem_id,
                        "role": "assistant",
                        "content": response,
                        "is_message_end": True,
                        "is_trajectory_end": False,
                        "timings": {"llm": round(_t_llm, 3)},
                    }]
                    if obs_str:
                        completed_turn.append({
                            "problem_id": self._problem_id,
                            "role": "tool",
                            "content": obs_str,
                            "is_message_end": True,
                            "is_trajectory_end": False,
                            "timings": {"tool": round(_t_tool, 3)},
                        })
                    evicted_payloads = _evict_graph_turn_payloads(
                        self._raw_graph_turn_window,
                        completed_turn,
                        self._graph_raw_turn_limit,
                    )

                    if evicted_payloads:
                        _bg_snapshot = await self.belief_client.push_turns(evicted_payloads)
                        if _bg_snapshot:
                            await self._record_graph_snapshot(_bg_snapshot, phase="evicted_turn")
                        logger.info(
                            "[Workflow] Moved %d evicted raw turn(s) into graph; "
                            "%d recent turn(s) remain verbatim",
                            sum(1 for item in evicted_payloads if item.get("role") == "assistant"),
                            len(self._raw_graph_turn_window),
                        )
                    elif self._graph_raw_turn_limit is None:
                        logger.info(
                            "[Workflow] Raw history is unbounded; completed turn "
                            "remains verbatim and is not added to graph"
                        )
                    else:
                        logger.info(
                            "[Workflow] Keeping completed turn verbatim (%d/%d); "
                            "not yet added to graph",
                            len(self._raw_graph_turn_window), self._graph_raw_turn_limit,
                        )
                elif self._belief_graph_interval == 1:
                    # Wait for the assistant push we kicked off before the tool (it
                    # likely finished during tool execution). This preserves ordering
                    # and gives us the assistant-only snapshot for the no-tool case.
                    _assistant_snapshot = (
                        await _assistant_push_task if _assistant_push_task is not None else None
                    )
                    if obs_str:
                        _bg_snapshot = await self.belief_client.push_turn(
                            problem_id=self._problem_id,
                            role="tool",
                            content=obs_str,
                            is_message_end=True,
                            is_trajectory_end=False,
                            # tool time is only known after env.step; the server preserves it
                            # in trajectory_stream.jsonl on the tool turn for live read.
                            meta={"timings": {"tool": round(_t_tool, 3)}},
                        )
                    else:
                        _bg_snapshot = _assistant_snapshot
                    if _bg_snapshot:
                        await self._record_graph_snapshot(_bg_snapshot, phase="turn")
                else:
                    self._pending_graph_turns.append({
                        "problem_id": self._problem_id,
                        "role": "assistant",
                        "content": response,
                        "is_message_end": True,
                        "is_trajectory_end": False,
                    })
                    if obs_str:
                        self._pending_graph_turns.append({
                            "problem_id": self._problem_id,
                            "role": "tool",
                            "content": obs_str,
                            "is_message_end": True,
                            "is_trajectory_end": False,
                        })

                    turn_index = len(self._model_io)  # 1-based: this turn just recorded
                    should_flush = (
                        turn_index % self._belief_graph_interval == 0 or done
                    )
                    if should_flush:
                        _bg_snapshot = await self.belief_client.push_turns(
                            self._pending_graph_turns
                        )
                        self._pending_graph_turns = []
                        if _bg_snapshot:
                            await self._record_graph_snapshot(_bg_snapshot, phase="turn")
                    else:
                        logger.info(
                            "[Workflow] Buffering turn %d for graph (interval=%d, "
                            "%d turn(s) pending)",
                            turn_index, self._belief_graph_interval, len(self._pending_graph_turns),
                        )
                _t_graph = _time.time() - _t_graph_start

            # Record per-turn timings.
            self._model_io[-1]["timings"] = {
                "llm": round(_t_llm, 3),
                "tool": round(_t_tool, 3),
                "graph": round(_t_graph, 3),
            }
            logger.info("[Workflow] Turn %d timings: llm=%.2fs tool=%.2fs graph=%.2fs",
                        len(self._model_io) - 1, _t_llm, _t_tool, _t_graph)

            if done:
                info["model_io"] = self._model_io
                info["token_usage"] = {
                    "turns": self._token_usage_turns,
                    "total_prompt_tokens": self._total_prompt_tokens,
                    "total_completion_tokens": self._total_completion_tokens,
                    "total_output_tokens": self._total_completion_tokens,
                    "num_model_calls": len(self._token_usage_turns),
                }
                info["graph_problem_id"] = self._problem_id
                if self._context_memory:
                    info["context_memory_mode"] = self._context_memory.mode
                    info["context_memory_history"] = self._context_memory.export_state()

            self.agent.update_from_env(next_obs, reward, done, info)
            if context_memory_message:
                set_memory = getattr(self.agent, "set_context_memory_message", None)
                if callable(set_memory):
                    set_memory(context_memory_message)

            # Inject graph after tool response is appended to messages.
            if _bg_snapshot:
                self.agent.inject_belief_graph(_bg_snapshot, self.belief_client)

            if output.finish_reason == "length":
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

            if done:
                # Finalize and fetch final complete graph
                if self.belief_client:
                    await self.belief_client.push_turn(
                        problem_id=self._problem_id,
                        role="assistant",
                        content="",
                        is_trajectory_end=True,
                    )
                    final_graph = await self.belief_client.get_graph(self._problem_id)
                    if final_graph:
                        await self._record_graph_snapshot(final_graph, phase="final")
                        info["belief_graph_final"] = final_graph
                    info["belief_graph_history"] = self._graph_snapshots
                    self._save_graph_snapshots()
                raise TerminationEvent(TerminationReason.ENV_DONE)

        # Flush any turns buffered for the graph (belief_graph_interval > 1)
        # that never hit a trigger turn before max_steps ran out.
        if self.belief_client and self._pending_graph_turns:
            snapshot = await self.belief_client.push_turns(self._pending_graph_turns)
            self._pending_graph_turns = []
            if snapshot:
                await self._record_graph_snapshot(snapshot, phase="flush")

        final_info = {
            "model_io": self._model_io,
            "token_usage": {
                "turns": self._token_usage_turns,
                "total_prompt_tokens": self._total_prompt_tokens,
                "total_completion_tokens": self._total_completion_tokens,
                "total_output_tokens": self._total_completion_tokens,
                "num_model_calls": len(self._token_usage_turns),
            },
            "graph_problem_id": self._problem_id,
        }
        if self._context_memory:
            final_info["context_memory_mode"] = self._context_memory.mode
            final_info["context_memory_history"] = self._context_memory.export_state()
        self.agent.update_from_env(None, 0, True, final_info)
        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

    def _save_graph_snapshots(self) -> None:
        """Save all graph snapshots to a per-problem JSON file."""
        if not self._graph_snapshots:
            return
        output_dir = Path(self._graph_output_dir) if self._graph_output_dir else Path("belief_graphs")
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_id = str(self._problem_id).replace("/", "_").replace(":", "_")
        fpath = output_dir / f"{safe_id}.json"
        with fpath.open("w", encoding="utf-8") as f:
            json.dump(self._graph_snapshots, f, ensure_ascii=False, indent=2)
        logger.info("[Workflow] Saved %d graph snapshots to %s", len(self._graph_snapshots), fpath)

    async def _record_graph_snapshot(self, snapshot: dict[str, Any], *, phase: str) -> None:
        self._graph_snapshots.append(snapshot)
        await self._sync_graph_snapshot_to_tonggraph(snapshot, phase=phase)

    async def _sync_graph_snapshot_to_tonggraph(self, snapshot: dict[str, Any], *, phase: str) -> None:
        cfg = self._tonggraph_sync_config
        if not cfg.get("enabled"):
            return
        base_url = str(cfg.get("base_url") or "").rstrip("/")
        token = str(cfg.get("token") or "")
        if not base_url or not token:
            logger.warning("[TongGraph] Sync enabled but base_url/token is missing; skipping")
            return

        from bcg.agent.tonggraph_sync import infer_logical_graph_id, sync_graph_payload

        logical_graph_id = infer_logical_graph_id(None, snapshot, str(cfg.get("logical_graph_id") or self._problem_id))
        try:
            result = await asyncio.to_thread(
                sync_graph_payload,
                snapshot,
                base_url=base_url,
                token=token,
                graph=str(cfg.get("graph") or "agent_workspace"),
                logical_graph_id=logical_graph_id,
                text_index=cfg.get("text_index"),
                embedding_url=str(cfg.get("embedding_url") or ""),
                embedding_model=str(cfg.get("embedding_model") or ""),
                embedding_index=cfg.get("embedding_index"),
                embedding_batch_size=int(cfg.get("embedding_batch_size") or 16),
                timeout=float(cfg.get("timeout") or 30.0),
                verify_readback=True,
            )
        except Exception as exc:
            logger.warning("[TongGraph] Graph snapshot sync failed phase=%s: %s", phase, exc)
            return
        logger.info(
            "[TongGraph] Synced latest graph snapshot phase=%s to %s/%s: "
            "nodes=%d created/%d reused/%d stale deleted, "
            "edges=%d created/%d reused/%d stale deleted, full_graph_readback=%s",
            phase,
            result.graph,
            result.logical_graph_id,
            result.nodes_created,
            result.nodes_reused,
            result.nodes_deleted,
            result.edges_created,
            result.edges_reused,
            result.edges_deleted,
            result.readback_verified,
        )

    async def _sync_final_graph_to_tonggraph(self, final_graph: dict[str, Any]) -> None:
        await self._sync_graph_snapshot_to_tonggraph(final_graph, phase="final")


__all__ = ["BeliefTracerWorkflow"]
