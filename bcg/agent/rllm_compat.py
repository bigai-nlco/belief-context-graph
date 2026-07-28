"""Compatibility agent/environment for the current rLLM workflow API."""

from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass
from typing import Any

from bcg.agent.context_memory import BELIEF_GRAPH_MODE, is_context_memory_baseline
from bcg.agent.tokenizer_compat import get_tool_parser_compat
from bcg.agent.tool_call_protocol import (
    canonicalize_tool_call_text,
    parse_tool_call_blocks,
    render_tool_results_xml,
)
from rllm.agents.agent import BaseAgent
from rllm.environments.base.base_env import BaseEnv
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import ToolCall, ToolOutput
from rllm.agents.agent import Action, Step, Trajectory


@dataclass
class LabeledToolCall(ToolCall):
    """A ``ToolCall`` tagged with its protocol-level ``call_id`` (e.g.
    ``"call_1"``), assigned by ``BeliefTracerAgent.update_from_model`` in the
    order ``<tool_call>`` blocks appear in that turn's text (or, when no
    textual blocks are present -- e.g. a local vllm/sglang backend whose chat
    template parser strips them out of ``content`` -- in the order the
    backend's own structured ``tool_calls`` list reports them).

    Still a plain dataclass subclass of ``ToolCall``, so existing
    ``isinstance(call, ToolCall)`` checks (e.g. in
    ``BeliefTracerEnvironment.step``) keep working unchanged.
    """

    id: str = ""


# Resolve the real field name for extra info on rllm's Step.
# Pydantic v2 models expose fields via model_fields; dataclasses via
# dataclasses.fields.  The constructor param must be the real field name
# ("metadata"), NOT the @property alias ("info").
try:
    _STEP_FIELDS = set(Step.model_fields.keys())
except Exception:
    try:
        import dataclasses as _dc
        _STEP_FIELDS = {f.name for f in _dc.fields(Step)}
    except Exception:
        _STEP_FIELDS = set()


class BeliefTracerAgent(BaseAgent):
    """Small chat agent compatible with rLLM's legacy MultiTurnWorkflow."""

    def __init__(
        self,
        system_prompt: str = "",
        parser_name: str = "qwen",
        model: str | None = None,
        tools: list[str] | None = None,
        tool_map: dict[str, Any] | None = None,
        enable_thinking: bool = False,
        tool_prompt: str = "",
        tool_prompt_brief: str = "",
        user_rules_prompt: str = "",
        belief_graph_mode: str = "augment",
        graph_format: str = "structured",
        deepseek_v4_payload_format: str = "json",
        graph_include_relations: bool = True,
        belief_graph_placement: str = "user",
        layered_context: bool = False,
        archive_enabled: bool = False,
        recent_turns: int = 0,
        context_memory_mode: str = BELIEF_GRAPH_MODE,
        **_: Any,
    ) -> None:
        self.context_memory_mode = context_memory_mode or BELIEF_GRAPH_MODE
        # Layered context: belief graph becomes a standalone message after the
        # first user turn, tool usage goes into the user message, and (optionally)
        # older turns are archived out.
        self.layered_context = bool(
            layered_context or archive_enabled or is_context_memory_baseline(self.context_memory_mode)
        )
        self._tool_prompt = tool_prompt or ""
        # Brief tool listing (system prompt) and detailed AVeriTeC rules (user
        # message) are only used in layered mode.
        self._tool_prompt_brief = tool_prompt_brief or ""
        self._user_rules_prompt = user_rules_prompt or ""
        if self.layered_context:
            # System prompt stays minimal (base rules only); tool usage is moved
            # into the user message (see update_from_env).
            self.system_prompt = system_prompt or ""
        else:
            # Legacy: tool usage is appended to the system prompt.
            self.system_prompt = _join_prompt_parts(system_prompt, tool_prompt)
        self.parser_name = parser_name or "qwen"
        self.tools = list(tools or [])
        self.tool_map = dict(tool_map or {})
        self.enable_thinking = bool(enable_thinking)
        self.belief_graph_mode = belief_graph_mode
        self.graph_format = graph_format
        if deepseek_v4_payload_format not in {"json", "xml", "markdown"}:
            deepseek_v4_payload_format = "json"
        self.deepseek_v4_payload_format = deepseek_v4_payload_format
        self.graph_include_relations = graph_include_relations
        if belief_graph_placement not in {"user", "system"}:
            belief_graph_placement = "user"
        self.belief_graph_placement = belief_graph_placement
        self.archive_enabled = bool(archive_enabled)
        parsed_recent_turns = int(recent_turns)
        self.recent_turns = -1 if parsed_recent_turns < 0 else parsed_recent_turns
        self._parser = get_tool_parser_compat(self.parser_name, model=model)
        self.reset()

    def reset(self) -> None:
        self._messages: list[dict[str, Any]] = []
        self._trajectory = Trajectory()
        self._pending_model_response = ""
        self._pending_action: Any = None
        self._pending_model_output: Any = None
        self._pending_reasoning = ""
        self._pending_raw_assistant_output = ""
        self._pending_canonical_assistant_output = ""
        self._base_system_prompt = self.system_prompt
        self._belief_graph_text = ""
        self._context_memory_message: dict[str, Any] | None = None
        self._archive_tool_urls: dict[str, str] = {}

    @property
    def chat_completions(self) -> list[dict[str, Any]]:
        if self.layered_context:
            return self._chat_completions_layered()

        mode = self.belief_graph_mode if self.context_memory_mode == BELIEF_GRAPH_MODE else "none"

        if mode == "only":
            # system (with graph embedded) + first user question
            result = []
            for m in self._messages:
                if m.get("role") == "system":
                    result.append(_message_for_model(m))
                    break
            for m in self._messages:
                if m.get("role") == "user":
                    result.append(_message_for_model(m))
                    break
            return result

        # "none" / "augment": full conversation history
        return [_message_for_model(m) for m in self._messages]

    def _chat_completions_layered(self) -> list[dict[str, Any]]:
        """Layered assembly: system(rules+guide+archive refs) -> first user ->
        optional belief-graph message -> recent conversation.

        By default, the belief graph is a standalone user message right after
        the question, taking the summary slot. ``belief_graph_placement=system``
        appends the same block to the system prompt instead.
        """
        if not self._messages:
            return []

        system_msg = {"role": "system", "content": self._compose_system_prompt()}

        # First user message (the question) stays fixed.
        first_user_idx = next(
            (i for i, m in enumerate(self._messages) if m.get("role") == "user"), None
        )
        out: list[dict[str, Any]] = [system_msg]
        if first_user_idx is not None:
            out.append(_message_for_model(self._messages[first_user_idx]))

        if self.context_memory_mode != BELIEF_GRAPH_MODE:
            if self._context_memory_message:
                out.append(_message_for_model(self._context_memory_message))
        # Belief graph standalone message (role=user), only when requested.
        elif (
            self.belief_graph_mode != "none"
            and self.belief_graph_placement == "user"
            and self._belief_graph_text
        ):
            out.append({
                "role": "user",
                "content": self._belief_graph_block(),
            })

        if self.context_memory_mode == BELIEF_GRAPH_MODE and self.belief_graph_mode == "only":
            # Belief graph replaces the intermediate conversation entirely.
            return out

        # Remaining conversation (everything after the first user turn).
        rest = (
            self._messages[first_user_idx + 1:]
            if first_user_idx is not None
            else list(self._messages)
        )
        rest = [m for m in rest if m.get("role") != "system"]

        # Optional trimming: keep only the last N turns (1 turn = assistant + its
        # tool result). Zero keeps no raw turns; -1 keeps the full raw history.
        if (
            self.belief_graph_mode != "none"
            and self.archive_enabled
            and self.recent_turns >= 0
        ):
            rest = self._keep_recent_turns(rest, self.recent_turns)

        out.extend(_message_for_model(m) for m in rest)
        return out

    @staticmethod
    def _keep_recent_turns(messages: list[dict[str, Any]], n_turns: int) -> list[dict[str, Any]]:
        """Keep the last ``n_turns`` assistant-led turns from the tail.

        A turn starts at an assistant message and includes any following tool
        messages. Older turns are dropped (they live in the archive instead).
        """
        if n_turns < 0:
            return messages
        if n_turns == 0:
            return []
        # Indices where a new turn begins (assistant messages).
        starts = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if len(starts) <= n_turns:
            return messages
        cut = starts[-n_turns]
        return messages[cut:]

    def _compose_system_prompt(self) -> str:
        """Assemble the layered system prompt: base rules + guide + archive refs
        + a brief tool listing.

        Belief graph placement is controlled by ``belief_graph_placement``.
        The context-blocks guide explains the archive references, and is added
        only when archiving is active.
        """
        parts = [self._base_system_prompt]

        guide_lines: list[str] = []
        if self.context_memory_mode == BELIEF_GRAPH_MODE and self.belief_graph_mode != "none":
            if self.graph_format == "deepseek_v4":
                guide_lines.append(
                    "- DeepSeek-V4 encoded dialogue context holds preliminary beliefs "
                    "from earlier turns. They are NOT verified evidence — use them to "
                    "decide what to search next, and always confirm with the search tool."
                )
            else:
                guide_lines.append(
                    "- A <belief_graph> block holds preliminary beliefs from earlier "
                    "turns. They are NOT verified evidence — use them to decide what "
                    "to search next, and always confirm with the search tool."
                )
        elif self.context_memory_mode != "none":
            guide_lines.append(
                "- A context-memory message (right after the question) holds compressed "
                "prior work for the selected baseline. It is a planning aid, not verified "
                "evidence; raw tool evidence and archive raw_url content are stronger."
            )
        if self.archive_enabled:
            guide_lines.append(
                "- <archive_tool_indexes> below lists per-tool archive URLs. Each "
                "file groups queries with summaries and raw_url pointers. To recall "
                "details, call read_file on a tool's index URL to see query summaries, "
                "then read_file on a result's raw_url for its full content."
            )
        if guide_lines:
            parts.append(
                "<context_blocks_guide>\n"
                "Your context may contain these blocks:\n"
                + "\n".join(guide_lines)
                + "\nRaw evidence > archive manifest summary > context-memory summary."
                "\n</context_blocks_guide>"
            )

        if self.archive_enabled and self._archive_tool_urls:
            lines = "\n".join(
                f"{name}: {url}" for name, url in self._archive_tool_urls.items()
            )
            parts.append(
                "<archive_tool_indexes>\n"
                f"{lines}\n"
                "</archive_tool_indexes>"
            )

        # Brief tool listing (JSON shape + arg names only). The detailed usage
        # with examples lives in the user message.
        if self._tool_prompt_brief:
            parts.append(self._tool_prompt_brief)
        if (
            self.context_memory_mode == BELIEF_GRAPH_MODE
            and self.belief_graph_mode != "none"
            and self.belief_graph_placement == "system"
            and self._belief_graph_text
        ):
            parts.append(self._belief_graph_block())
        return "\n\n".join(p for p in parts if p)

    def _belief_graph_block(self) -> str:
        if self.graph_format == "deepseek_v4":
            return self._belief_graph_text
        return f"<belief_graph>\n{self._belief_graph_text}\n</belief_graph>"

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def update_from_env(
        self, observation: Any, reward: float, done: bool, info: dict, **_: Any
    ) -> None:
        if not self._messages:
            if self.system_prompt:
                self._messages.append({"role": "system", "content": self.system_prompt})
            question = ""
            if isinstance(observation, dict):
                question = str(observation.get("question") or observation.get("prompt") or "")
            else:
                question = str(observation or "")
            # In layered mode, the detailed AVeriTeC rules (label definitions,
            # search policy, decision rules) and the detailed tool usage (with
            # good/bad query examples) live in the user message, appended after
            # the claim/metadata/label-space. The system prompt stays minimal.
            if self.layered_context:
                question = _join_prompt_parts(
                    question, self._user_rules_prompt
                )
            self._messages.append({"role": "user", "content": question})
            return

        if observation not in (None, "", {}, []):
            self._messages.append({"role": "tool", "content": str(observation)})

        step_kwargs = dict(
            chat_completions=list(self._messages),
            observation=observation,
            action=self._pending_action,
            model_response=self._pending_model_response,
            model_output=self._pending_model_output,
            thought=self._pending_reasoning,
            reward=float(reward),
            done=bool(done),
        )
        # rllm's Step stores extra info under either `metadata` or `info`
        # depending on version; pick whichever this build supports.
        info_field = "metadata" if "metadata" in _STEP_FIELDS else "info"
        step_kwargs[info_field] = dict(info or {})
        self._trajectory.steps.append(Step(**step_kwargs))

    def update_from_model(
        self, response: str, model_output: Any = None, **_: Any
    ) -> Action:
        response = response or ""
        self._pending_model_response = response
        self._pending_model_output = model_output

        raw_text = getattr(model_output, "text", None) if model_output else response
        raw_tool_calls = getattr(model_output, "tool_calls", None) if model_output else None
        structured_tool_calls = _normalize_tool_calls(raw_tool_calls)
        parsed_tool_calls = self._parser.parse(response)
        content, reasoning = _normalize_response_parts(
            raw_text or response,
            content=getattr(model_output, "content", None) if model_output else None,
            reasoning=getattr(model_output, "reasoning", None) if model_output else None,
            enable_thinking=self.enable_thinking,
            tool_call_begin=getattr(self._parser, "tool_call_begin", "<tool_call>"),
        )
        self._pending_reasoning = reasoning
        # Exposed for workflow.py's per-turn model_io logging: the model's
        # unmodified output text, and the canonicalized (id-tagged) version
        # of `content` that actually gets stored in _messages below.
        self._pending_raw_assistant_output = raw_text or response

        # Protocol layer: assign each <tool_call> block in `content` a local
        # call_id (call_1, call_2, ... in parse order, restarting every turn)
        # and rewrite the opening tags to carry it explicitly. This is the
        # text that gets stored in _messages and replayed to the model next
        # turn, so every turn the model sees is already in canonical form --
        # "raw" vs "canonical" is a distinction the caller (workflow.py) reads
        # off model_output/this message pair for logging, not something this
        # method threads through separately.
        parsed_blocks = parse_tool_call_blocks(content)
        canonical_content = canonicalize_tool_call_text(content, parsed_blocks)
        valid_blocks = [b for b in parsed_blocks if b.format_error is None]

        # The turn's own text is the ground truth for (name, arguments, id)
        # whenever it carries parseable <tool_call> blocks -- this covers the
        # api backend (content always carries the literal blocks) and any
        # backend whose chat-template parser leaves them in place. Only fall
        # back to numbering the backend's structured tool_calls list in
        # parse order when no textual blocks were found at all (e.g. a local
        # vllm/sglang backend whose chat-template parser already stripped
        # <tool_call> out of `content`, leaving pure ToolCall objects).
        if valid_blocks:
            labeled_tool_calls = [
                LabeledToolCall(name=block.name, arguments=block.arguments, id=block.id)
                for block in valid_blocks
            ]
        elif structured_tool_calls:
            labeled_tool_calls = [
                LabeledToolCall(name=call.name, arguments=call.arguments or {}, id=f"call_{i}")
                for i, call in enumerate(structured_tool_calls, start=1)
            ]
        else:
            labeled_tool_calls = []

        self._pending_canonical_assistant_output = canonical_content

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": canonical_content,
        }
        if reasoning:
            assistant_message["reasoning_content"] = reasoning
        if raw_text and raw_text != assistant_message["content"]:
            assistant_message["raw_content"] = raw_text

        if labeled_tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments or {},
                    },
                }
                for call in labeled_tool_calls
            ]

        self._messages.append(assistant_message)

        raw_action: Any = labeled_tool_calls or structured_tool_calls or parsed_tool_calls or content or response
        self._pending_action = Action(action=raw_action)
        return self._pending_action

    def inject_belief_graph(self, snapshot: dict, client) -> None:
        """Update the belief graph context with the latest snapshot.

        Layered mode: store the graph text in a standalone slot (rendered as its
        own message by ``_chat_completions_layered``); the system prompt is left
        untouched. Legacy mode: embed the graph into the system message (original
        behavior, kept for regression).
        """
        from logging import getLogger
        _logger = getLogger(__name__)
        if self.context_memory_mode != BELIEF_GRAPH_MODE:
            _logger.info(
                "[Agent] Ignoring belief graph injection because context_memory_mode=%s",
                self.context_memory_mode,
            )
            return
        graph_text = client.format_graph_for_prompt(
            snapshot, fmt=self.graph_format,
            include_relations=self.graph_include_relations,
            deepseek_v4_payload_format=self.deepseek_v4_payload_format,
        )
        n_relations = len(
            snapshot.get("forward_relations") or snapshot.get("relations") or []
        )

        if self.layered_context:
            # Keep only the latest snapshot (no history accumulation).
            self._belief_graph_text = graph_text or ""
            _logger.info(
                "[Agent] Belief graph %s context updated (%d beliefs, %d relations)",
                self.belief_graph_placement,
                len(snapshot.get("beliefs", [])),
                n_relations,
            )
            return

        # Legacy: embed in system message.
        if not self._messages or self._messages[0].get("role") != "system":
            return
        if graph_text:
            self._belief_graph_text = graph_text
            self._messages[0]["content"] = _join_prompt_parts(
                self._base_system_prompt, self._belief_graph_block()
            )
        else:
            self._belief_graph_text = ""
            self._messages[0]["content"] = self._base_system_prompt
        _logger.info(
            "[Agent] Injecting belief graph into system prompt (%d beliefs, %d relations)",
            len(snapshot.get("beliefs", [])),
            n_relations,
        )

    def set_archive_tool_urls(self, urls: dict[str, str]) -> None:
        """Point the layered system prompt at the per-tool archive indexes."""
        self._archive_tool_urls = dict(urls or {})

    def set_context_memory_message(self, message: dict[str, Any] | None) -> None:
        """Set the replacement context-memory slot used by layered baselines."""
        self._context_memory_message = dict(message) if message else None


class BeliefTracerEnvironment(BaseEnv):
    """QA environment with finish-tool scoring and optional rLLM tools."""

    def __init__(
        self,
        reward_fn,
        max_steps: int = 10,
        tools: list[str] | None = None,
        tool_map: dict[str, Any] | None = None,
        max_tool_workers: int = 1,
        **_: Any,
    ) -> None:
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.multi_tool = MultiTool(tools=tools, tool_map=tool_map) if (tools or tool_map) else None
        self.task: dict[str, Any] = {}
        self.num_steps = 0
        # >1 runs a turn's independent (non-finish) tool calls concurrently in
        # a thread pool. Default 1 preserves the historical sequential order
        # exactly (single-worker pool == same call order, no behavior change).
        self._max_tool_workers = max(1, int(max_tool_workers or 1))

    def reset(self, task: dict | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.task = dict(task or {})
        self.num_steps = 0
        self._set_tool_task(self.task)
        return {"question": self.task.get("question", "")}, {}

    def _set_tool_task(self, task: dict[str, Any]) -> None:
        if self.multi_tool is None:
            return
        for tool in getattr(self.multi_tool, "tool_map", {}).values():
            set_task = getattr(tool, "set_task", None)
            if callable(set_task):
                set_task(task)

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        self.num_steps += 1
        raw_action = action.action if isinstance(action, Action) else action

        if isinstance(raw_action, list):
            final_answer = None
            pending_calls: list[ToolCall] = []
            for call in raw_action:
                if isinstance(call, ToolCall) and call.name == "finish":
                    final_answer = _extract_finish_answer(call.arguments)
                    break
                pending_calls.append(call)

            # Every call reaching here should already carry a protocol-level
            # call_id from BeliefTracerAgent.update_from_model (LabeledToolCall).
            # A plain ToolCall without one only reaches this path via the
            # legacy self._parser.parse() fallback (update_from_model's last
            # resort, used only when neither <tool_call> text nor a backend's
            # structured tool_calls were found) -- number those by position so
            # every call still gets a usable id.
            pending_calls = [
                call if getattr(call, "id", "") else LabeledToolCall(
                    name=call.name, arguments=call.arguments or {}, id=f"call_{i}",
                )
                for i, call in enumerate(pending_calls, start=1)
            ]

            result_entries: list[dict[str, Any]] = []
            tool_metadata: list[dict[str, Any]] = []
            if pending_calls:
                # Non-finish calls in this turn run concurrently in a thread
                # pool (max_tool_workers=1, the default, makes this a no-op:
                # a single-worker pool executes them one at a time, in order,
                # identical to the old sequential for-loop). pool.map keeps
                # results aligned with pending_calls' input order regardless
                # of which call finishes first, so downstream archiving/belief
                # graph updates see the same ordering as before.
                #
                # (turn, call_index) is fixed here, BEFORE dispatch — this turn's
                # position and each call's position within it are known upfront,
                # unlike a shared counter tools would otherwise have to race on.
                # Passed via _run_tool so tools can label debug output without
                # needing their own cross-call coordination.
                max_workers = min(len(pending_calls), self._max_tool_workers)
                call_args = [
                    (call, self.num_steps, i) for i, call in enumerate(pending_calls, start=1)
                ]
                if max_workers > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                        results = list(pool.map(self._run_tool_labeled, call_args))
                else:
                    results = [self._run_tool_labeled(a) for a in call_args]

                for call, out in zip(pending_calls, results):
                    call_id = getattr(call, "id", "") or ""
                    args = getattr(call, "arguments", {}) or {}
                    # Surface structured tool metadata (e.g. per-evidence list) so
                    # the workflow can archive each evidence separately. Also tag
                    # each call's own output text and whether it feeds memory
                    # (archive + belief graph), so the workflow can route by tool
                    # capability instead of hardcoding tool names.
                    md = getattr(out, "metadata", None)
                    md = md if isinstance(md, dict) else {}
                    tool_obj = None
                    if self.multi_tool is not None:
                        tool_obj = getattr(self.multi_tool, "tool_map", {}).get(
                            getattr(call, "name", "")
                        )
                    feeds_memory = getattr(tool_obj, "FEEDS_MEMORY", True)
                    tool_metadata.append({
                        "tool_call_id": call_id,
                        "name": getattr(out, "name", getattr(call, "name", "")),
                        "arguments": args,
                        "metadata": md,
                        "output": str(out),
                        "feeds_memory": bool(feeds_memory),
                    })

                    # Evidence list for the per-call <tool_result> block: prefer
                    # the tool's own structured evidences (averitec_search's
                    # per-chunk list); fall back to the tool's raw text output
                    # as a single entry so tools without an evidences shape
                    # (read_file, errors) still surface their content instead
                    # of silently disappearing.
                    evidences = md.get("evidences") or []
                    if not evidences:
                        fallback_text = str(out)
                        evidences = [{"text": fallback_text}] if fallback_text else []
                    query = str(
                        args.get("query") or args.get("url") or args.get("path") or ""
                    )
                    result_entries.append({
                        "tool_call_id": call_id,
                        "name": getattr(out, "name", getattr(call, "name", "")),
                        "query": query,
                        "evidence": evidences,
                    })
            if final_answer is not None:
                return self._score(final_answer)
            done = self.num_steps >= self.max_steps
            next_obs = render_tool_results_xml(result_entries)
            return next_obs, 0.0, done, {"tool_metadata": tool_metadata}

        return self._score(str(raw_action or ""))

    def _run_tool_labeled(self, call_and_position: tuple[Any, int, int]) -> ToolOutput:
        """Stamp (turn, call_index) on the tool instance's thread-local slot
        (see HerOSearchTool._call_label) before invoking it, so tools that log
        per-call debug output can label it without racing on shared state.
        Runs inside the worker thread — thread-local, so concurrent calls in
        the same turn don't clobber each other's label.
        """
        call, turn, call_index = call_and_position
        if self.multi_tool is not None:
            tool_obj = getattr(self.multi_tool, "tool_map", {}).get(
                getattr(call, "name", "")
            )
            set_label = getattr(tool_obj, "set_call_label", None)
            if callable(set_label):
                set_label(turn, call_index)
        return self._run_tool(call)

    def _run_tool(self, call: Any) -> ToolOutput:
        if not isinstance(call, ToolCall):
            return ToolOutput(name="unknown", error=f"Unsupported action: {call!r}")
        if self.multi_tool is None:
            return ToolOutput(name=call.name, error="No tools configured")
        args = call.arguments or {}
        if "arguments" in args and isinstance(args["arguments"], dict):
            args = args["arguments"]
        return self.multi_tool.forward(tool_name=call.name, **args)

    def _score(self, answer: str) -> tuple[str, float, bool, dict]:
        result = self.reward_fn(self.task, answer)
        info = {
            "is_correct": bool(getattr(result, "is_correct", False)),
            "metadata": dict(getattr(result, "metadata", {}) or {}),
        }
        return "", float(getattr(result, "reward", 0.0)), True, info

    @staticmethod
    def from_dict(info: dict) -> "BeliefTracerEnvironment":
        return BeliefTracerEnvironment(**info)


def _message_for_model(message: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "images",
    }
    return {
        key: value
        for key, value in message.items()
        if key in allowed and value is not None
    }


def _join_prompt_parts(*parts: str | None) -> str:
    return "\n\n".join(str(part).strip() for part in parts if str(part or "").strip())


def _extract_finish_answer(arguments: dict[str, Any] | None) -> str:
    args = arguments or {}
    for key in ("answer", "final_answer", "response", "result"):
        if key in args:
            return str(args[key])
    return str(args)


def _normalize_response_parts(
    response: str,
    *,
    content: Any,
    reasoning: Any,
    enable_thinking: bool,
    tool_call_begin: str,
) -> tuple[str, str]:
    response_text = str(response or "")
    content_text = "" if content is None else str(content)
    reasoning_text = "" if reasoning is None else str(reasoning)

    if not enable_thinking:
        return content_text if content is not None else response_text, reasoning_text

    split = _split_thinking_text(response_text, tool_call_begin=tool_call_begin)
    if split is None:
        if content_text:
            return content_text, reasoning_text
        if reasoning_text == response_text:
            return response_text, ""
        return response_text, reasoning_text

    split_reasoning, split_content = split
    if _reasoning_needs_repair(reasoning_text, split_reasoning, response_text):
        reasoning_text = split_reasoning
    if _content_needs_repair(content, content_text, split_content):
        # Keep the chat history action-focused. The full raw model output is
        # still saved separately as raw_content for UI/debugging.
        content_text = split_content
    return content_text, reasoning_text


def _split_thinking_text(
    text: str,
    *,
    tool_call_begin: str = "<tool_call>",
) -> tuple[str, str] | None:
    start_token = "<think>"
    end_token = "</think>"
    start = text.find(start_token)
    end = text.find(end_token)

    if start >= 0 and (end == -1 or start < end):
        prefix = text[:start].strip()
        after_start = text[start + len(start_token) :]
        if end >= 0:
            reasoning = text[start + len(start_token) : end].strip()
            content = text[end + len(end_token) :].strip()
        else:
            tool_start = after_start.find(tool_call_begin) if tool_call_begin else -1
            if tool_start >= 0:
                reasoning = after_start[:tool_start].strip()
                content = after_start[tool_start:].strip()
            else:
                reasoning = after_start.strip()
                content = ""
        if prefix:
            content = f"{prefix}\n{content}".strip()
        return reasoning, content

    if end >= 0:
        return text[:end].strip(), text[end + len(end_token) :].strip()

    return None


def _reasoning_needs_repair(
    reasoning: str,
    split_reasoning: str,
    response: str,
) -> bool:
    return bool(split_reasoning) and (
        not reasoning
        or reasoning == response
        or "<think>" in reasoning
        or "</think>" in reasoning
    )


def _content_needs_repair(content: Any, content_text: str, split_content: str) -> bool:
    return (
        content is None
        or (not content_text and bool(split_content))
        or "<think>" in content_text
        or "</think>" in content_text
    )


def _normalize_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    if not raw_tool_calls:
        return []

    tool_calls: list[ToolCall] = []
    for raw_call in raw_tool_calls:
        call = _normalize_tool_call(raw_call)
        if call is not None:
            tool_calls.append(call)
    return tool_calls


def _normalize_tool_call(raw_call: Any) -> ToolCall | None:
    if isinstance(raw_call, ToolCall):
        return raw_call

    function = _get_field(raw_call, "function")
    source = function if function is not None else raw_call
    name = _get_field(source, "name")
    arguments = _get_field(source, "arguments")

    if not isinstance(name, str) or not name:
        return None

    return ToolCall(name=name, arguments=_normalize_tool_arguments(arguments))


def _get_field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


__all__ = ["BeliefTracerAgent", "BeliefTracerEnvironment"]
