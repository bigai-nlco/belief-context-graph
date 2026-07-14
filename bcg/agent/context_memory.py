"""Context-memory baselines that can replace Belief Graph prompt context.

These classes deliberately avoid Belief Graph state. They consume only the
conversation, tool observations, and optional archive references, then render a
standalone user message for the same layered prompt slot that normally holds
``<belief_graph>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
import logging
from typing import Any, Protocol


logger = logging.getLogger(__name__)

BELIEF_GRAPH_MODE = "belief_graph"
NO_CONTEXT_MEMORY_MODE = "none"
CONTEXT_MEMORY_BASELINE_MODES = {
    "claude_pipeline",
    "codex_handoff",
    "opencode_marker",
}
CONTEXT_MEMORY_MODES = {
    BELIEF_GRAPH_MODE,
    NO_CONTEXT_MEMORY_MODE,
    *CONTEXT_MEMORY_BASELINE_MODES,
}


CODEX_SUMMARY_PREFIX = (
    "Another context compaction step produced a handoff summary for the model "
    "that will continue this verification task. Use it to build on previous "
    "work and avoid duplicating tool calls:"
)

CLAUDE_SUMMARIZER_SYSTEM_PROMPT = (
    "You are a conversation summarizer. Be concise but preserve important details."
)

CLAUDE_SUMMARIZER_USER_PROMPT = """Summarize the verification conversation so far. First write a private <analysis> of what must be preserved, then output only a <summary> block with the required sections.

Preserve:
- the original claim/question and task constraints
- key decisions and current verdict hypothesis
- search queries already tried
- evidence that supports, refutes, or complicates the claim
- exact URLs, archive raw_url values, tool call ids, dates, names, numbers, and error strings
- unresolved questions and the next evidence to seek

<summary>
## Objective
...
## Current Verdict Hypothesis
...
## Evidence Kept
...
## Evidence Refs
...
## Discarded Or Snipped Context
...
## Constraints
...
## Next Focus
...
</summary>"""

CODEX_HANDOFF_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the verification task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, evidence snippets, search queries, dates, URLs, archive raw_url values, or tool call ids needed to continue

For fact verification, explicitly separate:
- evidence that supports the claim
- evidence that refutes the claim
- unresolved or conflicting evidence
- current verdict hypothesis, if any

Be concise, structured, and focused on helping the next LLM seamlessly continue the work."""

OPENCODE_SUMMARY_TEMPLATE = """Output exactly the Markdown structure shown inside <template> and keep the section order unchanged. Do not include the <template> tags in your response.
<template>
## Objective
- [one or two brief sentences describing the claim/question and what must be verified]

## Important Details
- [constraints/preferences, decisions and why, exact dates/names/numbers/URLs/tool ids needed to continue, or "(none)"]

## Evidence State
### Supports
- [evidence supporting the claim, with source/archive refs; otherwise "(none)"]

### Refutes
- [evidence refuting the claim, with source/archive refs; otherwise "(none)"]

### Unresolved
- [conflicts, missing facts, or weak evidence; otherwise "(none)"]

## Work State
### Completed
- [finished searches, verified facts, labels ruled in/out, or "(none)"]

### Active
- [current hypothesis, partial tool work, or investigation state; otherwise "(none)"]

### Blocked
- [blockers, failed searches, failing commands, unavailable pages, or "(none)"]

## Next Move
1. [immediate concrete search/verification action, or "(none)"]
2. [next action if known, or "(none)"]

## Relevant Files
- [archive index/raw_url/path: why it matters, or "(none)"]
</template>

Rules:
- Keep every section, even when empty.
- Use terse bullets, not prose paragraphs.
- Preserve exact file paths, symbols, commands, error strings, URLs, dates, names, numbers, and identifiers when known.
- Do not mention the summary process or that context was compacted."""

OPENCODE_UPDATE_PREFIX = """Update the anchored summary below using the conversation history above.
Preserve still-true details, remove stale details, and merge in the new facts.
<previous-summary>
{previous_summary}
</previous-summary>"""

CONTEXT_MEMORY_PROMPTS: dict[str, dict[str, str]] = {
    "claude_pipeline": {
        "system": CLAUDE_SUMMARIZER_SYSTEM_PROMPT,
        "user": CLAUDE_SUMMARIZER_USER_PROMPT,
    },
    "codex_handoff": {
        "user": CODEX_HANDOFF_PROMPT,
        "summary_prefix": CODEX_SUMMARY_PREFIX,
    },
    "opencode_marker": {
        "new": "Create a new anchored summary from the conversation history.",
        "update": OPENCODE_UPDATE_PREFIX,
        "template": OPENCODE_SUMMARY_TEMPLATE,
    },
}


def is_context_memory_baseline(mode: str) -> bool:
    return str(mode or BELIEF_GRAPH_MODE) in CONTEXT_MEMORY_BASELINE_MODES


def uses_belief_graph_service(context_memory_mode: str, belief_graph_mode: str) -> bool:
    """Return whether this run should instantiate/call the Belief Graph service."""
    return (
        str(context_memory_mode or BELIEF_GRAPH_MODE) == BELIEF_GRAPH_MODE
        and str(belief_graph_mode or "none") != "none"
    )


def context_memory_prompt_templates(mode: str) -> dict[str, str]:
    """Return the source-inspired prompt templates used for auditability."""
    return dict(CONTEXT_MEMORY_PROMPTS.get(str(mode or ""), {}))


@dataclass(frozen=True)
class ContextMemoryConfig:
    mode: str = BELIEF_GRAPH_MODE
    recent_observations: int = 3
    tail_turns: int = 2
    max_chars: int = 8000
    tool_summary_chars: int = 200
    interval: int = 1
    summarizer: str = "local"
    summarizer_model: str = ""
    summarizer_base_url: str = ""
    summarizer_api_key: str = "EMPTY"
    summarizer_max_tokens: int = 2048
    summarizer_timeout: float = 120.0
    summarizer_failure_limit: int = 3
    log_preview_chars: int = 0


class ContextMemory(Protocol):
    mode: str

    def observe_initial(self, *, system: str, question: str) -> None: ...

    def observe_assistant(self, *, content: str, model_io: dict[str, Any] | None = None) -> None: ...

    def observe_tool(self, *, content: str, tool_metadata: list[dict[str, Any]]) -> None: ...

    async def maybe_compact(self) -> None: ...

    def render_message(self) -> dict[str, str] | None: ...

    def export_state(self) -> dict[str, Any]: ...


@dataclass
class ToolObservation:
    turn: int
    name: str
    call_id: str
    query: str
    summary: str
    raw_content: str
    archive_entries: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AssistantObservation:
    turn: int
    content: str


class BaseContextMemory:
    mode = ""

    def __init__(self, config: ContextMemoryConfig) -> None:
        self.config = config
        self.question = ""
        self.system = ""
        self.assistant_turns: list[AssistantObservation] = []
        self.tool_observations: list[ToolObservation] = []
        self.compactions: list[dict[str, Any]] = []
        self._turn = 0
        self.running_summary = ""
        self._last_compacted_turn = 0
        self._last_compacted_observation_count = 0
        self._summarizer_failures = 0
        self._summarizer_disabled = False
        self._last_summarizer_error = ""
        self._last_effective_summarizer = "local"

    def observe_initial(self, *, system: str, question: str) -> None:
        self.system = str(system or "")
        self.question = str(question or "")
        logger.info(
            "[ContextMemory] init mode=%s summarizer=%s interval=%s max_chars=%s "
            "recent_observations=%s tail_turns=%s question_chars=%d system_chars=%d",
            self.mode,
            self.config.summarizer,
            self.config.interval,
            self.config.max_chars,
            self.config.recent_observations,
            self.config.tail_turns,
            len(self.question),
            len(self.system),
        )

    def observe_assistant(self, *, content: str, model_io: dict[str, Any] | None = None) -> None:
        self._turn += 1
        self.assistant_turns.append(
            AssistantObservation(turn=self._turn, content=_clean_text(content))
        )
        logger.debug(
            "[ContextMemory] observe_assistant mode=%s turn=%d chars=%d total_assistant_turns=%d",
            self.mode,
            self._turn,
            len(str(content or "")),
            len(self.assistant_turns),
        )

    def observe_tool(self, *, content: str, tool_metadata: list[dict[str, Any]]) -> None:
        before = len(self.tool_observations)
        for tm in tool_metadata or []:
            if tm.get("name") == "finish" or not tm.get("feeds_memory", True):
                continue
            md = tm.get("metadata") or {}
            args = tm.get("arguments") or {}
            output = str(tm.get("output") or "")
            query = str(md.get("query") or args.get("query") or args.get("path") or "")
            archive_entries = list(tm.get("archive_entries") or [])
            summary = _tool_summary(
                tm,
                max_chars=self.config.tool_summary_chars,
                fallback=output,
            )
            self.tool_observations.append(
                ToolObservation(
                    turn=self._turn,
                    name=str(tm.get("name") or "tool"),
                    call_id=str(tm.get("tool_call_id") or ""),
                    query=query,
                    summary=summary,
                    raw_content=_clean_text(output or content),
                    archive_entries=archive_entries,
                )
            )
        added = len(self.tool_observations) - before
        if added:
            recent = self.tool_observations[-added:]
            logger.info(
                "[ContextMemory] observe_tool mode=%s turn=%d added=%d total=%d tools=%s",
                self.mode,
                self._turn,
                added,
                len(self.tool_observations),
                [
                    {
                        "name": obs.name,
                        "call_id": obs.call_id,
                        "query_chars": len(obs.query),
                        "summary_chars": len(obs.summary),
                        "raw_chars": len(obs.raw_content),
                        "archive_refs": len(obs.archive_entries),
                    }
                    for obs in recent
                ],
            )
        else:
            logger.debug(
                "[ContextMemory] observe_tool mode=%s turn=%d added=0 metadata_items=%d",
                self.mode,
                self._turn,
                len(tool_metadata or []),
            )

    async def maybe_compact(self) -> None:
        should_compact, reason = self._compact_decision()
        if not should_compact:
            logger.debug(
                "[ContextMemory] compact_skip mode=%s turn=%d reason=%s total_observations=%d "
                "last_compacted_turn=%d last_compacted_observations=%d",
                self.mode,
                self._turn,
                reason,
                len(self.tool_observations),
                self._last_compacted_turn,
                self._last_compacted_observation_count,
            )
            return None
        logger.info(
            "[ContextMemory] compact_start mode=%s turn=%d summarizer=%s observations=%d interval=%s",
            self.mode,
            self._turn,
            self.config.summarizer,
            len(self.tool_observations),
            self.config.interval,
        )
        self._last_compacted_turn = self._turn
        self._last_compacted_observation_count = len(self.tool_observations)
        if str(self.config.summarizer or "local") != "llm":
            self.running_summary = self._summary_from_history()
            self._last_effective_summarizer = "local"
            logger.info(
                "[ContextMemory] compact_local mode=%s turn=%d summary_chars=%d",
                self.mode,
                self._turn,
                len(self.running_summary),
            )
            return None
        await self._update_llm_summary()
        return None

    def render_message(self) -> dict[str, str] | None:
        body = self._render_body()
        if not body.strip():
            return None
        logger.info(
            "[ContextMemory] render mode=%s turn=%d body_chars=%d effective_summarizer=%s "
            "running_summary_chars=%d recent_observations=%d closed_tags=%s",
            self.mode,
            self._turn,
            len(body),
            self._last_effective_summarizer,
            len(self.running_summary),
            len(self._recent_tools()),
            _context_memory_tags_closed(body),
        )
        audit = self._render_audit(body)
        if audit:
            logger.info(
                "[ContextMemory] render_audit mode=%s turn=%d audit=%s",
                self.mode,
                self._turn,
                audit,
            )
        self._log_preview("render_body", body)
        return {
            "role": "user",
            "content": f'<context_memory type="{escape(self.mode)}">\n{body}\n</context_memory>',
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "config": {
                "recent_observations": self.config.recent_observations,
                "tail_turns": self.config.tail_turns,
                "max_chars": self.config.max_chars,
                "tool_summary_chars": self.config.tool_summary_chars,
                "interval": self.config.interval,
                "summarizer": self.config.summarizer,
                "effective_summarizer": self._last_effective_summarizer,
                "summarizer_model": self.config.summarizer_model,
                "summarizer_base_url": self.config.summarizer_base_url,
                "summarizer_failures": self._summarizer_failures,
                "summarizer_disabled": self._summarizer_disabled,
                "last_summarizer_error": self._last_summarizer_error,
                "log_preview_chars": self.config.log_preview_chars,
            },
            "prompt_templates": context_memory_prompt_templates(self.mode),
            "question": self.question,
            "running_summary": self.running_summary,
            "assistant_turns": [obs.__dict__ for obs in self.assistant_turns],
            "tool_observations": [
                {
                    "turn": obs.turn,
                    "name": obs.name,
                    "call_id": obs.call_id,
                    "query": obs.query,
                    "summary": obs.summary,
                    "archive_entries": obs.archive_entries,
                }
                for obs in self.tool_observations
            ],
            "compactions": self.compactions,
        }

    def _render_body(self) -> str:
        raise NotImplementedError

    def _render_audit(self, body: str) -> dict[str, Any]:
        recent = self._recent_tools()
        return {
            "total_assistant_turns": len(self.assistant_turns),
            "total_tool_observations": len(self.tool_observations),
            "recent_tool_refs": _tool_refs(recent),
            "body_chars": len(body),
        }

    def _compact_decision(self) -> tuple[bool, str]:
        if self._turn <= 0 or self._last_compacted_turn == self._turn:
            return False, "no_new_turn_or_already_compacted"
        if len(self.tool_observations) <= self._last_compacted_observation_count:
            return False, "no_new_tool_observation"
        interval = max(1, int(self.config.interval or 1))
        if self._turn % interval != 0:
            return False, f"interval_wait turn_mod={self._turn % interval}"
        return True, "eligible"

    def _recent_tools(self) -> list[ToolObservation]:
        n = max(0, int(self.config.recent_observations or 0))
        if n <= 0:
            return []
        return self.tool_observations[-n:]

    def _summary_from_history(self) -> str:
        pieces: list[str] = []
        if self.question:
            pieces.append(f"Objective: {_truncate(_clean_text(self.question), 700)}")
        for obs in self.tool_observations[-6:]:
            label = obs.query or obs.name
            pieces.append(f"Checked {label}: {obs.summary}")
        if self.assistant_turns:
            pieces.append(f"Latest assistant state: {_truncate(self.assistant_turns[-1].content, 700)}")
        return _truncate("\n".join(p for p in pieces if p), self.config.max_chars // 2)

    def _summary_text(self) -> str:
        return self.running_summary or self._summary_from_history()

    async def _update_llm_summary(self) -> None:
        if self._summarizer_disabled:
            self._last_effective_summarizer = "local"
            self.running_summary = self._summary_from_history()
            logger.info(
                "[ContextMemory] compact_disabled mode=%s turn=%d fallback=local summary_chars=%d failures=%d",
                self.mode,
                self._turn,
                len(self.running_summary),
                self._summarizer_failures,
            )
            return
        if not self.config.summarizer_model or not self.config.summarizer_base_url:
            self._last_effective_summarizer = "local"
            self._last_summarizer_error = "summarizer model/base_url is not configured"
            self.running_summary = self._summary_from_history()
            logger.info(
                "[ContextMemory] compact_missing_llm_config mode=%s turn=%d fallback=local summary_chars=%d",
                self.mode,
                self._turn,
                len(self.running_summary),
            )
            return

        try:
            source_text = self._history_for_summarizer()
            messages = self._summarizer_messages(source_text=source_text)
            payload_chars = sum(len(m.get("content", "")) for m in messages)
            logger.info(
                "[ContextMemory] llm_summary_request mode=%s turn=%d model=%s base_url=%s "
                "messages=%d source_chars=%d payload_chars=%d max_tokens=%s timeout=%s",
                self.mode,
                self._turn,
                self.config.summarizer_model,
                self.config.summarizer_base_url,
                len(messages),
                len(source_text),
                payload_chars,
                self.config.summarizer_max_tokens,
                self.config.summarizer_timeout,
            )
            self._log_preview("compact_source", source_text)
            summary = await _call_chat_completion(
                model=self.config.summarizer_model,
                base_url=self.config.summarizer_base_url,
                api_key=self.config.summarizer_api_key or "EMPTY",
                messages=messages,
                max_tokens=max(256, int(self.config.summarizer_max_tokens or 2048)),
                timeout=float(self.config.summarizer_timeout or 120.0),
            )
            raw_summary_chars = len(summary)
            summary = self._normalize_llm_summary(summary)
            if not summary:
                raise RuntimeError("empty summarizer response")
            self.running_summary = _truncate(summary, max(1000, self.config.max_chars))
            self._summarizer_failures = 0
            self._last_summarizer_error = ""
            self._last_effective_summarizer = "llm"
            logger.info(
                "[ContextMemory] llm_summary_success mode=%s turn=%d raw_summary_chars=%d "
                "normalized_chars=%d stored_chars=%d",
                self.mode,
                self._turn,
                raw_summary_chars,
                len(summary),
                len(self.running_summary),
            )
            self._log_preview("running_summary", self.running_summary)
            self.compactions.append(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "turn": self._turn,
                    "summarizer": "llm",
                    "model": self.config.summarizer_model,
                    "summary_chars": len(self.running_summary),
                }
            )
        except Exception as exc:
            self._summarizer_failures += 1
            self._last_summarizer_error = str(exc)
            self._last_effective_summarizer = "local"
            self.running_summary = self._summary_from_history()
            if self._summarizer_failures >= max(1, int(self.config.summarizer_failure_limit or 3)):
                self._summarizer_disabled = True
            logger.warning(
                "[ContextMemory] llm_summary_failed mode=%s turn=%s failures=%d disabled=%s "
                "fallback=local error=%s",
                self.mode,
                self._turn,
                self._summarizer_failures,
                self._summarizer_disabled,
                exc,
            )
            self.compactions.append(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "turn": self._turn,
                    "summarizer": "local",
                    "fallback_from": "llm",
                    "error": self._last_summarizer_error,
                    "disabled": self._summarizer_disabled,
                }
            )

    def _summarizer_messages(self, *, source_text: str | None = None) -> list[dict[str, str]]:
        prompt = CONTEXT_MEMORY_PROMPTS.get(self.mode, {})
        system = prompt.get("system") or (
            "You are a context compaction summarizer. Preserve exact facts, "
            "URLs, file paths, dates, identifiers, and unresolved questions."
        )
        if self.mode == "claude_pipeline":
            user_prompt = CLAUDE_SUMMARIZER_USER_PROMPT
        elif self.mode == "codex_handoff":
            user_prompt = CODEX_HANDOFF_PROMPT
        elif self.mode == "opencode_marker":
            if self.running_summary:
                user_prompt = OPENCODE_UPDATE_PREFIX.format(previous_summary=self.running_summary)
            else:
                user_prompt = CONTEXT_MEMORY_PROMPTS["opencode_marker"]["new"]
            user_prompt = f"{user_prompt}\n\n{OPENCODE_SUMMARY_TEMPLATE}"
        else:
            user_prompt = "Summarize the context needed to continue."
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"{user_prompt}\n\n<conversation_state>\n{source_text if source_text is not None else self._history_for_summarizer()}\n</conversation_state>",
            },
        ]

    def _history_for_summarizer(self) -> str:
        lines: list[str] = []
        if self.question:
            lines.extend(["## Original Question", _truncate(_clean_text(self.question), 1200), ""])
        if self.running_summary:
            lines.extend(["## Previous Running Summary", _truncate(self.running_summary, 2500), ""])

        lines.append("## Recent Assistant Turns")
        assistant_scope = self.assistant_turns[-8:]
        for obs in assistant_scope:
            lines.append(f"[assistant turn {obs.turn}] {_truncate(obs.content, 1200)}")
        if len(lines) and lines[-1] == "## Recent Assistant Turns":
            lines.append("(none)")
        lines.append("")

        lines.append("## Tool Observations")
        observations = self.tool_observations[-12:]
        if not observations:
            lines.append("(none)")
        for obs in observations:
            lines.append(_format_tool_observation(obs))
            raw = obs.raw_content or obs.summary
            if raw:
                lines.append(f"raw: {_truncate(raw, 1800)}")
            for entry in obs.archive_entries[:5]:
                raw_url = entry.get("raw_url")
                if raw_url:
                    lines.append(f"archive raw_url: {raw_url} summary: {entry.get('summary') or ''}")
        max_chars = max(6000, min(30000, self.config.max_chars * 3))
        source_text = _truncate("\n".join(lines), max_chars)
        logger.info(
            "[ContextMemory] compact_source_scope mode=%s turn=%d previous_summary=%s "
            "assistant_turns=%s tool_refs=%s archive_refs=%d max_chars=%d source_chars=%d",
            self.mode,
            self._turn,
            bool(self.running_summary),
            [obs.turn for obs in assistant_scope],
            _tool_refs(observations),
            sum(len(obs.archive_entries) for obs in observations),
            max_chars,
            len(source_text),
        )
        return source_text

    def _normalize_llm_summary(self, text: str) -> str:
        return _extract_summary_block(text) if self.mode == "claude_pipeline" else _clean_summary_text(text)

    def _log_preview(self, label: str, text: str) -> None:
        n = max(0, int(self.config.log_preview_chars or 0))
        if n <= 0:
            return
        logger.info(
            "[ContextMemory] %s_preview mode=%s turn=%d chars=%d preview=%s",
            label,
            self.mode,
            self._turn,
            len(text),
            _clean_text(text)[:n],
        )


class ClaudePipelineMemory(BaseContextMemory):
    mode = "claude_pipeline"

    def _render_body(self) -> str:
        recent = _render_tool_bullets(self._recent_tools())
        snipped_count = max(0, len(self.tool_observations) - len(self._recent_tools()))
        snipped = (
            f"- {snipped_count} older tool observation(s) are summarized or represented by archive refs."
            if snipped_count
            else "- (none)"
        )
        return _render_claude_pipeline_body(
            summary=_xml_text(self._summary_text() or "No prior compacted work yet."),
            recent=recent,
            snipped=_xml_text(snipped),
            next_focus=_xml_text(_next_focus(self.tool_observations)),
            max_chars=self.config.max_chars,
        )

    def _render_audit(self, body: str) -> dict[str, Any]:
        recent = self._recent_tools()
        older = self.tool_observations[: max(0, len(self.tool_observations) - len(recent))]
        summary = _extract_tag_text(body, "summary")
        return {
            **super()._render_audit(body),
            "summary_chars": len(summary),
            "recent_tool_refs": _tool_refs(recent),
            "older_tool_refs": _tool_refs(older),
            "recent_section_refs": _tool_refs(recent),
            "snipped_count": len(older),
            "summary_mentions_recent_tools": _tool_overlap_hits(summary, recent),
            "summary_mentions_older_tools": _tool_overlap_hits(summary, older),
        }


class CodexHandoffMemory(BaseContextMemory):
    mode = "codex_handoff"

    def _render_body(self) -> str:
        summary = self._summary_text() or "No prior handoff summary yet."
        return _render_codex_handoff_body(
            summary=summary,
            recent_tools=self._recent_tools(),
            max_chars=self.config.max_chars,
        )

    def _render_audit(self, body: str) -> dict[str, Any]:
        recent = self._recent_tools()
        older = self.tool_observations[: max(0, len(self.tool_observations) - len(recent))]
        summary = _extract_tag_text(body, "handoff_summary")
        verbatim = _extract_tag_text(body, "verbatim_recent_observations")
        return {
            **super()._render_audit(body),
            "handoff_summary_chars": len(summary),
            "verbatim_recent_chars": len(verbatim),
            "recent_tool_refs": _tool_refs(recent),
            "older_tool_refs": _tool_refs(older),
            "summary_mentions_recent_tools": _tool_overlap_hits(summary, recent),
            "summary_mentions_older_tools": _tool_overlap_hits(summary, older),
            "verbatim_recent_tool_refs": _tool_refs(recent),
        }


class OpenCodeMarkerMemory(BaseContextMemory):
    mode = "opencode_marker"

    def __init__(self, config: ContextMemoryConfig) -> None:
        super().__init__(config)
        self.hidden_until_turn = 0
        self.tail_start_turn = 0
        self.replay_turn = 0

    async def maybe_compact(self) -> None:
        tail_turns = max(1, int(self.config.tail_turns or 1))
        self.tail_start_turn = max(1, self._turn - tail_turns + 1) if self._turn else 0
        self.hidden_until_turn = max(0, self.tail_start_turn - 1)
        self.replay_turn = self.tool_observations[-1].turn if self.tool_observations else 0
        retained_assistant_turns = [
            obs.turn for obs in self.assistant_turns
            if not self.tail_start_turn or obs.turn >= self.tail_start_turn
        ]
        retained_tool_refs = [
            obs for obs in self.tool_observations
            if not self.tail_start_turn or obs.turn >= self.tail_start_turn
        ]
        logger.info(
            "[ContextMemory] opencode_marker_state turn=%d tail_turns=%d "
            "hidden_until_turn=%d tail_start_turn=%d replay_turn=%d "
            "retained_assistant_turns=%s retained_tool_refs=%s",
            self._turn,
            tail_turns,
            self.hidden_until_turn,
            self.tail_start_turn,
            self.replay_turn,
            retained_assistant_turns,
            _tool_refs(retained_tool_refs),
        )
        self.compactions.append(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "hidden_until_turn": self.hidden_until_turn,
                "tail_start_turn": self.tail_start_turn,
                "replay_turn": self.replay_turn,
            }
        )
        await super().maybe_compact()

    def _render_body(self) -> str:
        summary = (
            self.running_summary
            if self.running_summary and self._last_effective_summarizer == "llm"
            else _render_opencode_summary(
                question=self.question,
                summary=self._summary_text(),
                tools=self.tool_observations,
            )
        )
        retained_tail = _render_tail(
            assistants=self.assistant_turns,
            tools=self.tool_observations,
            tail_start_turn=self.tail_start_turn,
            max_chars=self.config.max_chars // 3,
        )
        replay = self.tool_observations[-1] if self.tool_observations else None
        return _render_opencode_marker_body(
            summary=_xml_text(summary),
            retained_tail=retained_tail,
            replay=_xml_text(_format_tool_observation(replay) if replay else "(none)"),
            max_chars=self.config.max_chars,
        )

    def _render_audit(self, body: str) -> dict[str, Any]:
        retained_tools = [
            obs for obs in self.tool_observations
            if not self.tail_start_turn or obs.turn >= self.tail_start_turn
        ]
        hidden_tools = [
            obs for obs in self.tool_observations
            if self.tail_start_turn and obs.turn < self.tail_start_turn
        ]
        replay = self.tool_observations[-1:] if self.tool_observations else []
        summary = _extract_tag_text(body, "anchored_summary")
        retained_tail = _extract_tag_text(body, "retained_tail")
        replay_text = _extract_tag_text(body, "replayed_latest_observation")
        return {
            **super()._render_audit(body),
            "hidden_until_turn": self.hidden_until_turn,
            "tail_start_turn": self.tail_start_turn,
            "replay_turn": self.replay_turn,
            "anchored_summary_chars": len(summary),
            "retained_tail_chars": len(retained_tail),
            "replayed_latest_observation_chars": len(replay_text),
            "retained_tool_refs": _tool_refs(retained_tools),
            "hidden_tool_refs": _tool_refs(hidden_tools),
            "replay_tool_refs": _tool_refs(replay),
            "summary_mentions_retained_tools": _tool_overlap_hits(summary, retained_tools),
            "summary_mentions_replay_tool": _tool_overlap_hits(summary, replay),
            "replay_also_in_retained_tail": bool(replay and replay[0] in retained_tools),
        }

    def export_state(self) -> dict[str, Any]:
        state = super().export_state()
        state.update(
            {
                "hidden_until_turn": self.hidden_until_turn,
                "tail_start_turn": self.tail_start_turn,
                "replay_turn": self.replay_turn,
            }
        )
        return state


def build_context_memory(config: ContextMemoryConfig | dict[str, Any] | None) -> ContextMemory | None:
    if config is None:
        return None
    if isinstance(config, dict):
        config = ContextMemoryConfig(**config)
    mode = str(config.mode or BELIEF_GRAPH_MODE)
    if mode in {BELIEF_GRAPH_MODE, NO_CONTEXT_MEMORY_MODE}:
        return None
    if mode == "claude_pipeline":
        return ClaudePipelineMemory(config)
    if mode == "codex_handoff":
        return CodexHandoffMemory(config)
    if mode == "opencode_marker":
        return OpenCodeMarkerMemory(config)
    raise ValueError(f"Unknown context memory mode: {mode}")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(text: str, max_chars: int) -> str:
    text = str(text or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars < 20:
        return text[:max_chars]
    keep = max_chars - 18
    return text[:keep] + "...[truncated]"


def _xml_text(text: str) -> str:
    return escape(str(text or ""), quote=False)


def _tool_ref(obs: ToolObservation) -> str:
    call_id = obs.call_id or "-"
    return f"turn={obs.turn}:{obs.name}:{call_id}"


def _tool_refs(observations: list[ToolObservation]) -> list[str]:
    return [_tool_ref(obs) for obs in observations]


def _extract_tag_text(body: str, tag_name: str) -> str:
    open_tag = f"<{tag_name}>"
    close_tag = f"</{tag_name}>"
    start = body.find(open_tag)
    end = body.rfind(close_tag)
    if start == -1 or end == -1 or end <= start:
        return ""
    return body[start + len(open_tag):end].strip()


def _needle_in_text(needle: str, haystack: str, *, min_chars: int = 24) -> bool:
    cleaned = _clean_text(needle)
    if len(cleaned) < min_chars:
        return False
    return cleaned.lower() in haystack


def _tool_overlap_hits(text: str, observations: list[ToolObservation]) -> list[dict[str, Any]]:
    haystack = _clean_text(text).lower()
    hits: list[dict[str, Any]] = []
    for obs in observations:
        reasons: list[str] = []
        if _needle_in_text(obs.query, haystack):
            reasons.append("query")
        summary = _clean_text(obs.summary)
        if _needle_in_text(summary[:120], haystack):
            reasons.append("summary")
        for entry in obs.archive_entries:
            raw_url = str(entry.get("raw_url") or "")
            if raw_url and raw_url.lower() in haystack:
                reasons.append("raw_url")
                break
        if reasons:
            hits.append(
                {
                    "tool_ref": _tool_ref(obs),
                    "reasons": sorted(set(reasons)),
                }
            )
    return hits


def _tool_summary(tm: dict[str, Any], *, max_chars: int, fallback: str) -> str:
    md = tm.get("metadata") or {}
    evidences = md.get("evidences") or []
    parts: list[str] = []
    for ev in evidences[:3]:
        parts.append(str(ev.get("summary") or ev.get("text") or ""))
    if not parts:
        parts.append(str(fallback or tm.get("output") or ""))
    return _truncate(_clean_text(" | ".join(p for p in parts if p)), max_chars)


def _render_tool_bullets(observations: list[ToolObservation]) -> str:
    if not observations:
        return "- (none)"
    lines: list[str] = []
    for obs in observations:
        lines.append(_xml_text(_format_tool_observation(obs)))
        for entry in obs.archive_entries[:3]:
            raw_url = entry.get("raw_url")
            summary = entry.get("summary")
            if raw_url:
                lines.append(_xml_text(f"  raw_url: {raw_url} summary: {summary or ''}"))
    return "\n".join(f"- {line}" if not line.startswith("  ") else line for line in lines)


def _render_claude_pipeline_body(
    *,
    summary: str,
    recent: str,
    snipped: str,
    next_focus: str,
    max_chars: int,
) -> str:
    tool_budget = (
        "Older tool results may be summarized or replaced by archive references. "
        "Raw evidence and archive raw_url content are stronger than this summary."
    )

    def build(summary_part: str, recent_part: str, snipped_part: str, next_part: str) -> str:
        return f"""<summary>
{summary_part}
</summary>
<tool_result_budget>
{tool_budget}
</tool_result_budget>
<recent_evidence>
{recent_part}
</recent_evidence>
<snipped_context>
{snipped_part}
</snipped_context>
<next_focus>
{next_part}
</next_focus>"""

    body = build(summary, recent, snipped, next_focus)
    if max_chars <= 0 or len(body) <= max_chars:
        return body

    fixed = build("", "", "", "")
    available = max(0, max_chars - len(fixed))
    snipped_target = min(len(snipped), max(0, available // 12))
    next_target = min(len(next_focus), max(0, available // 10))
    recent_target = min(len(recent), max(0, available // 3))
    summary_target = max(0, available - snipped_target - next_target - recent_target)
    summary_part = _truncate_section(summary, summary_target)
    recent_part = _truncate_section(recent, recent_target)
    snipped_part = _truncate_section(snipped, snipped_target)
    next_part = _truncate_section(next_focus, next_target)
    body = build(summary_part, recent_part, snipped_part, next_part)

    for section in ("recent", "summary"):
        if len(body) <= max_chars:
            break
        overflow = len(body) - max_chars
        if section == "recent":
            recent_part = _truncate_section(recent_part, max(0, len(recent_part) - overflow))
        else:
            summary_part = _truncate_section(summary_part, max(0, len(summary_part) - overflow))
        body = build(summary_part, recent_part, snipped_part, next_part)
    return body


def _render_verbatim_observations(observations: list[ToolObservation], max_chars: int) -> str:
    if not observations:
        return "(none)"
    chunks = []
    budget = max(200, max_chars)
    each = max(100, budget // max(1, len(observations)))
    for obs in observations:
        raw = obs.raw_content or obs.summary
        chunks.append(_xml_text(f"[turn {obs.turn} {obs.name} {obs.call_id}]\n{_truncate(raw, each)}"))
    return "\n\n".join(chunks)


def _render_codex_handoff_body(
    *,
    summary: str,
    recent_tools: list[ToolObservation],
    max_chars: int,
) -> str:
    prefix = _xml_text(CODEX_SUMMARY_PREFIX)
    summary_text = _xml_text(summary)
    recent_budget = max_chars // 2 if max_chars > 0 else 0
    recent = _render_verbatim_observations(recent_tools, recent_budget)

    def build(summary_part: str, recent_part: str) -> str:
        return f"""<handoff_summary>
{prefix}

{summary_part}
</handoff_summary>
<verbatim_recent_observations>
{recent_part}
</verbatim_recent_observations>"""

    if max_chars <= 0:
        return build(summary_text, recent)

    fixed = build("", "")
    available = max(0, max_chars - len(fixed))
    recent_target = min(len(recent), available // 2)
    recent_part = _truncate_section(recent, recent_target)
    summary_part = _truncate_section(summary_text, max(0, available - len(recent_part)))
    body = build(summary_part, recent_part)

    if len(body) > max_chars:
        overflow = len(body) - max_chars
        recent_part = _truncate_section(recent_part, max(0, len(recent_part) - overflow))
        body = build(summary_part, recent_part)
    if len(body) > max_chars:
        overflow = len(body) - max_chars
        summary_part = _truncate_section(summary_part, max(0, len(summary_part) - overflow))
        body = build(summary_part, recent_part)
    return body


def _truncate_section(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return _truncate(text, max_chars)


def _format_tool_observation(obs: ToolObservation | None) -> str:
    if obs is None:
        return "(none)"
    label = f"turn={obs.turn} tool={obs.name}"
    if obs.call_id:
        label += f" call_id={obs.call_id}"
    if obs.query:
        label += f" query={obs.query}"
    return f"{label}: {obs.summary}"


def _next_focus(observations: list[ToolObservation]) -> str:
    if not observations:
        return "Start by searching for primary evidence relevant to the claim."
    return "Verify unresolved or conflicting evidence before calling finish."


def _render_opencode_summary(*, question: str, summary: str, tools: list[ToolObservation]) -> str:
    evidence_lines = [
        f"- {obs.query or obs.name}: {obs.summary}" for obs in tools[-5:]
    ] or ["- (none)"]
    return "\n".join(
        [
            "## Objective",
            f"- {_truncate(_clean_text(question), 600) if question else '(none)'}",
            "",
            "## Important Details",
            f"- {_truncate(summary, 1000) if summary else '(none)'}",
            "",
            "## Evidence State",
            "### Supports",
            "- (none)",
            "",
            "### Refutes",
            "- (none)",
            "",
            "### Unresolved",
            *evidence_lines,
            "",
            "## Work State",
            "### Completed",
            "- Prior tool calls summarized above.",
            "",
            "### Active",
            "- Continue fact verification from retained tail and replayed observation.",
            "",
            "### Blocked",
            "- (none)",
            "",
            "## Next Move",
            "1. Verify the strongest unresolved evidence.",
            "2. Finish only after the verdict is supported by raw evidence.",
            "",
            "## Relevant Files",
            *(_archive_ref_lines(tools) or ["- (none)"]),
        ]
    )


def _archive_ref_lines(tools: list[ToolObservation]) -> list[str]:
    lines: list[str] = []
    for obs in tools[-5:]:
        for entry in obs.archive_entries[:3]:
            raw_url = entry.get("raw_url")
            if raw_url:
                lines.append(f"- {raw_url}: {entry.get('summary') or obs.summary}")
    return lines


def _render_opencode_marker_body(
    *,
    summary: str,
    retained_tail: str,
    replay: str,
    max_chars: int,
) -> str:
    def build(summary_part: str, tail_part: str, replay_part: str) -> str:
        return f"""<anchored_summary>
{summary_part}
</anchored_summary>
<retained_tail>
{tail_part}
</retained_tail>
<replayed_latest_observation>
{replay_part}
</replayed_latest_observation>"""

    body = build(summary, retained_tail, replay)
    if max_chars <= 0 or len(body) <= max_chars:
        return body

    fixed = build("", "", "")
    available = max(0, max_chars - len(fixed))
    replay_target = min(len(replay), max(0, available // 5))
    tail_target = min(len(retained_tail), max(0, available // 3))
    summary_target = max(0, available - replay_target - tail_target)
    summary_part = _truncate_section(summary, summary_target)
    tail_part = _truncate_section(retained_tail, tail_target)
    replay_part = _truncate_section(replay, replay_target)
    body = build(summary_part, tail_part, replay_part)

    for section in ("tail", "summary"):
        if len(body) <= max_chars:
            break
        overflow = len(body) - max_chars
        if section == "tail":
            tail_part = _truncate_section(tail_part, max(0, len(tail_part) - overflow))
        else:
            summary_part = _truncate_section(summary_part, max(0, len(summary_part) - overflow))
        body = build(summary_part, tail_part, replay_part)
    return body


def _render_tail(
    *,
    assistants: list[AssistantObservation],
    tools: list[ToolObservation],
    tail_start_turn: int,
    max_chars: int,
) -> str:
    lines: list[str] = []
    for assistant in assistants:
        if tail_start_turn and assistant.turn < tail_start_turn:
            continue
        lines.append(f"[assistant turn {assistant.turn}] {_truncate(assistant.content, 500)}")
        for obs in tools:
            if obs.turn == assistant.turn:
                lines.append(f"[tool turn {obs.turn}] {_format_tool_observation(obs)}")
    return _xml_text(_truncate("\n".join(lines) or "(none)", max_chars))


async def _call_chat_completion(
    *,
    model: str,
    base_url: str,
    api_key: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
) -> str:
    import aiohttp

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or 'EMPTY'}",
    }
    url = base_url.rstrip("/") + "/chat/completions"
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"API returned {resp.status}: {body[:500]}")
            data = await resp.json()

    msg = (((data.get("choices") or [{}])[0]).get("message") or {})
    return str((msg.get("content") or "") or (msg.get("reasoning_content") or ""))


def _extract_summary_block(text: str) -> str:
    cleaned = _clean_summary_text(text)
    start = cleaned.find("<summary>")
    if start != -1:
        end = cleaned.rfind("</summary>")
        if end != -1 and end > start:
            return _clean_summary_text(cleaned[start + len("<summary>"):end])
        return _clean_summary_text(cleaned[start + len("<summary>"):])
    analysis_start = cleaned.find("<analysis>")
    analysis_end = cleaned.rfind("</analysis>")
    if analysis_start != -1 and analysis_end != -1 and analysis_end > analysis_start:
        cleaned = (cleaned[:analysis_start] + cleaned[analysis_end + len("</analysis>"):]).strip()
    return cleaned


def _clean_summary_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _context_memory_tags_closed(body: str) -> bool:
    tag_names = [
        "summary",
        "recent_evidence",
        "snipped_context",
        "handoff_summary",
        "verbatim_recent_observations",
        "anchored_summary",
        "retained_tail",
        "replayed_latest_observation",
    ]
    for name in tag_names:
        opens = body.count(f"<{name}>")
        closes = body.count(f"</{name}>")
        if opens != closes:
            return False
    return True


__all__ = [
    "BELIEF_GRAPH_MODE",
    "CONTEXT_MEMORY_BASELINE_MODES",
    "CONTEXT_MEMORY_MODES",
    "ContextMemory",
    "ContextMemoryConfig",
    "NO_CONTEXT_MEMORY_MODE",
    "build_context_memory",
    "context_memory_prompt_templates",
    "is_context_memory_baseline",
    "uses_belief_graph_service",
]
