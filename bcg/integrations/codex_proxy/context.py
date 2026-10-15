"""Transactional graph-before-eviction policy for full-history requests."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field

from .config import ProxyConfig
from .graph import CONTEXT_GUIDE, GraphBackend, GraphError, render_graph
from .protocol import UnsupportedHistory, partition


class RequestError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@dataclass
class Session:
    session_hash: str
    problem_id: str = ""
    acknowledged: list[str] = field(default_factory=list)
    snapshot: dict = field(default_factory=dict)
    tainted: bool = False
    requests: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reset_graph(self) -> None:
        self.problem_id = f"codex_{self.session_hash[:16]}_{uuid.uuid4().hex}"
        self.acknowledged = []
        self.snapshot = {}
        self.tainted = False


class SessionStore:
    def __init__(self, max_sessions: int):
        self.max_sessions = max_sessions
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()

    def get(self, session_id: str) -> Session:
        if (
            not session_id
            or len(session_id) > 200
            or any(ord(c) < 33 or ord(c) > 126 for c in session_id)
        ):
            raise RequestError(
                "Provide a unique X-BCG-Session-ID header (1–200 printable non-space ASCII characters)"
            )
        key = digest(session_id)
        with self.lock:
            if key not in self.sessions:
                if len(self.sessions) >= self.max_sessions:
                    raise RequestError(
                        "Session capacity reached; restart the prototype for a new set of sessions",
                        503,
                    )
                self.sessions[key] = Session(key)
            return self.sessions[key]


def normalize_request(body: dict, request_summary: bool) -> dict:
    if not isinstance(body, dict):
        raise RequestError("Request body must be a JSON object")
    if body.get("previous_response_id") or body.get("conversation"):
        raise RequestError(
            "Use stateless full-history input; previous_response_id/conversation could restore evicted history"
        )
    if body.get("background") or body.get("context_management"):
        raise RequestError(
            "Background execution and provider-managed compaction are unsupported"
        )
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise RequestError(
            "A model is required; the gateway does not choose one for you"
        )
    if "instructions" in body and not isinstance(
        body["instructions"], (str, type(None))
    ):
        raise RequestError("instructions must be a string or null")
    value = copy.deepcopy(body)
    inputs = value.get("input", [])
    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    if not isinstance(inputs, list) or any(
        not isinstance(item, dict) for item in inputs
    ):
        raise RequestError("input must be text or a complete array of Responses items")
    if any(item.get("type") == "item_reference" for item in inputs):
        raise RequestError(
            "Expand item_reference items; only explicit full-history replay is supported"
        )
    value["input"] = inputs
    value["store"] = False
    value.pop("previous_response_id", None)
    value.pop("conversation", None)
    reasoning = value.get("reasoning")
    if request_summary and isinstance(reasoning, dict):
        reasoning.setdefault("summary", "auto")
    include = value.get("include") or []
    if not isinstance(include, list) or any(
        not isinstance(item, str) for item in include
    ):
        raise RequestError("include must be an array of strings")
    if "reasoning.encrypted_content" not in include:
        include.append("reasoning.encrypted_content")
    value["include"] = include
    return value


class ContextEngine:
    def __init__(self, config: ProxyConfig, graph: GraphBackend):
        self.config = config
        self.graph = graph

    def prepare(self, body: dict, session: Session) -> tuple[dict, dict]:
        """Caller holds session.lock until the upstream response is complete.

        Only graph acknowledgement advances the ledger. Unknown write outcomes
        require a fresh graph ID on the next request (no duplicate ingestion).
        """
        outgoing = normalize_request(body, self.config.request_reasoning_summary)
        if self.config.replay_reasoning_context and isinstance(
            outgoing.get("reasoning"), dict
        ):
            has_prior_output = any(
                item.get("type") in {"reasoning", "function_call", "custom_tool_call"}
                or item.get("role") == "assistant"
                for item in outgoing["input"]
            )
            outgoing["reasoning"].setdefault(
                "context", "all_turns" if has_prior_output else "current_turn"
            )
        metadata = {
            "mode": self.config.mode,
            "session_hash": session.session_hash,
            "input_items_before": len(outgoing["input"]),
            "input_items_after": len(outgoing["input"]),
            "evicted_steps": 0,
            "graph_turns_written": 0,
            "graph_chars": 0,
            "selected_node_ids": [],
            "selected_relation_ids": [],
            "fallback": None,
            "fallback_detail": None,
            "node_char_budget": self.config.node_char_budget,
        }
        if self.config.mode == "default":
            return outgoing, metadata
        try:
            plan = partition(outgoing["input"], self.config.recent_turns)
        except UnsupportedHistory as exc:
            metadata["fallback"] = str(exc)
            return outgoing, metadata
        if not plan.question.strip():
            metadata["fallback"] = "no_text_user_request"
            return outgoing, metadata
        desired = [digest(turn) for turn in plan.graph_turns]
        # A fork, edit, or local Codex compaction must not reuse a future graph.
        if (
            not session.problem_id
            or session.tainted
            or session.acknowledged != desired[: len(session.acknowledged)]
        ):
            session.reset_graph()
        metadata["problem_id"] = session.problem_id
        pending = plan.graph_turns[len(session.acknowledged) :]
        if pending:
            try:
                snapshot = self.graph.push(session.problem_id, pending)
            except GraphError as exc:
                session.tainted = True
                metadata["fallback"] = "graph_write_failed"
                metadata["fallback_detail"] = exc.audit_code
                return outgoing, metadata
            session.snapshot = snapshot
            session.acknowledged = desired
            metadata["graph_turns_written"] = len(pending)
        metadata["graph_usage_cumulative"] = session.snapshot.get("token_usage", {})
        if not plan.evicted:
            return outgoing, metadata
        try:
            selection = self.graph.select(
                session.problem_id,
                plan.query,
                plan.question,
                self.config.node_char_budget,
            )
            graph_text, node_ids, relation_ids = render_graph(
                session.snapshot, selection
            )
            if not node_ids:
                raise GraphError("empty_selection")
        except GraphError as exc:
            metadata["fallback"] = "graph_selection_failed"
            metadata["fallback_detail"] = exc.audit_code
            return outgoing, metadata
        # System/developer items and every user input remain verbatim. The
        # Responses instructions field is the native system/developer channel.
        prefix = outgoing.get("instructions") or ""
        outgoing["instructions"] = (
            prefix + ("\n\n" if prefix else "") + CONTEXT_GUIDE + "\n\n" + graph_text
        )
        outgoing["input"] = plan.retained
        metadata.update(
            {
                "input_items_after": len(plan.retained),
                "evicted_steps": len(plan.evicted),
                "graph_chars": len(graph_text),
                "selected_node_ids": node_ids,
                "selected_relation_ids": relation_ids,
            }
        )
        return outgoing, metadata
