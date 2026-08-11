"""
extractor.py
============
Generative belief/decision node extraction from semantic chunks.

A turn is split into semantic chunks first. Once the complete chunk list is
known, every chunk is submitted **concurrently** to a small generative model
(Qwen, served behind an OpenAI-compatible endpoint such as vLLM). Each chunk may
yield zero, one, or several self-contained belief/decision nodes. Node type is
decided by the model, but ``decision`` is only honoured for the assistant role.

This replaces the former local DistilBART summarizer. Stance is still inferred
separately by the local four-class classifier (``stance.py``); entities are still
attached post-merge by the local NER (``named_entities.py``); ``event_time`` is
still stamped by the graph builder. The generative model therefore only produces
the node *text* and *node_type* — it is never trusted for stance, entities, or
time.
"""

from __future__ import annotations

import json
import math
import sys
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .._shared.roles import normalize_role
from .llm import (
    bind_prompt_log_path,
    bind_usage_tracker,
    call_model,
    current_prompt_log_path,
    current_usage_tracker,
    is_context_overflow_error,
    make_client,
    parse_json_response,
    resolve_config_api_key,
    temperature_request_value,
    thinking_request_options,
    unbind_prompt_log_path,
    unbind_usage_tracker,
)
from .prompts import build_chunk_extraction_prompt, format_graph_nodes_context

_SUPPORTED_ROLES = {"user", "assistant", "tool"}


def normalize_extractor_config(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a validated, JSON-serialisable generative-extractor config."""
    raw = dict(config or {})
    required = (
        "enabled",
        "provider",
        "base_url",
        "model",
        "temperature",
        "max_tokens",
        "max_concurrency",
        "request_timeout",
        "retries",
        "context_scope",
        "enable_thinking",
        "include_turn_content",
        "require_excerpt",
        "dynamic_node_cap",
        "node_cap_unit",
        "node_cap_ratio",
        "node_cap_min",
        "node_cap_max",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(
            "belief_graph.extractor is missing required field(s): " + ", ".join(missing)
        )
    resolve_config_api_key(
        raw,
        default_env="BELIEF_GRAPH_LOCAL_API_KEY",
        config_path="belief_graph.extractor",
    )
    return {
        "enabled": bool(raw["enabled"]),
        "provider": str(raw["provider"]),
        "base_url": str(raw["base_url"]),
        "api_key": str(raw["api_key"]),
        "api_key_env": str(raw["api_key_env"]),
        "model": str(raw["model"]),
        "temperature": float(raw["temperature"]),
        "max_tokens": max(16, int(raw["max_tokens"])),
        "max_concurrency": max(1, int(raw["max_concurrency"])),
        "request_timeout": float(raw["request_timeout"]),
        "retries": max(1, int(raw["retries"])),
        # Read-only context handed to the model: historical graph NODES only
        # (no relations). "graph" = existing nodes, "none" = no context.
        "context_scope": str(raw["context_scope"]),
        # Qwen3 is a thinking model. For structured JSON extraction, reasoning is
        # unnecessary and, under a fixed max_tokens, can consume the whole budget
        # so the JSON is truncated/absent (empty extraction). Off by default.
        "enable_thinking": bool(raw["enable_thinking"]),
        # Optional (off by default): also pass the full current turn as read-only
        # context to help resolve references that point outside the chunk.
        "include_turn_content": bool(raw["include_turn_content"]),
        # Optional (off by default): request a verbatim supporting excerpt per
        # node and drop any node lacking one found inside the chunk.
        "require_excerpt": bool(raw["require_excerpt"]),
        # Optional dynamic per-chunk node cap. When ON, the per-chunk node budget
        # is computed from the chunk's length (rule-ized) and injected/enforced per
        # chunk. When OFF (default), there is NO cap of any kind — extraction is
        # unconstrained, exactly as if this feature did not exist.
        #   cap = clamp(ceil(size * node_cap_ratio), node_cap_min, node_cap_max)
        #   size = sentence count (node_cap_unit="sentence") or char count ("char")
        "dynamic_node_cap": bool(raw["dynamic_node_cap"]),
        "node_cap_unit": (
            "char" if str(raw["node_cap_unit"]).lower() == "char" else "sentence"
        ),
        "node_cap_ratio": float(raw["node_cap_ratio"]),
        "node_cap_min": max(1, int(raw["node_cap_min"])),
        "node_cap_max": max(0, int(raw["node_cap_max"])),  # 0 = no upper clamp
    }


@dataclass(frozen=True)
class ExtractedNode:
    chunk_index: int
    node_type: str  # "belief" | "decision"
    text: str
    supporting_excerpts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "chunk_index": self.chunk_index,
            "node_type": self.node_type,
            "text": self.text,
        }
        if self.supporting_excerpts:
            out["supporting_excerpts"] = list(self.supporting_excerpts)
        return out


def _normalise_role(role: Any) -> str:
    value = str(role or "user").strip().lower()
    return normalize_role(value)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _clean_excerpts(raw: Any, chunk_text: str, *, require: bool) -> list[str] | None:
    """Return cleaned excerpts, or None when a required excerpt is missing."""
    excerpts = [
        str(item).strip()
        for item in (raw or [])
        if isinstance(item, str) and str(item).strip()
    ]
    if not require:
        return excerpts
    grounded = [ex for ex in excerpts if ex and ex in chunk_text]
    if not grounded:
        return None
    return grounded


class QwenChunkExtractor:
    """Concurrent per-chunk generative node extractor over an OpenAI endpoint."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = normalize_extractor_config(config)
        self.model = self.config["model"]
        self._client = None
        self._client_lock = threading.Lock()

    # -- client -------------------------------------------------------------
    def _ensure_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                self._client = make_client(
                    {
                        "base_url": self.config["base_url"],
                        "api_key": self.config["api_key"],
                        "model": self.model,
                    }
                )
        return self._client

    # -- parsing ------------------------------------------------------------
    def _parse_response(
        self, raw: str, *, chunk_index: int, role: str, chunk_text: str
    ) -> list[ExtractedNode]:
        parsed = parse_json_response(raw)
        if not isinstance(parsed, dict) or parsed.get("_parse_error"):
            # Make silent failures visible: a truncated/absent JSON (e.g. a
            # thinking model that spent the token budget on reasoning) is the
            # most common cause of unexpected zero-node turns.
            reason = (
                parsed.get("_parse_error") if isinstance(parsed, dict) else "not a dict"
            )
            snippet = (raw or "").strip().replace("\n", " ")[:200]
            print(
                f"    [extract] chunk {chunk_index} ({role}) JSON parse failed "
                f"({reason}); raw[:200]={snippet!r}",
                file=sys.stderr,
            )
            return []
        require = self.config["require_excerpt"]
        nodes: list[ExtractedNode] = []

        for item in parsed.get("beliefs") or []:
            if not isinstance(item, dict):
                continue
            text = _clean_text(item.get("belief"))
            if not text:
                continue
            excerpts = _clean_excerpts(
                item.get("supporting_excerpts"), chunk_text, require=require
            )
            if excerpts is None:
                continue
            nodes.append(ExtractedNode(chunk_index, "belief", text, excerpts))

        for item in parsed.get("decisions") or []:
            if not isinstance(item, dict):
                continue
            text = _clean_text(item.get("decision"))
            if not text:
                continue
            excerpts = _clean_excerpts(
                item.get("supporting_excerpts"), chunk_text, require=require
            )
            if excerpts is None:
                continue
            # node_type gate: decisions are honoured only for the assistant role.
            node_type = "decision" if role == "assistant" else "belief"
            nodes.append(ExtractedNode(chunk_index, node_type, text, excerpts))

        return nodes

    # -- per-chunk node budget ---------------------------------------------
    def _cap_for_chunk(self, chunk: Any) -> int:
        """Per-chunk node-count cap. Returns 0 when no cap applies.

        When ``dynamic_node_cap`` is off, extraction is unconstrained (0). When
        on, the cap is computed from the chunk's length.
        """
        if not self.config.get("dynamic_node_cap", False):
            return 0
        if self.config.get("node_cap_unit") == "char":
            size = len(str(getattr(chunk, "text", chunk) or ""))
        else:
            size = (
                len(
                    getattr(chunk, "sentence_indices", None)
                    or getattr(chunk, "sentences", None)
                    or []
                )
                or 1
            )
        cap = math.ceil(size * float(self.config.get("node_cap_ratio", 1.0)))
        cap = max(int(self.config.get("node_cap_min", 1)), cap)
        upper = int(self.config.get("node_cap_max", 0))
        if upper > 0:
            cap = min(cap, upper)
        return cap

    # -- extraction ---------------------------------------------------------
    def extract_turn(
        self,
        chunks: list[Any],
        role: Any,
        *,
        turn_content: str = "",
        graph_nodes: list[dict[str, Any]] | None = None,
        context_chars: int = 9000,
        turn_index: int | None = None,
    ) -> list[list[ExtractedNode]]:
        """Extract nodes for every chunk of one turn, all chunks concurrently.

        Returns a list aligned with ``chunks``; each element is that chunk's
        (possibly empty) list of ``ExtractedNode``. All chunk requests for the
        turn are submitted at once so they enter the model together.
        """
        role_key = _normalise_role(role)
        if role_key not in _SUPPORTED_ROLES or not chunks:
            return [[] for _ in chunks]

        # Historical-node context is a per-turn snapshot shared by every chunk
        # call (Phase 1 runs before this turn's own nodes exist, so these are
        # strictly prior-turn nodes). Relations are intentionally omitted.
        if self.config["context_scope"] == "graph":
            nodes_context = format_graph_nodes_context(
                graph_nodes or [], char_budget=context_chars
            )
        else:
            nodes_context = "[]"

        client = self._ensure_client()
        reasoning_effort, extra_body = thinking_request_options(
            self.model, enabled=self.config.get("enable_thinking", False)
        )

        turn_ctx = (
            turn_content if self.config.get("include_turn_content", False) else None
        )
        require_excerpt = self.config.get("require_excerpt", False)
        prompts: list[str | None] = []
        chunk_texts: list[str] = []
        chunk_caps: list[int] = []
        for chunk in chunks:
            chunk_text = str(getattr(chunk, "text", chunk) or "")
            chunk_texts.append(chunk_text)
            cap = self._cap_for_chunk(chunk)
            chunk_caps.append(cap)
            prompts.append(
                build_chunk_extraction_prompt(
                    role_key,
                    chunk_text=chunk_text,
                    graph_nodes=nodes_context,
                    turn_content=turn_ctx,
                    require_excerpt=require_excerpt,
                    max_nodes=cap,
                )
            )

        # Bind the caller's usage tracker / prompt-log path into worker threads
        # so token accounting and prompt audit remain correct under concurrency.
        tracker = current_usage_tracker()
        log_path = current_prompt_log_path()

        def _run_one(index: int) -> list[ExtractedNode]:
            prompt = prompts[index]
            if not prompt:
                return []
            tok_u = bind_usage_tracker(tracker) if tracker is not None else None
            tok_p = bind_prompt_log_path(log_path) if log_path is not None else None
            try:
                label = (
                    f"t{turn_index}.extract.c{index}"
                    if turn_index is not None
                    else f"extract.c{index}"
                )

                def _call(request_prompt: str, request_label: str) -> str:
                    return call_model(
                        client,
                        self.model,
                        request_prompt,
                        temperature=temperature_request_value(
                            self.model, self.config["temperature"]
                        ),
                        max_tokens=self.config["max_tokens"],
                        retries=self.config["retries"],
                        usage_label=request_label,
                        reasoning_effort=reasoning_effort,
                        extra_body=extra_body,
                        response_format={"type": "json_object"},
                    )

                try:
                    raw = _call(prompt, label)
                except Exception as exc:
                    if not is_context_overflow_error(exc) or nodes_context == "[]":
                        raise
                    # Historical nodes are reference-resolution context only.
                    # If their rendered block makes the model context overflow,
                    # retry the same complete evidence chunk without that
                    # read-only block instead of silently dropping the chunk.
                    fallback_prompt = build_chunk_extraction_prompt(
                        role_key,
                        chunk_text=chunk_texts[index],
                        graph_nodes="[]",
                        turn_content=turn_ctx,
                        require_excerpt=require_excerpt,
                        max_nodes=chunk_caps[index],
                    )
                    if not fallback_prompt:
                        raise
                    print(
                        f"    [extract] {label} exceeded the context window; "
                        "retrying without historical-node context",
                        file=sys.stderr,
                    )
                    raw = _call(fallback_prompt, f"{label}.without_history")
                nodes = self._parse_response(
                    raw,
                    chunk_index=int(getattr(chunks[index], "chunk_id", index)),
                    role=role_key,
                    chunk_text=chunk_texts[index],
                )
                cap = chunk_caps[index]
                if cap and len(nodes) > cap:
                    # Hard backstop: keep decisions first, then beliefs in order.
                    ordered = [n for n in nodes if n.node_type == "decision"] + [
                        n for n in nodes if n.node_type != "decision"
                    ]
                    nodes = ordered[:cap]
                return nodes
            except Exception as exc:
                # A single chunk's failure yields zero nodes for that chunk;
                # the turn may still produce nodes from its other chunks, and a
                # turn producing zero nodes overall is allowed. Surface it so it
                # is not silently invisible.
                cid = int(getattr(chunks[index], "chunk_id", index))
                print(
                    f"    [extract] chunk {cid} ({role_key}) request failed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return []
            finally:
                if tok_u is not None:
                    unbind_usage_tracker(tok_u)
                if tok_p is not None:
                    unbind_prompt_log_path(tok_p)

        workers = min(len(chunks), self.config["max_concurrency"])
        results: list[list[ExtractedNode]] = [[] for _ in chunks]
        if workers <= 1:
            for index in range(len(chunks)):
                results[index] = _run_one(index)
            return results
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one, index): index for index in range(len(chunks))
            }
            for future in futures:
                index = futures[future]
                results[index] = future.result()
        return results


_EXTRACTOR_CACHE: dict[str, QwenChunkExtractor] = {}
_EXTRACTOR_CACHE_LOCK = threading.Lock()


def get_extractor(
    config: Mapping[str, Any] | None = None,
) -> QwenChunkExtractor:
    """Return one shared extractor per normalized config (client is lazy)."""
    normalized = normalize_extractor_config(config)
    cache_key = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    with _EXTRACTOR_CACHE_LOCK:
        extractor = _EXTRACTOR_CACHE.get(cache_key)
        if extractor is None:
            extractor = QwenChunkExtractor(normalized)
            _EXTRACTOR_CACHE[cache_key] = extractor
        return extractor


def extracted_nodes_as_json(per_chunk_nodes: list[list[ExtractedNode]]) -> str:
    """Stable audit representation used by stream events/result logs."""
    flat = [node.to_dict() for group in per_chunk_nodes for node in group]
    return json.dumps(flat, ensure_ascii=False)
