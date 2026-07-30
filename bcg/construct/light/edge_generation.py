"""Conservative Qwen relation-edge generation for one two-turn window."""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Mapping, Optional

from .llm import (
    call_model,
    make_client,
    parse_json_response,
    resolve_config_api_key,
)
from .prompts import build_relation_prompt

VALID_RELATION_TYPES = {"depends_on", "supplements", "contradicts"}


def normalize_edge_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Validate the complete ``belief_graph.edge_generation`` config block."""
    raw = dict(config or {})
    required = (
        "enabled", "provider", "base_url", "model", "temperature",
        "max_tokens", "retries", "enable_thinking", "fail_on_error",
        "search_previous_turns",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(
            "belief_graph.edge_generation is missing required field(s): "
            + ", ".join(missing)
        )
    if str(raw["provider"]).lower() != "openai":
        raise ValueError("belief_graph.edge_generation.provider must be 'openai'")
    resolve_config_api_key(
        raw,
        default_env="BELIEF_GRAPH_LOCAL_API_KEY",
        config_path="belief_graph.edge_generation",
    )
    for key in ("base_url", "api_key", "model"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"belief_graph.edge_generation.{key} must be non-empty")
    return {
        "enabled": bool(raw["enabled"]),
        "provider": "openai",
        "base_url": raw["base_url"].strip(),
        "api_key": raw["api_key"].strip(),
        "api_key_env": raw["api_key_env"].strip(),
        "model": raw["model"].strip(),
        "temperature": float(raw["temperature"]),
        "max_tokens": max(16, int(raw["max_tokens"])),
        "retries": max(1, int(raw["retries"])),
        "enable_thinking": bool(raw["enable_thinking"]),
        "fail_on_error": bool(raw["fail_on_error"]),
        "search_previous_turns": bool(raw["search_previous_turns"]),
        "max_previous_windows": max(1, int(raw.get("max_previous_windows", 4))),
    }


class QwenEdgeGenerator:
    """Generate and parse necessary relation edges with a non-thinking Qwen model."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = normalize_edge_config(config)
        self.model = self.config["model"]
        self._client = None
        self._client_lock = threading.Lock()

    def _ensure_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = make_client(self.config)
        return self._client

    def generate_window(
        self,
        nodes: List[Dict[str, Any]],
        *,
        current_node_ids: set[int],
        turn_index: int,
        previous_turn_index: Optional[int],
    ) -> Dict[str, Any]:
        if not self.config["enabled"]:
            return {
                "relations": [],
                "diagnostics": {"skipped": True, "skip_reason": "edge generation disabled"},
            }
        prompt = build_relation_prompt(nodes, current_node_ids)
        extra_body = None
        reasoning_effort = "medium"
        if not self.config["enable_thinking"]:
            reasoning_effort = None
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        raw = call_model(
            self._ensure_client(),
            self.model,
            prompt,
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            retries=self.config["retries"],
            usage_label=f"t{turn_index}.edges.prev{previous_turn_index}",
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            response_format={"type": "json_object"},
        )
        parsed = parse_json_response(raw)
        relations = parsed.get("relations", []) if isinstance(parsed, dict) else []
        if not isinstance(relations, list):
            relations = []
        return {
            "relations": relations,
            "diagnostics": {
                "model": self.model,
                "raw_output": raw,
                "parse_error": parsed.get("_parse_error") if isinstance(parsed, dict) else "not an object",
                "n_returned": len(relations),
            },
        }


_CACHE: Dict[str, QwenEdgeGenerator] = {}
_CACHE_LOCK = threading.Lock()


def get_edge_generator(config: Mapping[str, Any]) -> Optional[QwenEdgeGenerator]:
    normalized = normalize_edge_config(config)
    if not normalized["enabled"]:
        return None
    key = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    with _CACHE_LOCK:
        generator = _CACHE.get(key)
        if generator is None:
            generator = QwenEdgeGenerator(normalized)
            _CACHE[key] = generator
        return generator
