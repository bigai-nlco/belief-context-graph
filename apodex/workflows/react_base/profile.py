"""react_base workflow profile — loads YAML config and builds an LLM.

Profiles live in ``workflows/react_base/profiles/`` and specify model,
temperature, token limits, turn budget, and ``keep_tool_result`` for
benchmark runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_harness.core.llm import LLMClient

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_profile(name: str) -> dict[str, Any]:
    """Load a profile YAML with environment variable resolution."""
    import yaml
    from dotenv import load_dotenv
    from agent_harness.infra.config import _resolve_env_vars

    load_dotenv(_PROJECT_ROOT / ".env", override=False)

    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"react_base profile not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _resolve_env_vars(raw)


def _clean_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip per-property metadata absent from FastMCP wire shape:
    drop ``title`` and ``description`` at every level, rename
    ``anyOf`` → ``oneOf``.
    """
    import copy

    schema = copy.deepcopy(schema)
    properties = schema.get("properties") if isinstance(schema, dict) else None

    def _walk(node: Any, *, in_properties: bool) -> None:
        if not isinstance(node, dict):
            return
        node.pop("title", None)
        if in_properties:
            node.pop("description", None)
        if "anyOf" in node:
            node["oneOf"] = node.pop("anyOf")
        for k, v in node.items():
            if isinstance(v, dict):
                _walk(v, in_properties=(k == "properties") or in_properties)
            elif isinstance(v, list):
                for item in v:
                    _walk(item, in_properties=in_properties)

    _walk(schema, in_properties=False)
    if isinstance(properties, dict):
        for v in properties.values():
            if isinstance(v, dict):
                v.pop("description", None)
    return schema


def _collapse_optional_unions(schema: dict[str, Any]) -> dict[str, Any]:
    """Collapse ``oneOf: [T, null]`` → ``{type: <T>, default: null, ...}``.

    FastMCP emits ``Optional[T]`` as a single-type schema with
    ``default: null`` (not a union). Non-null unions (e.g.
    ``Union[str, list[str]]``) are left as-is.
    """
    import copy
    schema = copy.deepcopy(schema)

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            for k, v in list(node.items()):
                node[k] = _walk(v)
            branches = node.get("oneOf")
            if isinstance(branches, list) and len(branches) == 2:
                null_b: dict | None = None
                other_b: dict | None = None
                for b in branches:
                    if (
                        isinstance(b, dict) and b.get("type") == "null"
                        and len(b) == 1
                    ):
                        null_b = b
                    else:
                        other_b = b if isinstance(b, dict) else None
                if null_b is not None and other_b is not None:
                    siblings = {
                        k: v for k, v in node.items() if k != "oneOf"
                    }
                    return {**other_b, **siblings}
            return node
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    return _walk(schema)


def _sort_keys_alphabetical(obj: Any) -> Any:
    """Sort dict keys alphabetically at every depth. List order preserved."""
    if isinstance(obj, dict):
        return {k: _sort_keys_alphabetical(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_sort_keys_alphabetical(item) for item in obj]
    return obj


def _raw_docstring_with_indent(func: Any) -> str:
    """Return ``func``'s docstring preserving original source indentation.

    Uses ``ast.get_docstring(node, clean=False)`` to reach past any
    decorator that may have already dedented ``__doc__``.
    """
    import ast
    import inspect
    import textwrap
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return (func.__doc__ or "").rstrip()
    try:
        node = ast.parse(textwrap.dedent(src)).body[0]
    except (SyntaxError, IndexError):
        return (func.__doc__ or "").rstrip()
    doc = ast.get_docstring(node, clean=False)
    return (doc or "").rstrip()


def _build_fastmcp_style_tool_spec(tool: Any) -> dict[str, Any]:
    """Build an OpenAI tool spec in FastMCP shape.

    - Description = the function's RAW docstring (preserving indent).
    - ``Optional[T]`` params collapsed to ``{type:<T>, default:null}``.
    - All dict keys sorted alphabetically for stable byte layout.
    """
    if isinstance(tool, dict):
        spec = dict(tool)
        underlying = None
    else:
        schema_fn = getattr(tool, "to_openai_schema", None)
        if not callable(schema_fn):
            return tool
        spec = schema_fn()
        underlying = getattr(tool, "func", None)

    fn = spec.get("function") or {}
    if not isinstance(fn, dict):
        fn = {}

    if underlying is not None:
        raw_doc = _raw_docstring_with_indent(underlying)
        if raw_doc:
            fn["description"] = raw_doc

    params = fn.get("parameters")
    if isinstance(params, dict):
        params = _clean_tool_schema(params)
        params = _collapse_optional_unions(params)
        fn["parameters"] = params

    return _sort_keys_alphabetical({"function": fn, "type": "function"})


def clean_tool_schemas(tools: list[Any]) -> list[dict[str, Any]]:
    """Rebuild tool schemas in FastMCP form for wire transmission."""
    return [_build_fastmcp_style_tool_spec(t) for t in tools]


def apply_fastmcp_schemas(tools: list[Any]) -> None:
    """Pin each Tool's wire schema to the FastMCP-shaped spec, in place."""
    for t in tools:
        if hasattr(t, "metadata"):
            t.metadata["openai_schema"] = _build_fastmcp_style_tool_spec(t)


class _ReasoningContentInliner(LLMClient):
    """Wraps an :class:`LLMClient` and inlines ``reasoning_content`` into
    ``content`` as a ``{reasoning}</think>\\n\\n{text}`` block.

    The SGLang chat template auto-prepends ``<think>\\n`` to assistant
    generation, so the response arrives as ``{reasoning}</think>\\n\\n{text}``
    *without* a leading opening tag. We mirror that shape so history
    matches the training distribution.
    """

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.model = inner.model

    @staticmethod
    def _inline(response: Any) -> Any:
        rc = getattr(response, "reasoning_content", "") or ""
        if not rc:
            return response
        if "</think>" in rc:
            rc = rc.replace("</think>", "</ think>")
        existing = response.content if isinstance(response.content, str) else ""
        response.content = (
            f"{rc}</think>\n\n{existing}" if existing else f"{rc}</think>\n\n"
        )
        return response

    async def chat(self, messages, **kwargs):
        return self._inline(await self.inner.chat(messages, **kwargs))

    async def stream(self, messages, **kwargs):
        async for delta in self.inner.stream(messages, **kwargs):
            yield delta


def create_llm(profile: dict[str, Any]) -> LLMClient:
    """Build an OpenAI-compatible :class:`LLMClient` from a profile dict.

    ``thinking_format: tag`` wraps the client with :class:`_ReasoningContentInliner`
    so SGLang ``reasoning_content`` ends up in the visible content as a
    ``</think>``-bounded block.
    """
    from agent_harness.infra.openai_client import OpenAIClient

    cfg = profile["llm"]

    temperature = cfg.get("temperature", 1.0)
    top_p = cfg.get("top_p", 0.95)
    max_tokens = cfg.get("max_tokens", 32_768)
    repetition_penalty = cfg.get("repetition_penalty", 1.05)

    extra_body: dict[str, Any] = dict(cfg.get("extra_body") or {})
    if "top_p" not in extra_body:
        extra_body["top_p"] = top_p
    if repetition_penalty != 1.0 and "repetition_penalty" not in extra_body:
        extra_body["repetition_penalty"] = repetition_penalty

    client = OpenAIClient(
        model=cfg["model"],
        api_key=cfg.get("api_key", "dummy"),
        base_url=cfg.get("base_url"),
        temperature=temperature,
        max_completion_tokens=max_tokens,
        default_headers={
            "HTTP-Referer": "agent_harness",
            "X-Title": "AgentHarness-ProfileReAct",
        },
        extra_body=extra_body or None,
    )

    if _resolve_thinking_format(profile) == "tag":
        return _ReasoningContentInliner(client)
    return client


def _resolve_thinking_format(profile: dict[str, Any]) -> str:
    """YAML explicit > infer-from-model-id > ``tag`` fallback."""
    from agent_harness.core.runtime.loop.model_profile import (
        infer_thinking_format,
    )
    explicit = (profile.get("agent") or {}).get("thinking_format")
    if explicit:
        return explicit
    model_id = (profile.get("llm") or {}).get("model")
    return infer_thinking_format(model_id, default="tag")


def build_model_profile(profile: dict[str, Any]) -> Any:
    """Build a kernel ``ModelProfile`` from a profile dict."""
    from agent_harness.core.runtime.loop.model_profile import ModelProfile
    return ModelProfile(
        model_id=profile["llm"]["model"],
        provider="openai",
        thinking_format=_resolve_thinking_format(profile),
    )
