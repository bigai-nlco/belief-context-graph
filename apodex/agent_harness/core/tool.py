"""Tool definitions — function + OpenAI function-schema.

A :class:`Tool` carries the metadata the loop needs (name, description,
JSON-schema parameters) plus the coroutine that executes it. Use
:func:`tool` to wrap an async function; ``parameters`` is inferred from
type hints and the Google-style docstring when omitted.
"""

from __future__ import annotations

import inspect
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union, get_args, get_origin

ToolFn = Callable[..., Awaitable[Any]]


@dataclass
class Tool:
    """A callable tool with an OpenAI-compatible JSON-schema signature."""

    name: str
    description: str
    parameters: dict[str, Any]   # JSON schema (OpenAI ``function.parameters``)
    func: ToolFn
    metadata: dict[str, Any] = field(default_factory=dict)

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        """Run the tool with keyword args. ``func`` must be async."""
        return await self.func(**args)

    def to_openai_schema(self) -> dict[str, Any]:
        """One entry of OpenAI's ``tools=`` array."""
        if (spec := self.metadata.get("openai_schema")) is not None:
            return spec
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_PRIMITIVE_TYPES: dict[Any, str] = {
    str: "string", int: "integer", float: "number", bool: "boolean",
    type(None): "null",
}


def _schema_for_type(t: Any) -> dict[str, Any]:
    """Build a JSON-schema fragment for a Python type hint."""
    if t in _PRIMITIVE_TYPES:
        return {"type": _PRIMITIVE_TYPES[t]}

    origin = get_origin(t)
    if origin is None:
        # Unannotated / unknown — accept anything.
        return {"type": "string"}

    if origin in (list, set, tuple):
        args = get_args(t)
        if args:
            return {"type": "array", "items": _schema_for_type(args[0])}
        return {"type": "array"}

    if origin is dict:
        return {"type": "object"}

    if origin in (Union, typing.Union):  # type: ignore[attr-defined]
        non_none = [a for a in get_args(t) if a is not type(None)]
        nullable = len(non_none) != len(get_args(t))
        if len(non_none) == 1:
            schema = _schema_for_type(non_none[0])
        else:
            schema = {"anyOf": [_schema_for_type(a) for a in non_none]}
        if nullable:
            # ``default: null`` is FastMCP's idiom; many proxies accept it.
            schema.setdefault("default", None)
        return schema

    # Python 3.10+ ``X | Y`` resolves to ``types.UnionType``; treated the
    # same as ``typing.Union`` above when get_origin returns it.
    if origin is type(int | str):  # types.UnionType
        return _schema_for_type(Union[get_args(t)])  # type: ignore[misc]

    return {"type": "string"}


_ARG_HEADER_RE = re.compile(r"^\s*(Args|Arguments|Parameters):\s*$", re.MULTILINE)
_RETURNS_HEADER_RE = re.compile(
    r"^\s*(Returns|Yields|Raises|Examples):\s*$", re.MULTILINE,
)


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return ``(summary, {arg_name: description})`` from a Google-style docstring."""
    if not doc:
        return "", {}
    text = doc.strip()
    arg_match = _ARG_HEADER_RE.search(text)
    if not arg_match:
        return text, {}
    summary = text[:arg_match.start()].rstrip()
    body = text[arg_match.end():]
    end = _RETURNS_HEADER_RE.search(body)
    if end:
        body = body[:end.start()]

    descriptions: dict[str, str] = {}
    current: str | None = None
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^([A-Za-z_][\w]*)(?:\s*\([^)]+\))?\s*:\s*(.*)", stripped)
        if m and not line.startswith(" " * 12):
            current = m.group(1)
            descriptions[current] = m.group(2)
        elif current:
            descriptions[current] = f"{descriptions[current]} {stripped}".strip()
    return summary, descriptions


def _infer_parameters(
    fn: Callable[..., Any],
    arg_docs: dict[str, str],
) -> dict[str, Any]:
    """Build a JSON schema from a function's annotations + docstring."""
    hints = typing.get_type_hints(fn)
    hints.pop("return", None)
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in {"self", "cls"} or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        schema = _schema_for_type(hints.get(name, str))
        if desc := arg_docs.get(name):
            schema["description"] = desc
        if param.default is inspect.Parameter.empty:
            required.append(name)
        elif "default" not in schema:
            schema["default"] = param.default
        properties[name] = schema
    out: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        out["required"] = required
    return out


def tool(
    fn: ToolFn | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Tool | Callable[[ToolFn], Tool]:
    """Decorator that wraps an async function as a :class:`Tool`.

    Usage::

        @tool
        async def web_search(query: str) -> str:
            ...

        @tool(name="run_python", description="Run code in a sandbox.")
        async def _runner(code: str) -> str:
            ...

    When ``parameters`` is omitted, a JSON schema is inferred from the
    function's type hints + Google-style docstring.
    """

    def _wrap(f: ToolFn) -> Tool:
        summary, arg_docs = _parse_docstring(f.__doc__ or "")
        return Tool(
            name=name or f.__name__,
            description=description or summary or "",
            parameters=parameters or _infer_parameters(f, arg_docs),
            func=f,
        )

    return _wrap(fn) if fn is not None else _wrap


__all__ = ["Tool", "ToolFn", "tool"]
