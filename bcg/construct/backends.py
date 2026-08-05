"""Lazy backend registry and the common construct adapter implementation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, Callable

from bcg.core.client_adapter import ConstructClientAdapter
from bcg.core.contracts import (
    ConstructBackend,
    ConstructSession,
    RunOptions,
    SessionSpec,
)

OptionsBuilder = Callable[[RunOptions, dict[str, Any] | None], Any]
SessionOptionsBuilder = Callable[[Any], Any]
OptionsSerializer = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class SessionBackendAdapter:
    """Declarative adapter for backends sharing the streaming session API."""

    name: str
    session_cls: type
    llm_module: ModuleType
    options_builder: OptionsBuilder
    session_options_builder: SessionOptionsBuilder
    options_serializer: OptionsSerializer

    def build_options(
        self,
        options: RunOptions,
        *,
        belief_graph_config: dict[str, Any] | None,
    ) -> Any:
        return self.options_builder(options, belief_graph_config)

    def session_options(self, options: Any) -> Any:
        return self.session_options_builder(options)

    def create_session(self, spec: SessionSpec) -> ConstructSession:
        return self.session_cls(
            spec.run_id,
            client=ConstructClientAdapter(spec.llm, self.llm_module),
            model=spec.model,
            output_root=spec.output_root,
            options=spec.options,
            embedder=spec.embedder,
            max_tokens=spec.max_tokens,
            item_meta=spec.item_meta,
            extra_meta=spec.extra_meta,
        )

    def finalize(self, session: ConstructSession) -> dict[str, Any]:
        return session.finalize()

    def result(self, session: ConstructSession) -> dict[str, Any]:
        return dict(session.result or {})

    def serialize_options(self, options: Any) -> dict[str, Any]:
        return self.options_serializer(options)


_BACKEND_MODULES = {
    "api_based": "bcg.construct.api_based.adapter",
    "light": "bcg.construct.light.adapter",
}
_BACKENDS: dict[str, ConstructBackend] = {}


def resolve_backend(name: str) -> ConstructBackend:
    """Resolve a backend on first use without importing concrete code here."""

    if name in _BACKENDS:
        return _BACKENDS[name]
    try:
        module_name = _BACKEND_MODULES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown backend {name!r}; choose one of: {', '.join(_BACKEND_MODULES)}"
        ) from exc
    backend = getattr(import_module(module_name), "BACKEND")
    if not isinstance(backend, ConstructBackend) or backend.name != name:
        raise TypeError(f"{module_name}.BACKEND does not implement ConstructBackend")
    _BACKENDS[name] = backend
    return backend


__all__ = ["SessionBackendAdapter", "resolve_backend"]
