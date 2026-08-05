"""Stable contracts between SDK orchestration and construct backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from bcg.graph import BCG


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Backend-neutral options accepted by the public runner."""

    evidence_mode: str = "sentence"
    incremental_merge: bool = True
    incremental_merge_threshold: float = 0.8
    verify_merge: bool = False
    context_chars: int = 9000
    io_context_chars: int = 6000
    min_content_len: int = 0


@dataclass(frozen=True, slots=True)
class Artifact:
    """One named file produced by a belief run."""

    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class RunPaths:
    """SDK paths alongside a construct backend's native artifacts."""

    run_dir: Path
    artifacts_dir: Path
    graph: Path
    memory: Path
    token_usage: Path
    events: Path
    segments: Path
    io_beliefs: Path
    reasoning_beliefs: Path
    forward_relations: Path
    backward_relations: Path
    merges: Path
    result: Path
    final_graph: Path
    trajectory: Path
    graph_stream: Path

    def to_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in self.__dataclass_fields__}

    def artifacts(self) -> tuple[Artifact, ...]:
        """Return file outputs while excluding the two directory fields."""

        return tuple(
            Artifact(name, getattr(self, name))
            for name in self.__dataclass_fields__
            if name not in {"run_dir", "artifacts_dir"}
        )


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    graph: BCG
    memory: dict[str, Any]
    output_paths: RunPaths
    token_usage: dict[str, Any]
    counts: dict[str, Any]
    construct_result: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """Inputs required by a backend to create one construct session."""

    run_id: str
    llm: Any
    model: str
    output_root: Path
    options: Any
    embedder: Any | None
    max_tokens: int | None
    item_meta: dict[str, Any]
    extra_meta: dict[str, Any]


class ConstructSession(Protocol):
    result: dict[str, Any] | None

    def push(self, turn: dict[str, Any]) -> dict[str, Any]: ...

    def finalize(self) -> dict[str, Any]: ...


@runtime_checkable
class ConstructBackend(Protocol):
    """Operations the runner needs from a concrete construct backend."""

    name: str

    def build_options(
        self,
        options: RunOptions,
        *,
        belief_graph_config: dict[str, Any] | None,
    ) -> Any: ...

    def session_options(self, options: Any) -> Any: ...

    def create_session(self, spec: SessionSpec) -> ConstructSession: ...

    def finalize(self, session: ConstructSession) -> dict[str, Any]: ...

    def result(self, session: ConstructSession) -> dict[str, Any]: ...

    def serialize_options(self, options: Any) -> dict[str, Any]: ...


# Preserve the original public SDK type names during the migration window.
BeliefGraphRunPaths = RunPaths
BeliefGraphRunResult = RunResult


__all__ = [
    "Artifact",
    "BeliefGraphRunPaths",
    "BeliefGraphRunResult",
    "ConstructBackend",
    "ConstructSession",
    "RunOptions",
    "RunPaths",
    "RunResult",
    "SessionSpec",
]
