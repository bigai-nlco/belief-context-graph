"""Core contracts shared by the SDK and construct backends."""

from bcg.core.client_adapter import ConstructClientAdapter
from bcg.core.contracts import (
    Artifact,
    BeliefGraphRunPaths,
    BeliefGraphRunResult,
    ConstructBackend,
    ConstructSession,
    RunOptions,
    RunPaths,
    RunResult,
    SessionSpec,
)

__all__ = [
    "Artifact",
    "BeliefGraphRunPaths",
    "BeliefGraphRunResult",
    "ConstructBackend",
    "ConstructClientAdapter",
    "ConstructSession",
    "RunOptions",
    "RunPaths",
    "RunResult",
    "SessionSpec",
]
