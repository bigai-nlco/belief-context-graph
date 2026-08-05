"""Unified exception hierarchy for BCG (step 8).

New SDK/backend code should raise these instead of bare exceptions. Legacy
code migrates incrementally; subclasses keep standard-exception bases so
existing ``except ValueError`` / ``except RuntimeError`` handlers keep
working.
"""

from __future__ import annotations


class BCGError(Exception):
    """Base class for all BCG-raised errors."""


class BCGConfigError(BCGError, ValueError):
    """Invalid or unsupported configuration (schema, migration, env keys)."""


class BCGUsageError(BCGError, RuntimeError):
    """Invalid API usage / invalid state transitions."""


class BCGArtifactError(BCGError, RuntimeError):
    """Corrupt or unsupported artifacts (graphs, memory documents, runs)."""


class BCGBackendError(BCGError, RuntimeError):
    """Construct backend failures (model calls, merges, sessions)."""


__all__ = [
    "BCGArtifactError",
    "BCGBackendError",
    "BCGConfigError",
    "BCGError",
    "BCGUsageError",
]
