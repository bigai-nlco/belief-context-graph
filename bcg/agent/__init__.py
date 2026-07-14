"""BeliefTracer: agent workflow rollout utilities."""

from __future__ import annotations

from bcg.agent.warnings_compat import suppress_known_warnings

suppress_known_warnings()

__version__ = "0.1.0"

try:
    from bcg.agent.workflow import BeliefTracerWorkflow
except ModuleNotFoundError:
    BeliefTracerWorkflow = None
