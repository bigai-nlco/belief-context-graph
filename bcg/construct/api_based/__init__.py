"""
bcg.construct.api_based (v3)
============================
Belief-graph construction backend that uses one large API-based chat model to
extract belief/decision nodes, relations, entities, and stance directly (a
single LLM call per turn). See the sibling ``construct.light`` package for
the alternative embedding + small local-model backend.

Streaming belief-graph construction from role-tagged turns (user / assistant /
tool). No scenarios, no sessions: every turn is processed by role in one
LLM call that emits new belief nodes + new forward edges incrementally; the
backward (evaluation) pass and merge/dedup run once at trajectory end.

``__version__`` is the package version (pyproject metadata is the single
source); the historical "3.0.0" backend lineage is kept in this docstring.
"""

from bcg import _package_version as _bcg_version

__version__ = _bcg_version()

from .graph import BeliefGraph
from .pipeline import (
    BeliefGraphOptions,
    BeliefGraphPipeline,
    BeliefGraphRunPaths,
    BeliefGraphRunResult,
)
from .stream import StreamingBeliefBuilder, StreamOptions

__all__ = [
    "BeliefGraph",
    "BeliefGraphOptions",
    "BeliefGraphPipeline",
    "BeliefGraphRunPaths",
    "BeliefGraphRunResult",
    "StreamingBeliefBuilder",
    "StreamOptions",
    "__version__",
]
