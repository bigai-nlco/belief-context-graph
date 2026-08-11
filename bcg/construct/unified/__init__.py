"""
bcg.construct.unified (v3)
============================
Belief-graph construction backend that uses one general-purpose graph LLM to
extract belief/decision nodes, entities, and stance in one model call per
turn, followed by bounded relation-window calls. See the sibling
``construct.hybrid`` package for the alternative modular pipeline.

Streaming belief-graph construction from role-tagged turns (user / assistant /
tool). No scenarios, no sessions: every turn is processed by role in one
node-extraction call followed by bounded relation linking; merge/dedup runs
incrementally as turns arrive.

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
