"""
construct_beliefs (v3)
======================
Streaming belief-graph construction from role-tagged turns (user / assistant /
tool). No scenarios, no sessions: every turn is processed by role in one
LLM call that emits new belief nodes + new forward edges incrementally; the
backward (evaluation) pass and merge/dedup run once at trajectory end. See
README.md for the full documentation.
"""

__version__ = "3.0.0"

from .factor import Factor
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
    "Factor",
    "StreamingBeliefBuilder",
    "StreamOptions",
    "__version__",
]
