"""
bcg.construct
==============
Belief Context Graph construction. Two independent backends live side by
side; pick ONE per run:

    bcg.construct.hybrid    — local embeddings + small generative model
                                (semantic chunking, local stance classifier,
                                local NER, small non-thinking relation model).
    bcg.construct.unified   — one general-purpose graph LLM extracts nodes,
                              stance, and entities, followed by bounded
                              relation-window calls.

Both expose the same shape of entry points (``pipeline.run_input`` /
``pipeline.run_item`` for batch use, ``online.SessionManager`` for the HTTP
server, and a ``StreamingBeliefBuilder`` / ``StreamOptions`` pair), so
``bcg/run.py`` and ``bcg/online_server.py`` can dispatch to whichever one the
user selects (see their ``--help`` for the ``hybrid`` / ``unified``
subcommands) without the two implementations needing to share internals.

This package intentionally does NOT re-export both backends' symbols at this
level (their ``StreamOptions``/``BeliefGraph`` classes are shaped differently
and are not meant to be used interchangeably) — import the specific backend
you want, e.g.::

    from bcg.construct.hybrid import StreamingBeliefBuilder, StreamOptions
    from bcg.construct.unified import StreamingBeliefBuilder, StreamOptions
"""

from __future__ import annotations

__all__ = ["hybrid", "unified"]
