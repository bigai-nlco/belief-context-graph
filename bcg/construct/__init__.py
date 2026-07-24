"""
bcg.construct
==============
Belief Context Graph construction. Two independent backends live side by
side; pick ONE per run:

    bcg.construct.light      — local embeddings + small generative model
                                (semantic chunking, local stance classifier,
                                local NER, small non-thinking relation model).
    bcg.construct.api_based   — one large API-based chat model does
                                extraction + stance + entities + relations in
                                a single call per turn, plus a Factor
                                abstraction for reusable mechanism notes.

Both expose the same shape of entry points (``pipeline.run_input`` /
``pipeline.run_item`` for batch use, ``online.SessionManager`` for the HTTP
server, and a ``StreamingBeliefBuilder`` / ``StreamOptions`` pair), so
``bcg/run.py`` and ``bcg/online_server.py`` can dispatch to whichever one the
user selects (see their ``--help`` for the ``light`` / ``api_based``
subcommands) without the two implementations needing to share internals.

This package intentionally does NOT re-export both backends' symbols at this
level (their ``StreamOptions``/``BeliefGraph`` classes are shaped differently
and are not meant to be used interchangeably) — import the specific backend
you want, e.g.::

    from bcg.construct.light import StreamingBeliefBuilder, StreamOptions
    from bcg.construct.api_based import StreamingBeliefBuilder, StreamOptions
"""

from __future__ import annotations

__all__ = ["light", "api_based"]
