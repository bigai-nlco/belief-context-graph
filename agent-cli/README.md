# BCG Agent

Interactive terminal Agent for
[Belief Context Graph](https://github.com/bigai-nlco/belief-context-graph).

The public command is `bcg`, provided by the Python package. This Node package
installs the internal `bcg-agent` runtime used by that launcher.

The runtime supports three session-level context modes. Default retains normal
full context with compaction. BCG and Summary both pin the initial user input,
retain a configurable number of recent completed turns, and evict older turns
in the same batches. BCG converts those batches into a belief graph; Summary
updates one rolling LLM summary. Either memory block is injected into the
system prompt.

Configuration and sessions live under `~/.bcg/agent/`. API credentials can be
entered with `/login`; custom OpenAI-compatible endpoints use
`~/.bcg/agent/models.json`. When launched through `bcg`, root `.env` values are
translated into this configuration without copying the actual API key.
