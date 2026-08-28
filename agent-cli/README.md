# BCG Agent

Interactive terminal Agent for
[Belief Context Graph](https://github.com/bigai-nlco/belief-context-graph).

The public command is `bcg`, provided by the Python package. This Node package
installs the internal `bcg-agent` runtime used by that launcher.

The runtime supports five session-level context modes. Every bounded mode
permanently pins the initial user input and retains a configurable number of
recent completed turns:

- **Default** keeps the full conversation with automatic compaction.
- **Recent-Only** drops older turns and leaves the system prompt unchanged.
- **RAG** stores dropped turns in a session-local SQLite FTS5 database, queries
  it with the recent raw turns, and injects retrieved history into the system
  prompt.
- **Summary** compresses dropped turns into one rolling LLM summary.
- **BCG** converts dropped turns into a confidence-aware belief graph.

Configuration and sessions live under `~/.bcg/agent/`. API credentials can be
entered with `/login`; custom OpenAI-compatible endpoints use
`~/.bcg/agent/models.json`. When launched through `bcg`, root `.env` values are
translated into this configuration without copying the actual API key.
