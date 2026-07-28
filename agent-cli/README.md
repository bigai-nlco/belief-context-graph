# BCG Agent

Interactive terminal Agent for
[Belief Context Graph](https://github.com/bigai-nlco/belief-context-graph).

The public command is `bcg`, provided by the Python package. This Node package
installs the internal `bcg-agent` runtime used by that launcher.

The runtime always keeps the initial user input and the latest two completed
turns as raw messages. Older evicted messages are sent to the Graph
Construction service, and its Markdown belief graph is injected into the
system prompt.

Configuration and sessions live under `~/.bcg/agent/`. API credentials can be
entered with `/login`; custom OpenAI-compatible endpoints use
`~/.bcg/agent/models.json`. When launched through `bcg`, root `.env` values are
translated into this configuration without copying the actual API key.

This package contains a modified distribution of the MIT-licensed Pi coding
agent. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSE](LICENSE).
