---
title: "Environment Variables"
description: "Credentials, model routing, graph service, context policy, and tracing."
icon: "key"
---

Common variables include:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Default model credential |
| `OPENAI_BASE_URL` | OpenAI-compatible base URL |
| `OPENAI_MODEL` | Default model |
| `MODEL` | Fallback model name |
| `BCG_HOME` | Override global BCG state root |
| `BELIEF_GRAPH_URL` | Graph Construction server used by the Agent |
| `BCG_RECENT_TURNS` | Raw completed-turn window in BCG mode |
| `BCG_AGENT_MODEL` | Benchmark/reference Agent model override |
| `SERPER_API_KEY` | Search access for supported benchmarks |
| Langfuse variables | Optional remote tracing |

## Project `.env`

BCG searches for a project environment file and loads values without executing shell syntax.

```bash
cp .env.example .env
```

Configuration JSON should reference environment variable names rather than contain secret values.

## Global configuration

`bcg setup` stores:

```text
~/.bcg/config.json
~/.bcg/.env
~/.bcg/model_config.json
```

Credential files are written with restricted permissions where supported.
